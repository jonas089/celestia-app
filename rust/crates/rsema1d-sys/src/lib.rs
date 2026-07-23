//! Reverse-FFI Rust bindings to the **real Go** `rsema1d` multilinear PCS.
//!
//! Rust calls the unchanged Go prover/verifier through a c-shared library
//! (`librsema1d.dylib`, built from `pkg/rsema1d/cshim`). Because commitments are
//! produced by the exact Go code the celestia-app DA layer runs — including the
//! Leopard GF(2^16) encoder — they are byte-identical, with no risky Rust
//! reimplementation.
//!
//! # Memory / cgo pointer rules
//! No Go pointer is retained on the Rust side. Inputs are copied into Go memory
//! on entry; outputs are copied into Rust-owned buffers. The variable-length
//! proof buffer is `malloc`'d by Go and freed via `rsema1d_free_buf` as soon as
//! it has been copied into a `Vec<u8>`. A committed square lives on the Go heap,
//! referenced by an opaque [`Handle`]; dropping the handle releases it.

use std::os::raw::c_int;

/// A GF128 element serialized as 16 little-endian bytes (`field.EncodeGF128`).
pub const GF128_SIZE: usize = 16;
/// A commitment is `SHA256(rowRoot || rlcRoot)` — 32 bytes.
pub const COMMITMENT_SIZE: usize = 32;

// Raw C ABI exported by librsema1d.dylib (see pkg/rsema1d/cshim/cshim.go).
extern "C" {
    fn rsema1d_commit(
        k: u32,
        n: u32,
        rows: *const u8,
        row_len: usize,
        num_rows: usize,
        out_commitment: *mut u8,
        out_handle: *mut u64,
    ) -> c_int;

    fn rsema1d_open_at(
        handle: u64,
        range_start: u32,
        range_len: u32,
        point: *const u8,
        point_len: usize,
        sample_count: u32,
        out_proof: *mut *mut u8,
        out_proof_len: *mut usize,
    ) -> c_int;

    fn rsema1d_verify_at(
        k: u32,
        n: u32,
        commitment: *const u8,
        proof: *const u8,
        proof_len: usize,
        point: *const u8,
        point_len: usize,
        out_value: *mut u8,
    ) -> c_int;

    fn rsema1d_open_at_full(
        handle: u64,
        range_start: u32,
        range_len: u32,
        rcol: *const u8,
        rcol_len: usize,
        rrow: *const u8,
        rrow_len: usize,
        sample_count: u32,
        out_proof: *mut *mut u8,
        out_proof_len: *mut usize,
    ) -> c_int;

    fn rsema1d_verify_at_full(
        k: u32,
        n: u32,
        commitment: *const u8,
        proof: *const u8,
        proof_len: usize,
        rcol: *const u8,
        rcol_len: usize,
        rrow: *const u8,
        rrow_len: usize,
        out_value: *mut u8,
    ) -> c_int;

    fn rsema1d_free_handle(handle: u64);
    fn rsema1d_free_buf(buf: *mut u8);

    fn rsema1d_encode_extended(
        k: u32,
        n: u32,
        rows: *const u8,
        row_len: usize,
        num_rows: usize,
        out_commitment: *mut u8,
        out_extended: *mut *mut u8,
        out_extended_len: *mut usize,
    ) -> c_int;

    fn rsema1d_load_extended(
        extended: *const u8,
        extended_len: usize,
        out_commitment: *mut u8,
        out_handle: *mut u64,
    ) -> c_int;

    fn rsema1d_encode_call_count() -> u64;
}

/// Number of Reed-Solomon encodes (`Coder.Encode`) executed by the linked Go
/// `librsema1d` in THIS process so far, across [`commit`] and [`encode_extended`].
///
/// The cross-process hand-off invariant is machine-checkable with this: a prover
/// process that only ever calls [`load_extended`] observes `0` — it consumed a
/// DA-side encoding and did no polynomial encoding of its own.
pub fn encode_call_count() -> u64 {
    // SAFETY: no arguments; reads a process-global atomic on the Go side.
    unsafe { rsema1d_encode_call_count() }
}

/// **DA-encoder step.** RS-encodes the `k + n` row matrix ONCE with the original
/// `Coder.Encode` (byte-identical to celestia-app DA sampling) and returns the
/// 32-byte commitment together with the SERIALIZED extended matrix (originals +
/// genuine RS parity, self-describing: `k, n, rowLen` header + rows). The
/// serialized bytes are what crosses a process boundary — no Go handle does.
/// Pass them to [`load_extended`] in the prover process.
pub fn encode_extended(
    k: u32,
    n: u32,
    rows: &[&[u8]],
) -> Result<([u8; COMMITMENT_SIZE], Vec<u8>), FfiError> {
    let num_rows = rows.len();
    let row_len = rows.first().map_or(0, |r| r.len());
    let mut flat = Vec::with_capacity(row_len * num_rows);
    for r in rows {
        flat.extend_from_slice(r);
    }

    let mut commitment = [0u8; COMMITMENT_SIZE];
    let mut out_extended: *mut u8 = std::ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: pointers reference live Rust buffers with the stated lengths; the
    // shim copies out of them and writes a malloc'd blob into out_extended.
    let rc = unsafe {
        rsema1d_encode_extended(
            k,
            n,
            flat.as_ptr(),
            row_len,
            num_rows,
            commitment.as_mut_ptr(),
            &mut out_extended as *mut *mut u8,
            &mut out_len as *mut usize,
        )
    };
    if rc != 0 {
        return Err(FfiError { func: "rsema1d_encode_extended", code: rc });
    }
    // SAFETY: out_extended points to out_len bytes malloc'd by the shim.
    let extended = unsafe { std::slice::from_raw_parts(out_extended, out_len).to_vec() };
    unsafe { rsema1d_free_buf(out_extended) };
    Ok((commitment, extended))
}

/// **Prover step.** Reconstructs the committed square from the serialized
/// extended matrix produced by [`encode_extended`], WITHOUT re-encoding (only the
/// Merkle/RLC commitment structures are rebuilt Go-side; [`encode_call_count`] is
/// NOT incremented). Returns the recomputed 32-byte commitment (identical to the
/// DA side's) and a live [`Handle`] to the reconstructed square, ready for
/// [`open_at_full`]. This is the honest hand-off: the RS-encoding cost was paid
/// once, on the DA side.
pub fn load_extended(extended: &[u8]) -> Result<([u8; COMMITMENT_SIZE], Handle), FfiError> {
    let mut commitment = [0u8; COMMITMENT_SIZE];
    let mut handle: u64 = 0;
    // SAFETY: extended is a valid slice; the shim copies out of it and writes the
    // commitment + handle in place.
    let rc = unsafe {
        rsema1d_load_extended(
            extended.as_ptr(),
            extended.len(),
            commitment.as_mut_ptr(),
            &mut handle as *mut u64,
        )
    };
    if rc != 0 {
        return Err(FfiError { func: "rsema1d_load_extended", code: rc });
    }
    Ok((commitment, Handle(handle)))
}

/// A row range within the shared square: rows `[start, start + len)`. `len` must
/// be a positive power of two and `start` aligned to it (enforced Go-side).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowRange {
    pub start: u32,
    pub len: u32,
}

impl RowRange {
    pub fn new(start: u32, len: u32) -> Self {
        Self { start, len }
    }
}

/// Error returned when the Go FFI layer rejects a call. The code is the raw
/// return value documented on the corresponding `rsema1d_*` function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FfiError {
    pub func: &'static str,
    pub code: i32,
}

impl std::fmt::Display for FfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} returned error code {}", self.func, self.code)
    }
}

impl std::error::Error for FfiError {}

/// An opaque handle to a committed square held on the Go heap. Dropping it
/// releases the underlying `StructuredCommitment` from the Go registry.
#[derive(Debug)]
pub struct Handle(u64);

impl Handle {
    /// The raw registry key (for debugging / logging only).
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: self.0 was produced by rsema1d_commit and not yet freed.
        unsafe { rsema1d_free_handle(self.0) }
    }
}

/// Commits the `k + n` row matrix with the ORIGINAL Go `Coder.Encode` (legacy
/// DeriveCoefficients RLC — the canonical DA/spec commitment) and returns the
/// 32-byte commitment together with a [`Handle`] to the committed square (kept
/// alive on the Go side for subsequent openings). The commitment is
/// byte-identical to celestia-app DA sampling and `cmd/testvectors`.
///
/// `rows` must contain exactly `k + n` equal-length rows: the `k` original rows
/// followed by `n` parity rows (zeroed), matching `Encode`'s contract.
pub fn commit(k: u32, n: u32, rows: &[&[u8]]) -> Result<([u8; COMMITMENT_SIZE], Handle), FfiError> {
    let num_rows = rows.len();
    let row_len = rows.first().map_or(0, |r| r.len());
    // Flatten into one contiguous buffer (copied into Go memory by the shim).
    let mut flat = Vec::with_capacity(row_len * num_rows);
    for r in rows {
        flat.extend_from_slice(r);
    }

    let mut commitment = [0u8; COMMITMENT_SIZE];
    let mut handle: u64 = 0;
    // SAFETY: pointers reference live Rust buffers with the stated lengths; the
    // shim copies out of them and never retains them.
    let rc = unsafe {
        rsema1d_commit(
            k,
            n,
            flat.as_ptr(),
            row_len,
            num_rows,
            commitment.as_mut_ptr(),
            &mut handle as *mut u64,
        )
    };
    if rc != 0 {
        return Err(FfiError { func: "rsema1d_commit", code: rc });
    }
    Ok((commitment, Handle(handle)))
}

/// Opens the subset evaluation for `range` at `point`, sampling `sample_count`
/// rows for the proximity check, and returns the serialized `EvalProof` bytes.
///
/// `point` is the little-endian concatenation of `log2(range.len)` GF128
/// elements (`16 * log2(range.len)` bytes).
///
/// Note: `sample_count` is an explicit parameter here (the Go prover needs it);
/// callers typically pass a small fixed count such as 8.
pub fn open_at(
    handle: &Handle,
    range: RowRange,
    point: &[u8],
    sample_count: u32,
) -> Result<Vec<u8>, FfiError> {
    let mut out_proof: *mut u8 = std::ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: handle.0 is a live registry key; point is a valid slice; the shim
    // writes a malloc'd buffer into out_proof / out_len.
    let rc = unsafe {
        rsema1d_open_at(
            handle.0,
            range.start,
            range.len,
            point.as_ptr(),
            point.len(),
            sample_count,
            &mut out_proof as *mut *mut u8,
            &mut out_len as *mut usize,
        )
    };
    if rc != 0 {
        return Err(FfiError { func: "rsema1d_open_at", code: rc });
    }

    // Copy the Go-owned buffer into a Rust Vec, then free it immediately.
    // SAFETY: out_proof points to out_len bytes malloc'd by the shim.
    let proof = unsafe { std::slice::from_raw_parts(out_proof, out_len).to_vec() };
    unsafe { rsema1d_free_buf(out_proof) };
    Ok(proof)
}

/// Opens the subset evaluation for `range` at the FULL point (`rcol`, `rrow`),
/// spanning both the column and row variables of the committed square, sampling
/// `sample_count` rows for the proximity check, and returns the serialized
/// `EvalProofFull` bytes.
///
/// `rcol` is the little-endian concatenation of `log2(numSymbols)` GF128 elements
/// (the column point); `rrow` is `log2(range.len)` GF128 elements (the row
/// point). Each element is 16 bytes.
///
/// This is the opening Expander's GKR needs: the transcript fixes the entire
/// evaluation point, not just the row axis.
pub fn open_at_full(
    handle: &Handle,
    range: RowRange,
    rcol: &[u8],
    rrow: &[u8],
    sample_count: u32,
) -> Result<Vec<u8>, FfiError> {
    let mut out_proof: *mut u8 = std::ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: handle.0 is a live registry key; rcol/rrow are valid slices; the
    // shim writes a malloc'd buffer into out_proof / out_len.
    let rc = unsafe {
        rsema1d_open_at_full(
            handle.0,
            range.start,
            range.len,
            rcol.as_ptr(),
            rcol.len(),
            rrow.as_ptr(),
            rrow.len(),
            sample_count,
            &mut out_proof as *mut *mut u8,
            &mut out_len as *mut usize,
        )
    };
    if rc != 0 {
        return Err(FfiError { func: "rsema1d_open_at_full", code: rc });
    }
    // SAFETY: out_proof points to out_len bytes malloc'd by the shim.
    let proof = unsafe { std::slice::from_raw_parts(out_proof, out_len).to_vec() };
    unsafe { rsema1d_free_buf(out_proof) };
    Ok(proof)
}

/// Verifies a serialized `EvalProofFull` against `commitment` at the full point
/// (`rcol`, `rrow`). Returns `Some(value)` (the 16-byte GF128 evaluation) on
/// success, or `None` when the Go verifier rejects the proof.
pub fn verify_at_full(
    k: u32,
    n: u32,
    commitment: &[u8; COMMITMENT_SIZE],
    proof: &[u8],
    rcol: &[u8],
    rrow: &[u8],
) -> Option<[u8; GF128_SIZE]> {
    let mut value = [0u8; GF128_SIZE];
    // SAFETY: all pointers reference live Rust buffers with the given lengths;
    // the shim copies out of them and writes value in place.
    let rc = unsafe {
        rsema1d_verify_at_full(
            k,
            n,
            commitment.as_ptr(),
            proof.as_ptr(),
            proof.len(),
            rcol.as_ptr(),
            rcol.len(),
            rrow.as_ptr(),
            rrow.len(),
            value.as_mut_ptr(),
        )
    };
    if rc == 0 {
        Some(value)
    } else {
        None
    }
}

/// Verifies a serialized `EvalProof` against `commitment` at `point`. Returns
/// `Some(value)` (the 16-byte GF128 evaluation) when verification succeeds, or
/// `None` when the Go verifier rejects the proof.
pub fn verify_at(
    k: u32,
    n: u32,
    commitment: &[u8; COMMITMENT_SIZE],
    proof: &[u8],
    point: &[u8],
) -> Option<[u8; GF128_SIZE]> {
    let mut value = [0u8; GF128_SIZE];
    // SAFETY: all pointers reference live Rust buffers with the given lengths;
    // the shim copies out of them and writes value in place.
    let rc = unsafe {
        rsema1d_verify_at(
            k,
            n,
            commitment.as_ptr(),
            proof.as_ptr(),
            proof.len(),
            point.as_ptr(),
            point.len(),
            value.as_mut_ptr(),
        )
    };
    if rc == 0 {
        Some(value)
    } else {
        None
    }
}
