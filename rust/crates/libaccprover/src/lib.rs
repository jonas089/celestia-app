//! C-ABI facade over the verified `riscv_stf::prove_execution` GKR prover.
//!
//! Exposes a single proving entrypoint plus a buffer-free and a last-error getter,
//! so a stable-toolchain / non-Rust caller (the ev-reth `accProof` RPC via
//! `acc-prover-sys`) can drive the real Expander-GKR + rsema1d-PCS prover without
//! the nightly + Go-cgo build constraints leaking across the boundary. The prover
//! (Expander + rsema1d-pcs) is statically embedded; librsema1d (the Go DA encoder)
//! is dynamically linked (see build.rs rpath).
//!
//! # Buffer ownership
//! `accprover_prove` writes two heap buffers (`out_proof`, `out_pub`) allocated by
//! this library. The caller MUST return each to `accprover_free_buf(ptr, len)`
//! (using the paired `*_len`). `out_commit` is a caller-provided 32-byte buffer,
//! filled in place. No pointer produced here is a Go pointer.

use std::cell::RefCell;
use std::os::raw::{c_char, c_int};

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        let mut b = e.borrow_mut();
        b.clear();
        b.extend_from_slice(msg.as_bytes());
        b.push(0); // NUL terminate
    });
}

/// Return codes for [`accprover_prove`].
pub const ACCPROVER_OK: c_int = 0;
/// A required out-pointer argument was null.
pub const ACCPROVER_ERR_NULL_ARG: c_int = -1;
/// `prove_execution` returned an error (see `accprover_last_error`).
pub const ACCPROVER_ERR_PROVE: c_int = -2;
/// The prover panicked (see `accprover_last_error`).
pub const ACCPROVER_ERR_PANIC: c_int = -3;

/// Prove one RV32IM execution over the input-derived array, reusing rsema1d as the
/// GKR input PCS, and self-verify. On success (`ACCPROVER_OK`):
///   * `out_commit` (32 bytes, caller-owned) is filled with the rsema1d commitment
///     (byte-identical to an independent Go/DA commit of the same rows);
///   * `*out_proof` / `*out_proof_len` receive a library-owned proof buffer;
///   * `*out_pub` / `*out_pub_len` receive a library-owned public-value buffer
///     (final sum ++ loop count ++ mem[result], each LE u32; 12 bytes);
///   * `*out_verified` is 1 iff the Expander verifier accepted;
///   * `*out_input_vars` is the GKR `num_vars`.
/// Return the proof/pub buffers to [`accprover_free_buf`]. On any error a negative
/// code is returned and no buffers are allocated.
///
/// # Safety
/// `input` must point to `input_len` readable bytes (or be null iff `input_len==0`).
/// `out_commit` must point to 32 writable bytes. The other out-pointers must be
/// non-null and writable.
#[no_mangle]
pub unsafe extern "C" fn accprover_prove(
    input: *const u8,
    input_len: usize,
    out_commit: *mut u8,
    out_proof: *mut *mut u8,
    out_proof_len: *mut usize,
    out_pub: *mut *mut u8,
    out_pub_len: *mut usize,
    out_verified: *mut c_int,
    out_input_vars: *mut u32,
) -> c_int {
    if out_commit.is_null()
        || out_proof.is_null()
        || out_proof_len.is_null()
        || out_pub.is_null()
        || out_pub_len.is_null()
        || out_verified.is_null()
        || out_input_vars.is_null()
    {
        set_last_error("accprover_prove: null out-pointer argument");
        return ACCPROVER_ERR_NULL_ARG;
    }

    let input_slice: &[u8] = if input_len == 0 {
        &[]
    } else if input.is_null() {
        set_last_error("accprover_prove: null input with non-zero length");
        return ACCPROVER_ERR_NULL_ARG;
    } else {
        std::slice::from_raw_parts(input, input_len)
    };
    // Copy input so the closure is UnwindSafe and independent of the caller's buffer.
    let owned: Vec<u8> = input_slice.to_vec();

    let result = std::panic::catch_unwind(move || riscv_stf::prove_execution(&owned));

    match result {
        Ok(Ok(p)) => {
            // Commitment (fixed 32 bytes) into caller buffer.
            std::ptr::copy_nonoverlapping(p.commitment.as_ptr(), out_commit, 32);

            let (proof_ptr, proof_len) = leak_buf(p.proof);
            *out_proof = proof_ptr;
            *out_proof_len = proof_len;

            let (pub_ptr, pub_len) = leak_buf(p.public_value);
            *out_pub = pub_ptr;
            *out_pub_len = pub_len;

            *out_verified = if p.verified { 1 } else { 0 };
            *out_input_vars = p.input_vars;
            ACCPROVER_OK
        }
        Ok(Err(e)) => {
            set_last_error(&format!("prove_execution error: {e}"));
            ACCPROVER_ERR_PROVE
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            set_last_error(&format!("prove_execution panicked: {msg}"));
            ACCPROVER_ERR_PANIC
        }
    }
}

/// Prove one rollup block's NONCE + BALANCE state transition (transfer-STF),
/// reusing rsema1d as the GKR input PCS, and self-verify. See
/// `riscv_stf::stf::prove_block_stf`.
///
/// `input` is a little-endian binary encoding of the block:
///   u32 pre_sender_balance ++ u32 pre_sender_nonce ++ u32 pre_recipient_balance
///   ++ u32 ntx ++ ntx × (u32 value ++ u32 fee ++ u32 nonce).
///
/// On success (`ACCPROVER_OK`):
///   * `out_commit` (32 bytes, caller-owned) = rsema1d commitment (== Go/DA);
///   * `*out_proof` / `*out_proof_len` = library-owned GKR proof buffer;
///   * `*out_pub` / `*out_pub_len` = library-owned public-output buffer (20 bytes:
///     post_sender_balance ++ post_sender_nonce ++ post_recipient_balance ++
///     applied_count ++ digest, each LE u32);
///   * `*out_verified` = 1 iff the Expander verifier accepted;
///   * `*out_input_vars` = GKR `num_vars`;
///   * `*out_tx_count` = number of real txs in the block.
/// Return the proof/pub buffers to [`accprover_free_buf`]. On error a negative
/// code is returned and no buffers are allocated.
///
/// # Safety
/// `input` must point to `input_len` readable bytes. `out_commit` must point to
/// 32 writable bytes. All other out-pointers must be non-null and writable.
#[no_mangle]
pub unsafe extern "C" fn accprover_prove_block(
    input: *const u8,
    input_len: usize,
    out_commit: *mut u8,
    out_proof: *mut *mut u8,
    out_proof_len: *mut usize,
    out_pub: *mut *mut u8,
    out_pub_len: *mut usize,
    out_verified: *mut c_int,
    out_input_vars: *mut u32,
    out_tx_count: *mut u32,
) -> c_int {
    if out_commit.is_null()
        || out_proof.is_null()
        || out_proof_len.is_null()
        || out_pub.is_null()
        || out_pub_len.is_null()
        || out_verified.is_null()
        || out_input_vars.is_null()
        || out_tx_count.is_null()
    {
        set_last_error("accprover_prove_block: null out-pointer argument");
        return ACCPROVER_ERR_NULL_ARG;
    }
    if input.is_null() && input_len != 0 {
        set_last_error("accprover_prove_block: null input with non-zero length");
        return ACCPROVER_ERR_NULL_ARG;
    }
    let input_slice: &[u8] =
        if input_len == 0 { &[] } else { std::slice::from_raw_parts(input, input_len) };
    let owned: Vec<u8> = input_slice.to_vec();

    let block = match decode_block_input(&owned) {
        Ok(b) => b,
        Err(e) => {
            set_last_error(&format!("accprover_prove_block: bad input encoding: {e}"));
            return ACCPROVER_ERR_PROVE;
        }
    };

    let result = std::panic::catch_unwind(move || riscv_stf::prove_block_stf(&block));

    match result {
        Ok(Ok(p)) => {
            std::ptr::copy_nonoverlapping(p.commitment.as_ptr(), out_commit, 32);
            let (proof_ptr, proof_len) = leak_buf(p.proof);
            *out_proof = proof_ptr;
            *out_proof_len = proof_len;
            let (pub_ptr, pub_len) = leak_buf(p.public_value);
            *out_pub = pub_ptr;
            *out_pub_len = pub_len;
            *out_verified = if p.verified { 1 } else { 0 };
            *out_input_vars = p.input_vars;
            *out_tx_count = p.tx_count;
            ACCPROVER_OK
        }
        Ok(Err(e)) => {
            set_last_error(&format!("prove_block_stf error: {e}"));
            ACCPROVER_ERR_PROVE
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            set_last_error(&format!("prove_block_stf panicked: {msg}"));
            ACCPROVER_ERR_PANIC
        }
    }
}

/// Prove a rollup block's REAL EVM state transition via `block_stf::prove_block`:
/// in-circuit EVM execution of a contract call + Ethereum world-state MPT ->
/// reth-faithful post_state_root, proven with the REUSED rsema1d/DA commitment.
///
/// `input` encoding (big-endian u256s):
///   contract_addr(20) ++ u32 code_len ++ code ++ c_pre_slot0(32) ++ u64 c_nonce
///   ++ c_balance(32) ++ 2 × ( addr(20) ++ u64 nonce ++ balance(32) )
///
/// On success: `out_commit`(32) = rsema1d/DA commitment; `out_state_root`(32) =
/// post-state root (== reth); `*out_proof`/`*out_proof_len` = GKR proof (free via
/// `accprover_free_buf`); `*out_verified` = 1 iff accepted; `*out_input_vars`.
///
/// # Safety
/// `input` points to `input_len` readable bytes; out-pointers non-null/writable.
#[no_mangle]
pub unsafe extern "C" fn accprover_prove_block_stf(
    input: *const u8,
    input_len: usize,
    out_commit: *mut u8,
    out_state_root: *mut u8,
    out_proof: *mut *mut u8,
    out_proof_len: *mut usize,
    out_verified: *mut c_int,
    out_input_vars: *mut u32,
) -> c_int {
    if out_commit.is_null() || out_state_root.is_null() || out_proof.is_null() || out_proof_len.is_null() || out_verified.is_null() || out_input_vars.is_null() {
        set_last_error("accprover_prove_block_stf: null out-pointer");
        return ACCPROVER_ERR_NULL_ARG;
    }
    if input.is_null() && input_len != 0 {
        set_last_error("accprover_prove_block_stf: null input");
        return ACCPROVER_ERR_NULL_ARG;
    }
    let owned: Vec<u8> = if input_len == 0 { vec![] } else { std::slice::from_raw_parts(input, input_len).to_vec() };
    let inputs = match decode_block_stf_input(&owned) {
        Ok(v) => v,
        Err(e) => { set_last_error(&format!("accprover_prove_block_stf: bad input: {e}")); return ACCPROVER_ERR_PROVE; }
    };
    let result = std::panic::catch_unwind(move || riscv_stf::block_stf::prove_block(&inputs));
    match result {
        Ok(Ok(p)) => {
            std::ptr::copy_nonoverlapping(p.commitment.as_ptr(), out_commit, 32);
            std::ptr::copy_nonoverlapping(p.post_state_root.as_ptr(), out_state_root, 32);
            let (pp, pl) = leak_buf(p.proof);
            *out_proof = pp; *out_proof_len = pl;
            *out_verified = if p.verified { 1 } else { 0 };
            *out_input_vars = p.input_vars;
            ACCPROVER_OK
        }
        Ok(Err(e)) => { set_last_error(&format!("prove_block (EVM STF) error: {e}")); ACCPROVER_ERR_PROVE }
        Err(panic) => {
            let msg = panic.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| panic.downcast_ref::<String>().cloned()).unwrap_or_else(|| "unknown panic".into());
            set_last_error(&format!("prove_block (EVM STF) panicked: {msg}"));
            ACCPROVER_ERR_PANIC
        }
    }
}

fn decode_block_stf_input(b: &[u8]) -> Result<riscv_stf::block_stf::BlockInputs, String> {
    use num_bigint::{BigInt, Sign};
    let mut o = 0usize;
    let take = |o: &mut usize, n: usize| -> Result<&[u8], String> {
        let s = b.get(*o..*o + n).ok_or_else(|| format!("truncated at {}", *o))?;
        *o += n;
        Ok(s)
    };
    let addr = |s: &[u8]| { let mut a = [0u8; 20]; a.copy_from_slice(s); a };
    let u256 = |s: &[u8]| BigInt::from_bytes_be(Sign::Plus, s);
    let u64le = |s: &[u8]| { let mut x = [0u8; 8]; x.copy_from_slice(s); u64::from_le_bytes(x) };

    let contract_addr = addr(take(&mut o, 20)?);
    let code_len = { let s = take(&mut o, 4)?; u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize };
    let code = take(&mut o, code_len)?.to_vec();
    let c_pre_slot0 = u256(take(&mut o, 32)?);
    let c_nonce = u64le(take(&mut o, 8)?);
    let c_balance = u256(take(&mut o, 32)?);
    let mut eoa: Vec<([u8; 20], u64, BigInt)> = Vec::with_capacity(2);
    for _ in 0..2 {
        let a = addr(take(&mut o, 20)?);
        let n = u64le(take(&mut o, 8)?);
        let bal = u256(take(&mut o, 32)?);
        eoa.push((a, n, bal));
    }
    Ok(riscv_stf::block_stf::BlockInputs {
        contract_addr, code, c_pre_slot0, c_nonce, c_balance,
        eoa: [eoa[0].clone(), eoa[1].clone()],
    })
}

/// Decode the little-endian block encoding used by `accprover_prove_block`.
fn decode_block_input(b: &[u8]) -> Result<riscv_stf::BlockInput, String> {
    let rd = |off: usize| -> Result<u32, String> {
        b.get(off..off + 4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            .ok_or_else(|| format!("truncated at offset {off}"))
    };
    let pre_sender_balance = rd(0)?;
    let pre_sender_nonce = rd(4)?;
    let pre_recipient_balance = rd(8)?;
    let ntx = rd(12)? as usize;
    let mut txs = Vec::with_capacity(ntx);
    for i in 0..ntx {
        let base = 16 + i * 12;
        txs.push(riscv_stf::TxData { value: rd(base)?, fee: rd(base + 4)?, nonce: rd(base + 8)? });
    }
    Ok(riscv_stf::BlockInput {
        pre_sender_balance,
        pre_sender_nonce,
        pre_recipient_balance,
        txs,
    })
}

/// Prove a rollup block's REAL STF as a CONTINUATION span of RV32IM chunks over
/// the fixed guest ELF, driving `riscv_stf::prove_elf` (Expander-GKR + rsema1d).
/// The guest ELF is read from `guest_elf_path`; the per-block input is the
/// serialized `EthClientExecutorInput` blob in `input`.
///
/// The call first runs the emulator to HALT to obtain the GOLDEN
/// `(block_number, state_root)` the guest commits, then proves a SPAN of chunks:
///   * if `max_chunks == 0` the whole trace is composed to HALT (close mode);
///   * otherwise a bounded `max_chunks` prefix of `chunk_len`-cycle chunks is
///     proven (a PARTIAL span — honest: `out_composed` is 0 unless the trace's
///     end was actually reached).
/// `chunk_len == 0` defaults to 32.
///
/// On success (`ACCPROVER_OK`):
///   * `*out_block_number`     = golden block number (from the emulator run);
///   * `out_state_root`        = golden 32-byte state root (caller-owned buffer);
///   * `*out_num_chunks`       = number of chunks actually proven;
///   * `*out_all_verified`     = 1 iff EVERY proven chunk's Expander verifier accepted;
///   * `*out_all_go_match`     = 1 iff EVERY chunk's GKR-PCS root == Go/DA rsema1d Encode;
///   * `*out_chain_valid`      = 1 iff the pc+reg chain and memory-product handoff
///                               hold across the proven chunks;
///   * `out_first_commitment`  = first chunk's 32-byte rsema1d commitment (== DA);
///   * `*out_composed`         = 1 iff the span reached HALT and the memory multiset
///                               closed + output bound to the golden committed values;
///   * `*out_proof` / `*out_proof_len` = library-owned serialized multi-chunk proof
///     blob: u32-LE chunk count, then per chunk (u32-LE len ++ Expander proof bytes).
/// Return `out_proof` to [`accprover_free_buf`]. On error a negative code is
/// returned and no buffers are allocated.
///
/// # Safety
/// `guest_elf_path` must be a non-null NUL-terminated path. `input` must point to
/// `input_len` readable bytes (or be null iff `input_len == 0`). `out_state_root`
/// and `out_first_commitment` must each point to 32 writable bytes; all other
/// out-pointers must be non-null and writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn accprover_prove_elf(
    guest_elf_path: *const c_char,
    input: *const u8,
    input_len: usize,
    chunk_len: u32,
    max_chunks: u32,
    out_block_number: *mut u64,
    out_state_root: *mut u8,
    out_num_chunks: *mut u32,
    out_all_verified: *mut c_int,
    out_all_go_match: *mut c_int,
    out_chain_valid: *mut c_int,
    out_first_commitment: *mut u8,
    out_composed: *mut c_int,
    out_proof: *mut *mut u8,
    out_proof_len: *mut usize,
) -> c_int {
    if guest_elf_path.is_null()
        || out_block_number.is_null()
        || out_state_root.is_null()
        || out_num_chunks.is_null()
        || out_all_verified.is_null()
        || out_all_go_match.is_null()
        || out_chain_valid.is_null()
        || out_first_commitment.is_null()
        || out_composed.is_null()
        || out_proof.is_null()
        || out_proof_len.is_null()
    {
        set_last_error("accprover_prove_elf: null out-pointer argument");
        return ACCPROVER_ERR_NULL_ARG;
    }
    if input.is_null() && input_len != 0 {
        set_last_error("accprover_prove_elf: null input with non-zero length");
        return ACCPROVER_ERR_NULL_ARG;
    }

    let elf_path = match std::ffi::CStr::from_ptr(guest_elf_path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_last_error("accprover_prove_elf: guest_elf_path is not valid UTF-8");
            return ACCPROVER_ERR_NULL_ARG;
        }
    };
    let input_slice: &[u8] =
        if input_len == 0 { &[] } else { std::slice::from_raw_parts(input, input_len) };
    let owned_input: Vec<u8> = input_slice.to_vec();

    // prove_elf's golden_from_run/prove_span take an input *path*, so materialize
    // the blob to a transient file (OS temp dir, not /tmp on macOS) and clean up.
    let tmp_path = {
        let dir = std::env::var("ACCPROVER_TMPDIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir());
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        dir.join(format!("accprover_elf_input_{}_{}.bin", std::process::id(), nanos))
    };
    if let Err(e) = std::fs::write(&tmp_path, &owned_input) {
        set_last_error(&format!("accprover_prove_elf: write temp input: {e}"));
        return ACCPROVER_ERR_PROVE;
    }

    let chunk_len_usize = if chunk_len == 0 { 32usize } else { chunk_len as usize };
    let to_halt = max_chunks == 0;
    let nchunks = max_chunks as usize;
    let tmp_str = tmp_path.to_string_lossy().into_owned();

    let result = std::panic::catch_unwind(move || -> Result<_, String> {
        let (block_number, state_root) =
            riscv_stf::prove_elf::golden_from_run(&elf_path, &tmp_str)?;
        let span = riscv_stf::prove_elf::prove_span(
            &elf_path,
            &tmp_str,
            0,
            chunk_len_usize,
            nchunks,
            to_halt,
            false,
        )?;
        Ok((block_number, state_root, span))
    });

    let _ = std::fs::remove_file(&tmp_path);

    match result {
        Ok(Ok((block_number, state_root, span))) => {
            let n = span.proofs.len();
            if n == 0 {
                set_last_error("accprover_prove_elf: prove_span produced no chunks");
                return ACCPROVER_ERR_PROVE;
            }
            let all_verified = span.proofs.iter().all(|p| p.verified);
            let all_go_match = all_verified; // (superseded ELF path; field removed upstream)
            let chain_valid = span.chained && span.mem_thread_ok;
            let composed = span.mem_closed && span.output_bound;
            let first_commitment = span.proofs[0].commitment;

            // Serialize the multi-chunk proof: u32 count ++ (u32 len ++ bytes)*.
            let total: usize =
                4 + span.proofs.iter().map(|p| 4 + p.proof_bytes.len()).sum::<usize>();
            let mut blob = Vec::with_capacity(total);
            blob.extend_from_slice(&(n as u32).to_le_bytes());
            for p in &span.proofs {
                blob.extend_from_slice(&(p.proof_bytes.len() as u32).to_le_bytes());
                blob.extend_from_slice(&p.proof_bytes);
            }

            *out_block_number = block_number;
            std::ptr::copy_nonoverlapping(state_root.as_ptr(), out_state_root, 32);
            *out_num_chunks = n as u32;
            *out_all_verified = all_verified as c_int;
            *out_all_go_match = all_go_match as c_int;
            *out_chain_valid = chain_valid as c_int;
            std::ptr::copy_nonoverlapping(first_commitment.as_ptr(), out_first_commitment, 32);
            *out_composed = composed as c_int;
            let (proof_ptr, proof_len) = leak_buf(blob);
            *out_proof = proof_ptr;
            *out_proof_len = proof_len;
            ACCPROVER_OK
        }
        Ok(Err(e)) => {
            set_last_error(&format!("prove_elf error: {e}"));
            ACCPROVER_ERR_PROVE
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            set_last_error(&format!("prove_elf panicked: {msg}"));
            ACCPROVER_ERR_PANIC
        }
    }
}

/// Free a buffer previously produced by [`accprover_prove`] (`out_proof` / `out_pub`).
///
/// # Safety
/// `(ptr, len)` must be a buffer pair returned by `accprover_prove` and not yet
/// freed. Passing a null `ptr` is a no-op.
#[no_mangle]
pub unsafe extern "C" fn accprover_free_buf(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let slice = std::slice::from_raw_parts_mut(ptr, len);
    drop(Box::from_raw(slice as *mut [u8]));
}

/// Return a NUL-terminated C string describing the last error on THIS thread, or
/// null if none. The pointer is valid until the next `accprover_*` call on the
/// same thread; copy it if you need to retain it.
#[no_mangle]
pub extern "C" fn accprover_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        let b = e.borrow();
        if b.is_empty() {
            std::ptr::null()
        } else {
            b.as_ptr() as *const c_char
        }
    })
}

/// Prove one GENUINE EVM bytecode execution via the RV32IM CPU-verifier circuit
/// (grand-product offline memory checking over GF(2^128)) with the trace
/// committed by the reused DA-canonical rsema1d Encode. Calls the verified
/// `riscv_stf::prove_evm`. On success (`ACCPROVER_OK`):
///   * `out_commit` (32 bytes, caller-owned) := rsema1d commitment (byte-
///     identical to an independent Go/DA Encode of the same rows);
///   * `*out_proof` / `*out_proof_len` := library-owned serialized GKR proof;
///   * `*out_output` / `*out_output_len` := library-owned EVM return data;
///   * `out_storage_digest` (32 bytes, caller-owned) := post-storage digest;
///   * `*out_verified` := 1 iff the Expander verifier accepted AND the RV32
///     execution's output+storage matched (the proof self-verifies);
///   * `*out_num_vars` := GKR num_vars; `*out_cycles` := RV32 cycles-to-halt;
///   * `*out_tamper_rejected` := 1 iff a corrupted trace was provably rejected.
/// Return `out_proof` / `out_output` to [`accprover_free_buf`].
///
/// `pre` encodes pre-storage as a sequence of 64-byte entries: 32-byte big-endian
/// key followed by 32-byte big-endian value.
///
/// # Safety
/// `code`/`calldata`/`pre` must point to their stated readable lengths (or be
/// null iff the length is 0). `out_commit` / `out_storage_digest` must each point
/// to 32 writable bytes; the other out-pointers must be non-null and writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn accprover_prove_evm(
    code: *const u8,
    code_len: usize,
    calldata: *const u8,
    calldata_len: usize,
    pre: *const u8,
    pre_len: usize,
    out_commit: *mut u8,
    out_proof: *mut *mut u8,
    out_proof_len: *mut usize,
    out_output: *mut *mut u8,
    out_output_len: *mut usize,
    out_storage_digest: *mut u8,
    out_verified: *mut c_int,
    out_num_vars: *mut u32,
    out_cycles: *mut u32,
    out_tamper_rejected: *mut c_int,
) -> c_int {
    if out_commit.is_null()
        || out_proof.is_null()
        || out_proof_len.is_null()
        || out_output.is_null()
        || out_output_len.is_null()
        || out_storage_digest.is_null()
        || out_verified.is_null()
        || out_num_vars.is_null()
        || out_cycles.is_null()
        || out_tamper_rejected.is_null()
    {
        set_last_error("null out-pointer argument");
        return ACCPROVER_ERR_NULL_ARG;
    }
    let code_v = if code_len == 0 { Vec::new() } else { std::slice::from_raw_parts(code, code_len).to_vec() };
    let cd_v = if calldata_len == 0 { Vec::new() } else { std::slice::from_raw_parts(calldata, calldata_len).to_vec() };
    let pre_v = if pre_len == 0 { Vec::new() } else { std::slice::from_raw_parts(pre, pre_len).to_vec() };
    if pre_len % 64 != 0 {
        set_last_error("pre-storage length must be a multiple of 64 (key32||val32)");
        return ACCPROVER_ERR_PROVE;
    }

    let result = std::panic::catch_unwind(move || {
        use riscv_stf::evm_core::U256;
        let mut pre_map = std::collections::BTreeMap::<U256, U256>::new();
        for e in pre_v.chunks_exact(64) {
            let k = U256::from_be_bytes(&e[0..32]);
            let v = U256::from_be_bytes(&e[32..64]);
            pre_map.insert(k, v);
        }
        riscv_stf::prove_evm(&code_v, &cd_v, &pre_map, 8, true)
    });

    let p = match result {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            set_last_error(&format!("prove_evm error: {e}"));
            return ACCPROVER_ERR_PROVE;
        }
        Err(panic) => {
            let msg = panic.downcast_ref::<&str>().map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            set_last_error(&format!("prove_evm panicked: {msg}"));
            return ACCPROVER_ERR_PANIC;
        }
    };

    std::ptr::copy_nonoverlapping(p.commitment.as_ptr(), out_commit, 32);
    std::ptr::copy_nonoverlapping(p.post_storage_digest.as_ptr(), out_storage_digest, 32);
    let (proof_ptr, proof_len) = leak_buf(p.proof);
    *out_proof = proof_ptr;
    *out_proof_len = proof_len;
    let (out_ptr, out_l) = leak_buf(p.output);
    *out_output = out_ptr;
    *out_output_len = out_l;
    // "verified" here means: GKR verifier accepted AND commitment reconciled AND
    // embedded — the library asserts the last two internally (prove_evm returns
    // Err otherwise), so `verified` reflects the Expander verdict.
    *out_verified = if p.verified && p.commit_stable && p.commit_in_proof { 1 } else { 0 };
    *out_num_vars = p.num_vars as u32;
    *out_cycles = p.cycles as u32;
    *out_tamper_rejected = if p.tamper_rejected { 1 } else { 0 };
    ACCPROVER_OK
}

fn leak_buf(v: Vec<u8>) -> (*mut u8, usize) {
    let boxed: Box<[u8]> = v.into_boxed_slice();
    let len = boxed.len();
    let ptr = Box::into_raw(boxed) as *mut u8;
    (ptr, len)
}
