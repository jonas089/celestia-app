//! `Rsema1dPCS`: Polyhedra Expander's *input* polynomial commitment, backed by
//! the **real Go `rsema1d`** multilinear PCS over reverse-FFI.
//!
//! This is the "accidental computer" input-commitment reuse: Expander's GKR
//! commits ONLY the circuit input layer, so the entire polynomial-commitment
//! surface of the proof system can be discharged by the data-availability
//! encoding rsema1d already computes. We implement
//! [`gkr_engine::ExpanderPCS`]`<GF2ExtConfig>` so GKR uses rsema1d as a drop-in
//! replacement for `RawExpanderGKR`.
//!
//! # How the two field encodings are bridged
//! Expander works over `GF2_128` (LE u128, bit `i` = coefficient of `x^i`);
//! rsema1d works over its own `GF128` encoding. The two are related by a fixed
//! `GF(2)`-linear field isomorphism (see [`field_iso`], validated 10000/10000
//! against both real libraries). Every challenge coordinate is mapped
//! `ISO_E_TO_R` on the way in; every verified value is mapped back `ISO_R_TO_E`.
//!
//! # Encode-once and the full-point problem
//! The commitment is the ORIGINAL Go `Coder.Encode` (legacy DeriveCoefficients
//! RLC) — the exact bytes celestia-app DA sampling and `cmd/testvectors`
//! produce, so the GKR input commitment IS the DA commitment (encode-once).
//! rsema1d's native tensor opening would derive the *column* challenge itself
//! (Fiat-Shamir) and only accept a *row* challenge, and it commits a DIFFERENT
//! (tensor) RLC. Expander's GKR instead fixes the ENTIRE input point — both
//! axes. We therefore drive rsema1d's additive `OpenAtFullLegacy` /
//! `VerifyAtFullLegacy` (pkg/rsema1d/pcs_full_legacy.go), which soundly opens
//! the ORIGINAL Encode commitment at an externally supplied `(rCol, rRow)` — see
//! that file for the proximity/soundness argument.
//!
//! # Layout and variable ordering (validated by the oracle gate)
//! The input layer is `2^(num_vars+3)` bits: `hypercube_basis()` is a
//! `Vec<GF2x8>` of length `2^num_vars`, each element packing 8 SIMD lanes. We
//! bit-pack it into a `K x numSymbols` square of `GF(2^16)` symbols in `{0,1}`,
//! `numSymbols = 32` (`rowBytes = 64`, one Leopard chunk), `K = 2^(num_vars-2)`,
//! `N = K`. The flat bit address is `a = g*8 + s` (`g` = hypercube index, `s` =
//! SIMD lane); row `j` and column `i` come from `a = j*numSymbols + i`. Matching
//! Expander's oracle (`single_core_eval_circuit_vals_at_expander_challenge`)
//! against rsema1d's fold order (`foldGF128` / `TensorCoefficients`, MSB-first
//! per challenge) fixes the split: the full challenge in address-bit order is
//! `phi = r_simd ++ rz` (the 3 SIMD vars bind the low address bits, `rz` the
//! rest), `rCol = reverse(phi[0..5])`, `rRow = reverse(phi[5..])`, each element
//! mapped through the field isomorphism.

pub mod field_iso;
use field_iso::{ISO_E_TO_R, ISO_R_TO_E};

/// Map an Expander `GF2_128` coefficient (LE u128, bit i = x^i) to rsema1d's
/// `GF128` encoding (LE u128). Public wrapper over the validated GF(2)-linear
/// isomorphism `ISO_E_TO_R`, for external callers (e.g. the block-data linkage)
/// that drive `rsema1d_sys` directly and need to translate an evaluation point.
#[inline]
pub fn iso_e_to_r(e: u128) -> u128 {
    iso_apply(&ISO_E_TO_R, e)
}

/// Inverse of [`iso_e_to_r`]: map an rsema1d `GF128` encoding back to Expander's
/// `GF2_128` coefficient encoding. Used to bring an rsema1d opening value into
/// the field the GKR trace proof works over.
#[inline]
pub fn iso_r_to_e(r: u128) -> u128 {
    iso_apply(&ISO_R_TO_E, r)
}

use std::marker::PhantomData;

use arith::SimdField;
use gf2::GF2x8;
use gf2_128::GF2_128;
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, GF2ExtConfig, GKREngine, GKRScheme,
    MPIConfig, MPIEngine, PolynomialCommitmentType, StructuredReferenceString, Transcript,
};
use gkr_hashers::SHA256hasher;
use polynomials::MultilinearExtension;
use rand::RngCore;
use rsema1d_sys::RowRange;
use serdes::ExpSerde;
use transcript::BytesHashTranscript;

/// log2 of the number of GF(2^16) symbols per row. `numSymbols = 32` gives
/// `rowBytes = 64` (exactly one Leopard chunk, a multiple of 64).
const LOG_COLS: usize = 5;
/// GF(2^16) symbols per row.
const NUM_SYMBOLS: usize = 1 << LOG_COLS; // 32
/// Bytes per Leopard-formatted row: 32 low bytes followed by 32 high bytes.
const ROW_BYTES: usize = 2 * NUM_SYMBOLS; // 64
/// Proximity samples per opening. `u32::MAX` is clamped Go-side to `K+N`, i.e.
/// we sample every row for the strongest soundness in tests.
const SAMPLE_COUNT: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Field-encoding bridge
// ---------------------------------------------------------------------------

/// Applies a stored (column-major) GF(2)-linear map to a 128-bit element:
/// `r = XOR over each set bit i of e of table[i]`.
#[inline]
fn iso_apply(table: &[u128; 128], e: u128) -> u128 {
    let mut r = 0u128;
    for (i, col) in table.iter().enumerate() {
        if (e >> i) & 1 == 1 {
            r ^= *col;
        }
    }
    r
}

/// Expander `GF2_128` -> its 128-bit coefficient encoding (LE u128, bit i = x^i).
#[inline]
fn gf2_128_to_u128(x: &GF2_128) -> u128 {
    let mut buf = [0u8; 16];
    x.serialize_into(&mut buf[..]).expect("GF2_128 serialize");
    u128::from_le_bytes(buf)
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// The rsema1d square parameters for a given Expander `num_vars`.
struct Layout {
    k: u32,
    n: u32,
    log_rows: usize,
}

fn layout(num_vars: usize) -> Layout {
    assert!(
        num_vars + 3 >= LOG_COLS,
        "num_vars={num_vars} too small: need num_vars+3 >= {LOG_COLS} (>= 2 vars)"
    );
    let log_rows = num_vars + 3 - LOG_COLS; // = num_vars - 2
    let k = 1u32 << log_rows;
    Layout { k, n: k, log_rows }
}

/// Bit-packs the input polynomial's hypercube basis into the `K+N` row matrix
/// rsema1d commits: the first `K` rows are Leopard-formatted `{0,1}` GF(2^16)
/// symbols, the trailing `N` parity rows are zeroed (filled by rsema1d's RS
/// encoder). Bit address `a = j*numSymbols + i` maps to hypercube index
/// `g = a>>3` and SIMD lane `s = a&7`.
fn build_rows(poly: &impl MultilinearExtension<GF2x8>, num_vars: usize) -> Vec<Vec<u8>> {
    let basis = poly.hypercube_basis(); // Vec<GF2x8>, len 2^num_vars
    assert_eq!(basis.len(), 1usize << num_vars);
    let lay = layout(num_vars);
    let k = lay.k as usize;
    let total = k + lay.n as usize;

    // Pre-unpack each GF2x8 into its 8 lane bits once.
    let bits: Vec<[u8; 8]> = basis
        .iter()
        .map(|e| {
            let lanes = e.unpack(); // Vec<GF2>, len 8
            let mut b = [0u8; 8];
            for (s, lane) in lanes.iter().enumerate() {
                b[s] = lane.v & 1;
            }
            b
        })
        .collect();

    // K data rows followed by N zeroed parity rows (total K+N).
    let mut rows = vec![vec![0u8; ROW_BYTES]; total];
    for (j, row) in rows.iter_mut().take(k).enumerate() {
        for i in 0..NUM_SYMBOLS {
            let a = j * NUM_SYMBOLS + i;
            let g = a >> 3;
            let s = a & 7;
            // Leopard chunk: low byte at [i], high byte at [32+i]. Symbol is a
            // bit in {0,1}, so the high byte stays zero.
            row[i] = bits[g][s];
        }
    }
    rows
}

/// Maps an Expander challenge to rsema1d's `(rCol, rRow)` point blobs (each a
/// concatenation of 16-byte little-endian GF128 encodings). See the module doc
/// for the derivation of the ordering.
fn map_point(x: &ExpanderSingleVarChallenge<GF2ExtConfig>, num_vars: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(x.r_simd.len(), 3, "GF2x8 SIMD => 3 simd challenges");
    assert_eq!(x.rz.len(), num_vars, "rz must have num_vars challenges");
    assert!(x.r_mpi.is_empty(), "single-process: r_mpi must be empty");

    let lay = layout(num_vars);
    let log_rows = lay.log_rows;

    // phi in address-bit order: the 3 SIMD vars bind the low bits, rz the rest.
    let mut phi: Vec<u128> = Vec::with_capacity(num_vars + 3);
    for e in &x.r_simd {
        phi.push(iso_apply(&ISO_E_TO_R, gf2_128_to_u128(e)));
    }
    for e in &x.rz {
        phi.push(iso_apply(&ISO_E_TO_R, gf2_128_to_u128(e)));
    }

    // rCol = reverse(phi[0..LOG_COLS]); rRow = reverse(phi[LOG_COLS..]).
    let mut rcol = Vec::with_capacity(LOG_COLS * 16);
    for kk in 0..LOG_COLS {
        rcol.extend_from_slice(&phi[LOG_COLS - 1 - kk].to_le_bytes());
    }
    let mut rrow = Vec::with_capacity(log_rows * 16);
    for kk in 0..log_rows {
        rrow.extend_from_slice(&phi[LOG_COLS + log_rows - 1 - kk].to_le_bytes());
    }
    (rcol, rrow)
}

// ---------------------------------------------------------------------------
// ExpanderPCS implementation
// ---------------------------------------------------------------------------

/// 32-byte rsema1d commitment `SHA256(rowRoot || rlcRoot)`.
#[derive(Clone, Debug, Default, ExpSerde)]
pub struct Rsema1dCommitment {
    pub root: [u8; 32],
}

/// Serialized rsema1d `EvalProofFull`.
#[derive(Clone, Debug, Default, ExpSerde)]
pub struct Rsema1dOpening {
    pub proof: Vec<u8>,
}

/// The DA commitment + live handle, produced ONCE by the DA encoder and REUSED
/// by the GKR prover. Holding the [`Handle`] keeps the Go-side committed square
/// alive so `open` can evaluate it WITHOUT re-encoding.
struct DaCommitment {
    num_vars: usize,
    root: [u8; 32],
    handle: rsema1d_sys::Handle,
}

static DA_COMMITMENT: std::sync::Mutex<Option<DaCommitment>> = std::sync::Mutex::new(None);

/// **The DA encoder step.** Encode the block-data rows ONCE (this is the
/// data-availability encoding that happens regardless of proving) and install the
/// resulting rsema1d commitment + live handle. After this call, the GKR prover's
/// [`Rsema1dPCS::commit`] and [`Rsema1dPCS::open`] REUSE this handle and perform
/// ZERO encoding — the accidental-computer property (the prover does no
/// input-commitment work; it only opens the DA commitment at the sumcheck point).
/// Returns the DA commitment root.
pub fn install_da_commitment(num_vars: usize, poly: &impl MultilinearExtension<GF2x8>) -> [u8; 32] {
    assert_eq!(poly.num_vars(), num_vars);
    let lay = layout(num_vars);
    let rows = build_rows(poly, num_vars);
    let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (root, handle) = rsema1d_sys::commit(lay.k, lay.n, &row_refs).expect("DA rsema1d encode failed");
    *DA_COMMITMENT.lock().unwrap() = Some(DaCommitment { num_vars, root, handle });
    root
}

/// Clear the installed DA commitment (frees the Go-side handle).
pub fn clear_da_commitment() {
    *DA_COMMITMENT.lock().unwrap() = None;
}

/// **DA-encoder side of the TRUE cross-process hand-off.** Bit-packs the input
/// polynomial's hypercube basis into the `K+N` row matrix and RS-encodes it ONCE
/// via the Go DA encoder ([`rsema1d_sys::encode_extended`]), returning the DA
/// commitment root together with the SERIALIZED extended matrix.
///
/// This performs NO GKR work and installs nothing. It is meant to run in a
/// SEPARATE process (or at a separate time) from proving: the caller emits
/// `(root, extended)` to disk / a pipe, and a prover process later reconstructs
/// the committed square from `extended` via
/// [`install_da_commitment_from_serialized`] without re-encoding. The polynomial
/// RS-encoding thus happens exactly once, here.
pub fn da_encode_serialized(
    num_vars: usize,
    poly: &impl MultilinearExtension<GF2x8>,
) -> ([u8; 32], Vec<u8>) {
    assert_eq!(poly.num_vars(), num_vars);
    let lay = layout(num_vars);
    let rows = build_rows(poly, num_vars);
    let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    rsema1d_sys::encode_extended(lay.k, lay.n, &row_refs).expect("DA rsema1d encode failed")
}

/// **Prover side of the TRUE cross-process hand-off.** Reconstructs the committed
/// square from the serialized extended matrix produced by [`da_encode_serialized`]
/// in another process — via [`rsema1d_sys::load_extended`], which rebuilds only
/// the Merkle/RLC commitment structures and performs NO RS-encoding — then
/// installs the resulting live handle so [`Rsema1dPCS::commit`] / [`open`] reuse
/// it exactly as if it had been encoded in-process.
///
/// Asserts the reconstructed commitment equals `expected_root` (the root the DA
/// encoder emitted), i.e. the prover is proving against the *same* DA commitment.
/// Returns the (verified-equal) commitment root. After this call,
/// [`rsema1d_sys::encode_call_count`] in this process is still whatever it was —
/// this path adds ZERO encodes.
pub fn install_da_commitment_from_serialized(
    num_vars: usize,
    expected_root: [u8; 32],
    extended: &[u8],
) -> [u8; 32] {
    let (root, handle) =
        rsema1d_sys::load_extended(extended).expect("DA rsema1d load_extended failed");
    assert_eq!(
        root, expected_root,
        "reconstructed DA commitment != emitted DA root (data corrupted in transit?)"
    );
    *DA_COMMITMENT.lock().unwrap() = Some(DaCommitment { num_vars, root, handle });
    root
}

/// If a DA commitment for `num_vars` is currently installed, return its root.
/// Used by the prover to REUSE a commitment pre-installed by the cross-process
/// hand-off instead of encoding one in-process.
pub fn installed_da_root(num_vars: usize) -> Option<[u8; 32]> {
    let g = DA_COMMITMENT.lock().unwrap();
    g.as_ref()
        .filter(|da| da.num_vars == num_vars)
        .map(|da| da.root)
}

/// The Expander input-PCS backed by the real Go rsema1d multilinear PCS. The
/// prover NEVER encodes here: `commit`/`open` consume the DA-installed handle.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Rsema1dPCS;

impl ExpanderPCS<GF2ExtConfig> for Rsema1dPCS {
    const NAME: &'static str = "Rsema1dPCS";
    const PCS_TYPE: PolynomialCommitmentType = PolynomialCommitmentType::Raw;

    type Params = usize;
    type ScratchPad = ();
    type SRS = ();
    type Commitment = Rsema1dCommitment;
    type Opening = Rsema1dOpening;
    type BatchOpening = ();

    fn gen_srs(_params: &Self::Params, _mpi: &impl MPIEngine, _rng: impl RngCore) -> Self::SRS {}

    fn gen_params(n_input_vars: usize, _world_size: usize) -> Self::Params {
        n_input_vars
    }

    fn init_scratch_pad(_params: &Self::Params, _mpi: &impl MPIEngine) -> Self::ScratchPad {}

    fn commit(
        params: &Self::Params,
        _mpi: &impl MPIEngine,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<GF2x8>,
        _scratch_pad: &mut Self::ScratchPad,
    ) -> Option<Self::Commitment> {
        // REUSE the DA commitment: return the root the DA encoder already
        // produced. The prover does ZERO encoding here.
        let num_vars = *params;
        let _ = poly; // not encoded — the committed layer IS the DA'd data
        let g = DA_COMMITMENT.lock().unwrap();
        let da = g.as_ref().expect("no DA commitment installed: call install_da_commitment (the DA encoder) before proving");
        assert_eq!(da.num_vars, num_vars, "installed DA commitment num_vars mismatch");
        Some(Rsema1dCommitment { root: da.root })
    }

    fn open(
        params: &Self::Params,
        _mpi: &impl MPIEngine,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<GF2x8>,
        x: &ExpanderSingleVarChallenge<GF2ExtConfig>,
        _transcript: &mut impl Transcript,
        _scratch_pad: &Self::ScratchPad,
    ) -> Option<Self::Opening> {
        // The PCS opening does not itself append to the transcript (GKR appends
        // the returned Opening). The GKR driver already brackets this call with
        // lock_proof/unlock_proof (gkr/src/prover/snark.rs), exactly as
        // RawExpanderGKR::open expects, so this method must NOT lock again (a
        // second lock_proof panics on `assert!(!self.proof_locked)`).
        // REUSE the DA handle: open the commitment the DA encoder already built,
        // at the GKR sumcheck point. The prover does NO encoding — only this
        // opening (the "nearly free" partial evaluation the DA encoding enables).
        let num_vars = *params;
        let _ = poly; // not re-encoded — we open the installed DA square
        let lay = layout(num_vars);
        let (rcol, rrow) = map_point(x, num_vars);
        let range = RowRange::new(0, lay.k); // the WHOLE input layer
        let g = DA_COMMITMENT.lock().unwrap();
        let da = g.as_ref().expect("no DA commitment installed for open()");
        assert_eq!(da.num_vars, num_vars, "installed DA commitment num_vars mismatch");
        let proof = rsema1d_sys::open_at_full(&da.handle, range, &rcol, &rrow, SAMPLE_COUNT).ok()?;
        Some(Rsema1dOpening { proof })
    }

    fn verify(
        params: &Self::Params,
        _verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        commitment: &Self::Commitment,
        x: &ExpanderSingleVarChallenge<GF2ExtConfig>,
        v: <GF2ExtConfig as FieldEngine>::ChallengeField,
        _transcript: &mut impl Transcript,
        opening: &Self::Opening,
    ) -> bool {
        // As in open(): the GKR verifier (gkr/src/verifier/snark.rs) already
        // brackets this call with lock_proof/unlock_proof, matching
        // RawExpanderGKR::verify, so this method must NOT lock again.
        let num_vars = *params;
        let lay = layout(num_vars);
        let (rcol, rrow) = map_point(x, num_vars);

        let result = match rsema1d_sys::verify_at_full(
            lay.k,
            lay.n,
            &commitment.root,
            &opening.proof,
            &rcol,
            &rrow,
        ) {
            // rsema1d returns the value in its own GF128 encoding; map it back to
            // Expander's GF2_128 and compare to the GKR-claimed value v.
            Some(val_bytes) => {
                let recovered = iso_apply(&ISO_R_TO_E, u128::from_le_bytes(val_bytes));
                recovered == gf2_128_to_u128(&v)
            }
            None => false,
        };

        result
    }
}

// ---------------------------------------------------------------------------
// Hand-written GKREngine wiring (mirrors gkr/src/gkr_configs.rs
// GF2ExtConfigSha2Raw, but with Rsema1dPCS as the PCS).
// ---------------------------------------------------------------------------

/// A GKR engine over `GF2ExtConfig` with an SHA256 Fiat-Shamir transcript and
/// the rsema1d input polynomial commitment. Single-process (`MPIConfig`,
/// `world_size = 1`), mirroring `RawExpanderGKR::is_single_process()`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Rsema1dGKRConfig<'a> {
    _phantom: PhantomData<&'a ()>,
}

impl<'a> GKREngine for Rsema1dGKRConfig<'a> {
    type FieldConfig = GF2ExtConfig;
    type MPIConfig = MPIConfig<'a>;
    type TranscriptConfig = BytesHashTranscript<SHA256hasher>;
    type PCSConfig = Rsema1dPCS;
    const SCHEME: GKRScheme = GKRScheme::Vanilla;
}
