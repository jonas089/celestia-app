//! Driver for the input-committed STF **executor** (`circuit_ac`): the correct
//! accidental-computer shape. The committed GKR input layer is the block's
//! DATA (pre-state + txs), NOT the execution trace. The transfer-STF is computed
//! forward in circuit; only the data is committed, by the (reused) rsema1d DA
//! encoder — the sole polynomial commitment, opened at the sumcheck point.
//!
//! Reuses the block model / reference semantics from [`crate::stf`]
//! (`BlockInput`, `TxData`, `apply_native`) so the native reference and the
//! trace-commit path (`circuit_stf`) attest the identical transition; only the
//! commitment shape differs (data-committed vs trace-committed).

use crate::circuit_ac::{build_assignment, StfAcCircuit, MAXTX, NOUT, NWORDS, XLEN};
use crate::stf::{apply_native, BlockInput, StfOutcome, TxData, NULL_NONCE};
use arith::{Field, SimdField};
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use polynomials::MultiLinearPoly;
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};
use std::sync::Mutex;

static PROVE_LOCK: Mutex<()> = Mutex::new(());

/// A real GKR proof of a block's nonce + balance transition where the SOLE
/// polynomial commitment is the rsema1d/DA commitment of the block DATA (the
/// committed input layer), opened at the sumcheck point.
#[derive(Clone, Debug)]
pub struct AcStfProof {
    /// 32-byte rsema1d commitment of the committed input layer (the block data).
    /// Byte-identical to an independent Go/DA rsema1d commit of the same rows.
    pub commitment: [u8; 32],
    /// Serialized Expander GKR proof bytes.
    pub proof: Vec<u8>,
    /// Public outputs (post sbal, snon, rbal, applied, digest), each LE u32.
    pub public_value: Vec<u8>,
    /// Whether the Expander verifier accepted (self-verified here).
    pub verified: bool,
    /// GKR num_vars of the committed input polynomial.
    pub input_vars: u32,
    pub post_sender_balance: u32,
    pub post_sender_nonce: u32,
    pub post_recipient_balance: u32,
    pub applied_count: u32,
    pub digest: u32,
    pub tx_count: u32,
    /// The committed data words reconstructed from the committed poly (proof that
    /// the committed input IS the block data, not a trace).
    pub committed_words: Vec<u32>,
}

/// The block's committed data words in canonical order.
fn data_words(input: &BlockInput) -> Result<[u32; NWORDS], String> {
    if input.txs.len() > MAXTX {
        return Err(format!("block has {} txs; max {}", input.txs.len(), MAXTX));
    }
    let mut txs = [TxData { value: 0, fee: 0, nonce: NULL_NONCE }; MAXTX];
    for (i, t) in input.txs.iter().enumerate() {
        txs[i] = *t;
    }
    let mut w = [0u32; NWORDS];
    w[0] = input.pre_sender_balance;
    w[1] = input.pre_sender_nonce;
    w[2] = input.pre_recipient_balance;
    for i in 0..MAXTX {
        w[3 + 3 * i] = txs[i].value;
        w[3 + 3 * i + 1] = txs[i].fee;
        w[3 + 3 * i + 2] = txs[i].nonce;
    }
    Ok(w)
}

/// Independent Go/DA rsema1d commit of the committed input layer, in the exact
/// bit-packed square rsema1d-pcs commits (num_vars=9 => K=N=128, numSymbols=32).
fn da_commit(input_vals: &[gf2::GF2x8], num_vars: usize) -> Result<[u8; 32], String> {
    const NUM_SYMBOLS: usize = 32;
    const ROW_BYTES: usize = 64;
    let k: u32 = 1u32 << (num_vars - 2);
    let n: u32 = k;
    let bits: Vec<[u8; 8]> = input_vals
        .iter()
        .map(|e| {
            let l = e.unpack();
            let mut b = [0u8; 8];
            for s in 0..8 {
                b[s] = l[s].v & 1;
            }
            b
        })
        .collect();
    let mut rows = vec![vec![0u8; ROW_BYTES]; (k + n) as usize];
    for j in 0..(k as usize) {
        for i in 0..NUM_SYMBOLS {
            let a = j * NUM_SYMBOLS + i;
            rows[j][i] = bits[a >> 3][a & 7];
        }
    }
    let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (c_da, _h) = rsema1d_sys::commit(k, n, &row_refs)
        .map_err(|e| format!("independent rsema1d commit failed: {e}"))?;
    Ok(c_da)
}

/// Reconstruct the committed data words from the committed poly (SIMD lane 0).
fn recon_words(input_vals: &[gf2::GF2x8]) -> Vec<u32> {
    let mut words = vec![0u32; NWORDS];
    for i in 0..NWORDS {
        let mut w = 0u32;
        for b in 0..XLEN {
            let bit = input_vals[i * XLEN + b].unpack()[0].v & 1;
            w |= (bit as u32) << b;
        }
        words[i] = w;
    }
    words
}

/// Prove a block's transfer transition with the block DATA as the sole committed
/// input (the accidental-computer shape). Self-verifies and reconciles the GKR
/// input commitment against an independent Go/DA rsema1d commit of the identical
/// rows (byte-identical => genuine reuse, prover commits nothing new).
pub fn prove_block_stf_ac(input: &BlockInput) -> Result<AcStfProof, String> {
    let _guard = PROVE_LOCK.lock().map_err(|e| format!("prove lock poisoned: {e}"))?;

    let words = data_words(input)?;
    let native: StfOutcome = apply_native(input)?;
    let out_vals: [u32; NOUT] = [
        native.post_sender_balance,
        native.post_sender_nonce,
        native.post_recipient_balance,
        native.applied_count,
        native.digest,
    ];

    // ---- Compile -------------------------------------------------------------
    let CompileResult { witness_solver, layered_circuit } =
        compile(&StfAcCircuit::default(), CompileOptions::default())
            .map_err(|e| format!("compile failed: {e:?}"))?;

    // ---- Witness -------------------------------------------------------------
    let assignment = build_assignment(&words, &out_vals);
    let assignments = vec![assignment.clone(); 8];
    let witness = witness_solver
        .solve_witnesses(&assignments)
        .map_err(|e| format!("witness solve failed: {e:?}"))?;
    let res = layered_circuit.run(&witness);
    if !res.iter().all(|x| *x) {
        return Err(format!("layered circuit self-eval failed: {res:?}"));
    }

    // ---- Export & install the block DATA as the committed input layer --------
    let mut ec = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = simd_input.clone();
    ec.public_input = simd_public_input.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let num_vars = ec.log_input_size();

    // Confirm the committed poly IS the block data words.
    let committed_words = recon_words(&input_vals);
    if committed_words[..NWORDS] != words[..] {
        return Err(format!(
            "committed poly words {committed_words:?} != block data words {words:?}"
        ));
    }

    // ---- Prove & verify (rsema1d = sole PCS) ---------------------------------
    let mpi = MPIConfig::prover_new(None, None);
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let verified = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v)
        && claimed_v.is_zero();

    // ---- Commitment reconciliation (PCS root == independent Go/DA commit) ----
    let params = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi.world_size());
    let mut sp = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
    let input_poly = MultiLinearPoly::new(input_vals.clone());
    let c_prover =
        <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &input_poly, &mut sp)
            .ok_or_else(|| "rsema1d PCS commit returned None".to_string())?;
    let c_da = da_commit(&input_vals, num_vars)?;
    if c_prover.root != c_da {
        return Err(format!(
            "commitment mismatch: GKR PCS root {} != independent Go/DA rsema1d commit {}",
            hex(&c_prover.root),
            hex(&c_da)
        ));
    }
    if !proof.bytes.windows(32).any(|w| w == c_prover.root) {
        return Err("commitment not embedded in proof transcript".to_string());
    }

    // ---- Reconstruct public outputs; check against native --------------------
    let mut recon = [0u32; NOUT];
    for kk in 0..NOUT {
        let mut w = 0u32;
        for b in 0..XLEN {
            w |= ((simd_public_input[kk * XLEN + b].unpack()[0].v & 1) as u32) << b;
        }
        recon[kk] = w;
    }
    if recon != out_vals {
        return Err(format!("circuit public output {recon:?} != native post-state {out_vals:?}"));
    }
    let mut public_value = Vec::with_capacity(NOUT * 4);
    for v in recon {
        public_value.extend_from_slice(&v.to_le_bytes());
    }

    Ok(AcStfProof {
        commitment: c_da,
        proof: proof.bytes,
        public_value,
        verified,
        input_vars: num_vars as u32,
        post_sender_balance: recon[0],
        post_sender_nonce: recon[1],
        post_recipient_balance: recon[2],
        applied_count: recon[3],
        digest: recon[4],
        tx_count: input.txs.len() as u32,
        committed_words,
    })
}

/// Soundness helper: commit a tampered copy of the block data and return its
/// rsema1d commitment (must differ from the honest one). Also confirms that a
/// tampered-data / honest-output witness is UNSATISFIABLE (the executor rejects
/// a false transition).
pub fn tamper_check(input: &BlockInput, word_idx: usize, xor_mask: u32) -> Result<TamperReport, String> {
    let _guard = PROVE_LOCK.lock().map_err(|e| format!("prove lock poisoned: {e}"))?;
    let mut words = data_words(input)?;
    let native = apply_native(input)?;
    let out_vals: [u32; NOUT] = [
        native.post_sender_balance,
        native.post_sender_nonce,
        native.post_recipient_balance,
        native.applied_count,
        native.digest,
    ];

    let CompileResult { witness_solver, layered_circuit } =
        compile(&StfAcCircuit::default(), CompileOptions::default())
            .map_err(|e| format!("compile failed: {e:?}"))?;

    // Honest witness: the committed input layer IS the SIMD witness input; no
    // export/evaluate needed to reproduce the rsema1d commitment of that layer.
    let honest = build_assignment(&words, &out_vals);
    let w0 = witness_solver
        .solve_witnesses(&vec![honest; 8])
        .map_err(|e| format!("witness solve failed: {e:?}"))?;
    let honest_ok = layered_circuit.run(&w0).iter().all(|x| *x);
    let (si, _sp) = w0.to_simd::<gf2::GF2x8>();
    let num_vars = si.len().trailing_zeros() as usize; // power of two
    let c_honest = da_commit(&si, num_vars)?;

    // Tamper one committed data word; keep the honest claimed output. The
    // executor recomputes the post-state from the tampered data, so the honest
    // output no longer matches => the constraint system is UNSATISFIABLE.
    words[word_idx] ^= xor_mask;
    let tampered = build_assignment(&words, &out_vals);
    let solved = witness_solver.solve_witnesses(&vec![tampered; 8]);
    let (sat, c_tampered) = match &solved {
        Ok(w) => {
            let sat = layered_circuit.run(w).iter().all(|x| *x);
            let (si2, _p2) = w.to_simd::<gf2::GF2x8>();
            (sat, Some(da_commit(&si2, num_vars)?))
        }
        Err(_) => (false, None),
    };

    Ok(TamperReport {
        commitment_honest: c_honest,
        commitment_tampered: c_tampered,
        tampered_satisfiable: sat,
        honest_satisfiable: honest_ok,
    })
}

#[derive(Clone, Debug)]
pub struct TamperReport {
    pub commitment_honest: [u8; 32],
    pub commitment_tampered: Option<[u8; 32]>,
    pub tampered_satisfiable: bool,
    pub honest_satisfiable: bool,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}
