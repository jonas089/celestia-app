//! R4 driver — data-parallel batch keccak-256 over the real block-31 DA blob.
//! Committed input = the blob rows; rsema1d/DA is the SOLE polynomial commitment
//! (opened at the sumcheck point). Verifies each row digest is a real keccak,
//! reconciles the GKR input commitment byte-for-byte against an independent
//! Go/DA rsema1d commit, and demonstrates tamper rejection.

use arith::SimdField;
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use polynomials::MultiLinearPoly;
use riscv_stf::batch_keccak::{build_assignment, BatchKeccakCircuit, N_ROWS, ROW};
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};
use std::time::Instant;
use tiny_keccak::Hasher;

const BLOB: &[u8] = include_bytes!("../testdata/block-31-blob.bin");

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn keccak256(bytes: &[u8]) -> [u8; 32] {
    let mut h = tiny_keccak::Keccak::v256();
    h.update(bytes);
    let mut o = [0u8; 32];
    h.finalize(&mut o);
    o
}

/// Independent Go/DA rsema1d commit of the committed input layer, in the exact
/// bit-packed square rsema1d-pcs commits (K=N=2^(num_vars-2), numSymbols=32).
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
    let (c_da, _h) = rsema1d_sys::commit(k, n, &row_refs).map_err(|e| format!("rsema1d commit failed: {e}"))?;
    Ok(c_da)
}

fn main() {
    println!("=== R4: data-parallel batch keccak over committed DA block data ===");
    println!("[input] block-31 blob = {} bytes; ROW = {} bytes; N_ROWS = {}", BLOB.len(), ROW, N_ROWS);

    // Split the real blob into N_ROWS rows of ROW bytes (last row zero-padded).
    let mut rows = [[0u8; ROW]; N_ROWS];
    for r in 0..N_ROWS {
        for c in 0..ROW {
            let idx = r * ROW + c;
            if idx < BLOB.len() {
                rows[r][c] = BLOB[idx];
            }
        }
    }
    // Native keccak per row (the reference the circuit must reproduce).
    let mut digests = [[0u8; 32]; N_ROWS];
    for r in 0..N_ROWS {
        digests[r] = keccak256(&rows[r]);
    }
    println!("[input] row 0 keccak256 = {}", hex(&digests[0]));
    println!("[input] row {} keccak256 = {}", N_ROWS - 1, hex(&digests[N_ROWS - 1]));

    // Compile.
    let t = Instant::now();
    let CompileResult { witness_solver, layered_circuit } =
        compile(&BatchKeccakCircuit::default(), CompileOptions::default()).expect("compile");
    println!("[compile] {} keccak instances compiled in {:?}", N_ROWS, t.elapsed());

    // Witness (8 identical SIMD copies).
    let assignment = build_assignment(&rows, &digests);
    let t = Instant::now();
    let witness = witness_solver.solve_witnesses(&vec![assignment; 8]).expect("solve");
    let ok = layered_circuit.run(&witness);
    assert!(ok.iter().all(|x| *x), "layered self-eval failed: {ok:?}");
    println!("[witness] solved + self-eval OK in {:?}", t.elapsed());

    // Export & install the committed block-data rows as the input layer.
    let mut ec = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = simd_input.clone();
    ec.public_input = simd_public_input.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let num_vars = ec.log_input_size();
    println!("[commit] committed input: {} GF2x8 (num_vars = {})", input_vals.len(), num_vars);

    // DA ENCODER STEP: encode once, install commitment+handle for the prover to reuse.
    let da_poly = MultiLinearPoly::new(input_vals.clone());
    let _da_root = rsema1d_pcs::install_da_commitment(num_vars, &da_poly);
    println!("[DA] encoded once + installed handle; GKR prover will reuse it (zero encoding)");

    // Prove (rsema1d = sole PCS, REUSING the DA handle) + verify.
    let mpi = MPIConfig::prover_new(None, None);
    let t = Instant::now();
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    println!("[prove] done in {:?}, proof bytes = {}", t.elapsed(), proof.bytes.len());
    let t = Instant::now();
    let verified = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v)
        && claimed_v.is_zero();
    println!("[verify] Expander verifier accepted (rsema1d sole PCS) = {} in {:?}", verified, t.elapsed());
    assert!(verified, "verifier REJECTED the honest proof");

    // Commitment reconciliation: GKR PCS root == independent Go/DA rsema1d commit.
    let params = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi.world_size());
    let mut sp = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
    let input_poly = MultiLinearPoly::new(input_vals.clone());
    let c_prover = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &input_poly, &mut sp)
        .expect("Rsema1dPCS::commit");
    let c_da = da_commit(&input_vals, num_vars).expect("da_commit");
    println!("[commit] GKR input commitment = {}", hex(&c_prover.root));
    println!("[commit] independent Go/DA    = {}", hex(&c_da));
    assert_eq!(c_prover.root, c_da, "GKR input commitment != Go/DA rsema1d commit");
    assert!(proof.bytes.windows(32).any(|w| w == c_prover.root), "commitment not embedded in proof");

    // Reconstruct each row digest from the circuit public outputs; check real.
    let mut all_ok = true;
    for r in 0..N_ROWS {
        let mut d = [0u8; 32];
        for b in 0..32 {
            for j in 0..8 {
                let bit = simd_public_input[r * 256 + b * 8 + j].unpack()[0].v & 1;
                d[b] |= bit << j;
            }
        }
        if d != digests[r] {
            all_ok = false;
            println!("[keccak] row {r} MISMATCH: circuit {} != native {}", hex(&d), hex(&digests[r]));
        }
    }
    assert!(all_ok, "some row digest was not a real keccak256");
    println!("[keccak] all {} row digests == keccak256(row): MATCH", N_ROWS);

    // Soundness: tamper one committed byte; the honest-output witness is unsat
    // and the commitment changes.
    println!("\n=== Soundness gate ===");
    let mut trows = rows;
    trows[0][0] ^= 0x01;
    let tampered = build_assignment(&trows, &digests); // keep honest digests
    let solved = witness_solver.solve_witnesses(&vec![tampered; 8]);
    let (sat, c_tampered) = match &solved {
        Ok(w) => {
            let sat = layered_circuit.run(w).iter().all(|x| *x);
            let (si2, _p) = w.to_simd::<gf2::GF2x8>();
            (sat, Some(da_commit(&si2, num_vars).expect("da_commit tampered")))
        }
        Err(_) => (false, None),
    };
    println!("[tamper] tampered-data / honest-output witness satisfiable = {} (must be false)", sat);
    assert!(!sat, "SOUNDNESS BREAK: tampered rows satisfied the honest digests");
    if let Some(c) = c_tampered {
        println!("[tamper] tampered commitment = {}", hex(&c));
        assert_ne!(c, c_prover.root, "tampered data produced the same commitment");
        println!("[tamper] commitment differs from honest: reuse detects data changes");
    }

    println!("\n=== ALL GATES PASSED (R4) ===");
    println!("(a) verifier ACCEPTED; rsema1d = the ONLY polynomial commitment");
    println!("(b) committed input IS the real block-31 blob rows ({} bytes)", BLOB.len());
    println!("(c) {} data-parallel keccak instances; every digest == keccak256(row)", N_ROWS);
    println!("(d) GKR input commitment == independent Go/DA rsema1d commit (byte-identical => reuse)");
    println!("(e) tamper => unsatisfiable AND changes the commitment");
}
