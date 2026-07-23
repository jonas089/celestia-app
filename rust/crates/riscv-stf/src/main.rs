//! RV32IM CPU-verifier -> Expander GKR -> verify, with rsema1d as the INPUT PCS.
//!
//! Proving spine copied from `stf-circuit` (the keccak template): ECC frontend
//! over GF2 -> `export_to_expander_flatten` -> install the trace as
//! `layers[0].input_vals` (GF2x8) -> `executor::prove::<Rsema1dGKRConfig>` ->
//! verify -> assert the GKR input commitment is byte-identical to the independent
//! Go/DA `rsema1d_sys::commit` over the same bit-packed rows.

mod circuit;
mod emulator;

use arith::{Field, SimdField};
use circuit::{RiscvCircuit, CYCLES, NOUT, OUT_REGS, XLEN};
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use polynomials::MultiLinearPoly;
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};
use std::time::Instant;

fn hex_str(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

/// Peak resident set size of this process, in MiB (macOS: ru_maxrss is bytes).
fn peak_rss_mib() -> f64 {
    unsafe {
        let mut ru: libc_rusage = std::mem::zeroed();
        getrusage(0, &mut ru);
        ru.ru_maxrss as f64 / (1024.0 * 1024.0)
    }
}
#[repr(C)]
struct libc_rusage {
    ru_utime: [i64; 2],
    ru_stime: [i64; 2],
    ru_maxrss: i64,
    _rest: [i64; 14],
}
extern "C" {
    fn getrusage(who: i32, usage: *mut libc_rusage) -> i32;
}

fn main() {
    println!("==== RV32IM CPU-verifier over Expander GKR + rsema1d input PCS ====");
    println!("[step] A: arithmetic-only straight-line program, register consistency via in-circuit state threading\n");

    // ---- 1. Assemble a REAL arithmetic program (RV32I) ------------------------
    use emulator::*;
    let program: Vec<u32> = vec![
        addi(1, 0, 5),   // x1 = 5
        addi(2, 0, 37),  // x2 = 37
        add(3, 1, 2),    // x3 = x1 + x2 = 42
        sub(4, 2, 1),    // x4 = x2 - x1 = 32
        xor(5, 1, 2),    // x5 = x1 ^ x2 = 32
        and(6, 1, 2),    // x6 = x1 & x2 = 5
        or(7, 1, 2),     // x7 = x1 | x2 = 37
        sll(8, 2, 1),    // x8 = x2 << (x1 & 31) = 37 << 5 = 1184
    ];
    assert_eq!(program.len(), CYCLES, "program length must equal CYCLES");
    for (i, w) in program.iter().enumerate() {
        unsafe { circuit::PROGRAM[i] = *w };
        println!("[program] pc={:#06x}  insn={:#010x}", i * 4, w);
    }

    // ---- 2. Emulate: native execution -> flat trace ---------------------------
    let mut cpu = Cpu::new(program.clone(), 0);
    let trace = cpu.run(CYCLES);
    println!("\n[emulate] {} cycles executed", trace.len());
    for r in &trace {
        println!(
            "[trace] c={} pc={:#06x} insn={:#010x} rd=x{} rs1=x{}({}) rs2=x{}({}) -> rd_val={}",
            r.cycle, r.pc, r.insn, r.rd_idx, r.rs1_idx, r.rs1_val, r.rs2_idx, r.rs2_val, r.rd_val
        );
    }
    // Native final register file (the ground-truth re-run).
    let native_regs = cpu.regs;
    println!("\n[emulate] native final regs of interest:");
    for &reg in OUT_REGS.iter() {
        println!("          x{} = {}", reg, native_regs[reg]);
    }

    // ---- 3. Compile the CPU-verifier circuit ----------------------------------
    let t_compile = Instant::now();
    let compile_result = compile(&RiscvCircuit::default(), CompileOptions::default()).unwrap();
    let CompileResult { witness_solver, layered_circuit } = compile_result;
    println!("\n[compile] layered circuit compiled in {:?}", t_compile.elapsed());

    // ---- 4. Build the witness assignment from the trace -----------------------
    // The committed columns are insn/rs1_val/rs2_val/rd_val per cycle. Public
    // outputs are the final values of OUT_REGS.
    let out_vals: Vec<u32> = OUT_REGS.iter().map(|&r| native_regs[r]).collect();
    let assignment = circuit::build_assignment(&trace, &out_vals);
    // 8 identical SIMD lanes (full GF2x8 pack), matching the keccak template.
    let assignments = vec![assignment.clone(); 8];

    let t_wit = Instant::now();
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();
    println!("[witness] solved {} witnesses in {:?}", witness.num_witnesses, t_wit.elapsed());
    println!(
        "[witness] num_inputs_per_witness = {}, num_public_inputs_per_witness = {}",
        witness.num_inputs_per_witness, witness.num_public_inputs_per_witness
    );

    let res = layered_circuit.run(&witness);
    assert!(res.iter().all(|x| *x), "layered circuit self-eval FAILED (trace not valid): {:?}", res);
    println!("[witness] layered_circuit.run() = {:?} (all CPU constraints satisfied)", res);

    // ---- 5. Export to Expander & install the trace as the input layer ---------
    let mut expander_circuit = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    expander_circuit.layers[0].input_vals = simd_input.clone();
    expander_circuit.public_input = simd_public_input.clone();
    expander_circuit.evaluate();

    let input_vals = expander_circuit.layers[0].input_vals.clone();
    let num_vars = expander_circuit.log_input_size();
    println!(
        "\n[export] num expander layers = {}, input_vals.len() = {}, num_vars = {}",
        expander_circuit.layers.len(),
        input_vals.len(),
        num_vars
    );
    println!("[export] committed trace rows (StepRecords) = {}, committed input bits = {}",
        CYCLES, input_vals.len() * 8);

    // ---- 6. Prove with rsema1d as the input PCS -------------------------------
    let mpi_config = MPIConfig::prover_new(None, None);
    println!("[mpi] world_size = {} (single-process = {})",
        mpi_config.world_size(), mpi_config.world_size() == 1);

    let t_prove = Instant::now();
    let (claimed_v, proof) =
        executor::prove::<Rsema1dGKRConfig<'static>>(&mut expander_circuit, mpi_config.clone());
    let prove_dur = t_prove.elapsed();
    println!("[prove] engine = Rsema1dGKRConfig (PCS = {})", Rsema1dPCS::NAME);
    println!("[prove] DONE in {:?}, proof bytes = {}", prove_dur, proof.bytes.len());

    let t_verify = Instant::now();
    let ok = executor::verify::<Rsema1dGKRConfig<'static>>(
        &mut expander_circuit, mpi_config.clone(), &proof, &claimed_v);
    let verify_dur = t_verify.elapsed();
    println!("[verify] result = {} in {:?}", ok, verify_dur);
    assert!(ok, "verifier REJECTED the honest proof");
    // For a zero-check constraint circuit the claimed output MLE must be zero
    // (all assert_is_equal satisfied). This is the value a verifier binds to.
    let honest_zero = claimed_v.is_zero();
    println!("[verify] VERIFIER ACCEPTED (rsema1d input commitment); claimed output == 0 : {}", honest_zero);
    assert!(honest_zero, "honest proof did not claim zero output");

    // ---- 7. Commitment reconciliation vs independent Go/DA rsema1d ------------
    let params = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi_config.world_size());
    let mut sp = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi_config);
    let input_poly = MultiLinearPoly::new(input_vals.clone());
    let c_prover = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::commit(
        &params, &mpi_config, &(), &input_poly, &mut sp)
        .expect("Rsema1dPCS::commit (prover path)");
    println!("\n[commit] GKR prover input commitment = {}", hex_str(&c_prover.root));

    let embedded = proof.bytes.windows(32).any(|w| w == c_prover.root);
    println!("[commit] commitment found verbatim in proof.bytes = {}", embedded);
    assert!(embedded, "prover commitment not found in proof transcript");

    // Independent DA path: bit-pack input_vals into K+N rsema1d rows exactly as
    // Rsema1dPCS does, and call the raw Go FFI. K = 2^(num_vars-2), N = K.
    const NUM_SYMBOLS: usize = 32;
    const ROW_BYTES: usize = 64;
    let k: u32 = 1u32 << (num_vars - 2);
    let n: u32 = k;
    let bits: Vec<[u8; 8]> = input_vals
        .iter()
        .map(|e| {
            let lanes = e.unpack();
            let mut b = [0u8; 8];
            for s in 0..8 {
                b[s] = lanes[s].v & 1;
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
    let (c_da, _handle) = rsema1d_sys::commit(k, n, &row_refs).expect("rsema1d_sys::commit (DA path)");
    println!("[commit] Go/DA rsema1d commitment      = {}", hex_str(&c_da));
    assert_eq!(c_prover.root, c_da, "prover commitment != Go/DA rsema1d commitment");
    println!("[commit] BYTE-IDENTICAL (K = {}, N = {})", k, n);

    // ---- 8. SOUNDNESS: tamper a committed trace value -> verify must FAIL ------
    println!("\n[soundness] tampering one committed rd_val bit (x3's result: 42 -> 43)...");
    let mut bad = assignment.clone();
    // Cycle 2 is `add x3,x1,x2` producing rd_val = 42; tamper the committed value to 43.
    assert_eq!(trace[2].rd_val, 42, "expected x3=42 at cycle 2");
    circuit::tamper_rd_val(&mut bad, 2, 43);
    let bad_assignments = vec![bad; 8];
    let bad_witness = witness_solver.solve_witnesses(&bad_assignments).unwrap();
    let bad_run = layered_circuit.run(&bad_witness);
    println!("[soundness] tampered layered_circuit.run() = {:?}", bad_run);
    let all_ok = bad_run.iter().all(|x| *x);
    println!("[soundness] tampered trace satisfies constraints = {}", all_ok);

    // Prove & verify the tampered circuit: the GKR proof of a constraint-violating
    // trace must be rejected by the verifier (output claim != all-zero).
    let mut bad_circuit = layered_circuit.export_to_expander_flatten();
    let (bad_simd_input, bad_simd_pub) = bad_witness.to_simd::<gf2::GF2x8>();
    bad_circuit.layers[0].input_vals = bad_simd_input;
    bad_circuit.public_input = bad_simd_pub;
    bad_circuit.evaluate();
    let (bad_v, _bad_proof) =
        executor::prove::<Rsema1dGKRConfig<'static>>(&mut bad_circuit, mpi_config.clone());
    // The GKR proof faithfully proves the (tampered) computation, so verify() of
    // the proof against its OWN claimed output is vacuously true. The soundness
    // signal is the claimed OUTPUT: a valid execution forces the zero-check output
    // to 0; the tampered trace forces it NON-zero, so any verifier that binds
    // "output must be 0" REJECTS. (Also the DA commitment changes, below.)
    let bad_zero = bad_v.is_zero();
    println!("[soundness] tampered-trace claimed output == 0 : {} (expected false)", bad_zero);

    // The tampered trace is also a DIFFERENT committed input => different DA commitment.
    let bad_input = bad_circuit.layers[0].input_vals.clone();
    let bad_bits: Vec<[u8; 8]> = bad_input.iter().map(|e| {
        let l = e.unpack(); let mut b = [0u8; 8]; for s in 0..8 { b[s] = l[s].v & 1; } b
    }).collect();
    let mut bad_rows = vec![vec![0u8; ROW_BYTES]; (k + n) as usize];
    for j in 0..(k as usize) { for i in 0..NUM_SYMBOLS { let a = j * NUM_SYMBOLS + i; bad_rows[j][i] = bad_bits[a >> 3][a & 7]; } }
    let bad_refs: Vec<&[u8]> = bad_rows.iter().map(|r| r.as_slice()).collect();
    let (c_bad, _h) = rsema1d_sys::commit(k, n, &bad_refs).expect("commit bad");
    println!("[soundness] tampered DA commitment = {} (honest {})",
        &hex_str(&c_bad)[..16], &hex_str(&c_da)[..16]);

    assert!(!all_ok, "SOUNDNESS BROKEN: tampered trace still satisfied constraints");
    assert!(!bad_zero, "SOUNDNESS BROKEN: tampered trace claimed zero output");
    assert_ne!(c_bad, c_da, "tampered commitment unexpectedly matched honest");
    println!("[soundness] tampered trace REJECTED: constraints unsatisfied (run=all-false),");
    println!("            claimed output != 0 (zero-check fails), and DA commitment differs");

    // ---- 9. Output-match gate: circuit public output == native re-run ---------
    let mut recon = [0u32; NOUT];
    for k in 0..NOUT {
        let mut w = 0u32;
        for b in 0..XLEN {
            let bit = simd_public_input[k * XLEN + b].unpack()[0].v & 1;
            w |= (bit as u32) << b;
        }
        recon[k] = w;
    }
    println!("\n[output] circuit public output vs native re-run:");
    for (k, &reg) in OUT_REGS.iter().enumerate() {
        let m = recon[k] == native_regs[reg];
        println!("         x{} : circuit={} native={} match={}", reg, recon[k], native_regs[reg], m);
        assert_eq!(recon[k], native_regs[reg], "public output != native re-run for x{}", reg);
    }

    println!("\n=== STEP A GATES ===");
    println!("(1) Expander verifier ACCEPTED the proof (rsema1d input PCS)   in {:?}", verify_dur);
    println!("(2) GKR input commitment == Go/DA rsema1d commitment           BYTE-IDENTICAL");
    println!("(3) tampered trace REJECTED (soundness)                        OK");
    println!("(4) circuit public output == native re-run                    OK");
    println!("[stats] num_vars = {}, trace rows = {}, prove = {:?}, verify = {:?}, peak RSS = {:.1} MiB",
        num_vars, CYCLES, prove_dur, verify_dur, peak_rss_mib());
}
