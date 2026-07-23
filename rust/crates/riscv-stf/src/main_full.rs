//! Full RV32IM-subset CPU-verifier pipeline (Steps B + C): a REAL looping program
//! (sum an array via a BEQ/JAL loop with LW loads and a final SW), proven over
//! Expander GKR with rsema1d as the input PCS. Same proving spine as the keccak
//! template / Step A.

mod circuit_full;
mod emulator;

use arith::{Field, SimdField};
use circuit_full as ckt;
use circuit_full::{RiscvCircuitFull, CYCLES, NOUT, XLEN};
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use polynomials::MultiLinearPoly;
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};
use std::time::Instant;

fn hex_str(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

#[repr(C)]
struct Rusage { _t: [i64; 4], ru_maxrss: i64, _rest: [i64; 14] }
extern "C" { fn getrusage(who: i32, usage: *mut Rusage) -> i32; }
fn peak_rss_mib() -> f64 {
    unsafe { let mut ru: Rusage = std::mem::zeroed(); getrusage(0, &mut ru); ru.ru_maxrss as f64 / (1024.0 * 1024.0) }
}

fn main() {
    println!("==== RV32IM CPU-verifier (Steps B+C): looping array-sum program ====");
    println!("[step] control flow (BEQ/JAL) + memory (LW/SW) + arithmetic; register/memory/pc consistency via in-circuit state threading + ROM fetch\n");

    use emulator::*;

    // ---- 1. Assemble a REAL looping program (sum array[0..4]) -----------------
    // regs: x1=ptr, x2=sum, x3=i, x4=N, x5=tmp, x6=result addr
    let program: Vec<u32> = vec![
        addi(4, 0, 4),                              // 0: x4 = N = 4
        addi(1, 0, ckt::DATA_BASE as i32),          // 1: x1 = DATA_BASE
        addi(6, 0, ckt::RESULT_ADDR as i32),        // 2: x6 = RESULT_ADDR
        lw(5, 1, 0),                                // 3: x5 = mem[x1]      (loop head)
        add(2, 2, 5),                               // 4: x2 += x5
        addi(1, 1, 4),                              // 5: x1 += 4
        addi(3, 3, 1),                              // 6: x3 += 1
        beq(3, 4, 8),                               // 7: if x3==x4 -> idx9 (pc+8)
        jal(0, -20),                                // 8: -> idx3 (pc-20)
        sw(2, 6, 0),                                // 9: mem[x6] = x2 (sum)
        jal(0, 0),                                  // 10: halt (self-loop)
    ];
    assert_eq!(program.len(), ckt::PROG_LEN);
    for (i, w) in program.iter().enumerate() {
        unsafe { ckt::PROGRAM[i] = *w };
        println!("[program] idx{:<2} pc={:#06x}  insn={:#010x}", i, i * 4, w);
    }

    let array: [u32; 4] = [10, 20, 30, 40];
    let mem_init = [array[0], array[1], array[2], array[3], 0u32];
    for k in 0..ckt::NMEM {
        unsafe { ckt::MEM_INIT[k] = mem_init[k] };
    }
    println!("\n[data] array @ {:#x} = {:?}, expected sum = {}", ckt::DATA_BASE, array, array.iter().sum::<u32>());

    // ---- 2. Emulate -----------------------------------------------------------
    let mut cpu = Cpu::new(program.clone(), ckt::BASE);
    for (k, &v) in array.iter().enumerate() {
        cpu.mem.store(ckt::DATA_BASE + 4 * k as u32, v);
    }
    let trace = cpu.run(CYCLES);
    println!("\n[emulate] {} cycles (incl. halt padding):", trace.len());
    for r in &trace {
        let tag = if r.is_load { "LW" } else if r.is_store { "SW" } else if r.opcode == OPC_BRANCH { "BEQ" } else if r.opcode == OPC_JAL { "JAL" } else { "alu" };
        println!("[trace] c={:<2} pc={:#06x} insn={:#010x} {:<3} rd=x{:<2} rs1=x{}({}) rs2=x{}({}) rd_val={} next_pc={:#06x} mem[{:#x}]={}",
            r.cycle, r.pc, r.insn, tag, r.rd_idx, r.rs1_idx, r.rs1_val, r.rs2_idx, r.rs2_val, r.rd_val, r.next_pc, r.mem_addr, r.mem_val);
    }
    let native_sum = cpu.regs[2];
    let native_i = cpu.regs[3];
    let native_mem_result = cpu.mem.load(ckt::RESULT_ADDR);
    println!("\n[emulate] native: x2(sum)={}, x3(i)={}, mem[{:#x}]={}", native_sum, native_i, ckt::RESULT_ADDR, native_mem_result);
    assert_eq!(native_sum, 100);
    let out_vals = [native_sum, native_i, native_mem_result];

    // ---- 3. Compile -----------------------------------------------------------
    let t_compile = Instant::now();
    let CompileResult { witness_solver, layered_circuit } =
        compile(&RiscvCircuitFull::default(), CompileOptions::default()).unwrap();
    println!("\n[compile] layered circuit compiled in {:?}", t_compile.elapsed());

    // ---- 4. Witness -----------------------------------------------------------
    let assignment = ckt::build_assignment(&trace, &out_vals);
    let assignments = vec![assignment.clone(); 8];
    let t_wit = Instant::now();
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();
    println!("[witness] solved {} witnesses in {:?}; num_inputs_per_witness = {}",
        witness.num_witnesses, t_wit.elapsed(), witness.num_inputs_per_witness);
    let res = layered_circuit.run(&witness);
    assert!(res.iter().all(|x| *x), "layered circuit self-eval FAILED: {:?}", res);
    println!("[witness] layered_circuit.run() = all true (execution valid)");

    // ---- 5. Export & install trace as input layer -----------------------------
    let mut ec = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = simd_input.clone();
    ec.public_input = simd_public_input.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let num_vars = ec.log_input_size();
    println!("\n[export] expander layers = {}, input_vals.len() = {}, num_vars = {}, committed input bits = {}",
        ec.layers.len(), input_vals.len(), num_vars, input_vals.len() * 8);

    // ---- 6. Prove & verify ----------------------------------------------------
    let mpi = MPIConfig::prover_new(None, None);
    println!("[mpi] world_size = {}", mpi.world_size());
    let t_prove = Instant::now();
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let prove_dur = t_prove.elapsed();
    println!("[prove] DONE in {:?}, proof bytes = {}", prove_dur, proof.bytes.len());
    let t_verify = Instant::now();
    let ok = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v);
    let verify_dur = t_verify.elapsed();
    assert!(ok, "verifier REJECTED honest proof");
    let honest_zero = claimed_v.is_zero();
    println!("[verify] result = {} in {:?}; claimed output == 0 : {}", ok, verify_dur, honest_zero);
    assert!(honest_zero);

    // ---- 7. Commitment reconciliation -----------------------------------------
    let params = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi.world_size());
    let mut sp = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
    let input_poly = MultiLinearPoly::new(input_vals.clone());
    let c_prover = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &input_poly, &mut sp).unwrap();
    println!("\n[commit] GKR prover input commitment = {}", hex_str(&c_prover.root));
    assert!(proof.bytes.windows(32).any(|w| w == c_prover.root), "commitment not in proof");

    const NUM_SYMBOLS: usize = 32;
    const ROW_BYTES: usize = 64;
    let k: u32 = 1u32 << (num_vars - 2);
    let n: u32 = k;
    let pack = |vals: &[gf2::GF2x8]| -> Vec<[u8; 8]> {
        vals.iter().map(|e| { let l = e.unpack(); let mut b = [0u8; 8]; for s in 0..8 { b[s] = l[s].v & 1; } b }).collect()
    };
    let to_rows = |bits: &[[u8; 8]]| -> Vec<Vec<u8>> {
        let mut rows = vec![vec![0u8; ROW_BYTES]; (k + n) as usize];
        for j in 0..(k as usize) { for i in 0..NUM_SYMBOLS { let a = j * NUM_SYMBOLS + i; rows[j][i] = bits[a >> 3][a & 7]; } }
        rows
    };
    let rows = to_rows(&pack(&input_vals));
    let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (c_da, _h) = rsema1d_sys::commit(k, n, &row_refs).unwrap();
    println!("[commit] Go/DA rsema1d commitment      = {}", hex_str(&c_da));
    assert_eq!(c_prover.root, c_da, "prover commitment != Go/DA commitment");
    println!("[commit] BYTE-IDENTICAL (K = {}, N = {})", k, n);

    // ---- 8. Soundness: tamper a memory-load value in the trace ----------------
    // Cycle of the first LW (idx3 executes at cycle 3) loaded 10 into x5; tamper
    // the committed rd_val there to 11.
    let lw_cycle = trace.iter().position(|r| r.is_load).unwrap();
    println!("\n[soundness] tampering first LW's committed rd_val at cycle {} ({} -> {})...",
        lw_cycle, trace[lw_cycle].rd_val, trace[lw_cycle].rd_val + 1);
    let mut bad = assignment.clone();
    ckt::tamper_rd_val(&mut bad, lw_cycle, trace[lw_cycle].rd_val + 1);
    let bad_witness = witness_solver.solve_witnesses(&vec![bad; 8]).unwrap();
    let bad_run = layered_circuit.run(&bad_witness);
    let all_ok = bad_run.iter().all(|x| *x);
    println!("[soundness] tampered layered_circuit.run() all-true = {}", all_ok);
    let mut bad_ec = layered_circuit.export_to_expander_flatten();
    let (bi, bp) = bad_witness.to_simd::<gf2::GF2x8>();
    bad_ec.layers[0].input_vals = bi.clone();
    bad_ec.public_input = bp;
    bad_ec.evaluate();
    let (bad_v, _bproof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut bad_ec, mpi.clone());
    let bad_zero = bad_v.is_zero();
    let bad_rows = to_rows(&pack(&bad_ec.layers[0].input_vals));
    let bad_refs: Vec<&[u8]> = bad_rows.iter().map(|r| r.as_slice()).collect();
    let (c_bad, _h2) = rsema1d_sys::commit(k, n, &bad_refs).unwrap();
    println!("[soundness] tampered claimed output == 0 : {} (expected false)", bad_zero);
    println!("[soundness] tampered DA commitment {} != honest {}", &hex_str(&c_bad)[..16], &hex_str(&c_da)[..16]);
    assert!(!all_ok && !bad_zero && c_bad != c_da, "SOUNDNESS BROKEN");
    println!("[soundness] tampered trace REJECTED (constraints fail + non-zero output + commitment differs)");

    // ---- 9. Output-match gate -------------------------------------------------
    let mut recon = [0u32; NOUT];
    for k in 0..NOUT {
        let mut w = 0u32;
        for b in 0..XLEN { w |= ((simd_public_input[k * XLEN + b].unpack()[0].v & 1) as u32) << b; }
        recon[k] = w;
    }
    println!("\n[output] circuit public output vs native re-run:");
    println!("         sum x2       : circuit={} native={} match={}", recon[0], native_sum, recon[0] == native_sum);
    println!("         count x3     : circuit={} native={} match={}", recon[1], native_i, recon[1] == native_i);
    println!("         mem[result]  : circuit={} native={} match={}", recon[2], native_mem_result, recon[2] == native_mem_result);
    assert_eq!(recon, [native_sum, native_i, native_mem_result]);

    println!("\n=== STEP C GATES (largest step reached) ===");
    println!("(1) Expander verifier ACCEPTED the proof (rsema1d input PCS)   OK in {:?}", verify_dur);
    println!("(2) GKR input commitment == Go/DA rsema1d commitment           BYTE-IDENTICAL");
    println!("(3) tampered trace REJECTED (soundness)                        OK");
    println!("(4) program output (sum={}) == native re-run                 OK", native_sum);
    println!("[stats] num_vars = {}, trace rows = {}, prove = {:?}, verify = {:?}, peak RSS = {:.1} MiB",
        num_vars, CYCLES, prove_dur, verify_dur, peak_rss_mib());
}
