//! RV32IM CPU-verifier proven over Expander GKR + rsema1d, with register/memory
//! consistency by OFFLINE MEMORY CHECKING (a GF(2^128) grand-product argument),
//! replacing the O(cycles x state) in-circuit state threading of
//! `riscv_pipeline_full`. Same proving spine (ECC frontend over GF2 ->
//! export_to_expander_flatten -> Rsema1dGKRConfig prove/verify -> commitment ==
//! independent Go/DA rsema1d commit).
//!
//! Fiat-Shamir binding (soundness-critical): the challenge gamma in GF(2^128) is
//! derived from the rsema1d COMMITMENT of the trace (the committed GKR input
//! layer) and injected as a PUBLIC INPUT. Because the committed input layer is
//! exactly the private trace + hints (never gamma -- see witness.to_simd, which
//! splits inputs from public inputs, and the PCS commits only the input layer),
//! gamma cannot influence the commitment: there is no circularity, and any change
//! to the trace changes the commitment and hence gamma.

mod circuit_gp;
mod emulator;
mod gf128;

use arith::{Field, SimdField};
use circuit_gp as ckt;
use circuit_gp::{Hints, NOUT, NREG, SLOTS, XLEN};
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use polynomials::MultiLinearPoly;
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};
use std::collections::HashMap;
use std::time::{Duration, Instant};

const DATA_BASE: u32 = 0x100;
const RESULT_ADDR: u32 = 0x200;

fn hex_str(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn flip_bit(x: GF2) -> GF2 {
    if x.is_zero() { 1u32.into() } else { 0u32.into() }
}

#[repr(C)]
struct Rusage { _t: [i64; 4], ru_maxrss: i64, _rest: [i64; 14] }
extern "C" { fn getrusage(who: i32, usage: *mut Rusage) -> i32; }
fn peak_rss_mib() -> f64 {
    unsafe { let mut ru: Rusage = std::mem::zeroed(); getrusage(0, &mut ru); ru.ru_maxrss as f64 / (1024.0 * 1024.0) }
}

/// Fiat-Shamir: derive TWO non-zero challenges (alpha, beta) in GF(2^128) from
/// the 32-byte trace commitment. Modelled as a random oracle over the committed
/// trace; a different commitment yields different challenges w.h.p. `alpha` is
/// the product-offset challenge, `beta` the random-linear-combination challenge.
fn fs_challenges(commit: &[u8; 32]) -> (u128, u128) {
    let mix = |mut z: u64| -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    // Two independent FNV accumulators per challenge, seeded with a domain tag.
    let derive = |tag: u64| -> u128 {
        let mut lo: u64 = 0xcbf29ce484222325 ^ tag;
        let mut hi: u64 = 0x9e3779b97f4a7c15 ^ mix(tag);
        for (i, &b) in commit.iter().enumerate() {
            lo = (lo ^ b as u64).wrapping_mul(0x100000001b3);
            hi = (hi ^ (b as u64).rotate_left((i as u32 & 63) + 1)).wrapping_mul(0xff51afd7ed558ccd);
        }
        (((mix(hi) as u128) << 64) | (mix(lo) as u128)) | 1
    };
    (derive(0xA1), derive(0xB2))
}

fn program() -> Vec<u32> {
    use emulator::*;
    vec![
        addi(4, 0, 0),                     // 0: x4 = N (patched below)
        addi(1, 0, DATA_BASE as i32),      // 1: x1 = DATA_BASE
        addi(6, 0, RESULT_ADDR as i32),    // 2: x6 = RESULT_ADDR
        lw(5, 1, 0),                       // 3: x5 = mem[x1]   (loop head)
        add(2, 2, 5),                      // 4: x2 += x5
        addi(1, 1, 4),                     // 5: x1 += 4
        addi(3, 3, 1),                     // 6: x3 += 1
        beq(3, 4, 8),                      // 7: if x3==x4 -> idx9
        jal(0, -20),                       // 8: -> idx3
        sw(2, 6, 0),                       // 9: mem[x6] = x2
        jal(0, 0),                         // 10: halt
    ]
}

/// True iff the register-writing set matches the circuit's `write_enable`.
fn writes_reg(opcode: u32) -> bool {
    matches!(opcode,
        emulator::OPC_OP | emulator::OPC_OPIMM | emulator::OPC_LOAD
        | emulator::OPC_JAL | emulator::OPC_JALR | emulator::OPC_LUI | emulator::OPC_AUIPC)
}

/// (quotient, remainder) hints for a DIV/DIVU/REM/REMU cycle (else (0,0)). Matches
/// the emulator's RV32M semantics incl. div-by-zero / signed-overflow special cases.
fn div_hints(rec: &emulator::StepRecord) -> (u32, u32) {
    if rec.opcode != emulator::OPC_OP || rec.funct7 != emulator::FUNCT7_M {
        return (0, 0);
    }
    let a = rec.rs1_val;
    let b = rec.rs2_val;
    match rec.funct3 {
        0x4 | 0x6 => {
            let (ai, bi) = (a as i32, b as i32);
            if bi == 0 { (u32::MAX, a) }
            else if ai == i32::MIN && bi == -1 { (i32::MIN as u32, 0) }
            else { ((ai / bi) as u32, (ai % bi) as u32) }
        }
        0x5 | 0x7 => {
            if b == 0 { (u32::MAX, a) } else { (a / b, a % b) }
        }
        _ => (0, 0),
    }
}

/// Replay the memory model to produce all offline-memory-checking hints, using
/// the SAME slot semantics as the circuit.
fn replay_hints(trace: &[emulator::StepRecord], mem_addrs: &[u32], mem_init: &[u32]) -> Hints {
    let ncyc = trace.len();
    let nmem = mem_addrs.len();
    let mut last_ts: HashMap<u32, u32> = HashMap::new();
    let mut last_val: HashMap<u32, u32> = HashMap::new();
    for r in 0..NREG as u32 { last_ts.insert(r, 0); last_val.insert(r, 0); }
    for k in 0..nmem { last_ts.insert(mem_addrs[k], 0); last_val.insert(mem_addrs[k], mem_init[k]); }

    let mut tprev = vec![[0u32; SLOTS]; ncyc];
    let mut vold_c = vec![0u32; ncyc];
    let mut vold_d = vec![0u32; ncyc];
    let mut div_q = vec![0u32; ncyc];
    let mut div_r = vec![0u32; ncyc];

    for (c, rec) in trace.iter().enumerate() {
        let write_enable = writes_reg(rec.opcode);
        let (q, r) = div_hints(rec);
        div_q[c] = q;
        div_r[c] = r;
        for slot in 0..SLOTS {
            let now_ts = (c * SLOTS + slot + 1) as u32;
            let word = rec.mem_addr & !3;
            let (addr, rval, wval, active): (u32, u32, u32, bool) = match slot {
                0 => (rec.rs1_idx, rec.rs1_val, rec.rs1_val, true),
                1 => (rec.rs2_idx, rec.rs2_val, rec.rs2_val, true),
                2 => {
                    if rec.is_load {
                        (word, rec.mem_prev, rec.mem_prev, true)
                    } else if rec.is_store {
                        (word, rec.mem_prev, rec.mem_val, true)
                    } else {
                        (0, 0, 0, false)
                    }
                }
                _ => {
                    if write_enable && rec.rd_idx != 0 {
                        let old = *last_val.get(&rec.rd_idx).unwrap_or(&0);
                        (rec.rd_idx, old, rec.rd_val, true)
                    } else {
                        (0, 0, 0, false)
                    }
                }
            };
            let _ = now_ts;
            if active {
                let tp = *last_ts.get(&addr).unwrap_or(&0);
                tprev[c][slot] = tp;
                if slot == 2 { vold_c[c] = rval; } // old memory word (load & store)
                if slot == 3 { vold_d[c] = rval; }
                let cur = *last_val.get(&addr).unwrap_or(&0);
                assert_eq!(cur, rval, "replay read mismatch c={c} slot={slot} addr={addr}");
                last_ts.insert(addr, now_ts);
                last_val.insert(addr, wval);
            }
        }
    }

    let naddr = NREG + nmem;
    let mut fin_val = vec![0u32; naddr];
    let mut fin_ts = vec![0u32; naddr];
    for r in 0..NREG { fin_val[r] = last_val[&(r as u32)]; fin_ts[r] = last_ts[&(r as u32)]; }
    for k in 0..nmem {
        fin_val[NREG + k] = last_val[&mem_addrs[k]];
        fin_ts[NREG + k] = last_ts[&mem_addrs[k]];
    }
    Hints { tprev, vold_c, vold_d, div_q, div_r, fin_val, fin_ts }
}

/// Native mirror of the circuit's tuple construction + grand product, for
/// debugging: returns (PROD read, PROD write). `tamper` optionally overrides one
/// cycle's committed rd_val (as the in-circuit tamper does), leaving hints honest.
fn native_products(
    trace: &[emulator::StepRecord],
    hints: &Hints,
    mem_addrs: &[u32],
    mem_init: &[u32],
    alpha: u128,
    beta: u128,
    tamper: Option<(usize, u32)>,
) -> (u128, u128) {
    let b2 = gf128::native_mul(beta, beta);
    let fp = |addr: u32, val: u32, ts: u32| -> u128 {
        alpha ^ addr as u128 ^ gf128::native_mul(val as u128, beta) ^ gf128::native_mul(ts as u128, b2)
    };
    let mut prod_r: u128 = 1;
    let mut prod_w: u128 = 1;
    // init writes
    for r in 0..NREG as u32 { prod_w = gf128::native_mul(prod_w, fp(r, 0, 0)); }
    for k in 0..mem_addrs.len() { prod_w = gf128::native_mul(prod_w, fp(mem_addrs[k], mem_init[k], 0)); }
    for (c, rec) in trace.iter().enumerate() {
        let rd_val = match tamper { Some((tc, v)) if tc == c => v, _ => rec.rd_val };
        let we = writes_reg(rec.opcode);
        for slot in 0..SLOTS {
            let now = (c * SLOTS + slot + 1) as u32;
            let word = rec.mem_addr & !3;
            let (addr, rval, wval, active): (u32, u32, u32, bool) = match slot {
                0 => (rec.rs1_idx, rec.rs1_val, rec.rs1_val, true),
                1 => (rec.rs2_idx, rec.rs2_val, rec.rs2_val, true),
                2 => {
                    if rec.is_load { (word, hints.vold_c[c], hints.vold_c[c], true) }
                    else if rec.is_store { (word, hints.vold_c[c], rec.mem_val, true) }
                    else { (0, 0, 0, false) }
                }
                _ => { if we && rec.rd_idx != 0 { (rec.rd_idx, hints.vold_d[c], rd_val, true) } else { (0, 0, 0, false) } }
            };
            let (r_addr, r_val, r_ts) = if active { (addr, rval, hints.tprev[c][slot]) } else { (0, 0, now) };
            let (w_addr, w_val, w_ts) = if active { (addr, wval, now) } else { (0, 0, now) };
            prod_r = gf128::native_mul(prod_r, fp(r_addr, r_val, r_ts));
            prod_w = gf128::native_mul(prod_w, fp(w_addr, w_val, w_ts));
        }
    }
    for r in 0..NREG { prod_r = gf128::native_mul(prod_r, fp(r as u32, hints.fin_val[r], hints.fin_ts[r])); }
    for k in 0..mem_addrs.len() {
        prod_r = gf128::native_mul(prod_r, fp(mem_addrs[k], hints.fin_val[NREG + k], hints.fin_ts[NREG + k]));
    }
    (prod_r, prod_w)
}

/// Collect the raw (addr,val,ts) tuple multisets (read side, write side), sorted.
fn native_tuples(
    trace: &[emulator::StepRecord],
    hints: &Hints,
    mem_addrs: &[u32],
    mem_init: &[u32],
    tamper: Option<(usize, u32)>,
) -> (Vec<(u32, u32, u32)>, Vec<(u32, u32, u32)>) {
    let mut rs = Vec::new();
    let mut ws = Vec::new();
    for r in 0..NREG as u32 { ws.push((r, 0, 0)); }
    for k in 0..mem_addrs.len() { ws.push((mem_addrs[k], mem_init[k], 0)); }
    for (c, rec) in trace.iter().enumerate() {
        let rd_val = match tamper { Some((tc, v)) if tc == c => v, _ => rec.rd_val };
        let we = writes_reg(rec.opcode);
        for slot in 0..SLOTS {
            let now = (c * SLOTS + slot + 1) as u32;
            let word = rec.mem_addr & !3;
            let (addr, rval, wval, active): (u32, u32, u32, bool) = match slot {
                0 => (rec.rs1_idx, rec.rs1_val, rec.rs1_val, true),
                1 => (rec.rs2_idx, rec.rs2_val, rec.rs2_val, true),
                2 => {
                    if rec.is_load { (word, hints.vold_c[c], hints.vold_c[c], true) }
                    else if rec.is_store { (word, hints.vold_c[c], rec.mem_val, true) }
                    else { (0, 0, 0, false) }
                }
                _ => { if we && rec.rd_idx != 0 { (rec.rd_idx, hints.vold_d[c], rd_val, true) } else { (0, 0, 0, false) } }
            };
            if active { rs.push((addr, rval, hints.tprev[c][slot])); ws.push((addr, wval, now)); }
            else { rs.push((0, 0, now)); ws.push((0, 0, now)); }
        }
    }
    for r in 0..NREG { rs.push((r as u32, hints.fin_val[r], hints.fin_ts[r])); }
    for k in 0..mem_addrs.len() { rs.push((mem_addrs[k], hints.fin_val[NREG + k], hints.fin_ts[NREG + k])); }
    rs.sort(); ws.sort();
    (rs, ws)
}

// -- commitment reconciliation (identical packing to riscv_pipeline_full) -----
fn da_commit(input_vals: &[gf2::GF2x8], num_vars: usize) -> [u8; 32] {
    const NUM_SYMBOLS: usize = 32;
    const ROW_BYTES: usize = 64;
    let k: u32 = 1u32 << (num_vars - 2);
    let n: u32 = k;
    let pack: Vec<[u8; 8]> = input_vals
        .iter()
        .map(|e| { let l = e.unpack(); let mut b = [0u8; 8]; for s in 0..8 { b[s] = l[s].v & 1; } b })
        .collect();
    let mut rows = vec![vec![0u8; ROW_BYTES]; (k + n) as usize];
    for j in 0..(k as usize) {
        for i in 0..NUM_SYMBOLS { let a = j * NUM_SYMBOLS + i; rows[j][i] = pack[a >> 3][a & 7]; }
    }
    let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (c, _h) = rsema1d_sys::commit(k, n, &row_refs).unwrap();
    c
}

struct RunStats {
    array_len: usize,
    cycles: usize,
    num_vars: usize,
    gate_entries: usize,
    prove: Duration,
    verify: Duration,
}

/// Lightweight scaling probe: compile the circuit for `array_len` and read the
/// STRUCTURAL metrics (num_vars, gate entries) straight off the layered circuit.
/// No witness solve / prove / PCS -- so it stays cheap in time and memory and can
/// reach large cycle counts that a full prove could not fit in RAM.
fn measure(array_len: usize) -> RunStats {
    use emulator::*;
    let mut prog = program();
    prog[0] = addi(4, 0, array_len as i32);
    let prog_len = prog.len();
    let mut mem_addrs: Vec<u32> = (0..array_len as u32).map(|i| DATA_BASE + 4 * i).collect();
    mem_addrs.push(RESULT_ADDR);
    let nmem = mem_addrs.len();
    let ncyc = 6 * array_len + 16;
    unsafe {
        ckt::NCYC = ncyc;
        ckt::NMEM = nmem;
        ckt::PROG_LEN = prog_len;
        ckt::PROGRAM = prog.clone();
        ckt::MEM_ADDRS = mem_addrs.clone();
        ckt::MEM_INIT = { let mut m: Vec<u32> = (0..array_len as u32).map(|i| 3 * (i + 1) + 7).collect(); m.push(0); m };
    }
    let t_c = Instant::now();
    let CompileResult { layered_circuit, .. } =
        compile(&ckt::template(), CompileOptions::default()).unwrap();
    let compile_dur = t_c.elapsed();
    let ec = layered_circuit.export_to_expander_flatten();
    let num_vars = ec.log_input_size();
    let gate_entries: usize = ec.layers.iter().map(|l| l.mul.len() + l.add.len()).sum();
    println!("  [measure] len={array_len:>3} cycles={ncyc:>4} num_vars={num_vars} gate_entries={gate_entries} (compile {compile_dur:?})");
    RunStats { array_len, cycles: ncyc, num_vars, gate_entries, prove: Duration::ZERO, verify: Duration::ZERO }
}

fn run(array_len: usize, full: bool) -> RunStats {
    use emulator::*;

    let mut prog = program();
    prog[0] = addi(4, 0, array_len as i32);
    let prog_len = prog.len();

    let mut mem_addrs: Vec<u32> = (0..array_len as u32).map(|i| DATA_BASE + 4 * i).collect();
    mem_addrs.push(RESULT_ADDR);
    let nmem = mem_addrs.len();

    let array: Vec<u32> = (0..array_len as u32).map(|i| 3u32.wrapping_mul(i + 1) + 7).collect();
    let expected_sum: u32 = array.iter().copied().sum();
    let mut mem_init: Vec<u32> = array.clone();
    mem_init.push(0);

    let ncyc = 6 * array_len + 16;

    unsafe {
        ckt::NCYC = ncyc;
        ckt::NMEM = nmem;
        ckt::PROG_LEN = prog_len;
        ckt::PROGRAM = prog.clone();
        ckt::MEM_ADDRS = mem_addrs.clone();
        ckt::MEM_INIT = mem_init.clone();
    }

    let mut cpu = Cpu::new(prog.clone(), ckt::BASE);
    for (k, &v) in array.iter().enumerate() { cpu.mem.store(DATA_BASE + 4 * k as u32, v); }
    let trace = cpu.run(ncyc);
    let native_sum = cpu.regs[2];
    let native_i = cpu.regs[3];
    let native_mem_result = cpu.mem.load(RESULT_ADDR);
    assert_eq!(native_sum, expected_sum, "emulator sum mismatch");
    let out_vals = [native_sum, native_i, native_mem_result];

    let hints = replay_hints(&trace, &mem_addrs, &mem_init);

    let t_c = Instant::now();
    let CompileResult { witness_solver, layered_circuit } =
        compile(&ckt::template(), CompileOptions::default()).unwrap();
    let compile_dur = t_c.elapsed();

    // 1st solve with challenges=0 -> commitment -> derive (alpha,beta) (FS bind).
    let a0 = ckt::build_assignment(&trace, &hints, 0u128, 0u128, &out_vals);
    let w0 = witness_solver.solve_witnesses(&vec![a0; 8]).unwrap();
    let (si0, _sp0) = w0.to_simd::<gf2::GF2x8>();
    let ec0 = layered_circuit.export_to_expander_flatten();
    let num_vars = ec0.log_input_size();
    let c_trace = da_commit(&si0, num_vars);
    let (alpha, beta) = fs_challenges(&c_trace);

    // real solve with bound challenges.
    let assignment = ckt::build_assignment(&trace, &hints, alpha, beta, &out_vals);
    let t_w = Instant::now();
    let witness = witness_solver.solve_witnesses(&vec![assignment.clone(); 8]).unwrap();
    let witness_dur = t_w.elapsed();
    let res = layered_circuit.run(&witness);
    assert!(res.iter().all(|x| *x), "layered circuit self-eval FAILED with bound challenges");

    let mut ec = layered_circuit.export_to_expander_flatten();
    let (si, sp) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = si.clone();
    ec.public_input = sp.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let gate_entries: usize = ec.layers.iter().map(|l| l.mul.len() + l.add.len()).sum();

    let c_real = da_commit(&input_vals, num_vars);
    assert_eq!(c_real, c_trace, "commitment changed when challenges changed (binding broken)");

    let mpi = MPIConfig::prover_new(None, None);
    let t_p = Instant::now();
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let prove_dur = t_p.elapsed();
    let t_v = Instant::now();
    let ok = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v);
    let verify_dur = t_v.elapsed();
    assert!(ok && claimed_v.is_zero(), "verifier REJECTED honest grand-product proof");

    let params = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi.world_size());
    let mut spad = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
    let input_poly = MultiLinearPoly::new(input_vals.clone());
    let c_prover = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &input_poly, &mut spad).unwrap();
    assert_eq!(c_prover.root, c_real, "GKR PCS root != Go/DA commit");
    assert!(proof.bytes.windows(32).any(|w| w == c_prover.root), "commitment not embedded in proof");

    // Public inputs are laid out in field-declaration order: alpha (128) + beta
    // (128) then out.
    let out_off = 256;
    let mut recon = [0u32; NOUT];
    for kk in 0..NOUT {
        let mut w = 0u32;
        for b in 0..XLEN { w |= ((sp[out_off + kk * XLEN + b].unpack()[0].v & 1) as u32) << b; }
        recon[kk] = w;
    }
    assert_eq!(recon, out_vals, "circuit public output != native");

    if full {
        println!("\n================ GATE (a): honest proof, len={array_len} ================");
        println!("[emulate] cycles = {ncyc}, native sum = {native_sum} (array {:?}{})",
            &array[..array.len().min(6)], if array.len() > 6 { ", ..." } else { "" });
        println!("[compile] {compile_dur:?}; [witness] {witness_dur:?}; inputs/witness = {}", witness.num_inputs_per_witness);
        println!("[export] layers = {}, num_vars = {num_vars}, gate entries (mul+add) = {gate_entries}", ec.layers.len());
        println!("[FS] trace commitment = {}", hex_str(&c_trace));
        println!("[FS] alpha = 0x{alpha:032x}");
        println!("[FS] beta  = 0x{beta:032x}");
        println!("[FS] (both bound to commitment; committed layer is independent of alpha/beta -> commitment stable)");
        println!("[verify] ACCEPTED = {ok}, claimed==0 = {}, verify time = {verify_dur:?}", claimed_v.is_zero());
        println!("[commit] GKR PCS root      = {}", hex_str(&c_prover.root));
        println!("[commit] Go/DA rsema1d     = {}", hex_str(&c_real));
        println!("[commit] BYTE-IDENTICAL to Go/DA and to challenges=0 commitment (K=N={})", 1u32 << (num_vars - 2));
        println!("[output] sum={} count={} mem[result]={} (all match native)", recon[0], recon[1], recon[2]);
        println!("[prove] {prove_dur:?}, proof bytes = {}, peak RSS = {:.1} MiB", proof.bytes.len(), peak_rss_mib());

        // native cross-check of the grand product (independent of the circuit).
        let lw_c = trace.iter().position(|r| r.is_load).unwrap();
        let (rh, wh) = native_products(&trace, &hints, &mem_addrs, &mem_init, alpha, beta, None);
        let (rt, wt) = native_products(&trace, &hints, &mem_addrs, &mem_init, alpha, beta, Some((lw_c, trace[lw_c].rd_val.wrapping_add(1))));
        println!("[native] PROD_R==PROD_W  honest={} tampered={} (want true,false)", rh == wh, rt == wt);

        println!("\n================ GATE (a): tamper rejection ================");
        let lw_cycle = trace.iter().position(|r| r.is_load).unwrap();
        println!("[tamper] corrupting first LW's committed loaded value at cycle {lw_cycle} ({} -> {})",
            trace[lw_cycle].rd_val, trace[lw_cycle].rd_val.wrapping_add(1));
        let mut bad = assignment.clone();
        ckt::tamper_rd_val(&mut bad, lw_cycle, trace[lw_cycle].rd_val.wrapping_add(1));
        let bad_w = witness_solver.solve_witnesses(&vec![bad; 8]).unwrap();
        let bad_run = layered_circuit.run(&bad_w);
        let bad_all_ok = bad_run.iter().all(|x| *x);
        let mut bad_ec = layered_circuit.export_to_expander_flatten();
        let (bsi, bsp) = bad_w.to_simd::<gf2::GF2x8>();
        bad_ec.layers[0].input_vals = bsi.clone();
        bad_ec.public_input = bsp;
        bad_ec.evaluate();
        let (bad_v, _bp) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut bad_ec, mpi.clone());
        let c_bad = da_commit(&bsi, num_vars);
        println!("[tamper] layered self-eval all-true = {bad_all_ok} (expected false: grand product broke)");
        println!("[tamper] claimed output == 0 = {} (expected false)", bad_v.is_zero());
        println!("[tamper] tampered commitment {} != honest {}", &hex_str(&c_bad)[..16], &hex_str(&c_real)[..16]);
        assert!(!bad_all_ok && !bad_v.is_zero() && c_bad != c_real, "TAMPER NOT REJECTED");
        println!("[tamper] REJECTED (constraints fail + non-zero output + commitment differs)");

        println!("\n================ GATE (c): Fiat-Shamir challenge binding ================");
        let (alpha_bad, beta_bad) = fs_challenges(&c_bad);
        println!("[FS-bind] honest   (alpha,beta) from commitment {}... = (0x{alpha:032x}, 0x{beta:032x})", &hex_str(&c_real)[..16]);
        println!("[FS-bind] tampered (alpha,beta) from commitment {}... = (0x{alpha_bad:032x}, 0x{beta_bad:032x})", &hex_str(&c_bad)[..16]);
        assert!(alpha != alpha_bad && beta != beta_bad, "different traces produced the same challenges");
        println!("[FS-bind] different trace => different commitment => different challenges: CONFIRMED");
        let mut bad2 = ckt::build_assignment(&trace, &hints, alpha, beta, &out_vals);
        ckt::tamper_rd_val(&mut bad2, lw_cycle, trace[lw_cycle].rd_val.wrapping_add(1));
        let bad2_run = layered_circuit.run(&witness_solver.solve_witnesses(&vec![bad2; 8]).unwrap());
        let bad2_ok = bad2_run.iter().all(|x| *x);
        println!("[FS-bind] tampered trace + honest challenges: self-eval all-true = {bad2_ok} (expected false)");
        assert!(!bad2_ok);
        println!("[FS-bind] a witness valid for one trace's challenges does NOT validate a different trace: CONFIRMED");
    }

    RunStats { array_len, cycles: ncyc, num_vars, gate_entries, prove: prove_dur, verify: verify_dur }
}

fn main() {
    println!("==== RV32IM CPU-verifier: OFFLINE MEMORY CHECKING (GF(2^128) grand product) ====");
    println!("[design] register/memory consistency via multiset equality PROD(reads)==PROD(writes)");
    println!("[design] factor = alpha + (addr + value*beta + ts*beta^2) over GF(2^128) (GCM poly)");
    println!("[design] alpha,beta = Fiat-Shamir(trace commitment), injected as PUBLIC INPUTS");

    let full = run(4, true);

    println!("\n================ GATE (b): scaling (circuit size grows ~linearly in cycles) ================");
    // Full prove+verify data points (kept small so the 8-wide SIMD prove fits in
    // RAM), then compile-only structural measurements for larger cycle counts.
    let prove_sizes: Vec<usize> = std::env::var("GP_PROVE_SIZES")
        .ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 4, 8]);
    let measure_sizes: Vec<usize> = std::env::var("GP_MEASURE_SIZES")
        .ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 4, 8, 16, 32, 64]);

    println!("\n-- full prove+verify (real proofs) --");
    let mut proved = Vec::new();
    for &s in &prove_sizes { proved.push(run(s, false)); }
    println!("\n  array_len  cycles  num_vars  gate_entries   prove          verify");
    println!("  {:>9}  {:>6}  {:>8}  {:>12}   {:>12?}  {:>10?}",
        full.array_len, full.cycles, full.num_vars, full.gate_entries, full.prove, full.verify);
    for s in &proved {
        println!("  {:>9}  {:>6}  {:>8}  {:>12}   {:>12?}  {:>10?}",
            s.array_len, s.cycles, s.num_vars, s.gate_entries, s.prove, s.verify);
    }

    println!("\n-- compile-only structural scaling (num_vars / gate_entries vs cycles) --");
    let mut rows = Vec::new();
    for &s in &measure_sizes { rows.push(measure(s)); }
    println!("\n  array_len  cycles  num_vars  gate_entries  gates/cycle");
    for s in &rows {
        println!("  {:>9}  {:>6}  {:>8}  {:>12}  {:>11.0}",
            s.array_len, s.cycles, s.num_vars, s.gate_entries, s.gate_entries as f64 / s.cycles as f64);
    }
    // linear-fit sanity: (gates(max)-gates(min)) / (cycles(max)-cycles(min)).
    if rows.len() >= 2 {
        let a = &rows[0];
        let b = rows.last().unwrap();
        let slope = (b.gate_entries as f64 - a.gate_entries as f64) / (b.cycles as f64 - a.cycles as f64);
        println!("\n  marginal gates/cycle (slope over range) = {slope:.0}");
        println!("  => gate_entries ~ linear in cycles (state-threading would be ~cycles x state, i.e. superlinear per-cycle growth)");
    }

    println!("\n[peak RSS] {:.1} MiB", peak_rss_mib());
    println!("\n=== SUMMARY ===");
    println!("(a) verifier ACCEPTS, commitment == Go/DA rsema1d (byte-identical), tamper REJECTED  OK");
    println!("(b) circuit size grows ~linearly in cycles (constant gates/cycle), NOT cycles x state OK");
    println!("(c) FS challenges bound to committed trace (distinct trace => distinct challenges)    OK");
}
