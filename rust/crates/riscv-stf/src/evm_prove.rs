//! Library entrypoint: prove GENUINE EVM bytecode execution via the RV32IM
//! CPU-verifier circuit (`circuit_gp`, GF(2^128) grand-product offline memory
//! checking) with the execution trace committed by the reused DA-canonical
//! rsema1d Encode. The proven RV32 program is a real EVM interpreter
//! (`evm_rv32`) whose output+storage reproduce `evm-core` (== revm 26.0.1).
//!
//! This is the clean entrypoint wired into `libaccprover` / `acc-prover-sys`
//! (and, through them, ev-reth's `accProof`).

use crate::circuit_gp as ckt;
use crate::circuit_gp::{Hints, NOUT, NREG, SLOTS, XLEN};
use crate::evm_core::{self, U256};
use crate::{emulator, evm_rv32};
use arith::{Field, SimdField};
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use polynomials::MultiLinearPoly;
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// `circuit_gp` reads its shape from process-global `static mut`s at compile()
/// time; serialize proving so concurrent callers cannot race those globals or
/// the single-threaded MPI prover.
static PROVE_LOCK: Mutex<()> = Mutex::new(());

/// A real GKR proof of one EVM bytecode execution.
#[derive(Clone)]
pub struct EvmProof {
    /// 32-byte rsema1d commitment of the committed trace — byte-identical to an
    /// independent Go/DA rsema1d Encode of the same rows (testvectors-style
    /// Encode, NOT EncodeStructured).
    pub commitment: [u8; 32],
    /// Serialized Expander GKR proof.
    pub proof: Vec<u8>,
    /// Expander verifier verdict (self-verified here).
    pub verified: bool,
    /// EVM return data of the proven execution.
    pub output: Vec<u8>,
    /// Final non-zero storage of the proven execution.
    pub post_storage: BTreeMap<U256, U256>,
    /// Order-independent 32-byte digest of the final storage (evm-core's fold).
    pub post_storage_digest: [u8; 32],
    /// halt code: 1 STOP, 2 RETURN, 3 REVERT.
    pub halt_code: u32,
    /// RV32 cycles executed up to halt.
    pub cycles: usize,
    /// padded trace length actually proven.
    pub ncyc: usize,
    /// distinct data-memory words (grand-product addresses).
    pub nmem: usize,
    /// GKR num_vars (log2 padded committed input length).
    pub num_vars: usize,
    /// total layered-circuit gate entries (mul+add).
    pub gate_entries: usize,
    /// in-circuit public outputs = [x2, x3, mem[RESULT_ADDR]] (result fold).
    pub out_vals: [u32; 3],
    /// number of EVM opcodes executed (native reference).
    pub opcodes_executed: u64,
    pub prove_time: Duration,
    pub verify_time: Duration,
    /// commitment stable when the FS challenge changes (input independent of it).
    pub commit_stable: bool,
    /// commitment embedded in the proof transcript.
    pub commit_in_proof: bool,
    /// tamper: a corrupted trace fails the circuit and commits differently.
    pub tamper_rejected: bool,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

/// Fiat-Shamir: derive (alpha,beta) in GF(2^128) from the trace commitment.
fn fs_challenges(commit: &[u8; 32]) -> (u128, u128) {
    let mix = |mut z: u64| -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
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

/// True iff the register-writing set matches the circuit's `write_enable`.
fn writes_reg(opcode: u32) -> bool {
    matches!(
        opcode,
        emulator::OPC_OP | emulator::OPC_OPIMM | emulator::OPC_LOAD
        | emulator::OPC_JAL | emulator::OPC_JALR | emulator::OPC_LUI | emulator::OPC_AUIPC
    )
}

/// (quotient, remainder) hints for a DIV/DIVU/REM/REMU cycle (else (0,0)).
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

fn replay_hints(trace: &[emulator::StepRecord], mem_addrs: &[u32], mem_init: &[u32]) -> Hints {
    let ncyc = trace.len();
    let nmem = mem_addrs.len();
    let mut last_ts: HashMap<u32, u32> = HashMap::new();
    let mut last_val: HashMap<u32, u32> = HashMap::new();
    for r in 0..NREG as u32 {
        last_ts.insert(r, 0);
        last_val.insert(r, 0);
    }
    for k in 0..nmem {
        last_ts.insert(mem_addrs[k], 0);
        last_val.insert(mem_addrs[k], mem_init[k]);
    }
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
            if active {
                let tp = *last_ts.get(&addr).unwrap_or(&0);
                tprev[c][slot] = tp;
                if slot == 2 {
                    vold_c[c] = rval; // old memory word (load & store)
                }
                if slot == 3 {
                    vold_d[c] = rval;
                }
                let cur = *last_val.get(&addr).unwrap_or(&0);
                assert_eq!(cur, rval, "replay read mismatch c={c} slot={slot} addr={addr:#x}");
                last_ts.insert(addr, now_ts);
                last_val.insert(addr, wval);
            }
        }
    }
    let naddr = NREG + nmem;
    let mut fin_val = vec![0u32; naddr];
    let mut fin_ts = vec![0u32; naddr];
    for r in 0..NREG {
        fin_val[r] = last_val[&(r as u32)];
        fin_ts[r] = last_ts[&(r as u32)];
    }
    for k in 0..nmem {
        fin_val[NREG + k] = last_val[&mem_addrs[k]];
        fin_ts[NREG + k] = last_ts[&mem_addrs[k]];
    }
    Hints { tprev, vold_c, vold_d, div_q, div_r, fin_val, fin_ts }
}

fn da_commit(input_vals: &[gf2::GF2x8], num_vars: usize) -> Result<[u8; 32], String> {
    const NUM_SYMBOLS: usize = 32;
    const ROW_BYTES: usize = 64;
    let k: u32 = 1u32 << (num_vars - 2);
    let n: u32 = k;
    let pack: Vec<[u8; 8]> = input_vals
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
            rows[j][i] = pack[a >> 3][a & 7];
        }
    }
    let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (c, _h) = rsema1d_sys::commit(k, n, &row_refs).map_err(|e| format!("rsema1d commit: {e}"))?;
    Ok(c)
}

/// Prove one EVM execution of `code` over `calldata` against `pre` storage. `pad`
/// spin cycles pad the trace after halt; `do_tamper` additionally checks that a
/// corrupted trace is rejected (self-eval failure + differing commitment).
pub fn prove_evm(
    code: &[u8],
    calldata: &[u8],
    pre: &BTreeMap<U256, U256>,
    pad: usize,
    do_tamper: bool,
) -> Result<EvmProof, String> {
    let _guard = PROVE_LOCK.lock().map_err(|e| format!("prove lock poisoned: {e}"))?;

    // 1. emulate to find halt, fix NCYC, then build the fixed-length trace.
    let probe = evm_rv32::run_evm(code, calldata, pre, 500_000);
    if !matches!(probe.halt_code, 1 | 2 | 3) {
        return Err(format!("interpreter did not halt cleanly (halt_code={})", probe.halt_code));
    }
    let ncyc = probe.cycles + pad;
    let run = evm_rv32::run_evm(code, calldata, pre, ncyc);
    let trace = &run.trace;
    let mem_addrs = run.touched.clone();
    let mem_init = run.mem_init.clone();
    let nmem = mem_addrs.len();
    let prog = evm_rv32::assemble_interpreter();
    let prog_len = prog.len();
    let out_vals = run.out_vals;

    let native = evm_core::execute(code, calldata, pre, 30_000_000);

    // 2. install circuit config.
    let out_idx = [
        2usize,
        3usize,
        NREG
            + mem_addrs
                .iter()
                .position(|&a| a == evm_rv32::RESULT_ADDR)
                .ok_or("RESULT_ADDR not touched")?,
    ];
    unsafe {
        ckt::NCYC = ncyc;
        ckt::NMEM = nmem;
        ckt::PROG_LEN = prog_len;
        ckt::PROGRAM = prog.clone();
        ckt::MEM_ADDRS = mem_addrs.clone();
        ckt::MEM_INIT = mem_init.clone();
        ckt::OUT_IDX = out_idx;
    }

    let hints = replay_hints(trace, &mem_addrs, &mem_init);

    // 3. compile.
    let CompileResult { witness_solver, layered_circuit } =
        compile(&ckt::template(), CompileOptions::default()).map_err(|e| format!("compile: {e:?}"))?;

    // 4. FS bind: challenges=0 solve -> commitment -> (alpha,beta).
    let a0 = ckt::build_assignment(trace, &hints, 0u128, 0u128, &out_vals);
    let w0 = witness_solver.solve_witnesses(&vec![a0; 8]).map_err(|e| format!("solve0: {e:?}"))?;
    let (si0, _sp0) = w0.to_simd::<gf2::GF2x8>();
    let ec0 = layered_circuit.export_to_expander_flatten();
    let num_vars = ec0.log_input_size();
    let c_trace = da_commit(&si0, num_vars)?;
    let (alpha, beta) = fs_challenges(&c_trace);

    // 5. real solve with bound challenges.
    let assignment = ckt::build_assignment(trace, &hints, alpha, beta, &out_vals);
    let witness = witness_solver.solve_witnesses(&vec![assignment.clone(); 8]).map_err(|e| format!("solve: {e:?}"))?;
    let res = layered_circuit.run(&witness);
    if !res.iter().all(|x| *x) {
        return Err("layered self-eval FAILED with bound challenges".into());
    }

    let mut ec = layered_circuit.export_to_expander_flatten();
    let (si, sp) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = si.clone();
    ec.public_input = sp.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let gate_entries: usize = ec.layers.iter().map(|l| l.mul.len() + l.add.len()).sum();

    let c_real = da_commit(&input_vals, num_vars)?;
    let commit_stable = c_real == c_trace;

    // 6. prove + verify.
    let mpi = MPIConfig::prover_new(None, None);
    let t_p = Instant::now();
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let prove_time = t_p.elapsed();
    let t_v = Instant::now();
    let ok = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v);
    let verify_time = t_v.elapsed();
    let verified = ok && claimed_v.is_zero();

    // 7. commitment reconciliation.
    let params = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi.world_size());
    let mut spad = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
    let input_poly = MultiLinearPoly::new(input_vals.clone());
    let c_prover = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &input_poly, &mut spad)
        .ok_or("rsema1d commit None")?;
    if c_prover.root != c_real {
        return Err(format!("GKR PCS root {} != Go/DA {}", hex(&c_prover.root), hex(&c_real)));
    }
    let commit_in_proof = proof.bytes.windows(32).any(|w| w == c_prover.root);

    // 8. reconstruct public outputs (layout: alpha 128 | beta 128 | out).
    let out_off = 256;
    let mut recon = [0u32; NOUT];
    for kk in 0..NOUT {
        let mut w = 0u32;
        for b in 0..XLEN {
            w |= ((sp[out_off + kk * XLEN + b].unpack()[0].v & 1) as u32) << b;
        }
        recon[kk] = w;
    }
    if recon != out_vals {
        return Err(format!("circuit public output {recon:?} != native {out_vals:?}"));
    }

    // 9. tamper (cheap): a corrupted committed load fails constraints + differs.
    let mut tamper_rejected = false;
    if do_tamper {
        if let Some(lw_cycle) = trace.iter().position(|r| r.is_load) {
            let mut bad = assignment.clone();
            ckt::tamper_rd_val(&mut bad, lw_cycle, trace[lw_cycle].rd_val.wrapping_add(1));
            let bad_w = witness_solver.solve_witnesses(&vec![bad; 8]).map_err(|e| format!("solve tamper: {e:?}"))?;
            let bad_run = layered_circuit.run(&bad_w);
            let bad_all_ok = bad_run.iter().all(|x| *x);
            let (bsi, _bsp) = bad_w.to_simd::<gf2::GF2x8>();
            let c_bad = da_commit(&bsi, num_vars)?;
            tamper_rejected = !bad_all_ok && c_bad != c_real;
        }
    }

    Ok(EvmProof {
        commitment: c_real,
        proof: proof.bytes,
        verified,
        output: run.output.clone(),
        post_storage: run.storage.clone(),
        post_storage_digest: native.storage_digest(),
        halt_code: probe.halt_code,
        cycles: probe.cycles,
        ncyc,
        nmem,
        num_vars,
        gate_entries,
        out_vals,
        opcodes_executed: native.steps,
        prove_time,
        verify_time,
        commit_stable,
        commit_in_proof,
        tamper_rejected,
    })
}
