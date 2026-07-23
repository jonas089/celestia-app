//! Callable proving entrypoint: `prove_execution(input)` runs the proven RV32IM
//! looping array-sum program (the exact circuit from `circuit_full`, unchanged),
//! parameterized by `input` bytes (the four summed array words are derived from
//! the input, so different inputs yield different traces / commitments / proofs),
//! proves it over Expander GKR with `Rsema1dGKRConfig`, self-verifies, and
//! reconciles the GKR input commitment against an INDEPENDENT Go/DA rsema1d
//! commit of the same bit-packed input rows.
//!
//! This is the exact proving spine of `main_full.rs`, extracted so it can be
//! called from a C-ABI shared library (`libaccprover`) and, through it, from the
//! ev-reth fork's `accProof` RPC. The proven circuit is untouched.

use crate::circuit_full as ckt;
use crate::circuit_full::{RiscvCircuitFull, CYCLES, NOUT, XLEN};
use crate::emulator::*;
use arith::{Field, SimdField};
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use polynomials::MultiLinearPoly;
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};
use std::sync::Mutex;

/// A real GKR proof of one RV32IM execution, with the input committed by the
/// (reused) rsema1d DA encoder.
#[derive(Clone, Debug)]
pub struct AccProof {
    /// The 32-byte rsema1d commitment of the committed GKR input layer (the
    /// execution trace). Byte-identical to the independent Go/DA `rsema1d`
    /// commit of the same rows.
    pub commitment: [u8; 32],
    /// The serialized Expander GKR proof bytes.
    pub proof: Vec<u8>,
    /// The program's public output: final x2 (sum) ++ x3 (loop count) ++
    /// mem[RESULT_ADDR], each as a little-endian u32 (12 bytes).
    pub public_value: Vec<u8>,
    /// Whether the Expander verifier accepted the proof (self-verified here).
    pub verified: bool,
    /// log2 of the padded committed input bit-length (GKR `num_vars`).
    pub input_vars: u32,
}

/// The proven circuit reads a program image + initial memory from process-global
/// `static mut`s (`ckt::PROGRAM` / `ckt::MEM_INIT`) at `compile()` time. Serialize
/// proving so concurrent callers (e.g. the RPC's `spawn_blocking` pool) cannot
/// race those globals or the (single-threaded) MPI prover.
static PROVE_LOCK: Mutex<()> = Mutex::new(());

/// Derive the four summed array words deterministically from arbitrary input
/// bytes. Distinct inputs produce distinct arrays with high probability, so the
/// resulting trace / commitment / proof differ per input. The proof genuinely
/// proves summing exactly these four words.
fn derive_array(input: &[u8]) -> [u32; 4] {
    let mut buf = [0u8; 16];
    // Mix the length in so trailing-zero / prefix inputs still differ.
    let len = input.len() as u32;
    for i in 0..16 {
        let mut v = (len.wrapping_mul(0x9E37 + i as u32) & 0xff) as u8;
        for (j, &b) in input.iter().enumerate() {
            v = v.wrapping_add(
                b.wrapping_mul((i as u8).wrapping_add(1))
                    .wrapping_add(j as u8),
            );
        }
        buf[i] = v;
    }
    [
        u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
        u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
        u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
        u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
    ]
}

/// Prove one RV32IM execution (the array-sum loop) over the four words derived
/// from `input`, using `Rsema1dGKRConfig` as the input PCS. Self-verifies and
/// confirms the GKR input commitment is byte-identical to an independent Go/DA
/// rsema1d commit of the same rows. Returns an [`AccProof`] or an error string.
pub fn prove_execution(input: &[u8]) -> Result<AccProof, String> {
    let _guard = PROVE_LOCK.lock().map_err(|e| format!("prove lock poisoned: {e}"))?;

    // ---- 1. Assemble the (fixed) looping array-sum program -------------------
    let program: Vec<u32> = vec![
        addi(4, 0, 4),                     // 0: x4 = N = 4
        addi(1, 0, ckt::DATA_BASE as i32), // 1: x1 = DATA_BASE
        addi(6, 0, ckt::RESULT_ADDR as i32), // 2: x6 = RESULT_ADDR
        lw(5, 1, 0),                       // 3: x5 = mem[x1]  (loop head)
        add(2, 2, 5),                      // 4: x2 += x5
        addi(1, 1, 4),                     // 5: x1 += 4
        addi(3, 3, 1),                      // 6: x3 += 1
        beq(3, 4, 8),                       // 7: if x3==x4 -> idx9
        jal(0, -20),                        // 8: -> idx3
        sw(2, 6, 0),                        // 9: mem[x6] = x2
        jal(0, 0),                          // 10: halt
    ];
    if program.len() != ckt::PROG_LEN {
        return Err(format!("program length {} != PROG_LEN {}", program.len(), ckt::PROG_LEN));
    }
    for (i, w) in program.iter().enumerate() {
        unsafe { ckt::PROGRAM[i] = *w };
    }

    // ---- input-derived data --------------------------------------------------
    let array = derive_array(input);
    let mem_init = [array[0], array[1], array[2], array[3], 0u32];
    for k in 0..ckt::NMEM {
        unsafe { ckt::MEM_INIT[k] = mem_init[k] };
    }

    // ---- 2. Emulate ----------------------------------------------------------
    let mut cpu = Cpu::new(program.clone(), ckt::BASE);
    for (k, &v) in array.iter().enumerate() {
        cpu.mem.store(ckt::DATA_BASE + 4 * k as u32, v);
    }
    let trace = cpu.run(CYCLES);
    let native_sum = cpu.regs[2];
    let native_i = cpu.regs[3];
    let native_mem_result = cpu.mem.load(ckt::RESULT_ADDR);
    let out_vals = [native_sum, native_i, native_mem_result];

    // ---- 3. Compile ----------------------------------------------------------
    let CompileResult { witness_solver, layered_circuit } =
        compile(&RiscvCircuitFull::default(), CompileOptions::default())
            .map_err(|e| format!("compile failed: {e:?}"))?;

    // ---- 4. Witness ----------------------------------------------------------
    let assignment = ckt::build_assignment(&trace, &out_vals);
    let assignments = vec![assignment.clone(); 8];
    let witness = witness_solver
        .solve_witnesses(&assignments)
        .map_err(|e| format!("witness solve failed: {e:?}"))?;
    let res = layered_circuit.run(&witness);
    if !res.iter().all(|x| *x) {
        return Err(format!("layered circuit self-eval failed: {res:?}"));
    }

    // ---- 5. Export & install trace as input layer ----------------------------
    let mut ec = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = simd_input.clone();
    ec.public_input = simd_public_input.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let num_vars = ec.log_input_size();

    // ---- 6. Prove & verify ---------------------------------------------------
    let mpi = MPIConfig::prover_new(None, None);
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let verified = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v)
        && claimed_v.is_zero();

    // ---- 7. Commitment reconciliation (PCS root == independent Go/DA commit) --
    let params = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi.world_size());
    let mut sp = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
    let input_poly = MultiLinearPoly::new(input_vals.clone());
    let c_prover =
        <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &input_poly, &mut sp)
            .ok_or_else(|| "rsema1d PCS commit returned None".to_string())?;

    const NUM_SYMBOLS: usize = 32;
    const ROW_BYTES: usize = 64;
    let k: u32 = 1u32 << (num_vars - 2);
    let n: u32 = k;
    let pack = |vals: &[gf2::GF2x8]| -> Vec<[u8; 8]> {
        vals.iter()
            .map(|e| {
                let l = e.unpack();
                let mut b = [0u8; 8];
                for s in 0..8 {
                    b[s] = l[s].v & 1;
                }
                b
            })
            .collect()
    };
    let to_rows = |bits: &[[u8; 8]]| -> Vec<Vec<u8>> {
        let mut rows = vec![vec![0u8; ROW_BYTES]; (k + n) as usize];
        for j in 0..(k as usize) {
            for i in 0..NUM_SYMBOLS {
                let a = j * NUM_SYMBOLS + i;
                rows[j][i] = bits[a >> 3][a & 7];
            }
        }
        rows
    };
    let rows = to_rows(&pack(&input_vals));
    let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (c_da, _h) = rsema1d_sys::commit(k, n, &row_refs)
        .map_err(|e| format!("independent rsema1d commit failed: {e}"))?;

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

    // ---- 8. Reconstruct public outputs from the circuit's public inputs ------
    let mut recon = [0u32; NOUT];
    for kk in 0..NOUT {
        let mut w = 0u32;
        for b in 0..XLEN {
            w |= ((simd_public_input[kk * XLEN + b].unpack()[0].v & 1) as u32) << b;
        }
        recon[kk] = w;
    }
    if recon != [native_sum, native_i, native_mem_result] {
        return Err(format!(
            "circuit public output {recon:?} != native re-run {out_vals:?}"
        ));
    }
    let mut public_value = Vec::with_capacity(NOUT * 4);
    for v in recon {
        public_value.extend_from_slice(&v.to_le_bytes());
    }

    Ok(AccProof {
        commitment: c_da,
        proof: proof.bytes,
        public_value,
        verified,
        input_vars: num_vars as u32,
    })
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}
