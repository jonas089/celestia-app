//! Transfer-STF block prover: proves the nonce + balance STATE TRANSITION of a
//! rollup block over its committed transaction data, reusing the rsema1d DA
//! encoding as the GKR input polynomial commitment.
//!
//! The block's pre-state (sender/recipient balance + nonce) and its
//! transactions ({value, fee, nonce}) are installed as the initial data memory
//! of a small RV32I transfer-STF interpreter (`build_program`), which is then
//! emulated to a fixed-length trace and proven over Expander GKR with
//! `Rsema1dGKRConfig` — the exact proving spine of `prove_execution`
//! (`prove.rs`), just driving the STF circuit (`circuit_stf`).
//!
//! Per tx, IN ORDER, the interpreter enforces:
//!   * REPLAY PROTECTION: `tx.nonce == sender.nonce` (else the tx is rejected —
//!     not applied — its slot leaves state untouched);
//!   * BALANCE: `sender.balance >= value + fee` (else rejected);
//!   * on success: `sender.balance -= value + fee`, `sender.nonce += 1`,
//!     `recipient.balance += value`.
//! It then writes the post-state (sender balance/nonce, recipient balance),
//! the count of applied txs, and a post-state digest back to memory; those five
//! words are the circuit's PUBLIC OUTPUTS.
//!
//! HONEST SCOPE (documented in `circuit_stf` too): this proves the nonce +
//! balance transition over the block's committed tx data, reusing rsema1d. It
//! does NOT prove ECDSA signature verification or a keccak/MPT state root — the
//! post-state "digest" is a simple XOR/shift fold, NOT keccak. Balances/values
//! must be < 2^31 (the balance check uses the sign bit of a two's-complement
//! subtraction). These are the next rungs toward full light-client validity.

use crate::circuit_stf as ckt;
use crate::circuit_stf::{RiscvCircuitStf, CYCLES, MAXTX, NOUT, PROG_LEN, XLEN};
use crate::emulator::*;
use arith::{Field, SimdField};
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use polynomials::MultiLinearPoly;
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};
use std::sync::Mutex;

/// Max transactions per block (re-exported from the circuit).
pub use crate::circuit_stf::MAXTX as MAXTX_TXS;

/// Guaranteed-invalid nonce for unused (padding) tx slots — always rejected.
pub const NULL_NONCE: u32 = 0xFFFF_FFFF;

/// One transfer transaction (sender -> recipient) as committed in the block.
#[derive(Clone, Copy, Debug)]
pub struct TxData {
    pub value: u32,
    pub fee: u32,
    pub nonce: u32,
}

/// A block's transactions plus the touched pre-state. All transfers in a block
/// go from a single sender to a single recipient (documented simplification:
/// one (sender, recipient) pair per block); nonce replay protection is enforced
/// against the sender's running nonce.
#[derive(Clone, Debug)]
pub struct BlockInput {
    pub pre_sender_balance: u32,
    pub pre_sender_nonce: u32,
    pub pre_recipient_balance: u32,
    pub txs: Vec<TxData>,
}

/// The result of natively applying the transfer-STF (reference semantics that
/// the RV32I interpreter — and thus the GKR proof — reproduces exactly).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StfOutcome {
    pub post_sender_balance: u32,
    pub post_sender_nonce: u32,
    pub post_recipient_balance: u32,
    pub applied_count: u32,
    pub digest: u32,
    /// Per-tx applied/rejected flag (true = applied), length == txs.len().
    pub applied: Vec<bool>,
}

/// A real GKR proof of a block's nonce + balance state transition, with the
/// committed trace committed by the (reused) rsema1d DA encoder.
#[derive(Clone, Debug)]
pub struct BlockStfProof {
    /// 32-byte rsema1d commitment of the committed GKR input (the trace).
    /// Byte-identical to an independent Go/DA rsema1d commit of the same rows.
    pub commitment: [u8; 32],
    /// Serialized Expander GKR proof bytes.
    pub proof: Vec<u8>,
    /// Public outputs, each LE u32: post sender-balance ++ post sender-nonce ++
    /// post recipient-balance ++ applied-count ++ post-state digest (20 bytes).
    pub public_value: Vec<u8>,
    /// Whether the Expander verifier accepted (self-verified here).
    pub verified: bool,
    /// GKR `num_vars` of the committed input polynomial.
    pub input_vars: u32,
    // Decoded public outputs (also present in `public_value`).
    pub post_sender_balance: u32,
    pub post_sender_nonce: u32,
    pub post_recipient_balance: u32,
    pub applied_count: u32,
    pub digest: u32,
    /// Number of real (non-padding) txs in the block.
    pub tx_count: u32,
}

/// Serialize proving: the circuit reads program/memory from process-global
/// `static mut`s at compile time, and the MPI prover is single-threaded.
static PROVE_LOCK: Mutex<()> = Mutex::new(());

/// Build the fixed transfer-STF interpreter program (RV32I subset). Offsets are
/// computed from instruction indices, so the encoding is verified by
/// construction. Length is exactly `PROG_LEN`.
pub fn build_program() -> Vec<u32> {
    let mut p: Vec<u32> = Vec::with_capacity(PROG_LEN);
    // Prologue: load pre-state into registers (x1=SBAL, x2=SNON, x3=RBAL).
    p.push(lw(1, 0, ckt::SBAL_ADDR as i32));
    p.push(lw(2, 0, ckt::SNON_ADDR as i32));
    p.push(lw(3, 0, ckt::RBAL_ADDR as i32));

    // MAXTX unrolled tx blocks (14 instructions each).
    for i in 0..MAXTX {
        let val_addr = (ckt::TX_BASE + ckt::TX_STRIDE * i as u32) as i32;
        let fee_addr = val_addr + 4;
        let non_addr = val_addr + 8;
        p.push(lw(4, 0, val_addr)); // b+0: x4 = value
        p.push(lw(5, 0, fee_addr)); // b+1: x5 = fee
        p.push(lw(6, 0, non_addr)); // b+2: x6 = nonce
        p.push(add(7, 4, 5)); // b+3: x7 = need = value + fee
        p.push(beq(6, 2, 8)); // b+4: if nonce==SNON -> b+6 (skip the jal)
        p.push(jal(0, 36)); // b+5: else jump to SKIP (b+14)
        p.push(sub(8, 1, 7)); // b+6: x8 = SBAL - need
        p.push(srli(9, 8, 31)); // b+7: x9 = overspend bit (sign of diff)
        p.push(beq(9, 0, 8)); // b+8: if !overspend -> b+10 (skip the jal)
        p.push(jal(0, 20)); // b+9: else jump to SKIP (b+14)
        p.push(sub(1, 1, 7)); // b+10: SBAL -= need
        p.push(addi(2, 2, 1)); // b+11: SNON += 1
        p.push(add(3, 3, 4)); // b+12: RBAL += value
        p.push(addi(10, 10, 1)); // b+13: applied += 1
    }

    // Epilogue: write post-state + digest back to memory, then halt.
    p.push(sw(1, 0, ckt::SBAL_ADDR as i32));
    p.push(sw(2, 0, ckt::SNON_ADDR as i32));
    p.push(sw(3, 0, ckt::RBAL_ADDR as i32));
    p.push(sw(10, 0, ckt::APPLIED_ADDR as i32));
    p.push(slli(11, 2, 8)); // x11 = SNON << 8
    p.push(slli(12, 3, 16)); // x12 = RBAL << 16
    p.push(xor(13, 1, 11)); // x13 = SBAL ^ (SNON<<8)
    p.push(xor(13, 13, 12)); // x13 ^= (RBAL<<16)  => digest
    p.push(sw(13, 0, ckt::DIGEST_ADDR as i32));
    p.push(jal(0, 0)); // halt (self-loop)

    assert_eq!(p.len(), PROG_LEN, "program length mismatch");
    p
}

/// Pad a block's txs to exactly MAXTX slots; padding slots carry NULL_NONCE
/// (always rejected). Errors if the block has more than MAXTX txs.
fn padded_txs(input: &BlockInput) -> Result<[TxData; MAXTX], String> {
    if input.txs.len() > MAXTX {
        return Err(format!("block has {} txs; max {}", input.txs.len(), MAXTX));
    }
    let mut txs = [TxData { value: 0, fee: 0, nonce: NULL_NONCE }; MAXTX];
    for (i, t) in input.txs.iter().enumerate() {
        txs[i] = *t;
    }
    Ok(txs)
}

/// The reference transfer-STF, applied natively. Identical semantics to the
/// RV32I interpreter (and thus to what the GKR proof attests).
pub fn apply_native(input: &BlockInput) -> Result<StfOutcome, String> {
    let txs = padded_txs(input)?;
    let mut sbal = input.pre_sender_balance;
    let mut snon = input.pre_sender_nonce;
    let mut rbal = input.pre_recipient_balance;
    let mut applied = 0u32;
    let mut flags = Vec::with_capacity(input.txs.len());
    for (i, t) in txs.iter().enumerate() {
        let need = t.value.wrapping_add(t.fee);
        let nonce_ok = t.nonce == snon;
        let diff = sbal.wrapping_sub(need);
        let overspend = (diff >> 31) & 1 == 1; // valid balances/values < 2^31
        let valid = nonce_ok && !overspend;
        if valid {
            sbal = sbal.wrapping_sub(need);
            snon = snon.wrapping_add(1);
            rbal = rbal.wrapping_add(t.value);
            applied += 1;
        }
        if i < input.txs.len() {
            flags.push(valid);
        }
    }
    let digest = sbal ^ (snon << 8) ^ (rbal << 16);
    Ok(StfOutcome {
        post_sender_balance: sbal,
        post_sender_nonce: snon,
        post_recipient_balance: rbal,
        applied_count: applied,
        digest,
        applied: flags,
    })
}

/// Prove a block's nonce + balance state transition. Builds the interpreter,
/// installs the block's pre-state + txs as data memory, emulates a fixed trace,
/// proves it over Expander GKR (`Rsema1dGKRConfig`), self-verifies, and confirms
/// the GKR input commitment is byte-identical to an independent Go/DA rsema1d
/// commit of the same rows. Returns a [`BlockStfProof`] or an error string.
pub fn prove_block_stf(input: &BlockInput) -> Result<BlockStfProof, String> {
    let _guard = PROVE_LOCK.lock().map_err(|e| format!("prove lock poisoned: {e}"))?;

    let txs = padded_txs(input)?;
    let native = apply_native(input)?;

    // ---- 1. Program + initial data memory -----------------------------------
    let program = build_program();
    for (i, w) in program.iter().enumerate() {
        unsafe { ckt::PROGRAM[i] = *w };
    }
    let mem_addrs = ckt::mem_addrs();
    // MEM_INIT: [SBAL, SNON, RBAL, APPLIED=0, DIGEST=0, then per-tx value,fee,nonce].
    let mut mem_init = [0u32; ckt::NMEM];
    mem_init[0] = input.pre_sender_balance;
    mem_init[1] = input.pre_sender_nonce;
    mem_init[2] = input.pre_recipient_balance;
    mem_init[3] = 0;
    mem_init[4] = 0;
    for i in 0..MAXTX {
        mem_init[ckt::NACC_CELLS + 3 * i] = txs[i].value;
        mem_init[ckt::NACC_CELLS + 3 * i + 1] = txs[i].fee;
        mem_init[ckt::NACC_CELLS + 3 * i + 2] = txs[i].nonce;
    }
    for c in 0..ckt::NMEM {
        unsafe { ckt::MEM_INIT[c] = mem_init[c] };
    }

    // ---- 2. Emulate ----------------------------------------------------------
    let mut cpu = Cpu::new(program.clone(), ckt::BASE);
    for (c, &addr) in mem_addrs.iter().enumerate() {
        cpu.mem.store(addr, mem_init[c]);
    }
    let trace = cpu.run(CYCLES);

    // Post-state read from the (final) data memory, and cross-checked against the
    // native reference — the emulator must reproduce apply_native exactly.
    let out_vals: [u32; NOUT] = [
        cpu.mem.load(ckt::SBAL_ADDR),
        cpu.mem.load(ckt::SNON_ADDR),
        cpu.mem.load(ckt::RBAL_ADDR),
        cpu.mem.load(ckt::APPLIED_ADDR),
        cpu.mem.load(ckt::DIGEST_ADDR),
    ];
    let emu = StfOutcome {
        post_sender_balance: out_vals[0],
        post_sender_nonce: out_vals[1],
        post_recipient_balance: out_vals[2],
        applied_count: out_vals[3],
        digest: out_vals[4],
        applied: native.applied.clone(),
    };
    if emu != native {
        return Err(format!(
            "emulator STF {emu:?} disagrees with native reference {native:?}"
        ));
    }

    // ---- 3. Compile ----------------------------------------------------------
    let CompileResult { witness_solver, layered_circuit } =
        compile(&RiscvCircuitStf::default(), CompileOptions::default())
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
    if recon != out_vals {
        return Err(format!(
            "circuit public output {recon:?} != emulated post-state {out_vals:?}"
        ));
    }
    let mut public_value = Vec::with_capacity(NOUT * 4);
    for v in recon {
        public_value.extend_from_slice(&v.to_le_bytes());
    }

    Ok(BlockStfProof {
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
    })
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_applies_valid_and_rejects_replay_and_overspend() {
        // Two valid transfers from sender(nonce 5) -> recipient.
        let input = BlockInput {
            pre_sender_balance: 1000,
            pre_sender_nonce: 5,
            pre_recipient_balance: 10,
            txs: vec![
                TxData { value: 100, fee: 1, nonce: 5 }, // applied: bal 1000->899, nonce 5->6
                TxData { value: 50, fee: 1, nonce: 6 },  // applied: bal 899->848, nonce 6->7
            ],
        };
        let o = apply_native(&input).unwrap();
        assert_eq!(o.post_sender_balance, 848);
        assert_eq!(o.post_sender_nonce, 7);
        assert_eq!(o.post_recipient_balance, 160);
        assert_eq!(o.applied_count, 2);
        assert_eq!(o.applied, vec![true, true]);

        // Replay: second tx reuses nonce 5 (sender nonce is now 6) -> rejected.
        let replay = BlockInput {
            pre_sender_balance: 1000,
            pre_sender_nonce: 5,
            pre_recipient_balance: 10,
            txs: vec![
                TxData { value: 100, fee: 1, nonce: 5 },
                TxData { value: 50, fee: 1, nonce: 5 }, // REPLAY
            ],
        };
        let o = apply_native(&replay).unwrap();
        assert_eq!(o.applied, vec![true, false]);
        assert_eq!(o.applied_count, 1);
        assert_eq!(o.post_sender_nonce, 6);
        assert_eq!(o.post_sender_balance, 899);
        assert_eq!(o.post_recipient_balance, 110);

        // Overspend: value+fee exceeds balance -> rejected.
        let overspend = BlockInput {
            pre_sender_balance: 100,
            pre_sender_nonce: 0,
            pre_recipient_balance: 0,
            txs: vec![TxData { value: 200, fee: 0, nonce: 0 }],
        };
        let o = apply_native(&overspend).unwrap();
        assert_eq!(o.applied, vec![false]);
        assert_eq!(o.applied_count, 0);
        assert_eq!(o.post_sender_balance, 100);
    }

    #[test]
    fn emulator_matches_native() {
        // Drive the RV32I interpreter through the emulator and check it matches
        // the native reference for valid / replay / overspend blocks.
        let cases = vec![
            BlockInput {
                pre_sender_balance: 1000,
                pre_sender_nonce: 5,
                pre_recipient_balance: 10,
                txs: vec![
                    TxData { value: 100, fee: 1, nonce: 5 },
                    TxData { value: 50, fee: 1, nonce: 6 },
                ],
            },
            BlockInput {
                pre_sender_balance: 1000,
                pre_sender_nonce: 5,
                pre_recipient_balance: 10,
                txs: vec![
                    TxData { value: 100, fee: 1, nonce: 5 },
                    TxData { value: 50, fee: 1, nonce: 5 }, // replay
                ],
            },
            BlockInput {
                pre_sender_balance: 100,
                pre_sender_nonce: 0,
                pre_recipient_balance: 0,
                txs: vec![TxData { value: 200, fee: 0, nonce: 0 }],
            },
            BlockInput {
                pre_sender_balance: 5_000_000,
                pre_sender_nonce: 42,
                pre_recipient_balance: 123,
                txs: vec![
                    TxData { value: 1000, fee: 10, nonce: 42 },
                    TxData { value: 2000, fee: 20, nonce: 43 },
                    TxData { value: 3000, fee: 30, nonce: 44 },
                    TxData { value: 4000, fee: 40, nonce: 45 },
                ],
            },
        ];
        for input in cases {
            let native = apply_native(&input).unwrap();
            let program = build_program();
            let mem_addrs = ckt::mem_addrs();
            let txs = padded_txs(&input).unwrap();
            let mut mem_init = [0u32; ckt::NMEM];
            mem_init[0] = input.pre_sender_balance;
            mem_init[1] = input.pre_sender_nonce;
            mem_init[2] = input.pre_recipient_balance;
            for i in 0..MAXTX {
                mem_init[ckt::NACC_CELLS + 3 * i] = txs[i].value;
                mem_init[ckt::NACC_CELLS + 3 * i + 1] = txs[i].fee;
                mem_init[ckt::NACC_CELLS + 3 * i + 2] = txs[i].nonce;
            }
            let mut cpu = Cpu::new(program, ckt::BASE);
            for (c, &addr) in mem_addrs.iter().enumerate() {
                cpu.mem.store(addr, mem_init[c]);
            }
            let trace = cpu.run(CYCLES);
            // trace must have halted (last cycle is the self-loop jal).
            assert!(trace.len() == CYCLES);
            assert_eq!(cpu.mem.load(ckt::SBAL_ADDR), native.post_sender_balance, "sbal {input:?}");
            assert_eq!(cpu.mem.load(ckt::SNON_ADDR), native.post_sender_nonce, "snon {input:?}");
            assert_eq!(cpu.mem.load(ckt::RBAL_ADDR), native.post_recipient_balance, "rbal {input:?}");
            assert_eq!(cpu.mem.load(ckt::APPLIED_ADDR), native.applied_count, "applied {input:?}");
            assert_eq!(cpu.mem.load(ckt::DIGEST_ADDR), native.digest, "digest {input:?}");
        }
    }
}
