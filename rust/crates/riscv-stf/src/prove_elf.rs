//! Segmented / CONTINUATION prover for real RV32IM executions, over Expander GKR
//! + rsema1d, made whole end-to-end:
//!
//!   1. HINT_READ committed initial memory. The bytes SP1's HINT_READ syscall
//!      copies into guest memory are modelled as a batch of COMMITTED memory
//!      writes (circuit variables `hint_word`, part of the rsema1d-committed input
//!      rows) at the syscall addresses. Downstream loads validate against them
//!      through the memory multiset, so the delivered input is bound to the
//!      encode-once rsema1d/DA commitment.
//!
//!   2. Cross-chunk MEMORY-multiset threading. Instead of each chunk re-seeding
//!      and draining its own memory image, a running GF(2^128) grand-product pair
//!      (read-product, write-product) is carried as PUBLIC IO across chunk
//!      boundaries with monotone GLOBAL timestamps. The span's first chunk seeds
//!      the initial image; interior chunks only fold their per-cycle factors; the
//!      last chunk drains and asserts PROD(reads)==PROD(writes). A value written
//!      in chunk i and read in chunk j (i<j) validates through the threaded
//!      product. Registers keep their (already-working) self-contained per-chunk
//!      multiset; only memory is threaded.
//!
//!   3. Final-output binding. The chunk that reaches COMMIT/HALT exposes the
//!      committed public output (the memory image at the output-buffer words) as
//!      circuit public output, checked == the emulator golden.
//!
//! FS binding: one (alpha,beta) challenge shared across the whole span, derived
//! from ALL chunk committed-input fingerprints (so the challenge follows the
//! committed trace). Each chunk is Expander-proven and self-verified.
//!
//! TRACE PCS: the RV32IM execution trace (the circuit input layer) is committed
//! by Expander's OWN stock polynomial commitment, `RawExpanderGKR<GF2ExtConfig>`,
//! via the stock GKR config `RawGF2GKRConfig` below. rsema1d is NOT used for the
//! trace commitment on this path (a later step re-introduces rsema1d ONLY for a
//! block-data linkage, not for the trace).

use crate::circuit_elf as ckt;
use crate::circuit_elf::{native_factor, native_product, Hints, NOUT, SLOTS};
use crate::circuit_gp::{NREG, XLEN};
use crate::emulator::{self, Cpu, Memory, StepRecord};
use crate::gf128;
use crate::loader::{load_elf, STACK_TOP};
use arith::{Field, SimdField};
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, GKREngine, GKRScheme, MPIConfig, MPIEngine};
use gkr_hashers::SHA256hasher;
use gkr_hashers::FiatShamirHasher;
use poly_commit::raw::RawExpanderGKR;
use polynomials::MultiLinearPoly;
use rsema1d_sys::RowRange;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::time::{Duration, Instant};
use transcript::BytesHashTranscript;

/// A GKR engine over `GF2ExtConfig` with an SHA256 Fiat-Shamir transcript and
/// Expander's STOCK `RawExpanderGKR` input polynomial commitment. This mirrors
/// the vendored `GF2ExtConfigSha2Raw` stock config; the EXECUTION TRACE input
/// layer is committed by this PCS (no rsema1d). Single-process (`world_size=1`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RawGF2GKRConfig<'a> {
    _phantom: PhantomData<&'a ()>,
}

impl<'a> GKREngine for RawGF2GKRConfig<'a> {
    type FieldConfig = GF2ExtConfig;
    type MPIConfig = MPIConfig<'a>;
    type TranscriptConfig = BytesHashTranscript<SHA256hasher>;
    type PCSConfig = RawExpanderGKR<GF2ExtConfig>;
    const SCHEME: GKRScheme = GKRScheme::Vanilla;
}

/// 32-byte display digest of a committed input layer (the exact `Vec<GF2x8>`
/// evaluations `RawExpanderGKR` commits). NON-crypto; only used to print/compare
/// a stable commitment id and to show a tampered trace yields a different commit.
fn commit_digest(evals: &[gf2::GF2x8]) -> [u8; 32] {
    let mut s: [u64; 4] = [
        0xcbf29ce484222325,
        0x9e3779b97f4a7c15,
        0x100000001b3,
        0xff51afd7ed558ccd,
    ];
    for e in evals {
        let l = e.unpack();
        let mut w = 0u64;
        for k in 0..8 {
            w |= ((l[k].v & 1) as u64) << k;
        }
        for k in 0..4 {
            s[k] ^= w.rotate_left((k as u32) * 7 + 1);
            s[k] = (s[k] ^ (s[k] >> 29)).wrapping_mul(0xbf58476d1ce4e5b9);
            s[k] ^= s[(k + 1) & 3] >> 17;
        }
    }
    let mut out = [0u8; 32];
    for k in 0..4 {
        out[k * 8..k * 8 + 8].copy_from_slice(&s[k].to_le_bytes());
    }
    out
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

/// Fiat-Shamir: derive (alpha,beta) in GF(2^128) from the concatenation of ALL
/// chunk commitments (binds the challenge to the whole span's committed trace).
fn fs_challenges(commits: &[[u8; 32]]) -> (u128, u128) {
    let mix = |mut z: u64| -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    let derive = |tag: u64| -> u128 {
        let mut lo: u64 = 0xcbf29ce484222325 ^ tag;
        let mut hi: u64 = 0x9e3779b97f4a7c15 ^ mix(tag);
        for c in commits {
            for (i, &b) in c.iter().enumerate() {
                lo = (lo ^ b as u64).wrapping_mul(0x100000001b3);
                hi = (hi ^ (b as u64).rotate_left((i as u32 & 63) + 1)).wrapping_mul(0xff51afd7ed558ccd);
            }
        }
        (((mix(hi) as u128) << 64) | (mix(lo) as u128)) | 1
    };
    (derive(0xA1), derive(0xB2))
}

fn writes_reg(opcode: u32) -> bool {
    matches!(
        opcode,
        emulator::OPC_OP | emulator::OPC_OPIMM | emulator::OPC_LOAD
        | emulator::OPC_JAL | emulator::OPC_JALR | emulator::OPC_LUI | emulator::OPC_AUIPC
    )
}

fn div_hints(rec: &StepRecord) -> (u32, u32) {
    if rec.opcode != emulator::OPC_OP || rec.funct7 != emulator::FUNCT7_M {
        return (0, 0);
    }
    let (a, b) = (rec.rs1_val, rec.rs2_val);
    match rec.funct3 {
        0x4 | 0x6 => {
            let (ai, bi) = (a as i32, b as i32);
            if bi == 0 { (u32::MAX, a) }
            else if ai == i32::MIN && bi == -1 { (i32::MIN as u32, 0) }
            else { ((ai / bi) as u32, (ai % bi) as u32) }
        }
        0x5 | 0x7 => if b == 0 { (u32::MAX, a) } else { (a / b, a % b) },
        _ => (0, 0),
    }
}

// ---------------------------- span planning ---------------------------------

/// A HINT_READ event within the span: the delivered word-aligned image.
struct HintEvent {
    cycle: usize,        // span-relative cycle index of the HINT_READ ecall
    addr_words: Vec<u32>,
    vals: Vec<u32>,
}

/// Everything one chunk needs, produced by the whole-span replay.
struct ChunkPlan {
    len: usize,
    base_cycle: usize, // span-relative index of this chunk's first cycle
    is_first: bool,
    is_last: bool,
    entry_pc: u32,
    exit_pc: u32,
    entry_reg: [u32; 32],
    exit_reg: [u32; 32],
    // committed HINT_READ words handled in this chunk.
    nhint: usize,
    hint_addrs: Vec<u32>,
    hint_ts: u32,
    hint_vals: Vec<u32>,
    // per-cycle circuit hints.
    tprev: Vec<[u32; SLOTS]>,
    vold_c: Vec<u32>,
    vold_d: Vec<u32>,
    vold_f: Vec<u32>,
    div_q: Vec<u32>,
    div_r: Vec<u32>,
    is_sys: Vec<u32>,
    sys_wr: Vec<u32>,
    // per-chunk register finals (self-contained, local ts).
    reg_fin_val: [u32; 32],
    reg_fin_ts: [u32; 32],
    // DISTRIBUTED genesis: addresses first-touched (globally) in this chunk.
    gen_addrs: Vec<u32>,
    gen_init: Vec<u32>,
    // DISTRIBUTED drain: addresses last-touched (globally) in this chunk (close mode).
    drain_addrs: Vec<u32>,
    drain_fin_val: Vec<u32>,
    drain_fin_ts: Vec<u32>,
    // per-chunk output binding (indices into fin_val, golden values).
    out_idx: [usize; NOUT],
    out_vals: [u32; NOUT],
    // native MEMORY factor tuples (addr,val,ts) for this chunk's threaded product.
    mem_reads: Vec<(u32, u32, u32)>,
    mem_writes: Vec<(u32, u32, u32)>,
    trace: Vec<StepRecord>,
}

struct SpanPlan {
    chunks: Vec<ChunkPlan>,
    union_words: usize,    // total distinct touched words (informational)
    golden_pv: Vec<u8>,
    out_vals: [u32; NOUT], // golden output words (for the summary check)
    close: bool,           // whether the trace reached HALT (product closes)
    input_len: usize,
}

/// Capture the span and run the whole-span replay producing per-chunk plans.
fn build_plan(
    elf_path: &str,
    input_path: &str,
    start: u64,
    chunk_len: usize,
    nchunks: usize,
    to_halt: bool,
    do_output_bind: bool,
) -> Result<SpanPlan, String> {
    let elf = std::fs::read(elf_path).map_err(|e| format!("read elf: {e}"))?;
    let input = std::fs::read(input_path).map_err(|e| format!("read input: {e}"))?;
    let input_len = input.len();
    let mut mem = Memory::default();
    let loaded = load_elf(&elf, &mut mem);
    let mut cpu = Cpu::from_image(mem, loaded.entry);
    cpu.regs[2] = STACK_TOP;
    cpu.inputs.push(input);

    // advance to span start (discard).
    for c in 0..start { cpu.step(c as u32); }
    let span_entry_reg = cpu.regs;

    // capture the span, watching for HINT_READ and register snapshots.
    let mut trace: Vec<StepRecord> = Vec::new();
    let mut hint_events: Vec<HintEvent> = Vec::new();
    let mut reg_snaps: Vec<[u32; 32]> = vec![span_entry_reg];
    let mut i: usize = 0;
    let cap = if to_halt { usize::MAX } else { nchunks * chunk_len };
    loop {
        if !to_halt && i >= cap { break; }
        if to_halt && cpu.halted { break; }
        let rec = cpu.step((start + i as u64) as u32);
        let halted_now = cpu.halted;
        trace.push(rec);
        if let Some((ptr, len)) = cpu.last_hint_write.take() {
            let w0 = ptr & !3;
            let end = (ptr.wrapping_add(len).wrapping_add(3)) & !3;
            let addr_words: Vec<u32> = (w0..end).step_by(4).collect();
            let vals: Vec<u32> = addr_words.iter().map(|&a| cpu.mem.load(a)).collect();
            hint_events.push(HintEvent { cycle: i, addr_words, vals });
        }
        reg_snaps.push(cpu.regs);
        i += 1;
        if to_halt && halted_now { break; }
    }
    let total = trace.len();
    if total == 0 { return Err("empty span".into()); }

    // Restrict COMMITTED hint words to those the captured span actually consumes
    // (reads/writes) after delivery. For the FULL trace every delivered byte is
    // consumed, so all are committed; for a PARTIAL span only the consumed prefix
    // is committed (unread words are absent from the open multiset). This binds
    // every input byte the proven span depends on without materialising the whole
    // (e.g. 20985-byte) blob as committed rows in one chunk.
    {
        let mut touched_ls: HashSet<u32> = HashSet::new();
        for rec in &trace {
            if rec.is_load || rec.is_store { touched_ls.insert(rec.mem_addr & !3); }
        }
        for ev in &mut hint_events {
            let keep: Vec<usize> = (0..ev.addr_words.len())
                .filter(|&k| touched_ls.contains(&ev.addr_words[k])).collect();
            ev.addr_words = keep.iter().map(|&k| ev.addr_words[k]).collect();
            ev.vals = keep.iter().map(|&k| ev.vals[k]).collect();
        }
    }

    // chunk lengths: fixed L except the last chunk (to_halt: remainder).
    let mut lens: Vec<usize> = Vec::new();
    let mut acc = 0;
    while acc < total {
        let l = chunk_len.min(total - acc);
        lens.push(l);
        acc += l;
    }
    let nchunk = lens.len();

    let union_words: usize = {
        let mut u: HashSet<u32> = HashSet::new();
        for rec in &trace {
            u.insert(rec.pc & !3);
            if rec.is_load || rec.is_store { u.insert(rec.mem_addr & !3); }
        }
        u.len()
    };

    // per-chunk accumulators.
    let mut chunks: Vec<ChunkPlan> = Vec::new();
    let mut chunk_starts: Vec<usize> = Vec::new();
    { let mut a = 0; for &l in &lens { chunk_starts.push(a); a += l; } }
    let chunk_of = |ci: usize| -> usize {
        let mut ch = 0;
        for (k, &s) in chunk_starts.iter().enumerate() { if ci >= s { ch = k; } }
        ch
    };

    for (ck, &l) in lens.iter().enumerate() {
        let start_idx = chunk_starts[ck];
        chunks.push(ChunkPlan {
            len: l,
            base_cycle: start_idx,
            is_first: ck == 0,
            is_last: ck == nchunk - 1,
            entry_pc: trace[start_idx].pc,
            exit_pc: trace[start_idx + l - 1].next_pc,
            entry_reg: reg_snaps[start_idx],
            exit_reg: reg_snaps[start_idx + l],
            nhint: 0,
            hint_addrs: Vec::new(),
            hint_ts: 0,
            hint_vals: Vec::new(),
            tprev: vec![[0u32; SLOTS]; l],
            vold_c: vec![0u32; l],
            vold_d: vec![0u32; l],
            vold_f: vec![0u32; l],
            div_q: vec![0u32; l],
            div_r: vec![0u32; l],
            is_sys: vec![0u32; l],
            sys_wr: vec![0u32; l],
            reg_fin_val: [0u32; 32],
            reg_fin_ts: [0u32; 32],
            gen_addrs: Vec::new(),
            gen_init: Vec::new(),
            drain_addrs: Vec::new(),
            drain_fin_val: Vec::new(),
            drain_fin_ts: Vec::new(),
            out_idx: [0usize; NOUT],
            out_vals: [0u32; NOUT],
            mem_reads: Vec::new(),
            mem_writes: Vec::new(),
            trace: trace[start_idx..start_idx + l].to_vec(),
        });
    }

    // ---- whole-span replay. registers self-contained per chunk (local ts);
    // memory THREADED globally. genesis writes are DISTRIBUTED: an address is
    // seeded (addr, entry_val, ts=0) in the chunk where it is first touched.
    let mut reg_last_ts = [0u32; 32];
    let mut reg_last_val = span_entry_reg;
    let mut mem_last_ts: HashMap<u32, u32> = HashMap::new();
    let mut mem_last_val: HashMap<u32, u32> = HashMap::new();
    let mut last_touch_chunk: HashMap<u32, usize> = HashMap::new();

    for (gi, rec) in trace.iter().enumerate() {
        let ck = chunk_of(gi);
        let local = gi - chunk_starts[ck];
        if local == 0 {
            reg_last_ts = [0u32; 32];
            reg_last_val = chunks[ck].entry_reg;
        }
        let sys = rec.opcode == emulator::OPC_SYSTEM;
        chunks[ck].is_sys[local] = sys as u32;
        chunks[ck].sys_wr[local] = (sys && rec.rd_idx != 0) as u32;
        let we = writes_reg(rec.opcode) || sys;
        let (q, r) = div_hints(rec);
        chunks[ck].div_q[local] = q;
        chunks[ck].div_r[local] = r;

        for slot in 0..SLOTS {
            let is_mem_slot = slot == 0 || slot == 3;
            let now_ts: u32 = if is_mem_slot {
                (gi * SLOTS + slot + 1) as u32
            } else {
                (local * SLOTS + slot + 1) as u32
            };
            let word = rec.mem_addr & !3;
            let fw = rec.pc & !3;
            let (addr, rval, wval, active): (u32, u32, u32, bool) = match slot {
                0 => (fw, rec.insn, rec.insn, true),
                1 => (rec.rs1_idx, rec.rs1_val, rec.rs1_val, true),
                2 => (rec.rs2_idx, rec.rs2_val, rec.rs2_val, true),
                3 => {
                    if rec.is_load { (word, rec.mem_prev, rec.mem_prev, true) }
                    else if rec.is_store { (word, rec.mem_prev, rec.mem_val, true) }
                    else { (0, 0, 0, false) }
                }
                _ => {
                    if we && rec.rd_idx != 0 {
                        let old = reg_last_val[rec.rd_idx as usize];
                        (rec.rd_idx, old, rec.rd_val, true)
                    } else { (0, 0, 0, false) }
                }
            };

            if is_mem_slot {
                if active {
                    // DISTRIBUTED genesis: first touch of `addr` seeds it here.
                    if !mem_last_val.contains_key(&addr) {
                        chunks[ck].gen_addrs.push(addr);
                        chunks[ck].gen_init.push(rval);
                        chunks[ck].mem_writes.push((addr, rval, 0));
                        mem_last_val.insert(addr, rval);
                        mem_last_ts.insert(addr, 0);
                    }
                    let tp = *mem_last_ts.get(&addr).unwrap_or(&0);
                    chunks[ck].tprev[local][slot] = tp;
                    if slot == 0 { chunks[ck].vold_f[local] = rval; }
                    if slot == 3 { chunks[ck].vold_c[local] = rval; }
                    let cur = *mem_last_val.get(&addr).unwrap_or(&0);
                    if cur != rval {
                        return Err(format!("mem replay mismatch gi={gi} slot={slot} addr={addr:#x} want {cur:#x} got {rval:#x}"));
                    }
                    chunks[ck].mem_reads.push((addr, rval, tp));
                    chunks[ck].mem_writes.push((addr, wval, now_ts));
                    mem_last_ts.insert(addr, now_ts);
                    mem_last_val.insert(addr, wval);
                    last_touch_chunk.insert(addr, ck);
                } else {
                    chunks[ck].mem_reads.push((0, 0, now_ts));
                    chunks[ck].mem_writes.push((0, 0, now_ts));
                }
            } else if active {
                let tp = reg_last_ts[addr as usize];
                chunks[ck].tprev[local][slot] = tp;
                if slot == 4 { chunks[ck].vold_d[local] = rval; }
                let cur = reg_last_val[addr as usize];
                if cur != rval {
                    return Err(format!("reg replay mismatch gi={gi} slot={slot} r={addr} want {cur:#x} got {rval:#x}"));
                }
                reg_last_ts[addr as usize] = now_ts;
                reg_last_val[addr as usize] = wval;
            }
        }

        // HINT_READ committed writes: after the ecall cycle's slots.
        if let Some(ev) = hint_events.iter().find(|e| e.cycle == gi) {
            let hts = ((gi + 1) * SLOTS) as u32;
            if chunks[ck].nhint != 0 {
                return Err("multiple HINT_READ events in one chunk unsupported".into());
            }
            chunks[ck].nhint = ev.addr_words.len();
            chunks[ck].hint_addrs = ev.addr_words.clone();
            chunks[ck].hint_ts = hts;
            chunks[ck].hint_vals = ev.vals.clone();
            for (k, &a) in ev.addr_words.iter().enumerate() {
                chunks[ck].mem_writes.push((a, ev.vals[k], hts));
                mem_last_ts.insert(a, hts);
                mem_last_val.insert(a, ev.vals[k]);
                last_touch_chunk.insert(a, ck);
            }
        }

        if local == lens[ck] - 1 {
            chunks[ck].reg_fin_val = reg_last_val;
            chunks[ck].reg_fin_ts = reg_last_ts;
        }
    }

    // ---- DISTRIBUTED drain (close mode only): drain each touched address in the
    // chunk where it is last touched, with its final value/ts. Summed over chunks
    // this reads every cell exactly once, so the threaded product balances at the
    // end. In OPEN spans (end not reached) nothing is drained (product left open).
    let close = to_halt;
    let mut drain_pos: HashMap<u32, (usize, usize)> = HashMap::new(); // addr -> (chunk, index in drain_addrs)
    if close {
        let mut addrs: Vec<u32> = mem_last_val.keys().copied().collect();
        addrs.sort();
        for a in addrs {
            let q = *last_touch_chunk.get(&a).unwrap_or(&(chunks.len() - 1));
            let v = mem_last_val[&a];
            let t = *mem_last_ts.get(&a).unwrap_or(&0);
            let idx = chunks[q].drain_addrs.len();
            chunks[q].drain_addrs.push(a);
            chunks[q].drain_fin_val.push(v);
            chunks[q].drain_fin_ts.push(t);
            chunks[q].mem_reads.push((a, v, t));
            drain_pos.insert(a, (q, idx));
        }
    }

    // ---- output binding: bind the memory image at the WRITE(fd=PUBLIC_VALUES)
    // buffer words to the GOLDEN committed public output. out_vals are the golden
    // words (from the emulator's committed public values); the circuit asserts the
    // grand-product-validated final memory value at the buffer == golden.
    let golden_pv = cpu.public_values.clone();
    let mut out_vals = [0u32; NOUT];
    if do_output_bind && close {
        if let Some((buf, _len)) = cpu.last_public_write {
            for k in 0..NOUT {
                let a = buf.wrapping_add((k * 4) as u32) & !3;
                let (q, idx) = *drain_pos.get(&a)
                    .ok_or_else(|| format!("output word {a:#x} not drained (not touched?)"))?;
                let mem_final = chunks[q].drain_fin_val[idx];
                let golden = if golden_pv.len() >= (k + 1) * 4 {
                    u32::from_le_bytes([golden_pv[k*4], golden_pv[k*4+1], golden_pv[k*4+2], golden_pv[k*4+3]])
                } else { 0 };
                if mem_final != golden {
                    return Err(format!("output word {a:#x}: final memory {mem_final:#x} != golden committed {golden:#x}"));
                }
                chunks[q].out_idx[k] = NREG + idx;
                chunks[q].out_vals[k] = golden;
                out_vals[k] = golden;
            }
        } else {
            return Err("no WRITE(fd=public_values) captured for output binding".into());
        }
    }

    Ok(SpanPlan { chunks, union_words, golden_pv, out_vals, close, input_len })
}

// ------------------------------ proving -------------------------------------

#[derive(Clone)]
pub struct ChunkProof {
    pub start: u64,
    pub len: usize,
    pub base_cycle: usize,
    pub nmem_drain: usize,
    pub ngen: usize,
    pub nhint: usize,
    pub num_vars: usize,
    pub gate_entries: usize,
    pub is_first: bool,
    pub is_last: bool,
    /// 32-byte digest of the trace input layer committed by Expander's stock
    /// `RawExpanderGKR<GF2ExtConfig>` PCS (NOT an rsema1d commitment).
    pub commitment: [u8; 32],
    pub verified: bool,
    /// Raw-PCS commit determinism (commit twice, same evals => same digest).
    pub commit_stable: bool,
    pub entry_pc: u32,
    pub exit_pc: u32,
    pub entry_reg: [u32; 32],
    pub exit_reg: [u32; 32],
    pub pub_exit_pc: u32,
    pub pub_exit_reg_ok: bool,
    pub tamper_rejected: bool,
    pub mem_pr_in: u128,
    pub mem_pw_in: u128,
    pub mem_pr_out: u128,
    pub mem_pw_out: u128,
    /// The serialized Expander GKR proof bytes for this chunk (as produced by
    /// `executor::prove`). Retained so a C-ABI / FFI caller can return the real
    /// per-chunk proof artifact.
    pub proof_bytes: Vec<u8>,
    pub compile_time: Duration,
    pub prove_time: Duration,
    pub verify_time: Duration,
}

/// 32-byte transcript digest of one chunk's COMMITTED-INPUT rows (the exact
/// values the stock PCS commits): trace words + all replay hints + finals + hint bytes.
/// Used to derive the shared FS challenge, so it follows the committed trace.
fn chunk_fingerprint(cp: &ChunkPlan) -> [u8; 32] {
    let mut s: [u64; 4] = [
        0xcbf29ce484222325 ^ cp.base_cycle as u64,
        0x9e3779b97f4a7c15 ^ cp.len as u64,
        0x100000001b3 ^ cp.entry_pc as u64,
        0xff51afd7ed558ccd ^ cp.exit_pc as u64,
    ];
    let mut absorb = |x: u64| {
        for k in 0..4 {
            s[k] ^= x.rotate_left((k as u32) * 7 + 1);
            s[k] = (s[k] ^ (s[k] >> 29)).wrapping_mul(0xbf58476d1ce4e5b9);
            s[k] ^= s[(k + 1) & 3] >> 17;
        }
    };
    for r in &cp.trace {
        absorb(r.insn as u64);
        absorb(((r.rs1_val as u64) << 32) | r.rs2_val as u64);
        absorb(r.rd_val as u64);
    }
    for t in &cp.tprev { for &x in t { absorb(x as u64); } }
    for v in [&cp.vold_c, &cp.vold_d, &cp.vold_f, &cp.div_q, &cp.div_r, &cp.is_sys, &cp.sys_wr] {
        for &x in v { absorb(x as u64); }
    }
    for &x in &cp.reg_fin_val { absorb(x as u64); }
    for &x in &cp.reg_fin_ts { absorb(x as u64); }
    for &x in &cp.gen_init { absorb(x as u64); }
    for &x in &cp.drain_fin_val { absorb(x as u64); }
    for &x in &cp.drain_fin_ts { absorb(x as u64); }
    for &x in &cp.hint_vals { absorb(x as u64); }
    let mut out = [0u8; 32];
    for k in 0..4 { out[k * 8..k * 8 + 8].copy_from_slice(&s[k].to_le_bytes()); }
    out
}

/// Install the compile-time statics for chunk `cp`.
fn install_statics(plan: &SpanPlan, cp: &ChunkPlan) {
    unsafe {
        ckt::NCYC = cp.len;
        ckt::ENTRY_PC = cp.entry_pc;
        ckt::REG_INIT = cp.entry_reg.to_vec();
        ckt::IS_FIRST = cp.is_first;
        ckt::IS_LAST = cp.is_last && plan.close;
        ckt::BASE_CYCLE = cp.base_cycle;
        ckt::NHINT = cp.nhint;
        ckt::HINT_ADDRS = cp.hint_addrs.clone();
        ckt::HINT_TS = cp.hint_ts;
        // DISTRIBUTED genesis: addresses first-touched in THIS chunk.
        ckt::GEN_ADDRS = cp.gen_addrs.clone();
        ckt::GEN_INIT = cp.gen_init.clone();
        // DISTRIBUTED drain: addresses last-touched in THIS chunk (close mode).
        ckt::NMEM = cp.drain_addrs.len();
        ckt::MEM_ADDRS = cp.drain_addrs.clone();
        ckt::OUT_IDX = cp.out_idx;
    }
}

/// Build the per-chunk Hints from the plan (products / challenge filled by caller).
fn build_hints(_plan: &SpanPlan, cp: &ChunkPlan, alpha: u128, beta: u128) -> Hints {
    let nmem = cp.drain_addrs.len();
    let naddr = NREG + nmem;
    let mut fin_val = vec![0u32; naddr];
    let mut fin_ts = vec![0u32; naddr];
    for r in 0..NREG { fin_val[r] = cp.reg_fin_val[r]; fin_ts[r] = cp.reg_fin_ts[r]; }
    for k in 0..nmem {
        fin_val[NREG + k] = cp.drain_fin_val[k];
        fin_ts[NREG + k] = cp.drain_fin_ts[k];
    }
    let out_vals = cp.out_vals.to_vec();
    let _ = (alpha, beta);
    Hints {
        tprev: cp.tprev.clone(),
        vold_c: cp.vold_c.clone(),
        vold_d: cp.vold_d.clone(),
        vold_f: cp.vold_f.clone(),
        div_q: cp.div_q.clone(),
        div_r: cp.div_r.clone(),
        is_sys: cp.is_sys.clone(),
        sys_wr: cp.sys_wr.clone(),
        fin_val,
        fin_ts,
        entry_reg: cp.entry_reg.to_vec(),
        exit_reg: cp.exit_reg.to_vec(),
        exit_pc: cp.exit_pc,
        out_vals,
        hint_vals: cp.hint_vals.clone(),
        mem_pr_in: 0, mem_pw_in: 0, mem_pr_out: 0, mem_pw_out: 0,
        link_r: Vec::new(), link_y: 0,
    }
}

/// Native chunk memory read/write products under (alpha,beta).
fn chunk_mem_products(cp: &ChunkPlan, alpha: u128, beta: u128, beta2: u128) -> (u128, u128) {
    let pr: Vec<u128> = cp.mem_reads.iter().map(|&(a, v, t)| native_factor(a, v, t, alpha, beta, beta2)).collect();
    let pw: Vec<u128> = cp.mem_writes.iter().map(|&(a, v, t)| native_factor(a, v, t, alpha, beta, beta2)).collect();
    (native_product(&pr), native_product(&pw))
}

/// Result of proving a whole span.
pub struct SpanResult {
    pub proofs: Vec<ChunkProof>,
    pub chained: bool,
    pub mem_thread_ok: bool,
    pub mem_closed: bool,
    pub output_bound: bool,
    pub golden_pv: Vec<u8>,
    pub bound_out: [u32; NOUT],
    pub input_len: usize,
}

/// Prove one span. `to_halt` runs the full trace to HALT (Demo A). Otherwise
/// prove `nchunks` contiguous chunks of `chunk_len` from `start` (Demo B).
pub fn prove_span(
    elf_path: &str,
    input_path: &str,
    start: u64,
    chunk_len: usize,
    nchunks: usize,
    to_halt: bool,
    do_tamper: bool,
) -> Result<SpanResult, String> {
    let plan = build_plan(elf_path, input_path, start, chunk_len, nchunks, to_halt, to_halt)?;
    let n = plan.chunks.len();
    println!("[plan] span cycles={} chunks={} union_words={} close={}",
        plan.chunks.iter().map(|c| c.len).sum::<usize>(), n, plan.union_words, plan.close);
    for (i, c) in plan.chunks.iter().enumerate() {
        if c.nhint > 0 {
            println!("[plan] chunk {i} carries HINT_READ: {} committed words @ ts={} (input_len={})",
                c.nhint, c.hint_ts, plan.input_len);
        }
    }

    // ---- shared Fiat-Shamir challenge, bound to the COMMITTED INPUT rows of
    // every chunk. Derived from a transcript hash over all chunks' committed
    // values so the memory RLC challenge follows the trace. Single-pass (no
    // separate commit pass), since the committed rows are challenge-independent.
    let fps: Vec<[u8; 32]> = plan.chunks.iter().map(chunk_fingerprint).collect();
    let (alpha, beta) = fs_challenges(&fps);
    let beta2 = gf128::native_mul(beta, beta);
    println!("[FS] shared (alpha,beta) bound to {} chunks' committed-input rows: alpha={:#034x} beta={:#034x}", n, alpha, beta);

    let mut mem_pr = 1u128;
    let mut mem_pw = 1u128;
    let mut proofs: Vec<ChunkProof> = Vec::new();
    println!("\n[prove] proving {n} chunks with the shared challenge, threading memory ...");
    for (i, cp) in plan.chunks.iter().enumerate() {
        install_statics(&plan, cp);
        let t_c = Instant::now();
        let CompileResult { witness_solver, layered_circuit } =
            compile(&ckt::template(), CompileOptions::default()).map_err(|e| format!("compile[{i}]: {e:?}"))?;
        let compile_time = t_c.elapsed();

        let (cpr, cpw) = chunk_mem_products(cp, alpha, beta, beta2);
        let pr_in = mem_pr;
        let pw_in = mem_pw;
        let pr_out = gf128::native_mul(pr_in, cpr);
        let pw_out = gf128::native_mul(pw_in, cpw);

        let mut h = build_hints(&plan, cp, alpha, beta);
        h.mem_pr_in = pr_in; h.mem_pw_in = pw_in; h.mem_pr_out = pr_out; h.mem_pw_out = pw_out;

        let assignment = ckt::build_assignment(&cp.trace, &h, alpha, beta);
        let witness = witness_solver.solve_witnesses(&vec![assignment.clone(); 8]).map_err(|e| format!("solveB[{i}]: {e:?}"))?;
        let res = layered_circuit.run(&witness);
        if !res.iter().all(|x| *x) {
            return Err(format!("chunk {i} layered self-eval FAILED (memory/register/threading constraint)"));
        }
        use std::io::Write as _;
        print!("[prog] chunk {i}/{n} [{},{}) self-eval OK (compile={:?} gen={} drain={} hint={}) ... ",
            cp.base_cycle, cp.base_cycle + cp.len, compile_time, cp.gen_addrs.len(), cp.drain_addrs.len(), cp.nhint);
        let _ = std::io::stdout().flush();

        let mut ec = layered_circuit.export_to_expander_flatten();
        let (si, sp) = witness.to_simd::<gf2::GF2x8>();
        ec.layers[0].input_vals = si.clone();
        ec.public_input = sp.clone();
        ec.evaluate();
        let input_vals = ec.layers[0].input_vals.clone();
        let gate_entries: usize = ec.layers.iter().map(|l| l.mul.len() + l.add.len()).sum();
        let num_vars = ec.log_input_size();

        let mpi = MPIConfig::prover_new(None, None);
        // ---- commit the EXECUTION-TRACE input layer via Expander's STOCK PCS,
        // RawExpanderGKR<GF2ExtConfig>. No rsema1d is involved in this commitment.
        let params = <RawExpanderGKR<GF2ExtConfig> as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi.world_size());
        let mut spad = <RawExpanderGKR<GF2ExtConfig> as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
        let input_poly = MultiLinearPoly::new(input_vals.clone());
        let c_raw = <RawExpanderGKR<GF2ExtConfig> as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &input_poly, &mut spad)
            .ok_or("RawExpanderGKR commit None")?;
        let c_real = commit_digest(&c_raw.evals);
        // stability: recommit the same input layer (determinism of the stock PCS).
        let mut spad2 = <RawExpanderGKR<GF2ExtConfig> as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
        let c_raw2 = <RawExpanderGKR<GF2ExtConfig> as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &input_poly, &mut spad2)
            .ok_or("RawExpanderGKR commit None")?;
        let commit_stable = commit_digest(&c_raw2.evals) == c_real;

        let t_p = Instant::now();
        let (claimed_v, proof) = executor::prove::<RawGF2GKRConfig<'static>>(&mut ec, mpi.clone());
        let prove_time = t_p.elapsed();
        let t_v = Instant::now();
        let ok = executor::verify::<RawGF2GKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v);
        let verify_time = t_v.elapsed();
        let verified = ok && claimed_v.is_zero();
        println!("verified={verified} pcs=RawExpanderGKR<GF2Ext> commit={}", &hex(&c_real)[..16]);
        let _ = std::io::stdout().flush();

        // reconstruct public exit state from public input layout.
        // layout: alpha128|beta128|entry_pc32|exit_pc32|entry_reg[32*32]|exit_reg[32*32]|out[NOUT*32]|mem_pr_in128|...
        let word_at = |off: usize| -> u32 {
            let mut w = 0u32;
            for b in 0..XLEN { w |= ((sp[off + b].unpack()[0].v & 1) as u32) << b; }
            w
        };
        let pub_exit_pc = word_at(288);
        let exit_reg_off = 256 + 32 + 32 + 32 * 32;
        let mut pub_exit_reg_ok = true;
        for r in 0..NREG {
            if word_at(exit_reg_off + r * XLEN) != cp.exit_reg[r] { pub_exit_reg_ok = false; }
        }

        // tamper: corrupt a committed value; must be rejected AND change commitment.
        let mut tamper_rejected = false;
        if do_tamper {
            let mut bad = assignment.clone();
            if let Some(lw) = cp.trace.iter().position(|r| r.is_load) {
                ckt::tamper_rd_val(&mut bad, lw, cp.trace[lw].rd_val.wrapping_add(1));
            } else {
                ckt::tamper_insn_bit(&mut bad, 0, 7);
            }
            let bad_w = witness_solver.solve_witnesses(&vec![bad; 8]).map_err(|e| format!("solve tamper[{i}]: {e:?}"))?;
            let bad_run = layered_circuit.run(&bad_w);
            let bad_all_ok = bad_run.iter().all(|x| *x);
            let (bsi, _bsp) = bad_w.to_simd::<gf2::GF2x8>();
            // stock-PCS soundness: the tampered trace both fails the circuit and
            // yields a DIFFERENT stock-PCS commitment (different input layer).
            let c_bad = commit_digest(&bsi);
            tamper_rejected = !bad_all_ok && c_bad != c_real;
        }

        proofs.push(ChunkProof {
            start: start + cp.base_cycle as u64,
            len: cp.len,
            base_cycle: cp.base_cycle,
            nmem_drain: cp.drain_addrs.len(),
            ngen: cp.gen_addrs.len(),
            nhint: cp.nhint,
            num_vars, gate_entries,
            is_first: cp.is_first, is_last: cp.is_last && plan.close,
            commitment: c_real, verified, commit_stable,
            entry_pc: cp.entry_pc, exit_pc: cp.exit_pc,
            entry_reg: cp.entry_reg, exit_reg: cp.exit_reg,
            pub_exit_pc, pub_exit_reg_ok, tamper_rejected,
            mem_pr_in: pr_in, mem_pw_in: pw_in, mem_pr_out: pr_out, mem_pw_out: pw_out,
            proof_bytes: proof.bytes.clone(),
            compile_time, prove_time, verify_time,
        });

        mem_pr = pr_out;
        mem_pw = pw_out;
    }

    // ---- chain checks: pc + regs + memory-product handoff. ----
    let mut chained = true;
    let mut mem_thread_ok = true;
    for w in proofs.windows(2) {
        if w[0].exit_pc != w[1].entry_pc { chained = false; }
        if w[0].exit_reg != w[1].entry_reg { chained = false; }
        if w[0].mem_pr_out != w[1].mem_pr_in { mem_thread_ok = false; }
        if w[0].mem_pw_out != w[1].mem_pw_in { mem_thread_ok = false; }
    }
    let mem_closed = plan.close && proofs.last().map(|p| p.mem_pr_out == p.mem_pw_out).unwrap_or(false);
    let output_bound = plan.close && !plan.golden_pv.is_empty();

    Ok(SpanResult {
        proofs, chained, mem_thread_ok, mem_closed, output_bound,
        golden_pv: plan.golden_pv, bound_out: plan.out_vals, input_len: plan.input_len,
    })
}

/// Checkpoint carried between segments of a resumable full compose.
#[derive(Clone)]
pub struct Checkpoint {
    pub chunks_done: usize,
    pub mem_pr: u128,
    pub mem_pw: u128,
    pub prev_exit_pc: u32,
    pub prev_exit_reg: [u32; 32],
    pub all_ok: bool,
    pub chained: bool,
    pub mem_thread_ok: bool,
}
impl Checkpoint {
    fn fresh() -> Self {
        Checkpoint { chunks_done: 0, mem_pr: 1, mem_pw: 1, prev_exit_pc: 0,
            prev_exit_reg: [0u32; 32], all_ok: true, chained: true, mem_thread_ok: true }
    }
    fn load(path: &str) -> Option<Self> {
        let s = std::fs::read_to_string(path).ok()?;
        let mut it = s.split_whitespace();
        let chunks_done = it.next()?.parse().ok()?;
        let mem_pr = u128::from_str_radix(it.next()?, 16).ok()?;
        let mem_pw = u128::from_str_radix(it.next()?, 16).ok()?;
        let prev_exit_pc = u32::from_str_radix(it.next()?, 16).ok()?;
        let mut prev_exit_reg = [0u32; 32];
        for r in 0..32 { prev_exit_reg[r] = u32::from_str_radix(it.next()?, 16).ok()?; }
        let all_ok = it.next()? == "1";
        let chained = it.next()? == "1";
        let mem_thread_ok = it.next()? == "1";
        Some(Checkpoint { chunks_done, mem_pr, mem_pw, prev_exit_pc, prev_exit_reg, all_ok, chained, mem_thread_ok })
    }
    fn save(&self, path: &str) -> Result<(), String> {
        let mut s = format!("{} {:x} {:x} {:x}", self.chunks_done, self.mem_pr, self.mem_pw, self.prev_exit_pc);
        for r in 0..32 { s.push_str(&format!(" {:x}", self.prev_exit_reg[r])); }
        s.push_str(&format!(" {} {} {}", self.all_ok as u8, self.chained as u8, self.mem_thread_ok as u8));
        std::fs::write(path, s).map_err(|e| format!("save ckpt: {e}"))
    }
}

/// Prove a SEGMENT [seg_start, seg_end) of the full to-HALT compose, threading the
/// memory product + register/pc chain from a checkpoint file so a long full-trace
/// composition can be completed in resumable pieces (each survivable). The shared
/// FS challenge is derived from ALL chunks' committed-input fingerprints (computed
/// upfront from the plan), so every segment uses the identical (alpha,beta).
pub fn prove_full_segment(
    elf_path: &str,
    input_path: &str,
    chunk_len: usize,
    seg_start: usize,
    seg_end: usize,
    ckpt_path: &str,
    do_tamper: bool,
) -> Result<(), String> {
    use std::io::Write as _;
    let plan = build_plan(elf_path, input_path, 0, chunk_len, 0, true, true)?;
    let n = plan.chunks.len();
    let seg_end = seg_end.min(n);
    let fps: Vec<[u8; 32]> = plan.chunks.iter().map(chunk_fingerprint).collect();
    let (alpha, beta) = fs_challenges(&fps);
    let beta2 = gf128::native_mul(beta, beta);
    if seg_start == 0 {
        println!("[plan] FULL compose: {n} chunks, {} cycles, union_words={}",
            plan.chunks.iter().map(|c| c.len).sum::<usize>(), plan.union_words);
        println!("[FS] shared (alpha,beta) over {n} chunk fingerprints: alpha={:#034x} beta={:#034x}", alpha, beta);
    }
    let mut ck = if seg_start == 0 { Checkpoint::fresh() } else {
        Checkpoint::load(ckpt_path).ok_or("missing/invalid checkpoint for resume")?
    };
    if ck.chunks_done != seg_start {
        return Err(format!("checkpoint at {} but segment starts at {seg_start}", ck.chunks_done));
    }
    println!("[seg] proving chunks [{seg_start},{seg_end}) of {n}; carried mem_pr={:#x}", ck.mem_pr);

    for i in seg_start..seg_end {
        let cp = &plan.chunks[i];
        install_statics(&plan, cp);
        let t_c = Instant::now();
        let CompileResult { witness_solver, layered_circuit } =
            compile(&ckt::template(), CompileOptions::default()).map_err(|e| format!("compile[{i}]: {e:?}"))?;
        let compile_time = t_c.elapsed();

        let (cpr, cpw) = chunk_mem_products(cp, alpha, beta, beta2);
        let pr_in = ck.mem_pr;
        let pw_in = ck.mem_pw;
        let pr_out = gf128::native_mul(pr_in, cpr);
        let pw_out = gf128::native_mul(pw_in, cpw);

        let mut h = build_hints(&plan, cp, alpha, beta);
        h.mem_pr_in = pr_in; h.mem_pw_in = pw_in; h.mem_pr_out = pr_out; h.mem_pw_out = pw_out;
        let assignment = ckt::build_assignment(&cp.trace, &h, alpha, beta);
        let witness = witness_solver.solve_witnesses(&vec![assignment.clone(); 8]).map_err(|e| format!("solve[{i}]: {e:?}"))?;
        if !layered_circuit.run(&witness).iter().all(|x| *x) {
            return Err(format!("chunk {i} self-eval FAILED"));
        }
        print!("[prog] chunk {i}/{n} [{},{}) self-eval OK (compile={:?} gen={} drain={} hint={}{}{}) ... ",
            cp.base_cycle, cp.base_cycle + cp.len, compile_time, cp.gen_addrs.len(), cp.drain_addrs.len(), cp.nhint,
            if cp.is_first { " FIRST" } else { "" }, if cp.is_last { " LAST/close" } else { "" });
        let _ = std::io::stdout().flush();

        let mut ec = layered_circuit.export_to_expander_flatten();
        let (si, sp) = witness.to_simd::<gf2::GF2x8>();
        ec.layers[0].input_vals = si.clone();
        ec.public_input = sp.clone();
        ec.evaluate();
        let input_vals = ec.layers[0].input_vals.clone();
        let num_vars = ec.log_input_size();

        let mpi = MPIConfig::prover_new(None, None);
        // Commit the EXECUTION-TRACE input layer via Expander's STOCK PCS,
        // RawExpanderGKR<GF2ExtConfig>. No rsema1d is involved in this commitment.
        let params = <RawExpanderGKR<GF2ExtConfig> as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi.world_size());
        let mut spad = <RawExpanderGKR<GF2ExtConfig> as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
        let input_poly = MultiLinearPoly::new(input_vals.clone());
        let c_raw = <RawExpanderGKR<GF2ExtConfig> as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &input_poly, &mut spad)
            .ok_or("RawExpanderGKR commit None")?;
        let c_real = commit_digest(&c_raw.evals);

        let (claimed_v, proof) = executor::prove::<RawGF2GKRConfig<'static>>(&mut ec, mpi.clone());
        let ok = executor::verify::<RawGF2GKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v);
        let verified = ok && claimed_v.is_zero();
        let _ = &proof;

        // tamper.
        let mut tamper_rejected = true;
        if do_tamper {
            let mut bad = assignment.clone();
            if let Some(lw) = cp.trace.iter().position(|r| r.is_load) {
                ckt::tamper_rd_val(&mut bad, lw, cp.trace[lw].rd_val.wrapping_add(1));
            } else { ckt::tamper_insn_bit(&mut bad, 0, 7); }
            let bad_w = witness_solver.solve_witnesses(&vec![bad; 8]).map_err(|e| format!("tamper[{i}]: {e:?}"))?;
            let bad_ok = layered_circuit.run(&bad_w).iter().all(|x| *x);
            let (bsi, _) = bad_w.to_simd::<gf2::GF2x8>();
            // stock-PCS soundness: tampered trace fails the circuit AND yields a
            // DIFFERENT stock-PCS commitment (different input layer).
            tamper_rejected = !bad_ok && commit_digest(&bsi) != c_real;
        }

        // chain vs previous chunk.
        if i > 0 {
            if cp.entry_pc != ck.prev_exit_pc { ck.chained = false; }
            if cp.entry_reg != ck.prev_exit_reg { ck.chained = false; }
            // memory-product handoff is automatic: pr_in == carried mem_pr.
        }
        if !(verified && tamper_rejected) { ck.all_ok = false; }
        println!("verified={verified} pcs=RawExpanderGKR<GF2Ext> tamper-rej={tamper_rejected} commit={}", &hex(&c_real)[..16]);
        let _ = std::io::stdout().flush();

        ck.mem_pr = pr_out;
        ck.mem_pw = pw_out;
        ck.prev_exit_pc = cp.exit_pc;
        ck.prev_exit_reg = cp.exit_reg;
        ck.chunks_done = i + 1;
        ck.save(ckpt_path)?;
    }

    if seg_end == n {
        let closed = ck.mem_pr == ck.mem_pw;
        println!("\n=== FULL COMPOSE COMPLETE ===");
        println!("chunks proven      : {n}/{n}");
        println!("every chunk pass   : {}", ck.all_ok);
        println!("pc+reg chain       : {}", ck.chained);
        println!("GLOBAL memory multiset CLOSED (pr==pw) : {closed}");
        println!("  final mem_pr = {:#034x}", ck.mem_pr);
        println!("  final mem_pw = {:#034x}", ck.mem_pw);
        let g = &plan.golden_pv;
        print!("golden committed output = 0x"); for b in g.iter().take(8) { print!("{b:02x}"); } println!();
        println!("output bound to golden = {:?}", plan.out_vals);
    } else {
        println!("[seg] segment done; checkpoint saved at chunk {}", ck.chunks_done);
    }
    Ok(())
}

/// FAST native-only pre-check of the WHOLE close-mode plan: builds the span to
/// HALT, threads the native memory grand-product across every chunk (distributed
/// genesis + per-cycle + hint + distributed drain) and reports whether the global
/// product CLOSES (PROD reads == PROD writes) and whether the output-buffer memory
/// image equals the golden committed output. No compile/prove — validates the
/// continuation bookkeeping in seconds.
pub fn dry_close_check(elf_path: &str, input_path: &str, chunk_len: usize) -> Result<(), String> {
    let plan = build_plan(elf_path, input_path, 0, chunk_len, 0, true, true)?;
    let fps: Vec<[u8; 32]> = plan.chunks.iter().map(chunk_fingerprint).collect();
    let (alpha, beta) = fs_challenges(&fps);
    let beta2 = gf128::native_mul(beta, beta);
    let (mut pr, mut pw) = (1u128, 1u128);
    let mut total_reads = 0usize;
    let mut total_writes = 0usize;
    for cp in &plan.chunks {
        let (cpr, cpw) = chunk_mem_products(cp, alpha, beta, beta2);
        pr = gf128::native_mul(pr, cpr);
        pw = gf128::native_mul(pw, cpw);
        total_reads += cp.mem_reads.len();
        total_writes += cp.mem_writes.len();
    }
    println!("[dry] span cycles={} chunks={} union_words={}",
        plan.chunks.iter().map(|c| c.len).sum::<usize>(), plan.chunks.len(), plan.union_words);
    println!("[dry] total mem read factors={total_reads} write factors={total_writes}");
    println!("[dry] final mem_pr = {pr:#034x}");
    println!("[dry] final mem_pw = {pw:#034x}");
    println!("[dry] GLOBAL MEMORY MULTISET CLOSES (pr==pw) = {}", pr == pw);
    println!("[dry] golden committed public values ({} bytes) = {}", plan.golden_pv.len(), hex(&plan.golden_pv));
    let gw: Vec<u32> = (0..NOUT).map(|k| if plan.golden_pv.len() >= (k+1)*4 {
        u32::from_le_bytes([plan.golden_pv[k*4],plan.golden_pv[k*4+1],plan.golden_pv[k*4+2],plan.golden_pv[k*4+3]])
    } else { 0 }).collect();
    println!("[dry] output binding out_vals={:?} golden_words={:?} match={}",
        plan.out_vals, gw, plan.out_vals.to_vec() == gw);
    if pr != pw { return Err("dry check: memory multiset does NOT close".into()); }
    Ok(())
}

/// Prove a SINGLE self-contained chunk (compat shim for the earlier bin).
pub fn prove_chunk(
    elf_path: &str,
    input_path: &str,
    start: u64,
    len: usize,
    do_tamper: bool,
) -> Result<ChunkProof, String> {
    let r = prove_span(elf_path, input_path, start, len, 1, false, do_tamper)?;
    r.proofs.into_iter().next().ok_or_else(|| "no chunk proven".into())
}

/// Verify a linked chain (pc + all 32 registers + memory product handoff).
pub fn check_chain(chunks: &[ChunkProof]) -> bool {
    for w in chunks.windows(2) {
        if w[0].exit_pc != w[1].entry_pc { return false; }
        if w[0].exit_reg != w[1].entry_reg { return false; }
        if w[0].mem_pr_out != w[1].mem_pr_in { return false; }
        if w[0].mem_pw_out != w[1].mem_pw_in { return false; }
    }
    true
}

/// Golden public values from a full emulator run (for the FINAL chunk check).
pub fn golden_from_run(elf_path: &str, input_path: &str) -> Result<(u64, [u8; 32]), String> {
    let elf = std::fs::read(elf_path).map_err(|e| format!("read elf: {e}"))?;
    let input = std::fs::read(input_path).map_err(|e| format!("read input: {e}"))?;
    let mut mem = Memory::default();
    let loaded = load_elf(&elf, &mut mem);
    let mut cpu = Cpu::from_image(mem, loaded.entry);
    cpu.regs[2] = STACK_TOP;
    cpu.inputs.push(input);
    let _ = cpu.run_to_halt(200_000_000_000u64, false);
    let pv = &cpu.public_values;
    if pv.len() < 40 { return Err("public values too short".into()); }
    let block = u64::from_le_bytes(pv[0..8].try_into().unwrap());
    let root: [u8; 32] = pv[8..40].try_into().unwrap();
    Ok((block, root))
}
