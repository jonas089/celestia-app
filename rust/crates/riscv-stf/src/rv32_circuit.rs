//! In-circuit RV32I INTERPRETER over GF2 — the "accidental computer" step
//! function, mirroring `evm_circuit.rs` but for the RISC-V base integer ISA.
//!
//! REGIME (read `block_stf.rs` / `rsema1d-pcs` for the whole point): Expander GKR
//! commits ONLY the circuit input (leaf) layer, which we make BE the DA-committed
//! block data. Here the committed input layer is:
//!     program words  +  initial registers  +  initial memory (addr+val)
//! (the "input bytes" live inside the initial memory region — see `rv32_prove`).
//! EVERYTHING ELSE — every per-cycle register value, pc, memory word, decoded
//! field and ALU result — is an INTERMEDIATE wire pinned by sumcheck, NEVER
//! committed. There is DELIBERATELY no committed execution trace.
//!
//! State (`Rv32State`) is threaded across `n_steps` unrolled `step`s, exactly as
//! `Evm` is threaded through `evm_circuit::run`. Each `step`:
//!   1. fetches the current instruction by muxing the committed `program` by pc,
//!   2. decodes rv32i (opcode / rd / funct3 / rs1 / rs2 / funct7 / I,S,B,U,J imm),
//!   3. executes one instruction over regs + a bounded word memory,
//!   4. advances pc.
//! x0 is hardwired to 0. Programs terminate on a `JAL x0, 0` self-loop, so padding
//! steps idle (matching the native emulator's fixed-cycle `run`). Deterministic —
//! no advice.

use crate::u256::{add as u_add, eq, is_zero as _is_zero, lt, select, sub as u_sub};
use expander_compiler::frontend::*;

pub const XLEN: usize = 32;
pub type W = Vec<Variable>; // 32-bit little-endian value

/// Bounds for the RV32 machine (committed sizes fixed at circuit-compile time).
pub struct Rv32Cfg {
    pub nreg: usize,       // number of registers (32 for RV32I)
    pub mem_slots: usize,  // bounded word-memory slots
    pub waddr_bits: usize, // committed word-address width
    pub prog_len: usize,   // committed program length (words)
}

/// The RV32 machine state threaded through `step`. `regs`, `pc` and `mem_val` are
/// INTERMEDIATE wires; `mem_addr` are the committed initial word addresses (they
/// never change — only the values stored at them do).
pub struct Rv32State {
    pub regs: Vec<W>,                 // nreg x 32
    pub pc: W,                        // 32-bit program counter (base assumed 0)
    pub mem_addr: Vec<Vec<Variable>>, // mem_slots x waddr_bits (committed, constant)
    pub mem_val: Vec<W>,              // mem_slots x 32 (threaded)
}

// ------------------------------ small helpers ------------------------------

fn zeros<C: Config>(api: &mut impl RootAPI<C>, n: usize) -> Vec<Variable> {
    vec![api.constant(0); n]
}
fn const_u32<C: Config>(api: &mut impl RootAPI<C>, v: u32) -> W {
    (0..XLEN).map(|b| api.constant((v >> b) & 1)).collect()
}
/// 1 iff the LE bit-vector `a` equals the constant `v` (compared over `a.len()` bits).
fn eq_const<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], v: u32) -> Variable {
    let mut acc = api.constant(1);
    for (b, &ab) in a.iter().enumerate() {
        let want = (v >> b) & 1;
        let t = if want == 1 { ab } else { api.sub(1, ab) };
        acc = api.mul(acc, t);
    }
    acc
}
/// sel ? a : b for a single bit.
fn bit_select<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: Variable, b: Variable) -> Variable {
    let d = api.add(a, b);
    let t = api.mul(sel, d);
    api.add(b, t)
}
/// Low 32 bits of a + b.
fn add32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> W {
    let s = u_add(api, a, b);
    s[0..XLEN].to_vec()
}
/// a - b mod 2^32.
fn sub32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> W {
    let (d, _borrow) = u_sub(api, a, b);
    d[0..XLEN].to_vec()
}
fn xor32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> W {
    (0..XLEN).map(|i| api.add(a[i], b[i])).collect()
}
fn and32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> W {
    (0..XLEN).map(|i| api.mul(a[i], b[i])).collect()
}
fn or32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> W {
    (0..XLEN)
        .map(|i| {
            let x = api.add(a[i], b[i]);
            let y = api.mul(a[i], b[i]);
            api.add(x, y)
        })
        .collect()
}
/// Signed less-than (two's complement) over 32-bit values.
fn signed_lt<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Variable {
    let sa = a[XLEN - 1];
    let sb = b[XLEN - 1];
    let ult = lt(api, a, b);
    let diff = api.add(sa, sb); // signs differ (XOR)
    let t1 = api.mul(diff, sa); // if signs differ, the negative one is smaller
    let nd = api.sub(1, diff);
    let t2 = api.mul(nd, ult);
    api.add(t1, t2)
}
/// Logical shift-left by a dynamic 5-bit amount `sh` (barrel, 5 stages).
fn shl32<C: Config>(api: &mut impl RootAPI<C>, val: &W, sh: &[Variable]) -> W {
    let zero = api.constant(0);
    let mut acc = val.clone();
    for k in 0..5 {
        let amt = 1usize << k;
        let shifted: W = (0..XLEN).map(|i| if i >= amt { acc[i - amt] } else { zero }).collect();
        acc = select(api, sh[k], &shifted, &acc);
    }
    acc
}
/// Logical shift-right by a dynamic 5-bit amount.
fn srl32<C: Config>(api: &mut impl RootAPI<C>, val: &W, sh: &[Variable]) -> W {
    let zero = api.constant(0);
    let mut acc = val.clone();
    for k in 0..5 {
        let amt = 1usize << k;
        let shifted: W = (0..XLEN).map(|i| if i + amt < XLEN { acc[i + amt] } else { zero }).collect();
        acc = select(api, sh[k], &shifted, &acc);
    }
    acc
}
/// Arithmetic shift-right by a dynamic 5-bit amount (fills vacated bits with sign).
fn sra32<C: Config>(api: &mut impl RootAPI<C>, val: &W, sh: &[Variable]) -> W {
    let sign = val[XLEN - 1];
    let mut acc = val.clone();
    for k in 0..5 {
        let amt = 1usize << k;
        let shifted: W = (0..XLEN).map(|i| if i + amt < XLEN { acc[i + amt] } else { sign }).collect();
        acc = select(api, sh[k], &shifted, &acc);
    }
    acc
}
/// Read register `idx` (5-bit LE) via a mux over the register file.
fn read_reg<C: Config>(api: &mut impl RootAPI<C>, regs: &[W], idx: &[Variable]) -> W {
    let mut out = zeros(api, XLEN);
    for (i, r) in regs.iter().enumerate() {
        let sel = eq_const(api, idx, i as u32);
        for b in 0..XLEN {
            let t = api.mul(sel, r[b]);
            out[b] = api.add(out[b], t);
        }
    }
    out
}
/// Extract byte `idx2` (2-bit) from a 32-bit word (returns 8 bits).
fn extract_byte<C: Config>(api: &mut impl RootAPI<C>, word: &W, idx2: &[Variable]) -> Vec<Variable> {
    let mut out = zeros(api, 8);
    for k in 0..4 {
        let sel = eq_const(api, idx2, k as u32);
        for b in 0..8 {
            let t = api.mul(sel, word[k * 8 + b]);
            out[b] = api.add(out[b], t);
        }
    }
    out
}
/// Zero- or sign-extend an `n`-bit field to 32 bits.
fn extend<C: Config>(api: &mut impl RootAPI<C>, field: &[Variable], signed: bool) -> W {
    let n = field.len();
    let fill = if signed { field[n - 1] } else { api.constant(0) };
    (0..XLEN).map(|i| if i < n { field[i] } else { fill }).collect()
}

// -------------------------------- the step ---------------------------------

/// One RV32I instruction step over the committed `program`, threading `st`. base
/// address is assumed 0 (the instruction index is pc>>2). Implements the full
/// RV32I base integer ISA (see module tests / `rv32_prove` for the exact list).
pub fn step<C: Config>(api: &mut impl RootAPI<C>, st: Rv32State, program: &[W], cfg: &Rv32Cfg) -> Rv32State {
    let zero = api.constant(0);

    // --- fetch: mux committed program by pc>>2 (12-bit window; base=0) ---
    let widx: Vec<Variable> = (2..2 + 12).map(|i| st.pc[i]).collect();
    let mut insn = zeros(api, XLEN);
    for j in 0..cfg.prog_len {
        let sel = eq_const(api, &widx, j as u32);
        for b in 0..XLEN {
            let t = api.mul(sel, program[j][b]);
            insn[b] = api.add(insn[b], t);
        }
    }

    // --- decode fields ---
    let op = insn[0..7].to_vec();
    let rd = insn[7..12].to_vec();
    let f3 = insn[12..15].to_vec();
    let rs1 = insn[15..20].to_vec();
    let rs2 = insn[20..25].to_vec();
    let f7 = insn[25..32].to_vec();
    let insn30 = insn[30]; // arithmetic-shift bit (SUB/SRA/SRAI)

    // immediates (32-bit LE, sign-extended where applicable)
    let i_imm: W = (0..XLEN).map(|i| if i < 12 { insn[20 + i] } else { insn[31] }).collect();
    let s_imm: W = (0..XLEN)
        .map(|i| if i < 5 { insn[7 + i] } else if i < 12 { insn[25 + (i - 5)] } else { insn[31] })
        .collect();
    let b_imm: W = (0..XLEN)
        .map(|i| {
            if i == 0 { zero }
            else if i < 5 { insn[8 + (i - 1)] }   // imm[4:1] = insn[11:8]
            else if i < 11 { insn[25 + (i - 5)] } // imm[10:5] = insn[30:25]
            else if i == 11 { insn[7] }
            else { insn[31] }                     // imm[12] + sign
        })
        .collect();
    let u_imm: W = (0..XLEN).map(|i| if i < 12 { zero } else { insn[i] }).collect();
    let j_imm: W = (0..XLEN)
        .map(|i| {
            if i == 0 { zero }
            else if i < 11 { insn[21 + (i - 1)] }  // imm[10:1] = insn[30:21]
            else if i == 11 { insn[20] }
            else if i < 20 { insn[12 + (i - 12)] } // imm[19:12] = insn[19:12]
            else { insn[31] }                      // imm[20] + sign
        })
        .collect();

    // opcode selectors (mutually exclusive)
    let is_op = eq_const(api, &op, 0b0110011);
    let is_opimm = eq_const(api, &op, 0b0010011);
    let is_load = eq_const(api, &op, 0b0000011);
    let is_store = eq_const(api, &op, 0b0100011);
    let is_branch = eq_const(api, &op, 0b1100011);
    let is_jal = eq_const(api, &op, 0b1101111);
    let is_jalr = eq_const(api, &op, 0b1100111);
    let is_lui = eq_const(api, &op, 0b0110111);
    let is_auipc = eq_const(api, &op, 0b0010111);

    // funct3 one-hot
    let f3_0 = eq_const(api, &f3, 0);
    let f3_1 = eq_const(api, &f3, 1);
    let f3_2 = eq_const(api, &f3, 2);
    let f3_3 = eq_const(api, &f3, 3);
    let f3_4 = eq_const(api, &f3, 4);
    let f3_5 = eq_const(api, &f3, 5);
    let f3_6 = eq_const(api, &f3, 6);
    let f3_7 = eq_const(api, &f3, 7);
    let f7_20 = eq_const(api, &f7, 0x20); // SUB / SRA marker

    let rs1_val = read_reg(api, &st.regs, &rs1);
    let rs2_val = read_reg(api, &st.regs, &rs2);
    let rs2_sh: Vec<Variable> = rs2_val[0..5].to_vec();
    let imm_sh: Vec<Variable> = i_imm[0..5].to_vec();

    // --- R-type ALU (rs1 op rs2) ---
    let add_r = add32(api, &rs1_val, &rs2_val);
    let sub_r = sub32(api, &rs1_val, &rs2_val);
    let addsub_r = select(api, f7_20, &sub_r, &add_r);
    let sll_r = shl32(api, &rs1_val, &rs2_sh);
    let slt_r = { let b = signed_lt(api, &rs1_val, &rs2_val); let mut v = zeros(api, XLEN); v[0] = b; v };
    let sltu_r = { let b = lt(api, &rs1_val, &rs2_val); let mut v = zeros(api, XLEN); v[0] = b; v };
    let xor_r = xor32(api, &rs1_val, &rs2_val);
    let srl_r = srl32(api, &rs1_val, &rs2_sh);
    let sra_r = sra32(api, &rs1_val, &rs2_sh);
    let srlsra_r = select(api, insn30, &sra_r, &srl_r);
    let or_r = or32(api, &rs1_val, &rs2_val);
    let and_r = and32(api, &rs1_val, &rs2_val);
    let mut r_val = addsub_r; // f3 == 0
    r_val = select(api, f3_1, &sll_r, &r_val);
    r_val = select(api, f3_2, &slt_r, &r_val);
    r_val = select(api, f3_3, &sltu_r, &r_val);
    r_val = select(api, f3_4, &xor_r, &r_val);
    r_val = select(api, f3_5, &srlsra_r, &r_val);
    r_val = select(api, f3_6, &or_r, &r_val);
    r_val = select(api, f3_7, &and_r, &r_val);

    // --- I-type ALU (rs1 op imm) ---
    let addi_v = add32(api, &rs1_val, &i_imm);
    let slli_v = shl32(api, &rs1_val, &imm_sh);
    let slti_v = { let b = signed_lt(api, &rs1_val, &i_imm); let mut v = zeros(api, XLEN); v[0] = b; v };
    let sltiu_v = { let b = lt(api, &rs1_val, &i_imm); let mut v = zeros(api, XLEN); v[0] = b; v };
    let xori_v = xor32(api, &rs1_val, &i_imm);
    let srli_v = srl32(api, &rs1_val, &imm_sh);
    let srai_v = sra32(api, &rs1_val, &imm_sh);
    let srlisrai_v = select(api, insn30, &srai_v, &srli_v);
    let ori_v = or32(api, &rs1_val, &i_imm);
    let andi_v = and32(api, &rs1_val, &i_imm);
    let mut i_val = addi_v; // f3 == 0
    i_val = select(api, f3_1, &slli_v, &i_val);
    i_val = select(api, f3_2, &slti_v, &i_val);
    i_val = select(api, f3_3, &sltiu_v, &i_val);
    i_val = select(api, f3_4, &xori_v, &i_val);
    i_val = select(api, f3_5, &srlisrai_v, &i_val);
    i_val = select(api, f3_6, &ori_v, &i_val);
    i_val = select(api, f3_7, &andi_v, &i_val);

    // --- LOAD: effective address, matching-slot word, sub-word extraction ---
    let laddr = add32(api, &rs1_val, &i_imm);
    let lwidx: Vec<Variable> = (2..2 + cfg.waddr_bits).map(|i| laddr[i]).collect();
    let lbyte_idx: Vec<Variable> = laddr[0..2].to_vec();
    let lhalf_sel = laddr[1];
    let mut loaded = zeros(api, XLEN);
    for j in 0..cfg.mem_slots {
        let m = eq(api, &lwidx, &st.mem_addr[j]);
        for b in 0..XLEN {
            let t = api.mul(m, st.mem_val[j][b]);
            loaded[b] = api.add(loaded[b], t);
        }
    }
    let lbyte = extract_byte(api, &loaded, &lbyte_idx);
    let lb_v = extend(api, &lbyte, true);
    let lbu_v = extend(api, &lbyte, false);
    let lhalf = { // 16-bit half selected by addr bit 1
        let lo = loaded[0..16].to_vec();
        let hi = loaded[16..32].to_vec();
        select(api, lhalf_sel, &hi, &lo)
    };
    let lh_v = extend(api, &lhalf, true);
    let lhu_v = extend(api, &lhalf, false);
    let mut load_val = loaded.clone();
    load_val = select(api, f3_0, &lb_v, &load_val);
    load_val = select(api, f3_1, &lh_v, &load_val);
    load_val = select(api, f3_2, &loaded, &load_val);
    load_val = select(api, f3_4, &lbu_v, &load_val);
    load_val = select(api, f3_5, &lhu_v, &load_val);

    // --- writeback value + enable ---
    let four = const_u32(api, 4);
    let pc4 = add32(api, &st.pc, &four);
    let auipc_val = add32(api, &st.pc, &u_imm);
    let mut wb = zeros(api, XLEN);
    wb = select(api, is_op, &r_val, &wb);
    wb = select(api, is_opimm, &i_val, &wb);
    wb = select(api, is_load, &load_val, &wb);
    wb = select(api, is_jal, &pc4, &wb);
    wb = select(api, is_jalr, &pc4, &wb);
    wb = select(api, is_lui, &u_imm, &wb);
    wb = select(api, is_auipc, &auipc_val, &wb);
    let writes_rd = {
        let mut s = is_op;
        for x in [is_opimm, is_load, is_jal, is_jalr, is_lui, is_auipc] {
            s = api.add(s, x);
        }
        s
    };

    // --- branch taken? ---
    let beq = eq(api, &rs1_val, &rs2_val);
    let bne = api.sub(1, beq);
    let blt = signed_lt(api, &rs1_val, &rs2_val);
    let bge = api.sub(1, blt);
    let bltu = lt(api, &rs1_val, &rs2_val);
    let bgeu = api.sub(1, bltu);
    let mut taken = beq; // f3 == 0
    taken = bit_select(api, f3_1, bne, taken);
    taken = bit_select(api, f3_4, blt, taken);
    taken = bit_select(api, f3_5, bge, taken);
    taken = bit_select(api, f3_6, bltu, taken);
    taken = bit_select(api, f3_7, bgeu, taken);

    // --- next pc ---
    let branch_pc = { let t = add32(api, &st.pc, &b_imm); select(api, taken, &t, &pc4) };
    let jalr_target = { let mut t = add32(api, &rs1_val, &i_imm); t[0] = api.constant(0); t };
    let jal_target = add32(api, &st.pc, &j_imm);
    let mut npc = pc4.clone();
    npc = select(api, is_branch, &branch_pc, &npc);
    npc = select(api, is_jal, &jal_target, &npc);
    npc = select(api, is_jalr, &jalr_target, &npc);

    // --- register writeback (x0 hardwired to 0) ---
    let mut new_regs = st.regs.clone();
    for i in 0..cfg.nreg {
        let sel0 = eq_const(api, &rd, i as u32);
        let sel = api.mul(sel0, writes_rd);
        new_regs[i] = select(api, sel, &wb, &st.regs[i]);
    }
    new_regs[0] = zeros(api, XLEN);

    // --- STORE: RMW per slot, write to the matching slot only ---
    let saddr = add32(api, &rs1_val, &s_imm);
    let swidx: Vec<Variable> = (2..2 + cfg.waddr_bits).map(|i| saddr[i]).collect();
    let sbyte_idx: Vec<Variable> = saddr[0..2].to_vec();
    let shalf_sel = saddr[1];
    let rs2_byte: Vec<Variable> = rs2_val[0..8].to_vec();
    let rs2_half: Vec<Variable> = rs2_val[0..16].to_vec();
    let mut new_mem_val = st.mem_val.clone();
    for j in 0..cfg.mem_slots {
        let cur = st.mem_val[j].clone();
        // SB: replace byte at sbyte_idx
        let mut sb_word = cur.clone();
        for k in 0..4 {
            let sel = eq_const(api, &sbyte_idx, k as u32);
            for b in 0..8 {
                sb_word[k * 8 + b] = bit_select(api, sel, rs2_byte[b], sb_word[k * 8 + b]);
            }
        }
        // SH: replace half at shalf_sel
        let not_h = api.sub(1, shalf_sel);
        let mut sh_word = cur.clone();
        for b in 0..16 {
            sh_word[b] = bit_select(api, not_h, rs2_half[b], sh_word[b]);
            sh_word[16 + b] = bit_select(api, shalf_sel, rs2_half[b], sh_word[16 + b]);
        }
        // SW / SB / SH select by funct3
        let mut store_word = rs2_val.clone(); // f3 == 2 (SW)
        store_word = select(api, f3_0, &sb_word, &store_word);
        store_word = select(api, f3_1, &sh_word, &store_word);
        let hit = eq(api, &swidx, &st.mem_addr[j]);
        let do_store = api.mul(hit, is_store);
        new_mem_val[j] = select(api, do_store, &store_word, &cur);
    }

    Rv32State { regs: new_regs, pc: npc, mem_addr: st.mem_addr, mem_val: new_mem_val }
}

/// Run `n_steps` unrolled RV32I steps from an initial state over committed program.
pub fn run<C: Config>(api: &mut impl RootAPI<C>, mut st: Rv32State, program: &[W], cfg: &Rv32Cfg, n_steps: usize) -> Rv32State {
    for _ in 0..n_steps {
        st = step(api, st, program, cfg);
    }
    st
}
