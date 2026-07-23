//! Generalized RV32IM CPU-verifier for the REAL ev-reth STF ELF, proven over
//! Expander GKR with rsema1d as the committed-input PCS. This is `circuit_gp`
//! made whole for a real program image + segmented (continuation) proving:
//!
//!   * FETCH IS A MEMORY READ. `circuit_gp` enumerated the whole program ROM in
//!     circuit (O(cycles x prog_len)); with a ~1.2M-instruction ELF that is
//!     impossible. Here instruction fetch is just another access into the SAME
//!     offline-memory-checking grand product: slot 0 reads (pc, insn) from the
//!     unified memory whose initial image is seeded with the ELF text words (and
//!     the guest's hint/input bytes). The committed `insn` is therefore pinned to
//!     the image value at `pc` by the multiset argument — no ROM enumeration.
//!
//!   * NON-ZERO BASE / DYNAMIC MEMORY. Registers are initialised to the chunk's
//!     ENTRY register values (e.g. x2 = STACK_TOP), not 0. Memory addresses are
//!     the runtime-witnessed touched set (fetch words U load/store words) at the
//!     0x78xx.. base, with per-address initial values from the entry image.
//!
//!   * SEGMENTED / CONTINUATION. entry_pc + 32 entry registers and exit_pc + 32
//!     exit registers are PUBLIC IO. Chaining chunk i -> i+1 asserts, outside the
//!     circuit, exit_state(i) == entry_state(i+1). The threaded pc is pinned to
//!     entry_pc at the start and to exit_pc at the end; exit registers are pinned
//!     to the grand-product-validated final register values.
//!
//!   * ECALL. A SYSTEM (ecall) cycle recomputes no ALU result and is NOT rd-checked
//!     (the syscall return is a committed value produced by the host/coprocessor,
//!     the standard zkVM trust boundary), but its committed t0 write IS recorded in
//!     the register multiset so downstream reads stay consistent (HINT_LEN). Memory
//!     bytes delivered by HINT_READ are modelled as part of the committed initial
//!     memory image (the encode-once rsema1d/DA input), per the segmented spec.
//!
//! Everything else (decode, immediates, ALU incl. RV32M with div/rem hint checks,
//! sub-word load/store against the memory-checked old word, all six branches,
//! JAL/JALR/LUI/AUIPC, and the GF(2^128) grand product PROD(reads)==PROD(writes))
//! is the exact, per-opcode-verified logic reused from `circuit_gp` via its now
//! `pub` bit gadgets.

use crate::circuit_gp::{
    abs32, add32, add_bits, and32, borrow_lt, const_word, eq32, eq_const, factor, is_zero32,
    ltu32, mul_low_bits, mux1, mux32, mux_vec, not1, or32, product_tree, sext64, sext_to32,
    sll32, srl32, sra32, sub32, xor32, zext64, zext_to32, NREG, TS, XLEN,
};
use crate::emulator::StepRecord;
use crate::gf128;
use expander_compiler::frontend::*;

/// fetch, rs1-read, rs2-read, mem, rd-write.
pub const SLOTS: usize = 5;
pub const NOUT: usize = 2; // committed outputs: block_number(lo/hi packed) etc.

// ---- runtime config (set by the driver before compile) ---------------------
pub static mut NCYC: usize = 0;
/// Number of DRAIN memory addresses (the FULL touched-word union of the span).
/// Only enumerated when IS_LAST (final memory drain reads).
pub static mut NMEM: usize = 0;
pub static mut ENTRY_PC: u32 = 0;
/// Full touched-word union (drain list, last chunk).
pub static mut MEM_ADDRS: Vec<u32> = Vec::new();
/// Genesis (initial-image) addresses (span, excluding HINT_READ-delivered words),
/// with per-address entry-image value; enumerated only when IS_FIRST.
pub static mut GEN_ADDRS: Vec<u32> = Vec::new();
pub static mut GEN_INIT: Vec<u32> = Vec::new();
pub static mut MEM_INIT: Vec<u32> = Vec::new(); // legacy alias (== GEN_INIT for single chunk)
pub static mut REG_INIT: Vec<u32> = Vec::new(); // 32 entry register values
/// Which of the `naddr = NREG + NMEM` final-value entries feed the NOUT outputs.
pub static mut OUT_IDX: [usize; NOUT] = [0; NOUT];

// ---- segmentation / continuation config ------------------------------------
/// This chunk seeds the span's initial memory image (genesis writes).
pub static mut IS_FIRST: bool = true;
/// This chunk drains the final memory image and closes the threaded product.
pub static mut IS_LAST: bool = true;
/// Span-relative cycle index of this chunk's first cycle, for the GLOBAL memory
/// timestamps (registers stay chunk-local; only memory is threaded).
pub static mut BASE_CYCLE: usize = 0;
/// HINT_READ-delivered committed memory words handled in THIS chunk.
pub static mut NHINT: usize = 0;
pub static mut HINT_ADDRS: Vec<u32> = Vec::new();
/// Global memory timestamp assigned to the HINT_READ write batch.
pub static mut HINT_TS: u32 = 0;

// ---- block-data LINKAGE config (the accidental-computer DA reuse) -----------
/// When true, the circuit exposes `link_y = MLE_{B_trace}(R)`: the multilinear
/// extension of the committed HINT_READ region bytes (the block data B the guest
/// executed), evaluated over GF(2^128) at the externally supplied point R (public
/// input `link_r`), as a PUBLIC OUTPUT. This soundly binds the opened rsema1d DA
/// commitment C_data of B to the executed trace (see `prove_elf::prove_linkage`).
pub static mut LINK_ON: bool = false;
/// Number of GF(2^128) coordinates in the evaluation point R, i.e. m =
/// ceil(log2(NHINT * XLEN)) (the number of multilinear variables over the
/// padded 2^m-element bit-vector of the HINT_READ region).
pub static mut LINK_M: usize = 0;

fn link_on() -> bool { unsafe { LINK_ON } }
fn link_m() -> usize { unsafe { LINK_M } }

fn ncyc() -> usize { unsafe { NCYC } }
fn nmem() -> usize { unsafe { NMEM } }
fn entry_pc() -> u32 { unsafe { ENTRY_PC } }
fn mem_addrs() -> &'static [u32] { unsafe { MEM_ADDRS.as_slice() } }
fn gen_addrs() -> &'static [u32] { unsafe { GEN_ADDRS.as_slice() } }
fn gen_init() -> &'static [u32] { unsafe { GEN_INIT.as_slice() } }
fn reg_init() -> &'static [u32] { unsafe { REG_INIT.as_slice() } }
fn is_first() -> bool { unsafe { IS_FIRST } }
fn is_last() -> bool { unsafe { IS_LAST } }
fn base_cycle() -> usize { unsafe { BASE_CYCLE } }
fn nhint() -> usize { unsafe { NHINT } }
fn hint_addrs() -> &'static [u32] { unsafe { HINT_ADDRS.as_slice() } }
fn hint_ts() -> u32 { unsafe { HINT_TS } }

declare_circuit!(RiscvCircuitElf {
    insn: [[Variable; XLEN]],    // NCYC (committed fetched instruction word)
    rs1_val: [[Variable; XLEN]], // NCYC
    rs2_val: [[Variable; XLEN]], // NCYC
    rd_val: [[Variable; XLEN]],  // NCYC
    tprev: [[Variable; TS]],     // NCYC*SLOTS (slot-major)
    vold_c: [[Variable; XLEN]],  // NCYC old memory WORD at addr&!3
    vold_d: [[Variable; XLEN]],  // NCYC old register value for rd write
    vold_f: [[Variable; XLEN]],  // NCYC old value at fetch word (== insn image)
    div_q: [[Variable; XLEN]],   // NCYC
    div_r: [[Variable; XLEN]],   // NCYC
    is_sys: [Variable],          // NCYC : 1 iff this cycle is a SYSTEM/ecall cycle
    sys_wr: [Variable],          // NCYC : 1 iff this ecall writes t0 (x5) a committed value
    fin_val: [[Variable; XLEN]], // naddr final value per address
    fin_ts: [[Variable; TS]],    // naddr final last-write ts per address
    hint_word: [[Variable; XLEN]], // NHINT committed HINT_READ-delivered words
    alpha: [PublicVariable; 128],
    beta: [PublicVariable; 128],
    entry_pc: [PublicVariable; XLEN],
    exit_pc: [PublicVariable; XLEN],
    entry_reg: [[PublicVariable; XLEN]], // NREG
    exit_reg: [[PublicVariable; XLEN]],  // NREG
    out: [[PublicVariable; XLEN]],       // NOUT committed outputs
    // ---- threaded MEMORY grand-product accumulators (public IO) ----
    mem_pr_in: [PublicVariable; 128],  // running read-fp product entering chunk
    mem_pw_in: [PublicVariable; 128],  // running write-fp product entering chunk
    mem_pr_out: [PublicVariable; 128], // running read-fp product leaving chunk
    mem_pw_out: [PublicVariable; 128], // running write-fp product leaving chunk
    // ---- block-data LINKAGE public IO (empty unless LINK_ON) ----
    link_r: [[PublicVariable; 128]],   // LINK_M evaluation-point coords R (GF(2^128))
    link_y: [[PublicVariable; 128]],   // len 1 when on: y_tr = MLE_{B_trace}(R)
});

// -------------------------------- hints -------------------------------------

pub struct Hints {
    pub tprev: Vec<[u32; SLOTS]>,
    pub vold_c: Vec<u32>,
    pub vold_d: Vec<u32>,
    pub vold_f: Vec<u32>,
    pub div_q: Vec<u32>,
    pub div_r: Vec<u32>,
    pub is_sys: Vec<u32>,
    pub sys_wr: Vec<u32>,
    pub fin_val: Vec<u32>,
    pub fin_ts: Vec<u32>,
    pub entry_reg: Vec<u32>, // 32
    pub exit_reg: Vec<u32>,  // 32
    pub exit_pc: u32,
    pub out_vals: Vec<u32>,  // NOUT
    pub hint_vals: Vec<u32>, // NHINT committed HINT_READ words
    // threaded MEMORY products (u128 GF(2^128) elements), public IO.
    pub mem_pr_in: u128,
    pub mem_pw_in: u128,
    pub mem_pr_out: u128,
    pub mem_pw_out: u128,
    // block-data linkage: R coords (len LINK_M) + y_tr (== MLE of the HINT_READ
    // region at R). Ignored unless LINK_ON.
    pub link_r: Vec<u128>,
    pub link_y: u128,
}

fn set_word<const N: usize>(dst: &mut [GF2], word: u32) {
    for i in 0..N {
        dst[i] = ((word >> i) & 1).into();
    }
}

pub fn template() -> RiscvCircuitElf<Variable> {
    let n = ncyc();
    let naddr = NREG + nmem();
    RiscvCircuitElf {
        insn: vec![[Variable::default(); XLEN]; n],
        rs1_val: vec![[Variable::default(); XLEN]; n],
        rs2_val: vec![[Variable::default(); XLEN]; n],
        rd_val: vec![[Variable::default(); XLEN]; n],
        tprev: vec![[Variable::default(); TS]; n * SLOTS],
        vold_c: vec![[Variable::default(); XLEN]; n],
        vold_d: vec![[Variable::default(); XLEN]; n],
        vold_f: vec![[Variable::default(); XLEN]; n],
        div_q: vec![[Variable::default(); XLEN]; n],
        div_r: vec![[Variable::default(); XLEN]; n],
        is_sys: vec![Variable::default(); n],
        sys_wr: vec![Variable::default(); n],
        fin_val: vec![[Variable::default(); XLEN]; naddr],
        fin_ts: vec![[Variable::default(); TS]; naddr],
        hint_word: vec![[Variable::default(); XLEN]; nhint()],
        alpha: [Variable::default(); 128],
        beta: [Variable::default(); 128],
        entry_pc: [Variable::default(); XLEN],
        exit_pc: [Variable::default(); XLEN],
        entry_reg: vec![[Variable::default(); XLEN]; NREG],
        exit_reg: vec![[Variable::default(); XLEN]; NREG],
        out: vec![[Variable::default(); XLEN]; NOUT],
        mem_pr_in: [Variable::default(); 128],
        mem_pw_in: [Variable::default(); 128],
        mem_pr_out: [Variable::default(); 128],
        mem_pw_out: [Variable::default(); 128],
        link_r: vec![[Variable::default(); 128]; if link_on() { link_m() } else { 0 }],
        link_y: vec![[Variable::default(); 128]; if link_on() { 1 } else { 0 }],
    }
}

pub fn build_assignment(
    trace: &[StepRecord],
    hints: &Hints,
    alpha: u128,
    beta: u128,
) -> RiscvCircuitElf<GF2> {
    let n = ncyc();
    let naddr = NREG + nmem();
    let mut a = RiscvCircuitElf::<GF2> {
        insn: vec![[GF2::default(); XLEN]; n],
        rs1_val: vec![[GF2::default(); XLEN]; n],
        rs2_val: vec![[GF2::default(); XLEN]; n],
        rd_val: vec![[GF2::default(); XLEN]; n],
        tprev: vec![[GF2::default(); TS]; n * SLOTS],
        vold_c: vec![[GF2::default(); XLEN]; n],
        vold_d: vec![[GF2::default(); XLEN]; n],
        vold_f: vec![[GF2::default(); XLEN]; n],
        div_q: vec![[GF2::default(); XLEN]; n],
        div_r: vec![[GF2::default(); XLEN]; n],
        is_sys: vec![GF2::default(); n],
        sys_wr: vec![GF2::default(); n],
        fin_val: vec![[GF2::default(); XLEN]; naddr],
        fin_ts: vec![[GF2::default(); TS]; naddr],
        hint_word: vec![[GF2::default(); XLEN]; nhint()],
        alpha: [GF2::default(); 128],
        beta: [GF2::default(); 128],
        entry_pc: [GF2::default(); XLEN],
        exit_pc: [GF2::default(); XLEN],
        entry_reg: vec![[GF2::default(); XLEN]; NREG],
        exit_reg: vec![[GF2::default(); XLEN]; NREG],
        out: vec![[GF2::default(); XLEN]; NOUT],
        mem_pr_in: [GF2::default(); 128],
        mem_pw_in: [GF2::default(); 128],
        mem_pr_out: [GF2::default(); 128],
        mem_pw_out: [GF2::default(); 128],
        link_r: vec![[GF2::default(); 128]; if link_on() { link_m() } else { 0 }],
        link_y: vec![[GF2::default(); 128]; if link_on() { 1 } else { 0 }],
    };
    for c in 0..n {
        set_word::<XLEN>(&mut a.insn[c], trace[c].insn);
        set_word::<XLEN>(&mut a.rs1_val[c], trace[c].rs1_val);
        set_word::<XLEN>(&mut a.rs2_val[c], trace[c].rs2_val);
        set_word::<XLEN>(&mut a.rd_val[c], trace[c].rd_val);
        set_word::<XLEN>(&mut a.vold_c[c], hints.vold_c[c]);
        set_word::<XLEN>(&mut a.vold_d[c], hints.vold_d[c]);
        set_word::<XLEN>(&mut a.vold_f[c], hints.vold_f[c]);
        set_word::<XLEN>(&mut a.div_q[c], hints.div_q[c]);
        set_word::<XLEN>(&mut a.div_r[c], hints.div_r[c]);
        a.is_sys[c] = hints.is_sys[c].into();
        a.sys_wr[c] = hints.sys_wr[c].into();
        for s in 0..SLOTS {
            set_word::<TS>(&mut a.tprev[c * SLOTS + s], hints.tprev[c][s]);
        }
    }
    for i in 0..naddr {
        set_word::<XLEN>(&mut a.fin_val[i], hints.fin_val[i]);
        set_word::<TS>(&mut a.fin_ts[i], hints.fin_ts[i]);
    }
    for i in 0..128 {
        a.alpha[i] = (((alpha >> i) & 1) as u32).into();
        a.beta[i] = (((beta >> i) & 1) as u32).into();
    }
    set_word::<XLEN>(&mut a.entry_pc, entry_pc());
    set_word::<XLEN>(&mut a.exit_pc, hints.exit_pc);
    for r in 0..NREG {
        set_word::<XLEN>(&mut a.entry_reg[r], hints.entry_reg[r]);
        set_word::<XLEN>(&mut a.exit_reg[r], hints.exit_reg[r]);
    }
    for k in 0..NOUT {
        set_word::<XLEN>(&mut a.out[k], hints.out_vals[k]);
    }
    for k in 0..nhint() {
        set_word::<XLEN>(&mut a.hint_word[k], hints.hint_vals[k]);
    }
    let set128 = |dst: &mut [GF2], x: u128| {
        for i in 0..128 { dst[i] = (((x >> i) & 1) as u32).into(); }
    };
    set128(&mut a.mem_pr_in, hints.mem_pr_in);
    set128(&mut a.mem_pw_in, hints.mem_pw_in);
    set128(&mut a.mem_pr_out, hints.mem_pr_out);
    set128(&mut a.mem_pw_out, hints.mem_pw_out);
    if link_on() {
        for p in 0..link_m() {
            set128(&mut a.link_r[p], hints.link_r[p]);
        }
        set128(&mut a.link_y[0], hints.link_y);
    }
    a
}

/// Tamper helper: corrupt a committed loaded value (a memory read result).
pub fn tamper_rd_val(a: &mut RiscvCircuitElf<GF2>, cycle: usize, word: u32) {
    set_word::<XLEN>(&mut a.rd_val[cycle], word);
}

/// Tamper helper: flip one bit of a committed fetched instruction. Breaks the
/// fetch-consistency check (insn == memory-read value) and changes the commitment.
pub fn tamper_insn_bit(a: &mut RiscvCircuitElf<GF2>, cycle: usize, bit: usize) {
    let x = a.insn[cycle][bit];
    a.insn[cycle][bit] = if x.is_zero() { 1u32.into() } else { 0u32.into() };
}

// ---- native mirror of `factor` + product (for computing the threaded public
// products host-side). MUST match the in-circuit convention exactly:
//   factor = alpha ^ addr ^ (val . beta) ^ (ts . beta^2)   in GF(2^128).
pub fn native_factor(addr: u32, val: u32, ts: u32, alpha: u128, beta: u128, beta2: u128) -> u128 {
    alpha ^ (addr as u128) ^ gf128::native_mul(val as u128, beta) ^ gf128::native_mul(ts as u128, beta2)
}
/// Native GF(2^128) product of a set of factors.
pub fn native_product(factors: &[u128]) -> u128 {
    let mut acc: u128 = 1;
    for &f in factors {
        acc = gf128::native_mul(acc, f);
    }
    acc
}

// -------------------------------- the circuit -------------------------------

impl Define<GF2Config> for RiscvCircuitElf<Variable> {
    fn define<Builder: RootAPI<GF2Config>>(&self, api: &mut Builder) {
        let zero = api.constant(0);
        let n = ncyc();
        let nm = nmem();
        let naddr = NREG + nm;

        let alpha: Vec<Variable> = self.alpha.to_vec();
        let beta: Vec<Variable> = self.beta.to_vec();
        let beta2 = gf128::mul(api, &beta, &beta);

        // Registers are checked with a SELF-CONTAINED per-chunk multiset (genesis
        // entry_reg writes + per-cycle reads/writes + final drain, closed here);
        // MEMORY uses a THREADED grand-product carried across chunks as public IO.
        let mut reg_write_fps: Vec<Vec<Variable>> = Vec::new();
        let mut reg_read_fps: Vec<Vec<Variable>> = Vec::new();
        let mut mem_write_fps: Vec<Vec<Variable>> = Vec::new();
        let mut mem_read_fps: Vec<Vec<Variable>> = Vec::new();

        // ---- register genesis writes at ts=0 (entry register values, public).
        for r in 0..NREG {
            let addr = const_word(api, r as u32);
            let val = self.entry_reg[r].to_vec();
            let ts = (0..TS).map(|_| zero).collect::<Vec<_>>();
            reg_write_fps.push(factor(api, &addr, &val, &ts, &alpha, &beta, &beta2));
        }
        // ---- memory genesis writes at ts=0 (span initial image: ELF text words +
        // pre-existing data). DISTRIBUTED: each chunk seeds ONLY the addresses it
        // touches for the FIRST time in the span, so no single chunk enumerates the
        // whole ~5k-word image. HINT_READ-delivered words are NOT seeded here (they
        // are created by the committed hint-write batch below).
        for k in 0..gen_addrs().len() {
            let addr = const_word(api, gen_addrs()[k]);
            let val = const_word(api, gen_init()[k]);
            let ts = (0..TS).map(|_| zero).collect::<Vec<_>>();
            mem_write_fps.push(factor(api, &addr, &val, &ts, &alpha, &beta, &beta2));
        }

        // ---- threaded pc, pinned to entry_pc at the start.
        let mut pc = self.entry_pc.to_vec();

        for c in 0..n {
            let insn = self.insn[c].to_vec();
            let rs1v = self.rs1_val[c].to_vec();
            let rs2v = self.rs2_val[c].to_vec();
            let rdv = self.rd_val[c].to_vec();
            let is_sys = self.is_sys[c];
            let sys_wr = self.sys_wr[c];
            // is_sys, sys_wr are boolean; sys_wr implies is_sys.
            {
                let t = api.mul(is_sys, is_sys);
                api.assert_is_equal(t, is_sys);
                let t = api.mul(sys_wr, sys_wr);
                api.assert_is_equal(t, sys_wr);
                // sys_wr => is_sys : sys_wr * (1 - is_sys) == 0.
                let nis = not1(api, is_sys);
                let g = api.mul(sys_wr, nis);
                api.assert_is_equal(g, 0);
            }

            // (1) Decode (fetch is validated by the grand product, slot 0 below).
            let opcode = &insn[0..7];
            let rd_idx = &insn[7..12];
            let funct3 = &insn[12..15];
            let rs1_idx = &insn[15..20];
            let rs2_idx = &insn[20..25];
            let funct7 = &insn[25..32];

            let imm_i = {
                let mut v = vec![zero; XLEN];
                for i in 0..12 { v[i] = insn[20 + i]; }
                for i in 12..XLEN { v[i] = insn[31]; }
                v
            };
            let imm_s = {
                let mut v = vec![zero; XLEN];
                for i in 0..5 { v[i] = insn[7 + i]; }
                for i in 5..12 { v[i] = insn[20 + i]; }
                for i in 12..XLEN { v[i] = insn[31]; }
                v
            };
            let imm_b = {
                let mut v = vec![zero; XLEN];
                v[0] = zero;
                for i in 1..5 { v[i] = insn[7 + i]; }
                for i in 5..11 { v[i] = insn[20 + i]; }
                v[11] = insn[7];
                for i in 12..XLEN { v[i] = insn[31]; }
                v
            };
            let imm_j = {
                let mut v = vec![zero; XLEN];
                v[0] = zero;
                for i in 1..11 { v[i] = insn[20 + i]; }
                v[11] = insn[20];
                for i in 12..20 { v[i] = insn[i]; }
                for i in 20..XLEN { v[i] = insn[31]; }
                v
            };
            let imm_u = {
                let mut v = vec![zero; XLEN];
                for i in 12..XLEN { v[i] = insn[i]; }
                v
            };

            // (2) class + sub-op selectors.
            let is_op = eq_const(api, opcode, 0b0110011);
            let is_opimm = eq_const(api, opcode, 0b0010011);
            let is_load = eq_const(api, opcode, 0b0000011);
            let is_store = eq_const(api, opcode, 0b0100011);
            let is_branch = eq_const(api, opcode, 0b1100011);
            let is_jal = eq_const(api, opcode, 0b1101111);
            let is_jalr = eq_const(api, opcode, 0b1100111);
            let is_lui = eq_const(api, opcode, 0b0110111);
            let is_auipc = eq_const(api, opcode, 0b0010111);
            let is_alu = api.add(is_op, is_opimm);
            let f3_0 = eq_const(api, funct3, 0x0);
            let f3_1 = eq_const(api, funct3, 0x1);
            let f3_2 = eq_const(api, funct3, 0x2);
            let f3_3 = eq_const(api, funct3, 0x3);
            let f3_4 = eq_const(api, funct3, 0x4);
            let f3_5 = eq_const(api, funct3, 0x5);
            let f3_6 = eq_const(api, funct3, 0x6);
            let f3_7 = eq_const(api, funct3, 0x7);
            let f7_zero = eq_const(api, funct7, 0x00);
            let f7_sub = eq_const(api, funct7, 0x20);
            let f7_m = eq_const(api, funct7, 0x01);

            let operand_b = mux32(api, is_opimm, &imm_i, &rs2v);
            let shamt: Vec<Variable> = operand_b[0..5].to_vec();

            // (3) ALU results.
            let add_res = add32(api, &rs1v, &operand_b);
            let sub_res = sub32(api, &rs1v, &rs2v);
            let xor_res = xor32(api, &rs1v, &operand_b);
            let or_res = or32(api, &rs1v, &operand_b);
            let and_res = and32(api, &rs1v, &operand_b);
            let sll_res = sll32(api, &rs1v, &shamt);
            let srl_res = srl32(api, &rs1v, &shamt);
            let sra_res = sra32(api, &rs1v, &shamt);

            let ltu_ob = ltu32(api, &rs1v, &operand_b);
            let lt_s = {
                let t = api.add(ltu_ob, rs1v[XLEN - 1]);
                api.add(t, operand_b[XLEN - 1])
            };
            let slt_res = { let mut v = vec![zero; XLEN]; v[0] = lt_s; v };
            let sltu_res = { let mut v = vec![zero; XLEN]; v[0] = ltu_ob; v };

            let a_u = zext64(api, &rs1v);
            let b_u = zext64(api, &rs2v);
            let a_s = sext64(api, &rs1v);
            let b_s = sext64(api, &rs2v);
            let prod_uu = mul_low_bits(api, &a_u, &b_u, 64);
            let prod_ss = mul_low_bits(api, &a_s, &b_s, 64);
            let prod_su = mul_low_bits(api, &a_s, &b_u, 64);
            let mul_res = prod_uu[0..XLEN].to_vec();
            let mulhu_res = prod_uu[XLEN..2 * XLEN].to_vec();
            let mulh_res = prod_ss[XLEN..2 * XLEN].to_vec();
            let mulhsu_res = prod_su[XLEN..2 * XLEN].to_vec();

            let div_q = self.div_q[c].to_vec();
            let div_r = self.div_r[c].to_vec();
            let b_zero = is_zero32(api, &rs2v);
            let a_is_min = eq_const(api, &rs1v, 0x8000_0000);
            let b_is_neg1 = eq_const(api, &rs2v, 0xFFFF_FFFF);
            let ovf = api.mul(a_is_min, b_is_neg1);
            let allones = const_word(api, 0xFFFF_FFFF);
            let intmin = const_word(api, 0x8000_0000);
            let zeros = const_word(api, 0);
            let divu_res = mux_vec(api, b_zero, &allones, &div_q);
            let remu_res = mux_vec(api, b_zero, &rs1v, &div_r);
            let div_norm = mux_vec(api, ovf, &intmin, &div_q);
            let div_res = mux_vec(api, b_zero, &allones, &div_norm);
            let rem_norm = mux_vec(api, ovf, &zeros, &div_r);
            let rem_res = mux_vec(api, b_zero, &rs1v, &rem_norm);

            let a2 = |api: &mut Builder, x, y| api.mul(x, y);
            let a3 = |api: &mut Builder, x, y, z| { let t = api.mul(x, y); api.mul(t, z) };
            let sel_add = { let t1 = a3(api, is_op, f3_0, f7_zero); let t2 = a2(api, is_opimm, f3_0); api.add(t1, t2) };
            let sel_sub = a3(api, is_op, f3_0, f7_sub);
            let sel_sll = { let t1 = a3(api, is_op, f3_1, f7_zero); let t2 = a2(api, is_opimm, f3_1); api.add(t1, t2) };
            let sel_slt = { let t1 = a3(api, is_op, f3_2, f7_zero); let t2 = a2(api, is_opimm, f3_2); api.add(t1, t2) };
            let sel_sltu = { let t1 = a3(api, is_op, f3_3, f7_zero); let t2 = a2(api, is_opimm, f3_3); api.add(t1, t2) };
            let sel_xor = { let t1 = a3(api, is_op, f3_4, f7_zero); let t2 = a2(api, is_opimm, f3_4); api.add(t1, t2) };
            let sel_srl = { let t1 = a3(api, is_op, f3_5, f7_zero); let t2 = a3(api, is_opimm, f3_5, f7_zero); api.add(t1, t2) };
            let sel_sra = { let t1 = a3(api, is_op, f3_5, f7_sub); let t2 = a3(api, is_opimm, f3_5, f7_sub); api.add(t1, t2) };
            let sel_or = { let t1 = a3(api, is_op, f3_6, f7_zero); let t2 = a2(api, is_opimm, f3_6); api.add(t1, t2) };
            let sel_and = { let t1 = a3(api, is_op, f3_7, f7_zero); let t2 = a2(api, is_opimm, f3_7); api.add(t1, t2) };
            let sel_mul = a3(api, is_op, f3_0, f7_m);
            let sel_mulh = a3(api, is_op, f3_1, f7_m);
            let sel_mulhsu = a3(api, is_op, f3_2, f7_m);
            let sel_mulhu = a3(api, is_op, f3_3, f7_m);
            let sel_div = a3(api, is_op, f3_4, f7_m);
            let sel_divu = a3(api, is_op, f3_5, f7_m);
            let sel_rem = a3(api, is_op, f3_6, f7_m);
            let sel_remu = a3(api, is_op, f3_7, f7_m);

            let sels = [
                sel_add, sel_sub, sel_xor, sel_or, sel_and, sel_sll, sel_srl, sel_sra,
                sel_slt, sel_sltu, sel_mul, sel_mulh, sel_mulhsu, sel_mulhu,
                sel_div, sel_divu, sel_rem, sel_remu,
            ];
            let ress: [&Vec<Variable>; 18] = [
                &add_res, &sub_res, &xor_res, &or_res, &and_res, &sll_res, &srl_res, &sra_res,
                &slt_res, &sltu_res, &mul_res, &mulh_res, &mulhsu_res, &mulhu_res,
                &div_res, &divu_res, &rem_res, &remu_res,
            ];
            let mut alu_rd = vec![zero; XLEN];
            for (sel, res) in sels.iter().zip(ress.iter()) {
                for b in 0..XLEN {
                    let t = api.mul(*sel, res[b]);
                    alu_rd[b] = api.add(alu_rd[b], t);
                }
            }

            // Division hint correctness (unchanged from circuit_gp).
            let one = api.constant(1);
            let not_bzero = not1(api, b_zero);
            let not_ovf = not1(api, ovf);
            let act_u = { let g = api.add(sel_divu, sel_remu); api.mul(g, not_bzero) };
            {
                let dq = zext64(api, &div_q);
                let bq = mul_low_bits(api, &b_u, &dq, 64);
                let dr = zext64(api, &div_r);
                let sum = add_bits(api, &bq, &dr);
                let target = zext64(api, &rs1v);
                for b in 0..64 {
                    let d = api.add(sum[b], target[b]);
                    let g = api.mul(act_u, d);
                    api.assert_is_equal(g, 0);
                }
                let lt = ltu32(api, &div_r, &rs2v);
                let nlt = api.sub(one, lt);
                let g = api.mul(act_u, nlt);
                api.assert_is_equal(g, 0);
            }
            let act_s = {
                let g = api.add(sel_div, sel_rem);
                let g = api.mul(g, not_bzero);
                api.mul(g, not_ovf)
            };
            {
                let dq = sext64(api, &div_q);
                let bq = mul_low_bits(api, &b_s, &dq, 64);
                let dr = sext64(api, &div_r);
                let sum = add_bits(api, &bq, &dr);
                let target = sext64(api, &rs1v);
                for b in 0..64 {
                    let d = api.add(sum[b], target[b]);
                    let g = api.mul(act_s, d);
                    api.assert_is_equal(g, 0);
                }
                let ar = abs32(api, &div_r);
                let ab = abs32(api, &rs2v);
                let lt = ltu32(api, &ar, &ab);
                let nlt = api.sub(one, lt);
                let g = api.mul(act_s, nlt);
                api.assert_is_equal(g, 0);
                let r_zero = is_zero32(api, &div_r);
                let sign_diff = api.add(div_r[XLEN - 1], rs1v[XLEN - 1]);
                let same_sign = not1(api, sign_diff);
                let sign_ok = { let x = api.add(same_sign, r_zero); let y = api.mul(same_sign, r_zero); api.add(x, y) };
                let nok = not1(api, sign_ok);
                let g = api.mul(act_s, nok);
                api.assert_is_equal(g, 0);
            }

            // (4) link / U-type derived values.
            let four = const_word(api, 4);
            let pc_plus4 = add32(api, &pc, &four);
            let auipc_res = add32(api, &pc, &imm_u);

            // (5) memory address + sub-word extract/insert against old word.
            let load_addr = add32(api, &rs1v, &imm_i);
            let store_addr = add32(api, &rs1v, &imm_s);
            let mem_addr = mux32(api, is_store, &store_addr, &load_addr);
            let off0 = mem_addr[0];
            let off1 = mem_addr[1];
            let word_addr = { let mut v = mem_addr.clone(); v[0] = zero; v[1] = zero; v };
            let old_word = self.vold_c[c].to_vec();

            let b0 = old_word[0..8].to_vec();
            let b1 = old_word[8..16].to_vec();
            let b2 = old_word[16..24].to_vec();
            let b3 = old_word[24..32].to_vec();
            let lo_b = mux_vec(api, off0, &b1, &b0);
            let hi_b = mux_vec(api, off0, &b3, &b2);
            let sel_byte = mux_vec(api, off1, &hi_b, &lo_b);
            let h0 = old_word[0..16].to_vec();
            let h1 = old_word[16..32].to_vec();
            let sel_half = mux_vec(api, off1, &h1, &h0);
            let lb_res = sext_to32(api, &sel_byte);
            let lbu_res = zext_to32(api, &sel_byte);
            let lh_res = sext_to32(api, &sel_half);
            let lhu_res = zext_to32(api, &sel_half);
            let lw_res = old_word.clone();
            let load_res = {
                let mut v = vec![zero; XLEN];
                let parts: [(Variable, &Vec<Variable>); 5] =
                    [(f3_0, &lb_res), (f3_1, &lh_res), (f3_2, &lw_res), (f3_4, &lbu_res), (f3_5, &lhu_res)];
                for (sel, r) in parts.iter() {
                    for b in 0..XLEN { let t = api.mul(*sel, r[b]); v[b] = api.add(v[b], t); }
                }
                v
            };

            let noff0 = not1(api, off0);
            let noff1 = not1(api, off1);
            let lane0 = api.mul(noff1, noff0);
            let lane1 = api.mul(noff1, off0);
            let lane2 = api.mul(off1, noff0);
            let lane3 = api.mul(off1, off0);
            let lanes = [lane0, lane1, lane2, lane3];
            let sb_word = {
                let mut v = old_word.clone();
                for j in 0..4 {
                    for k in 0..8 {
                        v[j * 8 + k] = mux1(api, lanes[j], rs2v[k], old_word[j * 8 + k]);
                    }
                }
                v
            };
            let sh_word = {
                let mut v = old_word.clone();
                for k in 0..16 {
                    v[k] = mux1(api, off1, old_word[k], rs2v[k]);
                    v[16 + k] = mux1(api, off1, rs2v[k], old_word[16 + k]);
                }
                v
            };
            let new_word = {
                let mut v = vec![zero; XLEN];
                let parts: [(Variable, &Vec<Variable>); 3] =
                    [(f3_0, &sb_word), (f3_1, &sh_word), (f3_2, &rs2v)];
                for (sel, r) in parts.iter() {
                    for b in 0..XLEN { let t = api.mul(*sel, r[b]); v[b] = api.add(v[b], t); }
                }
                v
            };

            // (6) computed rd for register-writing classes vs committed rd_val.
            let link = api.add(is_jal, is_jalr);
            let mut computed_rd = alu_rd.clone();
            for b in 0..XLEN {
                let t_link = api.mul(link, pc_plus4[b]);
                let t_lui = api.mul(is_lui, imm_u[b]);
                let t_auipc = api.mul(is_auipc, auipc_res[b]);
                let t_load = api.mul(is_load, load_res[b]);
                computed_rd[b] = api.add(computed_rd[b], t_link);
                computed_rd[b] = api.add(computed_rd[b], t_lui);
                computed_rd[b] = api.add(computed_rd[b], t_auipc);
                computed_rd[b] = api.add(computed_rd[b], t_load);
            }
            // rd is checked for ALU/JAL/JALR/LUI/AUIPC/LOAD; SYSTEM is NOT rd-checked
            // (syscall return is a committed value).
            let check_rd = {
                let t = api.add(is_alu, is_jal);
                let t = api.add(t, is_jalr);
                let t = api.add(t, is_lui);
                let t = api.add(t, is_auipc);
                api.add(t, is_load)
            };
            for b in 0..XLEN {
                let diff = api.add(computed_rd[b], rdv[b]);
                let g = api.mul(check_rd, diff);
                api.assert_is_equal(g, 0);
            }

            // (7) slots. A normal register-writing class writes rd_idx (from insn);
            // a SYSTEM/ecall cycle instead writes x5 (t0) a committed value, gated by
            // the committed sys_wr flag (the ecall insn's rd bits are 0, so the normal
            // path would miss it). The two are mutually exclusive (is_sys splits them).
            let write_enable = {
                let t = api.add(is_alu, is_load);
                let t = api.add(t, is_jal);
                let t = api.add(t, is_jalr);
                let t = api.add(t, is_lui);
                api.add(t, is_auipc)
            };
            let rd_is_zero = eq_const(api, rd_idx, 0);
            let rd_nonzero = not1(api, rd_is_zero);
            let active_d_normal = api.mul(write_enable, rd_nonzero);
            // final rd-slot activation + target: SYSTEM -> (sys_wr, x5); else normal.
            let active_d = mux1(api, is_sys, sys_wr, active_d_normal);
            let active_c = api.add(is_load, is_store);

            let rs1_addr = { let mut v = vec![zero; XLEN]; for i in 0..5 { v[i] = rs1_idx[i]; } v };
            let rs2_addr = { let mut v = vec![zero; XLEN]; for i in 0..5 { v[i] = rs2_idx[i]; } v };
            let rd_addr_normal = { let mut v = vec![zero; XLEN]; for i in 0..5 { v[i] = rd_idx[i]; } v };
            let x5_addr = const_word(api, 5); // t0
            let rd_addr = mux32(api, is_sys, &x5_addr, &rd_addr_normal);

            let vold_d = self.vold_d[c].to_vec();
            let vold_f = self.vold_f[c].to_vec();
            let null_val = vec![zero; XLEN];
            let one_v = api.constant(1);

            for slot in 0..SLOTS {
                // registers are chunk-LOCAL (self-contained multiset); memory is
                // GLOBAL (threaded across chunks) so its timestamps are monotone
                // over the whole span. slots 0 (fetch) & 3 (data) are memory;
                // slots 1,2 (rs reads) & 4 (rd write) are registers.
                let is_mem_slot = slot == 0 || slot == 3;
                let now_ts_c: u32 = if is_mem_slot {
                    ((base_cycle() + c) * SLOTS + slot + 1) as u32
                } else {
                    (c * SLOTS + slot + 1) as u32
                };
                let tprev = self.tprev[c * SLOTS + slot].to_vec();

                let (addr, read_val, write_val, active): (Vec<Variable>, Vec<Variable>, Vec<Variable>, Variable) =
                    match slot {
                        // slot 0: instruction fetch. Read the committed insn from the
                        // unified memory at the (word-aligned) pc; write it back
                        // unchanged (text is read-only, but the model rewrites the
                        // same value with a fresh ts). vold_f is the old value at the
                        // fetch word and must equal the committed insn (pinned below).
                        0 => {
                            let pc_word = { let mut v = pc.clone(); v[0] = zero; v[1] = zero; v };
                            (pc_word, vold_f.clone(), vold_f.clone(), one_v)
                        }
                        1 => (rs1_addr.clone(), rs1v.clone(), rs1v.clone(), one_v),
                        2 => (rs2_addr.clone(), rs2v.clone(), rs2v.clone(), one_v),
                        3 => {
                            let rval = old_word.clone();
                            let wval = mux32(api, is_store, &new_word, &old_word);
                            (word_addr.clone(), rval, wval, active_c)
                        }
                        _ => (rd_addr.clone(), vold_d.clone(), rdv.clone(), active_d),
                    };

                let now_ts_bits: Vec<Variable> = (0..TS).map(|i| api.constant((now_ts_c >> i) & 1)).collect();
                let borrow = borrow_lt(api, now_ts_c.wrapping_sub(1), &tprev);
                let g = api.mul(active, borrow);
                api.assert_is_equal(g, 0);

                let r_addr = mux32(api, active, &addr, &null_val);
                let r_val = mux32(api, active, &read_val, &null_val);
                let r_ts: Vec<Variable> = (0..TS).map(|i| mux1(api, active, tprev[i], now_ts_bits[i])).collect();
                let w_addr = mux32(api, active, &addr, &null_val);
                let w_val = mux32(api, active, &write_val, &null_val);
                let w_ts = now_ts_bits.clone();

                let rf = factor(api, &r_addr, &r_val, &r_ts, &alpha, &beta, &beta2);
                let wf = factor(api, &w_addr, &w_val, &w_ts, &alpha, &beta, &beta2);
                if is_mem_slot {
                    mem_read_fps.push(rf);
                    mem_write_fps.push(wf);
                } else {
                    reg_read_fps.push(rf);
                    reg_write_fps.push(wf);
                }
            }

            // Fetch consistency: committed insn == value read at pc (vold_f).
            for b in 0..XLEN {
                api.assert_is_equal(insn[b], vold_f[b]);
            }

            // (8) next pc + control flow.
            let eq = eq32(api, &rs1v, &rs2v);
            let neq = not1(api, eq);
            let ltu_rr = ltu32(api, &rs1v, &rs2v);
            let lts_rr = { let t = api.add(ltu_rr, rs1v[XLEN - 1]); api.add(t, rs2v[XLEN - 1]) };
            let nlts = not1(api, lts_rr);
            let nltu = not1(api, ltu_rr);
            let taken_cond = {
                let mut acc = api.mul(f3_0, eq);
                let t = api.mul(f3_1, neq); acc = api.add(acc, t);
                let t = api.mul(f3_4, lts_rr); acc = api.add(acc, t);
                let t = api.mul(f3_5, nlts); acc = api.add(acc, t);
                let t = api.mul(f3_6, ltu_rr); acc = api.add(acc, t);
                let t = api.mul(f3_7, nltu); acc = api.add(acc, t);
                acc
            };
            let branch_taken = api.mul(is_branch, taken_cond);
            let branch_target = add32(api, &pc, &imm_b);
            let jal_target = add32(api, &pc, &imm_j);
            let jalr_target = { let mut v = add32(api, &rs1v, &imm_i); v[0] = zero; v };
            let mut np = pc_plus4.clone();
            np = mux32(api, branch_taken, &branch_target, &np);
            np = mux32(api, is_jal, &jal_target, &np);
            np = mux32(api, is_jalr, &jalr_target, &np);
            pc = np;
        }

        // ---- exit pc pinned to the public exit_pc.
        for b in 0..XLEN {
            api.assert_is_equal(pc[b], self.exit_pc[b]);
        }

        // ---- HINT_READ committed initial-memory writes (THE DA-reuse crux).
        // The delivered input BYTES are committed circuit variables (hint_word,
        // part of the rsema1d-committed input rows), written into guest memory at
        // the syscall-copy addresses with the global HINT_TS. Downstream loads of
        // these words validate against these writes through the threaded product,
        // so the whole computation is bound to the committed (== DA) input.
        {
            let hts = hint_ts();
            let ts_bits: Vec<Variable> = (0..TS).map(|i| api.constant((hts >> i) & 1)).collect();
            for k in 0..nhint() {
                let addr = const_word(api, hint_addrs()[k]);
                let val = self.hint_word[k].to_vec();
                mem_write_fps.push(factor(api, &addr, &val, &ts_bits, &alpha, &beta, &beta2));
            }
        }

        // ---- register final drain (self-contained per chunk, ALWAYS).
        for r in 0..NREG {
            let addr = const_word(api, r as u32);
            let val = self.fin_val[r].to_vec();
            let ts = self.fin_ts[r].to_vec();
            reg_read_fps.push(factor(api, &addr, &val, &ts, &alpha, &beta, &beta2));
        }
        // ---- memory drain reads. DISTRIBUTED: each chunk drains the addresses
        // whose LAST touch in the span is in this chunk (final value/ts). Summed
        // over all chunks this drains every touched cell exactly once; the threaded
        // product therefore balances only after the final chunk. In OPEN spans
        // (end not reached) the per-chunk drain list is empty (nm==0).
        for k in 0..nm {
            let addr = const_word(api, mem_addrs()[k]);
            let val = self.fin_val[NREG + k].to_vec();
            let ts = self.fin_ts[NREG + k].to_vec();
            mem_read_fps.push(factor(api, &addr, &val, &ts, &alpha, &beta, &beta2));
        }

        // ---- register grand product: closed per chunk.
        let reg_pr = product_tree(api, reg_read_fps);
        let reg_pw = product_tree(api, reg_write_fps);
        for b in 0..128 {
            api.assert_is_equal(reg_pr[b], reg_pw[b]);
        }

        // ---- memory grand product: THREADED. Fold this chunk's read/write factors
        // into the running accumulators carried as public IO from the previous chunk.
        let mem_pr_in = self.mem_pr_in.to_vec();
        let mem_pw_in = self.mem_pw_in.to_vec();
        let chunk_mem_pr = product_tree(api, mem_read_fps);
        let chunk_mem_pw = product_tree(api, mem_write_fps);
        let mem_pr_out = gf128::mul(api, &mem_pr_in, &chunk_mem_pr);
        let mem_pw_out = gf128::mul(api, &mem_pw_in, &chunk_mem_pw);
        for b in 0..128 {
            api.assert_is_equal(mem_pr_out[b], self.mem_pr_out[b]);
            api.assert_is_equal(mem_pw_out[b], self.mem_pw_out[b]);
        }
        // Global memory consistency: the CLOSED product balances only at the end.
        if is_last() {
            for b in 0..128 {
                api.assert_is_equal(self.mem_pr_out[b], self.mem_pw_out[b]);
            }
        }

        // ---- exit registers pinned to the (per-chunk) grand-product-validated finals.
        for r in 0..NREG {
            for b in 0..XLEN {
                api.assert_is_equal(self.fin_val[r][b], self.exit_reg[r][b]);
            }
        }

        // ---- committed outputs (block_number/state_root or the demo checksum),
        // taken from the memory final image at the output-buffer words. Meaningful
        // in the LAST chunk (out_idx points into the memory drain entries).
        let out_idx = unsafe { OUT_IDX };
        for k in 0..NOUT {
            for b in 0..XLEN {
                api.assert_is_equal(self.fin_val[out_idx[k]][b], self.out[k][b]);
            }
        }

        // ---- BLOCK-DATA LINKAGE gadget (the accidental-computer DA reuse) -----
        // Expose y_tr = MLE_{B_trace}(R) as a PUBLIC OUTPUT, where B_trace is the
        // committed HINT_READ region (self.hint_word: the exact block-data bytes
        // the guest executed, pinned to the trace by the memory multiset above),
        // and R = self.link_r is the m-coordinate GF(2^128) evaluation point fed
        // in as PUBLIC INPUT. Because y_tr is computed IN-CIRCUIT from the SAME
        // committed hint_word variables, GKR soundly binds it to the executed B.
        //
        // The bit-vector coefficient at flat address a = k*XLEN + b is bit b of
        // hint word k (address bits: low log2(XLEN)=5 = bit index, higher = word
        // index), matching the rsema1d row/symbol layout used to build C_data
        // (row k symbol b = bit b of word k). The MLE is the standard tensor:
        //   y_tr = Σ_a c_a · Π_p ( bit_p(a) ? R[p] : 1+R[p] )   over GF(2^128).
        if link_on() {
            let m = link_m();
            // eq table: table[a] = Π_p ( (a>>p)&1 ? R[p] : 1+R[p] ), a in 0..2^m.
            // Round p appends address-bit p as the new high bit (first half bit=0,
            // second half bit=1), so table[a]'s bit p uses R[p].
            let one = gf128::const_gf(api, 1);
            let mut table: Vec<Vec<Variable>> = vec![one.clone()];
            for p in 0..m {
                let rp: Vec<Variable> = self.link_r[p].to_vec();
                let omr = gf128::add(api, &one, &rp); // 1 + R[p]
                let mut nt: Vec<Vec<Variable>> = Vec::with_capacity(table.len() * 2);
                for e in &table {
                    nt.push(gf128::mul(api, e, &omr));
                }
                for e in &table {
                    nt.push(gf128::mul(api, e, &rp));
                }
                table = nt;
            }
            // y_tr = Σ over committed hint bits of c_a * table[a] (c_a a single
            // bit => the product is c_a AND each coordinate of table[a]).
            let mut y = vec![zero; 128];
            for k in 0..nhint() {
                for b in 0..XLEN {
                    let a = k * XLEN + b;
                    let c_a = self.hint_word[k][b];
                    for i in 0..128 {
                        let t = api.mul(c_a, table[a][i]);
                        y[i] = api.add(y[i], t);
                    }
                }
            }
            for i in 0..128 {
                api.assert_is_equal(y[i], self.link_y[0][i]);
            }
        }

        let _ = (naddr, reg_init);
    }
}

/// Native mirror of the in-circuit linkage MLE: y_tr = MLE_{B_trace}(R) over
/// GF(2^128), with the EXACT same field (`gf128::native_mul`) and bit-vector
/// layout (flat address a = word*XLEN + bit) as the gadget above. Used by the
/// driver to (a) set the public output `link_y`, and (b) cross-check against the
/// rsema1d opening value (after the field isomorphism).
pub fn link_mle_native(hint_words: &[u32], r: &[u128]) -> u128 {
    let m = r.len();
    let mut table: Vec<u128> = vec![1u128]; // GF(2^128) one
    for &rp in r.iter().take(m) {
        let omr = 1u128 ^ rp; // 1 + R[p]
        let mut nt: Vec<u128> = Vec::with_capacity(table.len() * 2);
        for &e in &table {
            nt.push(gf128::native_mul(e, omr));
        }
        for &e in &table {
            nt.push(gf128::native_mul(e, rp));
        }
        table = nt;
    }
    let mut y = 0u128;
    for (k, &w) in hint_words.iter().enumerate() {
        for b in 0..XLEN {
            if (w >> b) & 1 == 1 {
                y ^= table[k * XLEN + b];
            }
        }
    }
    y
}
