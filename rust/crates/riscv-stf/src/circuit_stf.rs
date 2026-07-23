//! Transfer-STF CPU-verifier circuit: the SAME state-threaded RV32I CPU verifier
//! as `circuit_full.rs` (ROM fetch + register-file + data-memory threading +
//! BEQ/JAL control flow), specialized to prove a *block state transition*.
//!
//! The proven program (installed in `PROGRAM`) is a fixed transfer-STF
//! interpreter. The block's pre-state (sender/recipient balance+nonce) and its
//! transactions ({value, fee, nonce}) are installed as the initial data memory
//! (`MEM_INIT`); the interpreter applies each tx IN ORDER with replay protection
//! (`tx.nonce == sender.nonce`) and balance checks, and writes the post-state
//! back to memory. The committed GKR INPUT layer is the execution trace
//! (insn ++ rs1_val ++ rs2_val ++ rd_val per cycle) — whose loaded/stored words
//! ARE the block's tx data and pre/post-state — committed by rsema1d, byte-
//! identical to the Go/DA encoder.
//!
//! HONEST SCOPE: this proves the *nonce + balance* state transition over the
//! block's committed tx data. It does NOT prove ECDSA signature verification or
//! a keccak/MPT state root (RAM wall / large gadget work) — those are the next
//! rungs toward full light-client validity. See `stf.rs`.
//!
//! Everything is bit-level over GF2 (add == XOR, mul == AND). All gadgets are
//! identical to the verified `circuit_full` circuit.

use crate::emulator::StepRecord;
use expander_compiler::frontend::*;

pub const XLEN: usize = 32;
pub const NREG: usize = 32;

/// Max transactions per block (fixed unrolled slots; unused slots carry a
/// guaranteed-invalid nonce so they are rejected, not applied).
pub const MAXTX: usize = 4;

/// Cycles (trace rows). Chosen so the fully-applied MAXTX path fits with halt
/// padding, and CYCLES*128 is a clean power of two (64*128 = 2^13).
pub const CYCLES: usize = 64;

/// Program length of the transfer-STF interpreter (see `stf::build_program`).
pub const PROG_LEN: usize = 69;

pub const BASE: u32 = 0;

// ---- data-memory cell layout (all above the program address range) ---------
pub const SBAL_ADDR: u32 = 0x200; // sender balance
pub const SNON_ADDR: u32 = 0x204; // sender nonce
pub const RBAL_ADDR: u32 = 0x208; // recipient balance
pub const APPLIED_ADDR: u32 = 0x20c; // count of applied (valid) txs
pub const DIGEST_ADDR: u32 = 0x210; // post-state digest (simple fold)
pub const TX_BASE: u32 = 0x240; // tx i: value @ +i*12, fee @ +i*12+4, nonce @ +i*12+8
pub const TX_STRIDE: u32 = 12;

pub const NACC_CELLS: usize = 5;
pub const NMEM: usize = NACC_CELLS + 3 * MAXTX; // 5 + 12 = 17

/// Addresses of the threaded data-memory cells (must match the program's access
/// pattern exactly).
pub fn mem_addrs() -> [u32; NMEM] {
    let mut a = [0u32; NMEM];
    a[0] = SBAL_ADDR;
    a[1] = SNON_ADDR;
    a[2] = RBAL_ADDR;
    a[3] = APPLIED_ADDR;
    a[4] = DIGEST_ADDR;
    for i in 0..MAXTX {
        a[NACC_CELLS + 3 * i] = TX_BASE + TX_STRIDE * i as u32; // value
        a[NACC_CELLS + 3 * i + 1] = TX_BASE + TX_STRIDE * i as u32 + 4; // fee
        a[NACC_CELLS + 3 * i + 2] = TX_BASE + TX_STRIDE * i as u32 + 8; // nonce
    }
    a
}

/// Public outputs: post sender-balance, post sender-nonce, post recipient-balance,
/// applied count, post-state digest. All read from the threaded memory cells.
pub const NOUT: usize = 5;
pub const OUT_CELLS: [usize; NOUT] = [0, 1, 2, 3, 4];

// Program image + initial memory, set by the driver before compile().
pub static mut PROGRAM: [u32; PROG_LEN] = [0; PROG_LEN];
pub static mut MEM_INIT: [u32; NMEM] = [0; NMEM];

declare_circuit!(RiscvCircuitStf {
    insn: [[Variable; XLEN]; CYCLES],
    rs1_val: [[Variable; XLEN]; CYCLES],
    rs2_val: [[Variable; XLEN]; CYCLES],
    rd_val: [[Variable; XLEN]; CYCLES],
    out: [[PublicVariable; XLEN]; NOUT],
});

fn set_word(dst: &mut [GF2], word: u32) {
    for i in 0..XLEN {
        dst[i] = ((word >> i) & 1).into();
    }
}

pub fn build_assignment(trace: &[StepRecord], out_vals: &[u32]) -> RiscvCircuitStf<GF2> {
    let mut a = RiscvCircuitStf::<GF2>::default();
    for c in 0..CYCLES {
        set_word(&mut a.insn[c], trace[c].insn);
        set_word(&mut a.rs1_val[c], trace[c].rs1_val);
        set_word(&mut a.rs2_val[c], trace[c].rs2_val);
        set_word(&mut a.rd_val[c], trace[c].rd_val);
    }
    for k in 0..NOUT {
        set_word(&mut a.out[k], out_vals[k]);
    }
    a
}

/// Tamper a committed rd_val word (for the soundness gate).
pub fn tamper_rd_val(a: &mut RiscvCircuitStf<GF2>, cycle: usize, word: u32) {
    set_word(&mut a.rd_val[cycle], word);
}

// ------------------------------- gadgets -----------------------------------
// (identical to circuit_full)

fn not1<C: Config>(api: &mut impl RootAPI<C>, x: Variable) -> Variable {
    api.sub(1, x)
}
fn mux1<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: Variable, b: Variable) -> Variable {
    let d = api.add(a, b);
    let t = api.mul(sel, d);
    api.add(b, t)
}
fn mux32<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| mux1(api, sel, a[i], b[i])).collect()
}
fn xor32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| api.add(a[i], b[i])).collect()
}
fn and32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| api.mul(a[i], b[i])).collect()
}
fn or32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| { let x = api.add(a[i], b[i]); let y = api.mul(a[i], b[i]); api.add(x, y) }).collect()
}
fn add32_cin<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable], cin: Variable) -> Vec<Variable> {
    let mut sum = vec![api.constant(0); XLEN];
    let mut carry = cin;
    for i in 0..XLEN {
        let ab = api.add(a[i], b[i]);
        sum[i] = api.add(ab, carry);
        let and_ab = api.mul(a[i], b[i]);
        let c_and = api.mul(carry, ab);
        carry = api.add(and_ab, c_and);
    }
    sum
}
fn add32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    add32_cin(api, a, b, zero)
}
fn sub32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let nb: Vec<Variable> = b.iter().map(|&x| not1(api, x)).collect();
    let one = api.constant(1);
    add32_cin(api, a, &nb, one)
}
fn sll32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], shamt: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut cur = a.to_vec();
    for (k, &sbit) in shamt.iter().enumerate().take(5) {
        let sh = 1usize << k;
        let shifted: Vec<Variable> = (0..XLEN).map(|i| if i >= sh { cur[i - sh] } else { zero }).collect();
        cur = (0..XLEN).map(|i| mux1(api, sbit, shifted[i], cur[i])).collect();
    }
    cur
}
fn srl32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], shamt: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut cur = a.to_vec();
    for (k, &sbit) in shamt.iter().enumerate().take(5) {
        let sh = 1usize << k;
        let shifted: Vec<Variable> = (0..XLEN).map(|i| if i + sh < XLEN { cur[i + sh] } else { zero }).collect();
        cur = (0..XLEN).map(|i| mux1(api, sbit, shifted[i], cur[i])).collect();
    }
    cur
}
fn eq_const<C: Config>(api: &mut impl RootAPI<C>, bits: &[Variable], value: u32) -> Variable {
    let mut acc = api.constant(1);
    for (i, &b) in bits.iter().enumerate() {
        let want = (value >> i) & 1;
        let term = if want == 1 { b } else { not1(api, b) };
        acc = api.mul(acc, term);
    }
    acc
}
fn eq32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Variable {
    let mut acc = api.constant(1);
    for i in 0..XLEN {
        let x = api.add(a[i], b[i]);
        let eqb = not1(api, x);
        acc = api.mul(acc, eqb);
    }
    acc
}
fn const_word<C: Config>(api: &mut impl RootAPI<C>, word: u32) -> Vec<Variable> {
    (0..XLEN).map(|i| api.constant((word >> i) & 1)).collect()
}
fn reg_read<C: Config>(api: &mut impl RootAPI<C>, regfile: &[Vec<Variable>], idx_bits: &[Variable]) -> Vec<Variable> {
    let onehot: Vec<Variable> = (0..NREG).map(|r| eq_const(api, idx_bits, r as u32)).collect();
    let mut out = vec![api.constant(0); XLEN];
    for r in 0..NREG {
        for b in 0..XLEN {
            let sel = api.mul(onehot[r], regfile[r][b]);
            out[b] = api.add(out[b], sel);
        }
    }
    out
}

// ------------------------------ the circuit --------------------------------

impl Define<GF2Config> for RiscvCircuitStf<Variable> {
    fn define<Builder: RootAPI<GF2Config>>(&self, api: &mut Builder) {
        let mem_addr_vals = mem_addrs();
        let zero = api.constant(0);
        let mut regfile: Vec<Vec<Variable>> = vec![vec![zero; XLEN]; NREG];
        let mut mem: Vec<Vec<Variable>> =
            (0..NMEM).map(|c| const_word(api, unsafe { MEM_INIT[c] })).collect();
        let mem_addr_const: Vec<Vec<Variable>> =
            (0..NMEM).map(|c| const_word(api, mem_addr_vals[c])).collect();
        let mut pc = const_word(api, BASE);

        for c in 0..CYCLES {
            let insn = self.insn[c].to_vec();
            let rs1v = self.rs1_val[c].to_vec();
            let rs2v = self.rs2_val[c].to_vec();
            let rdv = self.rd_val[c].to_vec();

            // (0) FETCH: committed insn == ROM[pc]; pc hits exactly one slot.
            let mut insn_rom = vec![zero; XLEN];
            let mut hit = api.constant(0);
            for j in 0..PROG_LEN {
                let addr_j = const_word(api, BASE + 4 * j as u32);
                let m = eq32(api, &pc, &addr_j);
                hit = api.add(hit, m);
                let word = unsafe { PROGRAM[j] };
                for b in 0..XLEN {
                    let wbit = api.constant((word >> b) & 1);
                    let t = api.mul(m, wbit);
                    insn_rom[b] = api.add(insn_rom[b], t);
                }
            }
            api.assert_is_equal(hit, 1);
            for b in 0..XLEN {
                api.assert_is_equal(insn[b], insn_rom[b]);
            }

            // (1) Decode.
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

            // (2) Register reads must match committed values.
            let rs1_read = reg_read(api, &regfile, rs1_idx);
            let rs2_read = reg_read(api, &regfile, rs2_idx);
            for b in 0..XLEN {
                api.assert_is_equal(rs1_read[b], rs1v[b]);
                api.assert_is_equal(rs2_read[b], rs2v[b]);
            }

            // (3) Class + sub-op selectors.
            let is_op = eq_const(api, opcode, 0b0110011);
            let is_opimm = eq_const(api, opcode, 0b0010011);
            let is_load = eq_const(api, opcode, 0b0000011);
            let is_store = eq_const(api, opcode, 0b0100011);
            let is_branch = eq_const(api, opcode, 0b1100011);
            let is_jal = eq_const(api, opcode, 0b1101111);
            let is_alu = api.add(is_op, is_opimm);
            let f3_0 = eq_const(api, funct3, 0x0);
            let f3_1 = eq_const(api, funct3, 0x1);
            let f3_4 = eq_const(api, funct3, 0x4);
            let f3_5 = eq_const(api, funct3, 0x5);
            let f3_6 = eq_const(api, funct3, 0x6);
            let f3_7 = eq_const(api, funct3, 0x7);
            let f7_zero = eq_const(api, funct7, 0x00);
            let f7_sub = eq_const(api, funct7, 0x20);

            let operand_b = mux32(api, is_opimm, &imm_i, &rs2v);
            let shamt: Vec<Variable> = operand_b[0..5].to_vec();

            // (4) ALU results.
            let add_res = add32(api, &rs1v, &operand_b);
            let sub_res = sub32(api, &rs1v, &rs2v);
            let xor_res = xor32(api, &rs1v, &operand_b);
            let or_res = or32(api, &rs1v, &operand_b);
            let and_res = and32(api, &rs1v, &operand_b);
            let sll_res = sll32(api, &rs1v, &shamt);
            let srl_res = srl32(api, &rs1v, &shamt);

            let sel_add = { let t1 = api.mul(is_op, f3_0); let t1 = api.mul(t1, f7_zero); let t2 = api.mul(is_opimm, f3_0); api.add(t1, t2) };
            let sel_sub = { let t = api.mul(is_op, f3_0); api.mul(t, f7_sub) };
            let sel_xor = api.mul(is_alu, f3_4);
            let sel_or = api.mul(is_alu, f3_6);
            let sel_and = api.mul(is_alu, f3_7);
            let sel_sll = api.mul(is_alu, f3_1);
            let sel_srl = api.mul(is_alu, f3_5);
            let sels = [sel_add, sel_sub, sel_xor, sel_or, sel_and, sel_sll, sel_srl];
            let ress = [&add_res, &sub_res, &xor_res, &or_res, &and_res, &sll_res, &srl_res];
            let mut alu_rd = vec![zero; XLEN];
            for (sel, res) in sels.iter().zip(ress.iter()) {
                for b in 0..XLEN {
                    let t = api.mul(*sel, res[b]);
                    alu_rd[b] = api.add(alu_rd[b], t);
                }
            }

            // (5) LOAD: rd = mem[rs1+imm_i].
            let load_addr = add32(api, &rs1v, &imm_i);
            let mut load_val = vec![zero; XLEN];
            for cell in 0..NMEM {
                let m = eq32(api, &load_addr, &mem_addr_const[cell]);
                for b in 0..XLEN {
                    let t = api.mul(m, mem[cell][b]);
                    load_val[b] = api.add(load_val[b], t);
                }
            }

            // (6) JAL return address = pc + 4.
            let four = const_word(api, 4);
            let pc_plus4 = add32(api, &pc, &four);

            // (7) computed rd = alu | load | jal(link). (classes disjoint)
            let mut computed_rd = alu_rd.clone();
            for b in 0..XLEN {
                let lt = api.mul(is_load, load_val[b]);
                computed_rd[b] = api.add(computed_rd[b], lt);
                let jt = api.mul(is_jal, pc_plus4[b]);
                computed_rd[b] = api.add(computed_rd[b], jt);
            }
            for b in 0..XLEN {
                api.assert_is_equal(computed_rd[b], rdv[b]);
            }

            // (8) STORE: mem[rs1+imm_s] := rs2.
            let store_addr = add32(api, &rs1v, &imm_s);
            let mut mem_next: Vec<Vec<Variable>> = mem.clone();
            for cell in 0..NMEM {
                let m = eq32(api, &store_addr, &mem_addr_const[cell]);
                let do_store = api.mul(is_store, m);
                mem_next[cell] = mux32(api, do_store, &rs2v, &mem[cell]);
            }

            // (9) Register write for classes that write rd (alu|load|jal), x0 discarded.
            let we0 = api.add(is_alu, is_load);
            let write_enable = api.add(we0, is_jal);
            let onehot_rd: Vec<Variable> = (0..NREG).map(|r| eq_const(api, rd_idx, r as u32)).collect();
            let mut reg_next: Vec<Vec<Variable>> = vec![vec![zero; XLEN]; NREG];
            for r in 1..NREG {
                let do_write = api.mul(onehot_rd[r], write_enable);
                reg_next[r] = mux32(api, do_write, &rdv, &regfile[r]);
            }

            // (10) next_pc.
            let beq_taken = { let e = eq32(api, &rs1v, &rs2v); let b0 = api.mul(is_branch, f3_0); api.mul(b0, e) };
            let branch_target = add32(api, &pc, &imm_b);
            let jal_target = add32(api, &pc, &imm_j);
            let mut np = pc_plus4.clone();
            np = mux32(api, beq_taken, &branch_target, &np);
            np = mux32(api, is_jal, &jal_target, &np);

            regfile = reg_next;
            mem = mem_next;
            pc = np;
        }

        // (11) Public outputs: post-state cells.
        for k in 0..NOUT {
            let cell = OUT_CELLS[k];
            for b in 0..XLEN {
                api.assert_is_equal(mem[cell][b], self.out[k][b]);
            }
        }
    }
}
