//! Full RV32IM-subset CPU-verifier circuit (Steps B + C): arithmetic + LOAD/STORE
//! (memory) + BEQ/JAL (control flow), proving a REAL looping program.
//!
//! Everything remains bit-level over GF2 (add==XOR, mul==AND). Consistency is by
//! EXPLICIT in-circuit state threading (sound for a fixed trace):
//!   * register file (NREG x XLEN) threaded across cycles;
//!   * data memory modeled as NMEM word cells at fixed addresses, threaded;
//!   * the program counter threaded, with the instruction FETCHED from a ROM
//!     (the program image) indexed by pc — so control flow (BEQ/JAL) is real:
//!     the committed insn must equal ROM[pc], and pc evolves per the decoded op.
//!
//! Committed INPUT layer per cycle: insn[32] ++ rs1_val[32] ++ rs2_val[32] ++
//! rd_val[32] (128 bits). Loaded/stored values reuse rd_val (loads) / rs2_val
//! (stores), so no extra committed column is needed.

use crate::emulator::StepRecord;
use expander_compiler::frontend::*;

pub const XLEN: usize = 32;
pub const NREG: usize = 32;
pub const CYCLES: usize = 32; // 32*128 = 4096 = 2^12 committed inputs
pub const PROG_LEN: usize = 11;
pub const NMEM: usize = 5; // 4 array words + 1 result word
pub const BASE: u32 = 0;
pub const DATA_BASE: u32 = 0x100;
pub const RESULT_ADDR: u32 = 0x200;

/// Data-memory cell addresses (must match the program's access pattern).
pub const MEM_ADDRS: [u32; NMEM] =
    [DATA_BASE, DATA_BASE + 4, DATA_BASE + 8, DATA_BASE + 12, RESULT_ADDR];

/// Public outputs: final x2 (sum), final x3 (loop count), final mem[RESULT_ADDR].
pub const NOUT: usize = 3;

// Program image + initial memory, set by the driver before compile().
pub static mut PROGRAM: [u32; PROG_LEN] = [0; PROG_LEN];
pub static mut MEM_INIT: [u32; NMEM] = [0; NMEM];

declare_circuit!(RiscvCircuitFull {
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

pub fn build_assignment(trace: &[StepRecord], out_vals: &[u32]) -> RiscvCircuitFull<GF2> {
    let mut a = RiscvCircuitFull::<GF2>::default();
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

pub fn tamper_rd_val(a: &mut RiscvCircuitFull<GF2>, cycle: usize, word: u32) {
    set_word(&mut a.rd_val[cycle], word);
}

// ------------------------------- gadgets -----------------------------------

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
/// 1 iff two 32-bit words are bit-identical.
fn eq32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Variable {
    let mut acc = api.constant(1);
    for i in 0..XLEN {
        let x = api.add(a[i], b[i]); // 0 iff equal
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

impl Define<GF2Config> for RiscvCircuitFull<Variable> {
    fn define<Builder: RootAPI<GF2Config>>(&self, api: &mut Builder) {
        let zero = api.constant(0);
        let mut regfile: Vec<Vec<Variable>> = vec![vec![zero; XLEN]; NREG];
        // Threaded data-memory cells, initialized from MEM_INIT.
        let mut mem: Vec<Vec<Variable>> =
            (0..NMEM).map(|c| const_word(api, unsafe { MEM_INIT[c] })).collect();
        let mem_addr_const: Vec<Vec<Variable>> =
            (0..NMEM).map(|c| const_word(api, MEM_ADDRS[c])).collect();
        // Threaded program counter, initialized to BASE.
        let mut pc = const_word(api, BASE);

        for c in 0..CYCLES {
            let insn = self.insn[c].to_vec();
            let rs1v = self.rs1_val[c].to_vec();
            let rs2v = self.rs2_val[c].to_vec();
            let rdv = self.rd_val[c].to_vec();

            // (0) FETCH: committed insn must equal ROM[pc]. Also require pc to hit
            // exactly one ROM slot (in-range), which control flow must preserve.
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
            api.assert_is_equal(hit, 1); // pc in range, unique fetch
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

            // Immediates (pure wire selection).
            let imm_i = {
                let mut v = vec![zero; XLEN];
                for i in 0..12 { v[i] = insn[20 + i]; }
                for i in 12..XLEN { v[i] = insn[31]; }
                v
            };
            let imm_s = {
                let mut v = vec![zero; XLEN];
                for i in 0..5 { v[i] = insn[7 + i]; }       // imm[4:0]=insn[11:7]
                for i in 5..12 { v[i] = insn[20 + i]; }      // imm[11:5]=insn[31:25]
                for i in 12..XLEN { v[i] = insn[31]; }
                v
            };
            let imm_b = {
                let mut v = vec![zero; XLEN];
                v[0] = zero;
                for i in 1..5 { v[i] = insn[7 + i]; }         // imm[4:1]=insn[11:8]
                for i in 5..11 { v[i] = insn[20 + i]; }       // imm[10:5]=insn[30:25]
                v[11] = insn[7];                              // imm[11]=insn[7]
                for i in 12..XLEN { v[i] = insn[31]; }        // imm[12]&sext=insn[31]
                v
            };
            let imm_j = {
                let mut v = vec![zero; XLEN];
                v[0] = zero;
                for i in 1..11 { v[i] = insn[20 + i]; }        // imm[10:1]=insn[30:21]
                v[11] = insn[20];                              // imm[11]=insn[20]
                for i in 12..20 { v[i] = insn[i]; }            // imm[19:12]=insn[19:12]
                for i in 20..XLEN { v[i] = insn[31]; }         // imm[20]&sext=insn[31]
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

            // (7) Compose computed rd = alu | load | jal(link). (classes disjoint)
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

            // (10) next_pc: JAL -> pc+imm_j; BEQ taken -> pc+imm_b; else pc+4.
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

        // (11) Public outputs: final x2 (sum), final x3 (loop count), mem[RESULT_ADDR].
        let result_cell = NMEM - 1; // RESULT_ADDR
        for b in 0..XLEN {
            api.assert_is_equal(regfile[2][b], self.out[0][b]);
            api.assert_is_equal(regfile[3][b], self.out[1][b]);
            api.assert_is_equal(mem[result_cell][b], self.out[2][b]);
        }
    }
}
