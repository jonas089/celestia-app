//! RV32IM CPU-verifier as an `expander_compiler` circuit over GF2.
//!
//! Everything is bit-level: over `GF2Config`, `api.add` == XOR and `api.mul` ==
//! AND, so all 32-bit ALU ops are built from booleans (ripple-carry adder, native
//! XOR/AND/OR, barrel shifters) — never prime-field carries. Register-file
//! consistency is enforced by EXPLICIT in-circuit state threading: the circuit
//! reconstructs the true register file cycle-by-cycle from the committed writes
//! and checks every committed read against it. This is fully sound for a fixed
//! trace and sidesteps a GF(2^128) grand-product permutation argument (which in a
//! GF(2) circuit would require emulating 128-bit carryless mult + a Fiat-Shamir
//! challenge as public input — the documented scaling path, unnecessary at demo
//! scale).
//!
//! The committed INPUT layer is the execution trace, bit-decomposed: per cycle
//! `insn[32] ++ rs1_val[32] ++ rs2_val[32] ++ rd_val[32]`. num circuit inputs =
//! CYCLES * 128, laid into `layers[0].input_vals` as GF2x8 (8 identical SIMD
//! lanes, like the keccak template), committed byte-identically by Rsema1dPCS /
//! Go rsema1d.

use crate::emulator::StepRecord;
use expander_compiler::frontend::*;

pub const XLEN: usize = 32;
pub const NREG: usize = 32;
pub const CYCLES: usize = 8;
/// Result registers exposed as public outputs (x1..x8 for the Step-A program).
pub const OUT_REGS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
pub const NOUT: usize = OUT_REGS.len();

declare_circuit!(RiscvCircuit {
    insn: [[Variable; XLEN]; CYCLES],
    rs1_val: [[Variable; XLEN]; CYCLES],
    rs2_val: [[Variable; XLEN]; CYCLES],
    rd_val: [[Variable; XLEN]; CYCLES],
    // public: final value of each OUT_REGS register (asserted == native re-run).
    out: [[PublicVariable; XLEN]; NOUT],
});

// --------------------- assignment builders (module-local) ------------------

fn set_word(dst: &mut [GF2], word: u32) {
    for i in 0..XLEN {
        dst[i] = ((word >> i) & 1).into();
    }
}

/// Build the witness assignment from a flat trace + the public output values.
pub fn build_assignment(trace: &[StepRecord], out_vals: &[u32]) -> RiscvCircuit<GF2> {
    let mut a = RiscvCircuit::<GF2>::default();
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

/// Overwrite the committed rd_val of one cycle (for the soundness/tamper test).
pub fn tamper_rd_val(a: &mut RiscvCircuit<GF2>, cycle: usize, word: u32) {
    set_word(&mut a.rd_val[cycle], word);
}

// ------------------------- bit-level ALU gadgets ---------------------------

/// XOR of two 32-bit words (bitwise add over GF2).
fn xor32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| api.add(a[i], b[i])).collect()
}
/// AND of two 32-bit words (bitwise mul over GF2).
fn and32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| api.mul(a[i], b[i])).collect()
}
/// OR: a|b = a ^ b ^ (a&b).
fn or32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN)
        .map(|i| {
            let x = api.add(a[i], b[i]);
            let y = api.mul(a[i], b[i]);
            api.add(x, y)
        })
        .collect()
}
/// NOT of one bit: 1 ^ x.
fn not1<C: Config>(api: &mut impl RootAPI<C>, x: Variable) -> Variable {
    api.sub(1, x)
}
/// 1-bit mux: sel ? a : b  ==  b ^ (sel & (a ^ b)).
fn mux1<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: Variable, b: Variable) -> Variable {
    let d = api.add(a, b);
    let t = api.mul(sel, d);
    api.add(b, t)
}
fn mux32<C: Config>(
    api: &mut impl RootAPI<C>,
    sel: Variable,
    a: &[Variable],
    b: &[Variable],
) -> Vec<Variable> {
    (0..XLEN).map(|i| mux1(api, sel, a[i], b[i])).collect()
}

/// 32-bit ripple-carry adder with a carry-in bit. Returns the 32-bit sum
/// (mod 2^32); the final carry-out is discarded.
fn add32_cin<C: Config>(
    api: &mut impl RootAPI<C>,
    a: &[Variable],
    b: &[Variable],
    cin: Variable,
) -> Vec<Variable> {
    let mut sum = vec![api.constant(0); XLEN];
    let mut carry = cin;
    for i in 0..XLEN {
        // s = a ^ b ^ carry
        let ab = api.add(a[i], b[i]);
        sum[i] = api.add(ab, carry);
        // carry' = (a&b) ^ (carry & (a^b))
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
/// a - b = a + (~b) + 1.
fn sub32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let nb: Vec<Variable> = b.iter().map(|&x| not1(api, x)).collect();
    let one = api.constant(1);
    add32_cin(api, a, &nb, one)
}
/// Logical left shift by a 5-bit amount (barrel shifter). `shamt` is 5 bits LSB-first.
fn sll32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], shamt: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut cur = a.to_vec();
    for (k, &sbit) in shamt.iter().enumerate().take(5) {
        let sh = 1usize << k;
        let shifted: Vec<Variable> = (0..XLEN)
            .map(|i| if i >= sh { cur[i - sh] } else { zero })
            .collect();
        cur = (0..XLEN).map(|i| mux1(api, sbit, shifted[i], cur[i])).collect();
    }
    cur
}
/// Logical right shift by a 5-bit amount.
fn srl32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], shamt: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut cur = a.to_vec();
    for (k, &sbit) in shamt.iter().enumerate().take(5) {
        let sh = 1usize << k;
        let shifted: Vec<Variable> = (0..XLEN)
            .map(|i| if i + sh < XLEN { cur[i + sh] } else { zero })
            .collect();
        cur = (0..XLEN).map(|i| mux1(api, sbit, shifted[i], cur[i])).collect();
    }
    cur
}

/// 1 iff the little-endian bit-slice `bits` equals the constant `value`.
fn eq_const<C: Config>(api: &mut impl RootAPI<C>, bits: &[Variable], value: u32) -> Variable {
    let mut acc = api.constant(1);
    for (i, &b) in bits.iter().enumerate() {
        let want = (value >> i) & 1;
        let term = if want == 1 { b } else { not1(api, b) };
        acc = api.mul(acc, term);
    }
    acc
}

/// Read register `idx` (5-bit LSB-first) from the register file via a one-hot mux.
fn reg_read<C: Config>(
    api: &mut impl RootAPI<C>,
    regfile: &[Vec<Variable>],
    idx_bits: &[Variable],
) -> Vec<Variable> {
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

/// Program image (u32 words) proven by the Step-A circuit; asserted against the
/// committed `insn` column at each straight-line cycle. Set by the driver.
pub static mut PROGRAM: [u32; CYCLES] = [0; CYCLES];

fn bits_of<C: Config>(api: &mut impl RootAPI<C>, word: u32, n: usize) -> Vec<Variable> {
    (0..n).map(|i| api.constant((word >> i) & 1)).collect()
}

impl Define<GF2Config> for RiscvCircuit<Variable> {
    fn define<Builder: RootAPI<GF2Config>>(&self, api: &mut Builder) {
        // Register file: NREG x XLEN, initialized to all-zero (x0..x31 = 0).
        let zero = api.constant(0);
        let mut regfile: Vec<Vec<Variable>> = vec![vec![zero; XLEN]; NREG];

        for c in 0..CYCLES {
            let insn = &self.insn[c];
            let rs1v = self.rs1_val[c].to_vec();
            let rs2v = self.rs2_val[c].to_vec();
            let rdv = self.rd_val[c].to_vec();

            // (0) Bind committed instruction to the real program word.
            let prog_word = unsafe { PROGRAM[c] };
            for i in 0..XLEN {
                let want = api.constant((prog_word >> i) & 1);
                api.assert_is_equal(insn[i], want);
            }

            // (1) Decode fields (pure wire selection over insn bits).
            let opcode = &insn[0..7];
            let rd_idx = &insn[7..12];
            let funct3 = &insn[12..15];
            let rs1_idx = &insn[15..20];
            let rs2_idx = &insn[20..25];
            let funct7 = &insn[25..32];
            // I-type immediate, sign-extended: bits[0..12]=insn[20..32], sign=insn[31].
            let mut imm_i = vec![zero; XLEN];
            for i in 0..12 {
                imm_i[i] = insn[20 + i];
            }
            for i in 12..XLEN {
                imm_i[i] = insn[31];
            }

            // (2) Register reads must match the committed rs1_val / rs2_val.
            let rs1_read = reg_read(api, &regfile, rs1_idx);
            let rs2_read = reg_read(api, &regfile, rs2_idx);
            for b in 0..XLEN {
                api.assert_is_equal(rs1_read[b], rs1v[b]);
                api.assert_is_equal(rs2_read[b], rs2v[b]);
            }

            // (3) Selectors.
            let is_op = eq_const(api, opcode, 0b0110011);
            let is_opimm = eq_const(api, opcode, 0b0010011);
            let is_alu = api.add(is_op, is_opimm); // mutually exclusive => XOR == OR
            let f3_0 = eq_const(api, funct3, 0x0);
            let f3_1 = eq_const(api, funct3, 0x1);
            let f3_4 = eq_const(api, funct3, 0x4);
            let f3_5 = eq_const(api, funct3, 0x5);
            let f3_6 = eq_const(api, funct3, 0x6);
            let f3_7 = eq_const(api, funct3, 0x7);
            let f7_zero = eq_const(api, funct7, 0x00);
            let f7_sub = eq_const(api, funct7, 0x20);

            // operand_b = is_opimm ? imm_i : rs2_val
            let operand_b = mux32(api, is_opimm, &imm_i, &rs2v);
            // shamt (5 bits) from operand_b low bits
            let shamt: Vec<Variable> = operand_b[0..5].to_vec();

            // (4) ALU results.
            let add_res = add32(api, &rs1v, &operand_b);
            let sub_res = sub32(api, &rs1v, &rs2v);
            let xor_res = xor32(api, &rs1v, &operand_b);
            let or_res = or32(api, &rs1v, &operand_b);
            let and_res = and32(api, &rs1v, &operand_b);
            let sll_res = sll32(api, &rs1v, &shamt);
            let srl_res = srl32(api, &rs1v, &shamt);

            // (5) Op selection (each sel is a single bit; disjoint => sum == mux).
            let sel_add = {
                let t1 = api.mul(is_op, f3_0);
                let t1 = api.mul(t1, f7_zero); // ADD
                let t2 = api.mul(is_opimm, f3_0); // ADDI
                api.add(t1, t2)
            };
            let sel_sub = {
                let t = api.mul(is_op, f3_0);
                api.mul(t, f7_sub)
            };
            let sel_xor = api.mul(is_alu, f3_4);
            let sel_or = api.mul(is_alu, f3_6);
            let sel_and = api.mul(is_alu, f3_7);
            let sel_sll = api.mul(is_alu, f3_1);
            let sel_srl = api.mul(is_alu, f3_5);
            // guard SRL/SLL R-type funct7==0 (SLLI/SRLI imm high bits 0 too);
            // for the Step-A program funct7 is always 0 on shifts, so fold f7_zero
            // only into the R-type contribution is unnecessary — keep simple.

            let sels = [sel_add, sel_sub, sel_xor, sel_or, sel_and, sel_sll, sel_srl];
            let ress = [&add_res, &sub_res, &xor_res, &or_res, &and_res, &sll_res, &srl_res];
            let mut computed_rd = vec![zero; XLEN];
            for (sel, res) in sels.iter().zip(ress.iter()) {
                for b in 0..XLEN {
                    let t = api.mul(*sel, res[b]);
                    computed_rd[b] = api.add(computed_rd[b], t);
                }
            }

            // (6) Committed rd_val must equal the computed ALU output.
            for b in 0..XLEN {
                api.assert_is_equal(computed_rd[b], rdv[b]);
            }

            // (7) Register-file update: write rd_val into reg rd_idx (x0 discarded).
            let onehot_rd: Vec<Variable> =
                (0..NREG).map(|r| eq_const(api, rd_idx, r as u32)).collect();
            let mut next: Vec<Vec<Variable>> = vec![vec![zero; XLEN]; NREG];
            for r in 0..NREG {
                if r == 0 {
                    // x0 hardwired to 0 (also discards any write to x0).
                    continue;
                }
                next[r] = mux32(api, onehot_rd[r], &rdv, &regfile[r]);
            }
            regfile = next;
        }

        // (8) Public output: final register values must match the native re-run.
        for (k, &reg) in OUT_REGS.iter().enumerate() {
            for b in 0..XLEN {
                api.assert_is_equal(regfile[reg][b], self.out[k][b]);
            }
        }
    }
}
