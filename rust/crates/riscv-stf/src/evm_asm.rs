//! Tiny label-resolving assembler over the base RV32I subset that the GKR CPU
//! circuit (`circuit_gp`) verifies in-circuit: ADD/SUB/XOR/OR/AND/SLL/SRL (+ imm
//! forms), LW, SW, BEQ, JAL. NO LUI / JALR / MUL / SLT / byte-loads — the EVM
//! interpreter is written entirely within this subset so the proven circuit needs
//! ZERO new instructions (constants are built with ADDI+SLLI, BNE is emulated as
//! BEQ+JAL, multi-limb carry/borrow use bit tricks, code/calldata/memory are word
//! -per-byte so only word LW/SW are used).
//!
//! The assembler lets the interpreter be written with symbolic labels; a two-pass
//! resolve turns labels into concrete BEQ/JAL byte offsets.

use crate::emulator as e;

#[derive(Clone, Debug)]
pub enum Ins {
    /// A fully-formed instruction word (no label to resolve).
    Word(u32),
    /// BEQ rs1, rs2, <label>
    Beq(u32, u32, String),
    /// JAL rd, <label>
    Jal(u32, String),
    /// A named location (emits no word).
    Label(String),
}

#[derive(Default)]
pub struct Asm {
    pub items: Vec<Ins>,
}

impl Asm {
    pub fn new() -> Self {
        Asm { items: Vec::new() }
    }

    // ---- structural ----
    pub fn label(&mut self, name: &str) {
        self.items.push(Ins::Label(name.to_string()));
    }
    pub fn beq(&mut self, rs1: u32, rs2: u32, label: &str) {
        self.items.push(Ins::Beq(rs1, rs2, label.to_string()));
    }
    pub fn jal(&mut self, rd: u32, label: &str) {
        self.items.push(Ins::Jal(rd, label.to_string()));
    }
    /// Unconditional jump to a label (JAL x0).
    pub fn j(&mut self, label: &str) {
        self.jal(0, label);
    }
    /// BNE rs1,rs2 -> label, emulated as: beq rs1,rs2, __skip ; j label ; __skip:
    pub fn bne(&mut self, rs1: u32, rs2: u32, label: &str, uniq: &str) {
        let skip = format!("__bne_skip_{uniq}");
        self.beq(rs1, rs2, &skip);
        self.j(label);
        self.label(&skip);
    }

    // ---- raw single-word instructions ----
    pub fn raw(&mut self, w: u32) {
        self.items.push(Ins::Word(w));
    }
    pub fn addi(&mut self, rd: u32, rs1: u32, imm: i32) {
        self.raw(e::addi(rd, rs1, imm));
    }
    pub fn add(&mut self, rd: u32, rs1: u32, rs2: u32) {
        self.raw(e::add(rd, rs1, rs2));
    }
    pub fn sub(&mut self, rd: u32, rs1: u32, rs2: u32) {
        self.raw(e::sub(rd, rs1, rs2));
    }
    pub fn xor(&mut self, rd: u32, rs1: u32, rs2: u32) {
        self.raw(e::xor(rd, rs1, rs2));
    }
    pub fn or(&mut self, rd: u32, rs1: u32, rs2: u32) {
        self.raw(e::or(rd, rs1, rs2));
    }
    pub fn and(&mut self, rd: u32, rs1: u32, rs2: u32) {
        self.raw(e::and(rd, rs1, rs2));
    }
    pub fn sll(&mut self, rd: u32, rs1: u32, rs2: u32) {
        self.raw(e::sll(rd, rs1, rs2));
    }
    pub fn srl(&mut self, rd: u32, rs1: u32, rs2: u32) {
        self.raw(e::srl(rd, rs1, rs2));
    }
    pub fn xori(&mut self, rd: u32, rs1: u32, imm: i32) {
        self.raw(e::xori(rd, rs1, imm));
    }
    pub fn andi(&mut self, rd: u32, rs1: u32, imm: i32) {
        self.raw(e::i_type(imm as u32, rs1, 0x7, rd, e::OPC_OPIMM));
    }
    pub fn ori(&mut self, rd: u32, rs1: u32, imm: i32) {
        self.raw(e::i_type(imm as u32, rs1, 0x6, rd, e::OPC_OPIMM));
    }
    pub fn slli(&mut self, rd: u32, rs1: u32, sh: u32) {
        self.raw(e::slli(rd, rs1, sh));
    }
    pub fn srli(&mut self, rd: u32, rs1: u32, sh: u32) {
        self.raw(e::srli(rd, rs1, sh));
    }
    pub fn lw(&mut self, rd: u32, rs1: u32, imm: i32) {
        self.raw(e::lw(rd, rs1, imm));
    }
    pub fn sw(&mut self, rs2: u32, rs1: u32, imm: i32) {
        self.raw(e::sw(rs2, rs1, imm));
    }

    /// Load an arbitrary 32-bit constant into `rd` using only ADDI + SLLI + OR
    /// (no LUI). Small constants (< 2048) are a single ADDI; otherwise build the
    /// value one byte at a time from the most-significant non-zero byte down.
    pub fn li(&mut self, rd: u32, tmp: u32, val: u32) {
        assert_ne!(rd, 0);
        if val < 0x800 {
            self.addi(rd, 0, val as i32);
            return;
        }
        // Highest non-zero byte index (0..=3).
        let top = (3 - (val.leading_zeros() / 8)) as i32;
        self.addi(rd, 0, ((val >> (8 * top)) & 0xff) as i32);
        for b in (0..top).rev() {
            self.slli(rd, rd, 8);
            let byte = ((val >> (8 * b)) & 0xff) as i32;
            if byte != 0 {
                self.addi(tmp, 0, byte);
                self.or(rd, rd, tmp);
            }
        }
    }

    /// Expand every BEQ into a near-branch + JAL trampoline so conditional
    /// targets are never limited by BEQ's +-4 KiB reach (JAL reaches +-1 MiB):
    ///   beq rs1,rs2, S ; j CONT ; S: j TARGET ; CONT:
    fn expand_branches(&self) -> Vec<Ins> {
        let mut out = Vec::with_capacity(self.items.len() * 2);
        let mut ctr = 0usize;
        for it in &self.items {
            match it {
                Ins::Beq(rs1, rs2, target) => {
                    let s = format!("__br_s_{ctr}");
                    let cont = format!("__br_c_{ctr}");
                    ctr += 1;
                    out.push(Ins::Beq(*rs1, *rs2, s.clone()));
                    out.push(Ins::Jal(0, cont.clone()));
                    out.push(Ins::Label(s));
                    out.push(Ins::Jal(0, target.clone()));
                    out.push(Ins::Label(cont));
                }
                other => out.push(other.clone()),
            }
        }
        out
    }

    /// Resolve labels and return the flat program image (one word per real insn).
    pub fn assemble(&self, base: u32) -> Vec<u32> {
        let items = self.expand_branches();
        // Pass 1: address of each item (labels take zero space).
        let mut addr: Vec<u32> = Vec::with_capacity(items.len());
        let mut pc = base;
        let mut labels = std::collections::HashMap::<String, u32>::new();
        for it in &items {
            addr.push(pc);
            match it {
                Ins::Label(name) => {
                    if labels.insert(name.clone(), pc).is_some() {
                        panic!("duplicate label {name}");
                    }
                }
                _ => pc += 4,
            }
        }
        // Pass 2: emit.
        let mut out = Vec::new();
        for (i, it) in items.iter().enumerate() {
            match it {
                Ins::Word(w) => out.push(*w),
                Ins::Label(_) => {}
                Ins::Beq(rs1, rs2, name) => {
                    let target = *labels
                        .get(name)
                        .unwrap_or_else(|| panic!("undefined label {name}"));
                    let off = target as i64 - addr[i] as i64;
                    out.push(e::beq(*rs1, *rs2, off as i32));
                }
                Ins::Jal(rd, name) => {
                    let target = *labels
                        .get(name)
                        .unwrap_or_else(|| panic!("undefined label {name}"));
                    let off = target as i64 - addr[i] as i64;
                    out.push(e::jal(*rd, off as i32));
                }
            }
        }
        out
    }
}
