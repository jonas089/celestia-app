//! RV32IM emulator (full base + M extension).
//!
//! Both the emulator AND the active GKR CPU-verifier circuit (`circuit_gp`) now
//! implement the complete RV32IM user-level ISA:
//!   RV32I: LUI, AUIPC, JAL, JALR, BEQ/BNE/BLT/BGE/BLTU/BGEU,
//!          LB/LH/LW/LBU/LHU, SB/SH/SW, ADDI/SLTI/SLTIU/XORI/ORI/ANDI/SLLI/
//!          SRLI/SRAI, ADD/SUB/SLL/SLT/SLTU/XOR/SRL/SRA/OR/AND, FENCE.
//!   RV32M: MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU.
//! FENCE/FENCE.I are nops (single-hart, in-order). ECALL/EBREAK are STUBBED as
//! nops (never panic) pending a later syscall/halt stage — no ECALL program is
//! proven yet.
//!
//! Runs a fixed program image (a `Vec<u32>` of instruction words loaded at a
//! base address) and emits a FLAT per-cycle trace of [`StepRecord`]s. The
//! register file and data memory are modeled explicitly; each StepRecord captures
//! everything the CPU-verifier circuit needs to re-derive and check the step.

/// One executed instruction, fully self-describing for the circuit.
#[derive(Clone, Debug, Default)]
pub struct StepRecord {
    pub cycle: u32,
    pub pc: u32,
    pub next_pc: u32,
    pub insn: u32,
    pub opcode: u32,
    pub funct3: u32,
    pub funct7: u32,
    pub rd_idx: u32,
    pub rs1_idx: u32,
    pub rs2_idx: u32,
    pub imm: u32, // sign-extended immediate (as u32 bit pattern)
    pub rs1_val: u32,
    pub rs2_val: u32,
    pub rd_val: u32, // value written to rd (0 if no rd write)
    // Memory op (LW/SW). is_mem=false => all zero, addr=0.
    pub is_load: bool,
    pub is_store: bool,
    pub mem_addr: u32,
    pub mem_val: u32,      // value loaded (LW) or stored (SW)
    pub mem_prev: u32,     // memory word at addr BEFORE this step
}

/// Aggregate statistics from a [`Cpu::run_to_halt`] run (for the circuit stage).
#[derive(Clone, Debug, Default)]
pub struct RunStats {
    pub cycles: u64,
    pub ecalls: u64,
    pub loads: u64,
    pub stores: u64,
    pub enter_unconstrained: u64,
    pub exit_unconstrained: u64,
    pub halted: bool,
    pub exit_code: u8,
    /// Distinct word addresses accessed (fetch ∪ load ∪ store ∪ hint-write).
    pub touched_words: usize,
    pub touched_min: Option<u32>,
    pub touched_max: Option<u32>,
    /// Distinct word addresses holding state at end (memory footprint).
    pub mem_words: usize,
    pub mem_min: Option<u32>,
    pub mem_max: Option<u32>,
    /// Full per-cycle trace (only if `keep_trace` was set).
    pub trace: Vec<StepRecord>,
}

pub const OPC_OP: u32 = 0b0110011; // R-type
pub const OPC_OPIMM: u32 = 0b0010011; // I-type ALU
pub const OPC_LOAD: u32 = 0b0000011;
pub const OPC_STORE: u32 = 0b0100011;
pub const OPC_BRANCH: u32 = 0b1100011;
pub const OPC_JAL: u32 = 0b1101111;
pub const OPC_JALR: u32 = 0b1100111;
pub const OPC_LUI: u32 = 0b0110111;
pub const OPC_AUIPC: u32 = 0b0010111;
pub const OPC_MISCMEM: u32 = 0b0001111; // FENCE / FENCE.I
pub const OPC_SYSTEM: u32 = 0b1110011; // ECALL / EBREAK
pub const FUNCT7_M: u32 = 0x01; // RV32M (MUL family)

fn sext(val: u32, bits: u32) -> u32 {
    let shift = 32 - bits;
    (((val << shift) as i32) >> shift) as u32
}

/// A tiny word-addressable data memory (byte address -> u32 word at addr&!3).
#[derive(Clone, Default)]
pub struct Memory {
    // word index -> value; word index = byte_addr >> 2.
    words: std::collections::BTreeMap<u32, u32>,
}
impl Memory {
    pub fn load(&self, addr: u32) -> u32 {
        *self.words.get(&(addr >> 2)).unwrap_or(&0)
    }
    pub fn store(&mut self, addr: u32, val: u32) {
        self.words.insert(addr >> 2, val);
    }
    /// Load one byte (zero-extended) from a byte address.
    pub fn load8(&self, addr: u32) -> u32 {
        let w = self.load(addr);
        (w >> (8 * (addr & 3))) & 0xff
    }
    /// Store one byte into the enclosing word, preserving the other three bytes.
    pub fn store8(&mut self, addr: u32, val: u32) {
        let w = self.load(addr);
        let sh = 8 * (addr & 3);
        let masked = (w & !(0xffu32 << sh)) | ((val & 0xff) << sh);
        self.store(addr, masked);
    }

    /// Write an arbitrary byte image at `vaddr` (used by the ELF loader and by
    /// HINT_READ). Handles unaligned start/tail; whole aligned words are written
    /// directly for speed, partial edges via read-modify-write.
    pub fn write_image(&mut self, vaddr: u32, data: &[u8]) {
        let mut i = 0usize;
        let mut addr = vaddr;
        // Leading unaligned bytes.
        while i < data.len() && (addr & 3) != 0 {
            self.store8(addr, data[i] as u32);
            addr = addr.wrapping_add(1);
            i += 1;
        }
        // Aligned whole words.
        while i + 4 <= data.len() {
            let w = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
            self.store(addr, w);
            addr = addr.wrapping_add(4);
            i += 4;
        }
        // Trailing bytes.
        while i < data.len() {
            self.store8(addr, data[i] as u32);
            addr = addr.wrapping_add(1);
            i += 1;
        }
    }

    /// Read `len` bytes starting at `addr` (little-endian words underneath).
    pub fn read_bytes(&self, addr: u32, len: usize) -> Vec<u8> {
        (0..len).map(|k| self.load8(addr.wrapping_add(k as u32)) as u8).collect()
    }

    /// Number of distinct word addresses that hold state, and the min/max byte
    /// address among them (None if empty).
    pub fn footprint(&self) -> (usize, Option<u32>, Option<u32>) {
        let n = self.words.len();
        let min = self.words.keys().next().map(|w| w << 2);
        let max = self.words.keys().next_back().map(|w| w << 2);
        (n, min, max)
    }
}

// ---- SP1 syscall ids (sp1-zkvm/src/syscalls/mod.rs) ------------------------
pub const SYS_HALT: u32 = 0x00;
pub const SYS_WRITE: u32 = 0x02;
pub const SYS_ENTER_UNCONSTRAINED: u32 = 0x03;
pub const SYS_EXIT_UNCONSTRAINED: u32 = 0x04;
pub const SYS_COMMIT: u32 = 0x10;
pub const SYS_COMMIT_DEFERRED_PROOFS: u32 = 0x1a;
pub const SYS_HINT_LEN: u32 = 0xf0;
pub const SYS_HINT_READ: u32 = 0xf1;

/// RISC-V register aliases used by the SP1 ecall ABI.
const REG_T0: usize = 5; // x5 — syscall id / return value
const REG_A0: usize = 10; // x10 — arg 1
const REG_A1: usize = 11; // x11 — arg 2
const REG_A2: usize = 12; // x12 — arg 3 (WRITE only)

// SP1 v6.3.1 offsets all named file descriptors by LOWEST_ALLOWED_FD (=10):
// FD_PUBLIC_VALUES = 3 + 10 = 13, FD_HINT = 4 + 10 = 14
// (sp1-primitives-6.3.1/src/consts.rs:101,118). fd 1/2 remain raw stdout/stderr.
pub const FD_PUBLIC_VALUES: u32 = 13;
pub const FD_HINT: u32 = 14;

pub struct Cpu {
    pub regs: [u32; 32],
    pub pc: u32,
    pub mem: Memory,
    pub base: u32,
    pub program: Vec<u32>,

    // ---- SP1 environment state ----
    /// Set when a HALT syscall executes; the run loop stops.
    pub halted: bool,
    pub exit_code: u8,
    /// Input hint buffers (one bincode blob per `io::read`); `input_idx` is the
    /// next buffer to be consumed by HINT_READ.
    pub inputs: Vec<Vec<u8>>,
    pub input_idx: usize,
    /// Bytes written to FD_PUBLIC_VALUES (fd=3) — the committed public values.
    pub public_values: Vec<u8>,
    /// COMMIT (0x10) digest words, indexed by word_idx.
    pub commit_words: [u32; 8],
    /// COMMIT_DEFERRED_PROOFS (0x1a) words, indexed by idx.
    pub deferred_words: [u32; 8],
    /// stdout/stderr bytes (fd 1/2).
    pub stdout: Vec<u8>,
    /// The (ptr, len) of the most recent HINT_READ, for touched-address tracking.
    pub last_hint_write: Option<(u32, u32)>,
    /// The (buf, len) of the most recent WRITE(fd=FD_PUBLIC_VALUES) — the committed
    /// public-output buffer, used for final-output binding.
    pub last_public_write: Option<(u32, u32)>,
    /// Whether currently inside an ENTER/EXIT_UNCONSTRAINED region.
    pub in_unconstrained: bool,
    // ---- counters ----
    pub ecall_count: u64,
    pub load_count: u64,
    pub store_count: u64,
    pub enter_unconstrained_count: u64,
    pub exit_unconstrained_count: u64,
}

impl Cpu {
    pub fn new(program: Vec<u32>, base: u32) -> Self {
        let mut mem = Memory::default();
        for (i, w) in program.iter().enumerate() {
            mem.store(base.wrapping_add((i as u32) << 2), *w);
        }
        Cpu {
            regs: [0u32; 32],
            pc: base,
            mem,
            base,
            program,
            halted: false,
            exit_code: 0,
            inputs: Vec::new(),
            input_idx: 0,
            public_values: Vec::new(),
            commit_words: [0u32; 8],
            deferred_words: [0u32; 8],
            stdout: Vec::new(),
            last_hint_write: None,
            last_public_write: None,
            in_unconstrained: false,
            ecall_count: 0,
            load_count: 0,
            store_count: 0,
            enter_unconstrained_count: 0,
            exit_unconstrained_count: 0,
        }
    }

    /// Build a CPU from a pre-populated memory image (ELF loader path).
    pub fn from_image(mem: Memory, entry: u32) -> Self {
        let mut cpu = Cpu::new(Vec::new(), entry);
        cpu.mem = mem;
        cpu.pc = entry;
        cpu
    }

    fn fetch(&self, pc: u32) -> u32 {
        self.mem.load(pc)
    }

    /// Service an SP1 `ecall`. Returns the value to place in `t0` (if any).
    /// Reads the syscall id from t0 and args from a0/a1/a2, exactly matching
    /// the SP1 v6.3.1 ABI.
    fn handle_ecall(&mut self) -> Option<u32> {
        self.ecall_count += 1;
        let id = self.regs[REG_T0];
        let a0 = self.regs[REG_A0];
        let a1 = self.regs[REG_A1];
        if std::env::var("RISCV_TRACE_ECALL").is_ok() {
            eprintln!(
                "[ecall #{}] id={:#x} a0={:#x} a1={:#x} a2={:#x} pc={:#x}",
                self.ecall_count, id, a0, a1, self.regs[REG_A2], self.pc
            );
        }
        match id {
            SYS_HALT => {
                self.halted = true;
                self.exit_code = a0 as u8;
                None
            }
            SYS_WRITE => {
                let fd = a0;
                let buf = a1;
                let len = self.regs[REG_A2] as usize;
                let bytes = self.mem.read_bytes(buf, len);
                if fd == FD_PUBLIC_VALUES {
                    self.public_values.extend_from_slice(&bytes);
                    self.last_public_write = Some((buf, len as u32));
                } else if fd == 1 || fd == 2 {
                    self.stdout.extend_from_slice(&bytes);
                } else if fd == FD_HINT {
                    // A guest-produced hint buffer (pushed to the front of the queue).
                    self.inputs.insert(self.input_idx, bytes);
                }
                // else: hook fds — not used by this STF.
                None
            }
            SYS_COMMIT => {
                let idx = a0 as usize;
                if idx < 8 {
                    self.commit_words[idx] = a1;
                }
                None
            }
            SYS_COMMIT_DEFERRED_PROOFS => {
                let idx = a0 as usize;
                if idx < 8 {
                    self.deferred_words[idx] = a1;
                }
                None
            }
            SYS_HINT_LEN => {
                // Peek the length of the next input buffer (does not consume).
                if self.input_idx < self.inputs.len() {
                    Some(self.inputs[self.input_idx].len() as u32)
                } else {
                    Some(u32::MAX) // stream exhausted
                }
            }
            SYS_HINT_READ => {
                // a0=ptr, a1=len : copy the current buffer into guest memory and
                // advance to the next buffer.
                let ptr = a0;
                let len = a1 as usize;
                if self.input_idx < self.inputs.len() {
                    let buf = self.inputs[self.input_idx].clone();
                    let n = len.min(buf.len());
                    self.mem.write_image(ptr, &buf[..n]);
                    self.last_hint_write = Some((ptr, len as u32));
                    self.input_idx += 1;
                }
                None
            }
            SYS_ENTER_UNCONSTRAINED => {
                self.enter_unconstrained_count += 1;
                self.in_unconstrained = true;
                // Executor default: run the unconstrained block (return 1 in t0).
                Some(1)
            }
            SYS_EXIT_UNCONSTRAINED => {
                self.exit_unconstrained_count += 1;
                self.in_unconstrained = false;
                None
            }
            other => panic!("unhandled SP1 syscall id {other:#x} at pc={:#x}", self.pc),
        }
    }

    /// Run for exactly `max_cycles` steps, producing a flat trace. Stops early by
    /// looping on a self-jump if the program terminates before max_cycles (so the
    /// trace has fixed length; padding steps are no-ops that keep state stable).
    pub fn run(&mut self, max_cycles: usize) -> Vec<StepRecord> {
        let mut trace = Vec::with_capacity(max_cycles);
        for c in 0..max_cycles {
            trace.push(self.step(c as u32));
        }
        trace
    }

    /// Run until a HALT syscall (or `max_cycles` guard) is reached, servicing
    /// SP1 syscalls. Does not retain the full StepRecord trace (a real block STF
    /// is far too large to hold in RAM); instead it accumulates the statistics
    /// the circuit stage needs. Returns [`RunStats`].
    ///
    /// If `keep_trace` is true, every StepRecord is also collected into
    /// `stats.trace` (only enable for small programs).
    pub fn run_to_halt(&mut self, max_cycles: u64, keep_trace: bool) -> RunStats {
        use std::collections::HashSet;
        let mut touched: HashSet<u32> = HashSet::new();
        let mut trace: Vec<StepRecord> = Vec::new();
        let mut cycles: u64 = 0;
        while !self.halted && cycles < max_cycles {
            let pc = self.pc;
            touched.insert(pc >> 2); // instruction fetch word
            let rec = self.step(cycles as u32);
            if rec.is_load || rec.is_store {
                touched.insert(rec.mem_addr >> 2);
            }
            if let Some((ptr, len)) = self.last_hint_write.take() {
                let start = ptr >> 2;
                let end = (ptr.wrapping_add(len).wrapping_add(3)) >> 2;
                for w in start..end {
                    touched.insert(w);
                }
            }
            if keep_trace {
                trace.push(rec);
            }
            cycles += 1;
        }
        let (mem_words, mem_min, mem_max) = self.mem.footprint();
        let touched_min = touched.iter().min().map(|w| w << 2);
        let touched_max = touched.iter().max().map(|w| w << 2);
        RunStats {
            cycles,
            ecalls: self.ecall_count,
            loads: self.load_count,
            stores: self.store_count,
            enter_unconstrained: self.enter_unconstrained_count,
            exit_unconstrained: self.exit_unconstrained_count,
            halted: self.halted,
            exit_code: self.exit_code,
            touched_words: touched.len(),
            touched_min,
            touched_max,
            mem_words,
            mem_min,
            mem_max,
            trace,
        }
    }

    /// Chunked trace capture for SEGMENTED proving: advance `start` cycles while
    /// servicing SP1 syscalls (discarding those records, so RAM stays bounded),
    /// snapshot the register file at the chunk boundary, then capture the next
    /// `len` [`StepRecord`]s. Returns (entry_regs, captured_trace, exit_regs).
    /// The emulator's `mem` is left at the post-window state.
    pub fn run_window(&mut self, start: u64, len: usize) -> ([u32; 32], Vec<StepRecord>, [u32; 32]) {
        for c in 0..start {
            self.step(c as u32);
        }
        let entry_regs = self.regs;
        let mut trace = Vec::with_capacity(len);
        for i in 0..len {
            trace.push(self.step((start + i as u64) as u32));
        }
        (entry_regs, trace, self.regs)
    }

    pub fn step(&mut self, cycle: u32) -> StepRecord {
        let pc = self.pc;
        let insn = self.fetch(pc);
        let opcode = insn & 0x7f;
        let rd_idx = (insn >> 7) & 0x1f;
        let funct3 = (insn >> 12) & 0x7;
        let rs1_idx = (insn >> 15) & 0x1f;
        let rs2_idx = (insn >> 20) & 0x1f;
        let funct7 = (insn >> 25) & 0x7f;
        let rs1_val = self.regs[rs1_idx as usize];
        let rs2_val = self.regs[rs2_idx as usize];

        let mut rec = StepRecord {
            cycle,
            pc,
            insn,
            opcode,
            funct3,
            funct7,
            rd_idx,
            rs1_idx,
            rs2_idx,
            rs1_val,
            rs2_val,
            ..Default::default()
        };

        let mut next_pc = pc.wrapping_add(4);
        let mut rd_val = 0u32;

        match opcode {
            OPC_OP => {
                let shamt = rs2_val & 0x1f;
                rd_val = match (funct3, funct7) {
                    (0x0, 0x00) => rs1_val.wrapping_add(rs2_val), // ADD
                    (0x0, 0x20) => rs1_val.wrapping_sub(rs2_val), // SUB
                    (0x4, 0x00) => rs1_val ^ rs2_val,             // XOR
                    (0x6, 0x00) => rs1_val | rs2_val,             // OR
                    (0x7, 0x00) => rs1_val & rs2_val,             // AND
                    (0x1, 0x00) => rs1_val << shamt,              // SLL
                    (0x5, 0x00) => rs1_val >> shamt,              // SRL
                    (0x5, 0x20) => ((rs1_val as i32) >> shamt) as u32, // SRA
                    (0x2, 0x00) => ((rs1_val as i32) < (rs2_val as i32)) as u32, // SLT
                    (0x3, 0x00) => (rs1_val < rs2_val) as u32,    // SLTU
                    // ---- RV32M ----
                    (0x0, FUNCT7_M) => rs1_val.wrapping_mul(rs2_val), // MUL (low 32)
                    (0x1, FUNCT7_M) => {
                        // MULH: high 32 bits of signed*signed product.
                        (((rs1_val as i32 as i64) * (rs2_val as i32 as i64)) >> 32) as u32
                    }
                    (0x2, FUNCT7_M) => {
                        // MULHSU: high 32 bits of signed(rs1)*unsigned(rs2).
                        (((rs1_val as i32 as i64) * (rs2_val as u64 as i64)) >> 32) as u32
                    }
                    (0x3, FUNCT7_M) => {
                        // MULHU: high 32 bits of unsigned product
                        (((rs1_val as u64) * (rs2_val as u64)) >> 32) as u32
                    }
                    (0x4, FUNCT7_M) => {
                        // DIV: signed. div-by-zero -> -1; INT_MIN/-1 overflow -> INT_MIN.
                        let a = rs1_val as i32;
                        let b = rs2_val as i32;
                        if b == 0 {
                            u32::MAX
                        } else if a == i32::MIN && b == -1 {
                            i32::MIN as u32
                        } else {
                            (a / b) as u32
                        }
                    }
                    (0x5, FUNCT7_M) => {
                        // DIVU: unsigned. div-by-zero -> all ones.
                        if rs2_val == 0 { u32::MAX } else { rs1_val / rs2_val }
                    }
                    (0x6, FUNCT7_M) => {
                        // REM: signed. div-by-zero -> rs1; INT_MIN/-1 overflow -> 0.
                        let a = rs1_val as i32;
                        let b = rs2_val as i32;
                        if b == 0 {
                            rs1_val
                        } else if a == i32::MIN && b == -1 {
                            0
                        } else {
                            (a % b) as u32
                        }
                    }
                    (0x7, FUNCT7_M) => {
                        // REMU: unsigned. div-by-zero -> rs1.
                        if rs2_val == 0 { rs1_val } else { rs1_val % rs2_val }
                    }
                    _ => panic!("unsupported R-type funct3={funct3:#x} funct7={funct7:#x}"),
                };
            }
            OPC_OPIMM => {
                let imm = sext(insn >> 20, 12);
                rec.imm = imm;
                let shamt = imm & 0x1f;
                let funct7_i = (insn >> 25) & 0x7f;
                rd_val = match funct3 {
                    0x0 => rs1_val.wrapping_add(imm), // ADDI
                    0x4 => rs1_val ^ imm,             // XORI
                    0x6 => rs1_val | imm,             // ORI
                    0x7 => rs1_val & imm,             // ANDI
                    0x1 => rs1_val << shamt,          // SLLI
                    0x5 if funct7_i == 0x20 => ((rs1_val as i32) >> shamt) as u32, // SRAI
                    0x5 => rs1_val >> shamt,          // SRLI
                    0x2 => ((rs1_val as i32) < (imm as i32)) as u32, // SLTI
                    0x3 => (rs1_val < imm) as u32,    // SLTIU
                    _ => panic!("unsupported I-type funct3={funct3:#x}"),
                };
            }
            OPC_LOAD => {
                // imm[11:0] = insn[31:20]
                let imm = sext(insn >> 20, 12);
                rec.imm = imm;
                let addr = rs1_val.wrapping_add(imm);
                let word = self.mem.load(addr);
                rd_val = match funct3 {
                    0x2 => word,                                   // LW
                    0x4 => self.mem.load8(addr),                   // LBU
                    0x0 => sext(self.mem.load8(addr), 8),          // LB
                    0x5 => {
                        let sh = 8 * (addr & 2);
                        (word >> sh) & 0xffff
                    } // LHU
                    0x1 => {
                        let sh = 8 * (addr & 2);
                        sext((word >> sh) & 0xffff, 16)
                    } // LH
                    _ => panic!("unsupported load funct3={funct3:#x}"),
                };
                rec.is_load = true;
                rec.mem_addr = addr;
                rec.mem_val = rd_val;
                rec.mem_prev = word;
            }
            OPC_STORE => {
                // imm[11:5]=insn[31:25], imm[4:0]=insn[11:7]
                let imm_raw = ((insn >> 25) << 5) | ((insn >> 7) & 0x1f);
                let imm = sext(imm_raw, 12);
                rec.imm = imm;
                let addr = rs1_val.wrapping_add(imm);
                let prev = self.mem.load(addr);
                rec.is_store = true;
                rec.mem_addr = addr;
                rec.mem_prev = prev;
                match funct3 {
                    0x2 => {
                        rec.mem_val = rs2_val;
                        self.mem.store(addr, rs2_val); // SW
                    }
                    0x0 => {
                        self.mem.store8(addr, rs2_val); // SB
                        rec.mem_val = self.mem.load(addr);
                    }
                    0x1 => {
                        // SH
                        let sh = 8 * (addr & 2);
                        let masked = (prev & !(0xffffu32 << sh)) | ((rs2_val & 0xffff) << sh);
                        self.mem.store(addr, masked);
                        rec.mem_val = masked;
                    }
                    _ => panic!("unsupported store funct3={funct3:#x}"),
                }
                // stores write no register
            }
            OPC_BRANCH => {
                // imm[12|10:5]=insn[31|30:25], imm[4:1|11]=insn[11:8|7]
                let imm_raw = (((insn >> 31) & 1) << 12)
                    | (((insn >> 7) & 1) << 11)
                    | (((insn >> 25) & 0x3f) << 5)
                    | (((insn >> 8) & 0xf) << 1);
                let imm = sext(imm_raw, 13);
                rec.imm = imm;
                let taken = match funct3 {
                    0x0 => rs1_val == rs2_val,                       // BEQ
                    0x1 => rs1_val != rs2_val,                       // BNE
                    0x4 => (rs1_val as i32) < (rs2_val as i32),      // BLT
                    0x5 => (rs1_val as i32) >= (rs2_val as i32),     // BGE
                    0x6 => rs1_val < rs2_val,                        // BLTU
                    0x7 => rs1_val >= rs2_val,                       // BGEU
                    _ => panic!("unsupported branch funct3={funct3:#x}"),
                };
                if taken {
                    next_pc = pc.wrapping_add(imm);
                }
                // branches write no register
            }
            OPC_JALR => {
                let imm = sext(insn >> 20, 12);
                rec.imm = imm;
                rd_val = pc.wrapping_add(4);
                next_pc = rs1_val.wrapping_add(imm) & !1;
            }
            OPC_LUI => {
                rd_val = insn & 0xffff_f000;
                rec.imm = rd_val;
            }
            OPC_AUIPC => {
                rd_val = pc.wrapping_add(insn & 0xffff_f000);
                rec.imm = insn & 0xffff_f000;
            }
            OPC_JAL => {
                let imm_raw = (((insn >> 31) & 1) << 20)
                    | (((insn >> 12) & 0xff) << 12)
                    | (((insn >> 20) & 1) << 11)
                    | (((insn >> 21) & 0x3ff) << 1);
                let imm = sext(imm_raw, 21);
                rec.imm = imm;
                rd_val = pc.wrapping_add(4);
                next_pc = pc.wrapping_add(imm);
            }
            // FENCE / FENCE.I: memory-ordering hints; a nop for this single-hart,
            // in-order model (advances pc, writes no register).
            OPC_MISCMEM => {}
            // ECALL / EBREAK / CSR. ECALL (funct3==0, imm==0) is an SP1 syscall
            // and is serviced against the guest ABI. EBREAK and CSR ops are nops
            // for this single-hart model (the STF guest issues neither).
            OPC_SYSTEM => {
                let is_ecall = funct3 == 0 && (insn >> 20) == 0;
                if is_ecall {
                    if let Some(ret) = self.handle_ecall() {
                        // Syscall return value goes to t0 (x5); x0 stays hardwired.
                        self.regs[REG_T0] = ret;
                        rec.rd_idx = REG_T0 as u32;
                        rec.rd_val = ret;
                    }
                }
            }
            _ => panic!("unsupported opcode {opcode:#09b} (insn={insn:#010x}) at pc={pc:#x}"),
        }

        // Track memory-access counts (informational for the circuit stage).
        if rec.is_load {
            self.load_count += 1;
        }
        if rec.is_store {
            self.store_count += 1;
        }

        // Register write (x0 is hardwired to 0; writes to it are discarded).
        // OPC_SYSTEM writes t0 directly above, so it is excluded here.
        let writes_rd = matches!(
            opcode,
            OPC_OP | OPC_OPIMM | OPC_LOAD | OPC_JAL | OPC_JALR | OPC_LUI | OPC_AUIPC
        );
        if writes_rd && rd_idx != 0 {
            self.regs[rd_idx as usize] = rd_val;
        }
        if writes_rd {
            rec.rd_val = rd_val;
        }

        self.pc = next_pc;
        rec.next_pc = next_pc;
        rec
    }
}

// ---- assembler helpers (keep emulator & circuit driving off real encodings) --

pub fn r_type(funct7: u32, rs2: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}
pub fn i_type(imm: u32, rs1: u32, funct3: u32, rd: u32, opcode: u32) -> u32 {
    ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
}
pub fn s_type(imm: u32, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
    let imm = imm & 0xfff;
    ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((imm & 0x1f) << 7) | opcode
}
pub fn b_type(imm: u32, rs2: u32, rs1: u32, funct3: u32, opcode: u32) -> u32 {
    // imm is a signed byte offset, multiple of 2
    (((imm >> 12) & 1) << 31)
        | (((imm >> 5) & 0x3f) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (((imm >> 1) & 0xf) << 8)
        | (((imm >> 11) & 1) << 7)
        | opcode
}
pub fn j_type(imm: u32, rd: u32, opcode: u32) -> u32 {
    (((imm >> 20) & 1) << 31)
        | (((imm >> 1) & 0x3ff) << 21)
        | (((imm >> 11) & 1) << 20)
        | (((imm >> 12) & 0xff) << 12)
        | (rd << 7)
        | opcode
}

// Convenience mnemonics.
pub fn addi(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x0, rd, OPC_OPIMM) }
pub fn xori(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x4, rd, OPC_OPIMM) }
pub fn add(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(0x00, rs2, rs1, 0x0, rd, OPC_OP) }
pub fn sub(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(0x20, rs2, rs1, 0x0, rd, OPC_OP) }
pub fn xor(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(0x00, rs2, rs1, 0x4, rd, OPC_OP) }
pub fn or(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(0x00, rs2, rs1, 0x6, rd, OPC_OP) }
pub fn and(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(0x00, rs2, rs1, 0x7, rd, OPC_OP) }
pub fn sll(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(0x00, rs2, rs1, 0x1, rd, OPC_OP) }
pub fn srl(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(0x00, rs2, rs1, 0x5, rd, OPC_OP) }
pub fn slli(rd: u32, rs1: u32, shamt: u32) -> u32 { i_type(shamt & 0x1f, rs1, 0x1, rd, OPC_OPIMM) }
pub fn srli(rd: u32, rs1: u32, shamt: u32) -> u32 { i_type(shamt & 0x1f, rs1, 0x5, rd, OPC_OPIMM) }
pub fn lw(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x2, rd, OPC_LOAD) }
pub fn lbu(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x4, rd, OPC_LOAD) }
pub fn sw(rs2: u32, rs1: u32, imm: i32) -> u32 { s_type(imm as u32, rs2, rs1, 0x2, OPC_STORE) }
pub fn sb(rs2: u32, rs1: u32, imm: i32) -> u32 { s_type(imm as u32, rs2, rs1, 0x0, OPC_STORE) }
pub fn beq(rs1: u32, rs2: u32, imm: i32) -> u32 { b_type(imm as u32, rs2, rs1, 0x0, OPC_BRANCH) }
pub fn bne(rs1: u32, rs2: u32, imm: i32) -> u32 { b_type(imm as u32, rs2, rs1, 0x1, OPC_BRANCH) }
pub fn blt(rs1: u32, rs2: u32, imm: i32) -> u32 { b_type(imm as u32, rs2, rs1, 0x4, OPC_BRANCH) }
pub fn bge(rs1: u32, rs2: u32, imm: i32) -> u32 { b_type(imm as u32, rs2, rs1, 0x5, OPC_BRANCH) }
pub fn bltu(rs1: u32, rs2: u32, imm: i32) -> u32 { b_type(imm as u32, rs2, rs1, 0x6, OPC_BRANCH) }
pub fn bgeu(rs1: u32, rs2: u32, imm: i32) -> u32 { b_type(imm as u32, rs2, rs1, 0x7, OPC_BRANCH) }
pub fn jal(rd: u32, imm: i32) -> u32 { j_type(imm as u32, rd, OPC_JAL) }
pub fn jalr(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x0, rd, OPC_JALR) }
pub fn lui(rd: u32, imm: u32) -> u32 { (imm & 0xffff_f000) | (rd << 7) | OPC_LUI }
pub fn auipc(rd: u32, imm: u32) -> u32 { (imm & 0xffff_f000) | (rd << 7) | OPC_AUIPC }
pub fn mul(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(FUNCT7_M, rs2, rs1, 0x0, rd, OPC_OP) }
pub fn mulh(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(FUNCT7_M, rs2, rs1, 0x1, rd, OPC_OP) }
pub fn mulhsu(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(FUNCT7_M, rs2, rs1, 0x2, rd, OPC_OP) }
pub fn mulhu(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(FUNCT7_M, rs2, rs1, 0x3, rd, OPC_OP) }
pub fn div(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(FUNCT7_M, rs2, rs1, 0x4, rd, OPC_OP) }
pub fn divu(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(FUNCT7_M, rs2, rs1, 0x5, rd, OPC_OP) }
pub fn rem(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(FUNCT7_M, rs2, rs1, 0x6, rd, OPC_OP) }
pub fn remu(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(FUNCT7_M, rs2, rs1, 0x7, rd, OPC_OP) }
pub fn slt(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(0x00, rs2, rs1, 0x2, rd, OPC_OP) }
pub fn sltu(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(0x00, rs2, rs1, 0x3, rd, OPC_OP) }
pub fn slti(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x2, rd, OPC_OPIMM) }
pub fn sltiu(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x3, rd, OPC_OPIMM) }
pub fn andi(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x7, rd, OPC_OPIMM) }
pub fn ori(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x6, rd, OPC_OPIMM) }
pub fn sra(rd: u32, rs1: u32, rs2: u32) -> u32 { r_type(0x20, rs2, rs1, 0x5, rd, OPC_OP) }
pub fn srai(rd: u32, rs1: u32, shamt: u32) -> u32 { ((0x20 << 5) | (shamt & 0x1f)) << 20 | (rs1 << 15) | (0x5 << 12) | (rd << 7) | OPC_OPIMM }
pub fn lh(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x1, rd, OPC_LOAD) }
pub fn lhu(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x5, rd, OPC_LOAD) }
pub fn lb(rd: u32, rs1: u32, imm: i32) -> u32 { i_type(imm as u32, rs1, 0x0, rd, OPC_LOAD) }
pub fn sh(rs2: u32, rs1: u32, imm: i32) -> u32 { s_type(imm as u32, rs2, rs1, 0x1, OPC_STORE) }

#[cfg(test)]
mod isa_tests {
    use super::*;

    fn run1(prog: Vec<u32>, setup: impl Fn(&mut Cpu)) -> Cpu {
        let mut cpu = Cpu::new(prog, 0);
        setup(&mut cpu);
        // Run enough cycles; last instruction is a self-loop halt.
        cpu.run(64);
        cpu
    }

    #[test]
    fn rv32m_and_compares() {
        // x1=7, x2=6 ; x3=MUL, x4=SLTU(x1,x2), x5=SLT, x6=MULHU big
        let prog = vec![
            addi(1, 0, 7),
            addi(2, 0, 6),
            mul(3, 1, 2),   // 42
            sltu(4, 2, 1),  // 6<7 -> 1
            slt(5, 1, 2),   // 7<6 -> 0
            jal(0, 0),
        ];
        let cpu = run1(prog, |_| {});
        assert_eq!(cpu.regs[3], 42);
        assert_eq!(cpu.regs[4], 1);
        assert_eq!(cpu.regs[5], 0);
    }

    #[test]
    fn mulhu_high_bits() {
        // 0x80000000 * 2 = 0x1_0000_0000 -> low=0, high=1
        let prog = vec![
            lui(1, 0x80000000),
            addi(2, 0, 2),
            mul(3, 1, 2),
            mulhu(4, 1, 2),
            jal(0, 0),
        ];
        let cpu = run1(prog, |_| {});
        assert_eq!(cpu.regs[3], 0);
        assert_eq!(cpu.regs[4], 1);
    }

    #[test]
    fn branches_bne_bltu() {
        // count x1 from 0 to 3 with bne loop
        let prog = vec![
            addi(1, 0, 0),  // 0: i=0
            addi(2, 0, 3),  // 1: n=3
            addi(1, 1, 1),  // 2: i++  (loop head)
            bne(1, 2, -4),  // 3: if i!=n goto 2
            jal(0, 0),      // 4: halt
        ];
        let cpu = run1(prog, |_| {});
        assert_eq!(cpu.regs[1], 3);
    }

    #[test]
    fn jalr_and_bytes() {
        // store byte 0xAB at addr 0x40+1, load it back with lbu
        let prog = vec![
            addi(1, 0, 0xAB),
            addi(2, 0, 0x40),
            sb(1, 2, 1),
            lbu(3, 2, 1),
            jal(0, 0),
        ];
        let cpu = run1(prog, |_| {});
        assert_eq!(cpu.regs[3], 0xAB);
    }
}
