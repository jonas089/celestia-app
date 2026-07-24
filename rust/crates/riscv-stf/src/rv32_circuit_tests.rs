//! FAITHFULNESS tests for the in-circuit RV32I GKR interpreter (`rv32_circuit`)
//! against the native emulator golden (`emulator`), driven through
//! `rv32_prove::prove_rv32_block`.
//!
//! Every test runs the EMULATOR and `prove_rv32_block` on the SAME program and
//! asserts:
//!   (a) the in-circuit post_regs / post_mem EQUAL the emulator's golden, AND
//!       equal hand-computed RV32I-spec expectations,
//!   (b) `proof.verified == true` (GKR self-verify),
//!   (c) any expected public output.
//!
//! `prove_rv32_block` binds the circuit's public output to the emulator golden
//! and internally `assert_is_equal`s in-circuit final state to it, so
//! `verified == true` already certifies (in-circuit state == emulator state);
//! the hand-computed constants below additionally certify the emulator itself is
//! spec-correct for each opcode.

use crate::emulator::*;
use crate::rv32_prove::{prove_rv32_block, Rv32Proof};

const STEPS: usize = 16; // must match rv32_prove::STEPS
const INPUT_ADDR: u32 = 0x100; // must match rv32_prove::INPUT_ADDR

/// Independent (fresh-CPU) emulator golden: reproduce `prove_rv32_block`'s slot
/// layout (input words at INPUT_ADDR, then `pre_mem`) and return the post
/// register file and the post values of exactly those slot addresses (in the
/// same order `prove_rv32_block` reports them).
fn emu_expected(
    program: &[u32],
    input: &[u8],
    pre_regs: &[u32; 32],
    pre_mem: &[(u32, u32)],
) -> ([u32; 32], Vec<(u32, u32)>) {
    let mut input_words: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < input.len() {
        let mut w = 0u32;
        for k in 0..4 {
            if i + k < input.len() {
                w |= (input[i + k] as u32) << (8 * k);
            }
        }
        input_words.push(w);
        i += 4;
    }
    let mut addrs: Vec<u32> = Vec::new();
    for k in 0..input_words.len() {
        addrs.push(INPUT_ADDR + 4 * (k as u32));
    }
    for &(a, _) in pre_mem {
        addrs.push(a);
    }

    let mut cpu = Cpu::new(program.to_vec(), 0);
    cpu.regs = *pre_regs;
    cpu.regs[0] = 0;
    for (k, w) in input_words.iter().enumerate() {
        cpu.mem.store(INPUT_ADDR + 4 * (k as u32), *w);
    }
    for &(a, v) in pre_mem {
        cpu.mem.store(a, v);
    }
    cpu.run(STEPS);
    let post_regs = cpu.regs;
    let post_mem: Vec<(u32, u32)> = addrs.iter().map(|&a| (a, cpu.mem.load(a))).collect();
    (post_regs, post_mem)
}

/// Prove a program, assert GKR verified and that the in-circuit post-state equals
/// the independent emulator golden. Returns the proof for extra per-test asserts.
fn check(name: &str, program: &[u32], input: &[u8], pre_regs: &[u32; 32], pre_mem: &[(u32, u32)]) -> Rv32Proof {
    let (exp_regs, exp_mem) = emu_expected(program, input, pre_regs, pre_mem);
    let p = prove_rv32_block(program, 0, input, pre_regs, pre_mem, STEPS)
        .unwrap_or_else(|e| panic!("[{name}] prove_rv32_block failed: {e}"));
    assert!(p.verified, "[{name}] GKR self-verify failed");
    assert_eq!(p.post_regs, exp_regs, "[{name}] in-circuit post_regs != emulator golden");
    // Compare committed memory as an unordered SET of real (addr,val) slots. The
    // prover lays out committed memory as [STATE region (pre_mem + padding) ..
    // INPUT region], whereas the emulator golden lists [INPUT .. pre_mem]; and the
    // prover pads the STATE region up to STATE_SLOTS with prover-internal dummy
    // slots at word addresses >= 0xE0000 (value 0, meaningless). Neither the order
    // nor the padding is part of the state transition, so normalize both sides
    // (drop padding, sort by address) before comparing.
    let norm = |m: &[(u32, u32)]| -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> = m.iter().copied().filter(|&(a, _)| a < (0xE0000u32 << 2)).collect();
        v.sort_by_key(|&(a, _)| a);
        v
    };
    assert_eq!(norm(&p.post_mem), norm(&exp_mem), "[{name}] in-circuit post_mem != emulator golden");
    println!("[{name}] OK verified num_cycles={}", p.num_cycles);
    p
}

const ZERO_REGS: [u32; 32] = [0u32; 32];

// ---------------------------------------------------------------------------
// 1. Every RV32I base opcode (differential vs emulator + hand-computed)
// ---------------------------------------------------------------------------

#[test]
fn op_lui_auipc() {
    // idx0 pc=0: LUI x1 = 0x12345000
    // idx1 pc=4: AUIPC x2 = pc + 0x1000 = 0x1004
    let prog = vec![lui(1, 0x12345000), auipc(2, 0x1000), jal(0, 0)];
    let p = check("lui_auipc", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[1], 0x12345000, "LUI");
    assert_eq!(p.post_regs[2], 0x0000_1004, "AUIPC = pc(4) + 0x1000");
}

#[test]
fn op_jal_link() {
    // JAL forward jump; link (rd) = pc+4; skipped instruction must not execute.
    let prog = vec![
        jal(1, 8),        // idx0 pc=0: x1 = 4, jump to idx2
        addi(5, 0, 99),   // idx1: SKIPPED
        addi(6, 0, 7),    // idx2 pc=8
        jal(0, 0),        // idx3 halt
    ];
    let p = check("jal_link", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[1], 4, "JAL link = pc+4");
    assert_eq!(p.post_regs[5], 0, "instruction after taken JAL must be skipped");
    assert_eq!(p.post_regs[6], 7);
}

#[test]
fn op_jalr_link_and_lsb_clear() {
    // JALR: link = pc+4, target = (rs1 + imm) with LSB forced to 0.
    let prog = vec![
        addi(10, 0, 21),  // idx0: x10 = 21 (ODD target => must be cleared to 20)
        addi(11, 0, 0),   // idx1
        jalr(1, 10, 0),   // idx2 pc=8: x1 = 12, target = 21 & ~1 = 20 = idx5
        addi(11, 0, 111), // idx3: SKIPPED
        addi(11, 0, 222), // idx4: SKIPPED
        addi(12, 0, 7),   // idx5 pc=20
        jal(0, 0),        // idx6 halt
    ];
    let p = check("jalr_link_lsb", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[1], 12, "JALR link = pc+4");
    assert_eq!(p.post_regs[10], 21);
    assert_eq!(p.post_regs[11], 0, "JALR must clear LSB (21 -> 20), skipping idx3/idx4");
    assert_eq!(p.post_regs[12], 7);
}

#[test]
fn op_branches_beq_bne() {
    // BEQ taken (forward, skip), BEQ not-taken (fall through).
    let prog = vec![
        addi(1, 0, 5),    // idx0
        addi(2, 0, 5),    // idx1
        beq(1, 2, 8),     // idx2 pc=8: taken -> idx4
        addi(3, 0, 50),   // idx3: SKIPPED
        addi(4, 0, 5),    // idx4
        addi(5, 0, 6),    // idx5
        beq(4, 5, 8),     // idx6 pc=24: NOT taken (5!=6) -> idx7
        addi(6, 0, 60),   // idx7: executed
        jal(0, 0),        // idx8 halt
    ];
    let p = check("beq_bne", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[3], 0, "taken BEQ must skip next instruction");
    assert_eq!(p.post_regs[6], 60, "not-taken BEQ must fall through");
}

#[test]
fn op_branch_backward_loop_bne() {
    // BNE backward loop (taken repeatedly, then not-taken to exit).
    let prog = vec![
        addi(1, 0, 0),    // idx0: i=0
        addi(2, 0, 3),    // idx1: n=3
        addi(1, 1, 1),    // idx2: i++ (loop head, pc=8)
        bne(1, 2, -4),    // idx3: if i!=n goto idx2
        jal(0, 0),        // idx4 halt
    ];
    let p = check("bne_loop", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[1], 3, "backward BNE loop must count to 3");
}

#[test]
fn op_branch_signed_blt_bltu() {
    // Signed vs unsigned branch on a high-bit-set (negative) operand.
    // x1 = -1 (0xffffffff), x2 = 1.
    //   BLT  (signed):   -1 < 1        => taken
    //   BLTU (unsigned): 0xffffffff<1  => NOT taken
    let prog = vec![
        addi(1, 0, -1),   // idx0
        addi(2, 0, 1),    // idx1
        addi(3, 0, 0),    // idx2
        blt(1, 2, 8),     // idx3 pc=12: taken -> idx5
        addi(3, 0, 100),  // idx4: SKIPPED
        addi(4, 0, 0),    // idx5
        bltu(1, 2, 8),    // idx6 pc=24: NOT taken -> idx7
        addi(4, 0, 200),  // idx7: executed
        jal(0, 0),        // idx8 halt
    ];
    let p = check("blt_bltu", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[3], 0, "BLT signed(-1<1) must be taken");
    assert_eq!(p.post_regs[4], 200, "BLTU unsigned(0xffffffff<1) must NOT be taken");
}

#[test]
fn op_branch_signed_bge_bgeu() {
    // x1 = -1, x2 = 1.
    //   BGE  (signed):   -1 >= 1        => NOT taken
    //   BGEU (unsigned): 0xffffffff>=1  => taken
    let prog = vec![
        addi(1, 0, -1),   // idx0
        addi(2, 0, 1),    // idx1
        bge(1, 2, 8),     // idx2 pc=8: NOT taken -> idx3
        addi(3, 0, 1),    // idx3: executed
        bgeu(1, 2, 8),    // idx4 pc=16: taken -> idx6
        addi(4, 0, 1),    // idx5: SKIPPED
        jal(0, 0),        // idx6 halt
    ];
    let p = check("bge_bgeu", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[3], 1, "BGE signed(-1>=1) must NOT be taken");
    assert_eq!(p.post_regs[4], 0, "BGEU unsigned(0xffffffff>=1) must be taken");
}

#[test]
fn op_alu_immediate_group() {
    // Group all I-type ALU ops into one proof.
    // x1 = 0x123, x2 = -1 (0xffffffff).
    let prog = vec![
        addi(1, 0, 0x123), // idx0
        addi(2, 0, -1),    // idx1
        xori(3, 1, 0xf0),  // 0x123 ^ 0x0f0 = 0x1D3
        ori(4, 1, 0xf0),   // 0x123 | 0x0f0 = 0x1F3
        andi(5, 1, 0xf0),  // 0x123 & 0x0f0 = 0x020
        slli(6, 1, 4),     // 0x123 << 4    = 0x1230
        srli(7, 1, 4),     // 0x123 >> 4    = 0x012
        srai(8, 2, 4),     // (-1) >>a 4    = 0xffffffff
        slti(9, 2, 1),     // signed(-1 < 1)     = 1
        sltiu(10, 2, 1),   // unsigned(0xffffffff < 1) = 0
        addi(11, 1, -1),   // 0x123 - 1     = 0x122
        jal(0, 0),
    ];
    let p = check("alu_imm", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[1], 0x123);
    assert_eq!(p.post_regs[2], 0xffff_ffff);
    assert_eq!(p.post_regs[3], 0x1D3, "XORI");
    assert_eq!(p.post_regs[4], 0x1F3, "ORI");
    assert_eq!(p.post_regs[5], 0x020, "ANDI");
    assert_eq!(p.post_regs[6], 0x1230, "SLLI");
    assert_eq!(p.post_regs[7], 0x012, "SRLI");
    assert_eq!(p.post_regs[8], 0xffff_ffff, "SRAI of -1");
    assert_eq!(p.post_regs[9], 1, "SLTI signed");
    assert_eq!(p.post_regs[10], 0, "SLTIU unsigned");
    assert_eq!(p.post_regs[11], 0x122, "ADDI negative imm");
}

#[test]
fn op_alu_register_group() {
    // Group all R-type ALU ops into one proof.
    // x1 = 12, x2 = 5, x3 = -3 (0xfffffffd).
    let prog = vec![
        addi(1, 0, 12),  // idx0
        addi(2, 0, 5),   // idx1
        addi(3, 0, -3),  // idx2
        add(4, 1, 2),    // 17
        sub(5, 1, 2),    // 7
        xor(6, 1, 2),    // 12 ^ 5 = 9
        or(7, 1, 2),     // 12 | 5 = 13
        and(8, 1, 2),    // 12 & 5 = 4
        sll(9, 1, 2),    // 12 << (5 & 31) = 384
        srl(10, 1, 2),   // 12 >> 5 = 0
        sra(11, 3, 2),   // (-3) >>a 5 = -1 = 0xffffffff
        slt(12, 3, 1),   // signed(-3 < 12) = 1
        sltu(13, 3, 1),  // unsigned(0xfffffffd < 12) = 0
        jal(0, 0),
    ];
    let p = check("alu_reg", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[4], 17, "ADD");
    assert_eq!(p.post_regs[5], 7, "SUB");
    assert_eq!(p.post_regs[6], 9, "XOR");
    assert_eq!(p.post_regs[7], 13, "OR");
    assert_eq!(p.post_regs[8], 4, "AND");
    assert_eq!(p.post_regs[9], 384, "SLL");
    assert_eq!(p.post_regs[10], 0, "SRL");
    assert_eq!(p.post_regs[11], 0xffff_ffff, "SRA of negative");
    assert_eq!(p.post_regs[12], 1, "SLT signed");
    assert_eq!(p.post_regs[13], 0, "SLTU unsigned");
}

#[test]
fn op_shift_negative_sra_vs_srl() {
    // SRA vs SRL of a negative value; SLL for completeness. x1 = -16.
    let prog = vec![
        addi(1, 0, -16), // 0xfffffff0
        addi(2, 0, 2),
        srl(3, 1, 2),    // logical  0xfffffff0 >> 2 = 0x3ffffffc
        sra(4, 1, 2),    // arith    (-16) >> 2      = 0xfffffffc (-4)
        srli(5, 1, 2),   // logical                  = 0x3ffffffc
        srai(6, 1, 2),   // arith                    = 0xfffffffc
        sll(7, 1, 2),    // 0xfffffff0 << 2          = 0xffffffc0
        slli(8, 1, 2),   //                          = 0xffffffc0
        jal(0, 0),
    ];
    let p = check("shift_neg", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[3], 0x3fff_fffc, "SRL negative = logical");
    assert_eq!(p.post_regs[4], 0xffff_fffc, "SRA negative = arithmetic");
    assert_eq!(p.post_regs[5], 0x3fff_fffc, "SRLI negative");
    assert_eq!(p.post_regs[6], 0xffff_fffc, "SRAI negative");
    assert_eq!(p.post_regs[7], 0xffff_ffc0, "SLL");
    assert_eq!(p.post_regs[8], 0xffff_ffc0, "SLLI");
}

#[test]
fn op_loads_all_widths_sign_zero_extend() {
    // Sub-word load extension: LB/LBU/LH/LHU on high-bit-set byte & half, LW.
    // mem[0x200] = 0x8FFF0080 (byte0 = 0x80, low half = 0x0080, high half = 0x8FFF).
    let pre_mem = vec![(0x200u32, 0x8FFF_0080u32)];
    let prog = vec![
        lb(1, 0, 0x200),   // byte0 0x80 sign-ext  = 0xffffff80
        lbu(2, 0, 0x200),  // byte0 0x80 zero-ext  = 0x00000080
        lh(3, 0, 0x202),   // high half 0x8FFF sext= 0xffff8fff
        lhu(4, 0, 0x202),  // high half 0x8FFF zext= 0x00008fff
        lw(5, 0, 0x200),   // whole word           = 0x8fff0080
        lh(6, 0, 0x200),   // low half 0x0080 sext = 0x00000080
        jal(0, 0),
    ];
    let p = check("loads", &prog, &[], &ZERO_REGS, &pre_mem);
    assert_eq!(p.post_regs[1], 0xffff_ff80, "LB sign-extends");
    assert_eq!(p.post_regs[2], 0x0000_0080, "LBU zero-extends");
    assert_eq!(p.post_regs[3], 0xffff_8fff, "LH sign-extends high half");
    assert_eq!(p.post_regs[4], 0x0000_8fff, "LHU zero-extends high half");
    assert_eq!(p.post_regs[5], 0x8fff_0080, "LW");
    assert_eq!(p.post_regs[6], 0x0000_0080, "LH low half (positive)");
}

#[test]
fn op_stores_lane_isolation_sb_sh_sw() {
    // Sub-word store lane isolation: SB writes only the addressed byte, SH only
    // the addressed half; other bytes stay intact. mem[0x200] starts 0xAABBCCDD.
    let pre_mem = vec![(0x200u32, 0xAABB_CCDDu32)];
    let prog = vec![
        addi(1, 0, 0x11),  // byte value
        addi(2, 0, 0x7AB), // half value
        sb(1, 0, 0x200),   // byte0 DD->11        => 0xAABBCC11
        sh(2, 0, 0x202),   // high half AABB->07AB => 0x07ABCC11
        lw(3, 0, 0x200),   // read back           => 0x07ABCC11
        jal(0, 0),
    ];
    let p = check("stores", &prog, &[], &ZERO_REGS, &pre_mem);
    assert_eq!(p.post_regs[3], 0x07AB_CC11, "SB/SH lane isolation (byte1 0xCC intact)");
    assert!(
        p.post_mem.iter().any(|&(a, v)| a == 0x200 && v == 0x07AB_CC11),
        "post_mem[0x200] must reflect isolated SB+SH"
    );
}

#[test]
fn op_sw_output_binding() {
    // SW to a declared result slot and confirm the public `output` bytes.
    let pre_mem = vec![(0x200u32, 0u32)];
    let prog = vec![
        addi(1, 0, 0x2A), // 42
        sw(1, 0, 0x200),
        jal(0, 0),
    ];
    let p = check("sw_output", &prog, &[], &ZERO_REGS, &pre_mem);
    assert_eq!(p.output, 42u32.to_le_bytes().to_vec(), "public output = stored word");
    assert!(p.post_mem.iter().any(|&(a, v)| a == 0x200 && v == 42));
}

// ---------------------------------------------------------------------------
// 2. Machinery / edge cases
// ---------------------------------------------------------------------------

#[test]
fn x0_hardwired_zero() {
    // Writes to x0 are discarded; x0 always reads 0.
    let prog = vec![
        addi(0, 0, 5),        // discarded
        add(0, 0, 0),         // discarded
        lui(0, 0x12345000),   // discarded
        addi(1, 0, 7),        // x1 = x0(0) + 7 = 7 (proves x0 still 0)
        jal(0, 0),
    ];
    let p = check("x0_hardwired", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[0], 0, "x0 must stay 0 despite writes");
    assert_eq!(p.post_regs[1], 7, "reading x0 yields 0");
}

#[test]
fn memory_aliasing_consistency() {
    // Two accesses to the same declared address stay consistent across a store.
    let pre_mem = vec![(0x200u32, 100u32)];
    let prog = vec![
        lw(1, 0, 0x200),  // x1 = 100
        addi(3, 1, 1),    // x3 = 101
        sw(3, 0, 0x200),  // mem[0x200] = 101
        lw(2, 0, 0x200),  // x2 = 101 (must see the store)
        jal(0, 0),
    ];
    let p = check("mem_alias", &prog, &[], &ZERO_REGS, &pre_mem);
    assert_eq!(p.post_regs[1], 100);
    assert_eq!(p.post_regs[2], 101, "load after store to same addr must be consistent");
    assert!(p.post_mem.iter().any(|&(a, v)| a == 0x200 && v == 101));
}

#[test]
fn halt_self_loop_before_steps() {
    // A program shorter than PROG_LEN that halts (self-loop) well before STEPS
    // still verifies with the correct frozen post-state.
    let prog = vec![addi(1, 0, 5), jal(0, 0)];
    let p = check("halt_selfloop", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[1], 5);
    assert_eq!(p.num_cycles, 2, "self-loop reached at cycle 2 (idx0 exec, idx1 self-loop)");
}

#[test]
fn fence_is_nop_state_uncorrupted() {
    // FENCE (opcode MISCMEM) is a no-op in both emulator and circuit; state must
    // be unchanged around it. Raw encoding: fence pred=succ=0 -> 0x0000000F.
    let prog = vec![addi(1, 0, 7), 0x0000_000F, addi(2, 0, 9), jal(0, 0)];
    let p = check("fence_nop", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[1], 7, "FENCE must not corrupt state");
    assert_eq!(p.post_regs[2], 9);
}

#[test]
fn ecall_halt_state_uncorrupted() {
    // ECALL (opcode SYSTEM, 0x00000073) with t0(x5)=0 => SP1 SYS_HALT in the
    // emulator (no register/memory mutation); the circuit treats SYSTEM as an
    // idle step. Either way regs/mem must be uncorrupted. (ECALL is supported by
    // the emulator; the circuit does not model syscalls, only that they leave the
    // committed state untouched.)
    let prog = vec![addi(1, 0, 7), 0x0000_0073, addi(2, 0, 9), jal(0, 0)];
    let p = check("ecall_halt", &prog, &[], &ZERO_REGS, &[]);
    assert_eq!(p.post_regs[1], 7, "ECALL(HALT) must not corrupt state");
    assert_eq!(p.post_regs[2], 9);
}

#[test]
fn store_to_undeclared_slot_is_rejected() {
    // SOUNDNESS: a store to an address that is NOT a committed memory slot must
    // be rejected by prove_rv32_block (the circuit would silently drop it and the
    // masked divergence would still verify). 0x900 is neither an input word nor a
    // pre_mem slot.
    let prog = vec![addi(1, 0, 42), sw(1, 0, 0x900), jal(0, 0)];
    let res = prove_rv32_block(&prog, 0, &[], &ZERO_REGS, &[], STEPS);
    assert!(
        res.is_err(),
        "store to undeclared slot must be rejected, got Ok (soundness hole)"
    );
    let msg = res.err().unwrap();
    assert!(msg.contains("undeclared"), "unexpected error message: {msg}");
    println!("[undeclared_store] correctly rejected: {msg}");
}

#[test]
fn pc_out_of_range_no_false_verify() {
    // PC leaving [0, PROG_LEN): a program with no self-loop runs off the end into
    // the zero-padded (illegal, opcode 0) region. The emulator (golden reference,
    // per the RV32I spec where the all-zero encoding is an illegal instruction)
    // TRAPS on the illegal fetch, so `prove_rv32_block` never returns a proof:
    // there is no path to a false `verified == true` for an out-of-range PC. We
    // assert the golden traps (catch_unwind) rather than fabricating a result.
    // The complementary "well-defined halt" behavior is covered by
    // `halt_self_loop_before_steps` — programs must terminate via a self-loop.
    let prog = vec![addi(1, 0, 1), addi(2, 0, 2)]; // no self-loop -> runs off end
    let regs = ZERO_REGS;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prove_rv32_block(&prog, 0, &[], &regs, &[], STEPS)
    }));
    match outcome {
        Err(_) => println!("[pc_out_of_range] golden traps illegal fetch (no false verify)"),
        Ok(Ok(p)) => {
            // If the golden ever DID return (it does not today), it must at least
            // not be a false-verified wrong result: verify against a fresh golden.
            let (exp_regs, exp_mem) = emu_expected(&prog, &[], &regs, &[]);
            assert!(p.verified, "out-of-range run must not falsely fail-open");
            assert_eq!(p.post_regs, exp_regs);
            assert_eq!(p.post_mem, exp_mem);
        }
        Ok(Err(e)) => println!("[pc_out_of_range] rejected with Err: {e}"),
    }
}

// ---------------------------------------------------------------------------
// 3. Differential fuzzer
// ---------------------------------------------------------------------------

/// Deterministic 64-bit LCG (constant seed; NO entropy/time). PCG-style multiplier.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn range(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

#[test]
fn differential_fuzzer() {
    // Generate many random VALID RV32I programs within the bounds and diff the
    // emulator against prove_rv32_block for each. Programs are constrained to be
    // well-formed & terminating: register/immediate ALU ops plus loads/stores to
    // ONLY declared slots, no branches/jumps except a final self-loop (so the PC
    // never leaves range and never stores out of bounds). Broad, cheap coverage.
    const N_PROGRAMS: usize = 60;
    let mut rng = Lcg(0x1234_5678_9abc_def0);

    // Declared, in-range memory addresses (word-aligned): two input words at
    // 0x100/0x104 and one pre_mem result slot at 0x200. Sub-word ops may target
    // byte offsets within the 0x200 word.
    let load_addrs = [0x100i32, 0x104, 0x200];

    let mut ok = 0usize;
    for pi in 0..N_PROGRAMS {
        // Random pre-state.
        let mut pre_regs = [0u32; 32];
        for r in pre_regs.iter_mut().take(11).skip(1) {
            *r = rng.next_u32();
        }
        pre_regs[0] = 0;
        let mut input = [0u8; 8];
        for b in input.iter_mut() {
            *b = (rng.range(256)) as u8;
        }
        let pre_mem = vec![(0x200u32, rng.next_u32())];

        // Random instruction stream (no control flow); terminate with self-loop.
        let nins = 4 + rng.range(9) as usize; // 4..=12 real instructions
        let mut prog: Vec<u32> = Vec::with_capacity(nins + 1);
        for _ in 0..nins {
            let rd = 1 + rng.range(10); // x1..x10 (avoid clobbering x0 semantics test)
            let rs1 = rng.range(11); // x0..x10
            let rs2 = rng.range(11);
            let imm12 = (rng.range(4096) as i32) - 2048; // signed 12-bit
            let shamt = rng.range(32);
            let insn = match rng.range(24) {
                0 => add(rd, rs1, rs2),
                1 => sub(rd, rs1, rs2),
                2 => xor(rd, rs1, rs2),
                3 => or(rd, rs1, rs2),
                4 => and(rd, rs1, rs2),
                5 => sll(rd, rs1, rs2),
                6 => srl(rd, rs1, rs2),
                7 => sra(rd, rs1, rs2),
                8 => slt(rd, rs1, rs2),
                9 => sltu(rd, rs1, rs2),
                10 => addi(rd, rs1, imm12),
                11 => xori(rd, rs1, imm12),
                12 => ori(rd, rs1, imm12),
                13 => andi(rd, rs1, imm12),
                14 => slli(rd, rs1, shamt),
                15 => srli(rd, rs1, shamt),
                16 => srai(rd, rs1, shamt),
                17 => slti(rd, rs1, imm12),
                18 => sltiu(rd, rs1, imm12),
                19 => lw(rd, 0, load_addrs[rng.range(3) as usize]),
                20 => lbu(rd, 0, load_addrs[rng.range(3) as usize] + rng.range(4) as i32),
                21 => lh(rd, 0, load_addrs[rng.range(3) as usize] + 2 * rng.range(2) as i32),
                22 => sw(rs2, 0, 0x200), // declared
                23 => sb(rs2, 0, 0x200 + rng.range(4) as i32), // declared word
                _ => sh(rs2, 0, 0x200 + 2 * rng.range(2) as i32), // declared word
            };
            prog.push(insn);
        }
        prog.push(jal(0, 0)); // self-loop halt

        let name = format!("fuzz#{pi}");
        check(&name, &prog, &input, &pre_regs, &pre_mem);
        ok += 1;
    }
    println!("[fuzzer] {ok}/{N_PROGRAMS} random programs matched emulator + verified");
    assert_eq!(ok, N_PROGRAMS);
}
