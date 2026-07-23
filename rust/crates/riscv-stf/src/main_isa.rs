//! Per-opcode verification driver for the RV32IM CPU-verifier (`circuit_gp`,
//! GF(2^128) grand-product offline memory checking).
//!
//! For each case it assembles a tiny RV32IM program that exercises one or more of
//! the newly-supported opcodes (with edge cases), runs the emulator, proves the
//! trace through `circuit_gp` over Expander GKR with rsema1d as the input PCS, and
//! asserts:
//!   (a) the Expander verifier ACCEPTS the honest proof (and claimed output == 0),
//!   (b) the in-circuit result register(s) == the emulator's result (the circuit
//!       INDEPENDENTLY recomputes computed_rd and asserts it equals the committed
//!       rd_val for EVERY reg-writing instruction, so acceptance already proves
//!       circuit-result == emulator-result bit-for-bit for each opcode; the three
//!       public outputs additionally cross-check final memory-checked values),
//!   (c) the GKR PCS commitment == an independent Go/DA rsema1d Encode of the same
//!       committed rows.
//!
//! Same proving spine as `main_gp` / `prove_evm`.

mod circuit_gp;
mod emulator;
mod gf128;

use arith::{Field, SimdField};
use circuit_gp as ckt;
use circuit_gp::{Hints, NOUT, NREG, SLOTS, XLEN};
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use polynomials::MultiLinearPoly;
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};
use std::collections::HashMap;

fn hex_str(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn fs_challenges(commit: &[u8; 32]) -> (u128, u128) {
    let mix = |mut z: u64| -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    let derive = |tag: u64| -> u128 {
        let mut lo: u64 = 0xcbf29ce484222325 ^ tag;
        let mut hi: u64 = 0x9e3779b97f4a7c15 ^ mix(tag);
        for (i, &b) in commit.iter().enumerate() {
            lo = (lo ^ b as u64).wrapping_mul(0x100000001b3);
            hi = (hi ^ (b as u64).rotate_left((i as u32 & 63) + 1)).wrapping_mul(0xff51afd7ed558ccd);
        }
        (((mix(hi) as u128) << 64) | (mix(lo) as u128)) | 1
    };
    (derive(0xA1), derive(0xB2))
}

fn writes_reg(opcode: u32) -> bool {
    matches!(opcode,
        emulator::OPC_OP | emulator::OPC_OPIMM | emulator::OPC_LOAD
        | emulator::OPC_JAL | emulator::OPC_JALR | emulator::OPC_LUI | emulator::OPC_AUIPC)
}

fn div_hints(rec: &emulator::StepRecord) -> (u32, u32) {
    if rec.opcode != emulator::OPC_OP || rec.funct7 != emulator::FUNCT7_M {
        return (0, 0);
    }
    let a = rec.rs1_val;
    let b = rec.rs2_val;
    match rec.funct3 {
        0x4 | 0x6 => {
            let (ai, bi) = (a as i32, b as i32);
            if bi == 0 { (u32::MAX, a) }
            else if ai == i32::MIN && bi == -1 { (i32::MIN as u32, 0) }
            else { ((ai / bi) as u32, (ai % bi) as u32) }
        }
        0x5 | 0x7 => { if b == 0 { (u32::MAX, a) } else { (a / b, a % b) } }
        _ => (0, 0),
    }
}

fn replay_hints(trace: &[emulator::StepRecord], mem_addrs: &[u32], mem_init: &[u32]) -> Hints {
    let ncyc = trace.len();
    let nmem = mem_addrs.len();
    let mut last_ts: HashMap<u32, u32> = HashMap::new();
    let mut last_val: HashMap<u32, u32> = HashMap::new();
    for r in 0..NREG as u32 { last_ts.insert(r, 0); last_val.insert(r, 0); }
    for k in 0..nmem { last_ts.insert(mem_addrs[k], 0); last_val.insert(mem_addrs[k], mem_init[k]); }

    let mut tprev = vec![[0u32; SLOTS]; ncyc];
    let mut vold_c = vec![0u32; ncyc];
    let mut vold_d = vec![0u32; ncyc];
    let mut div_q = vec![0u32; ncyc];
    let mut div_r = vec![0u32; ncyc];

    for (c, rec) in trace.iter().enumerate() {
        let write_enable = writes_reg(rec.opcode);
        let (q, r) = div_hints(rec);
        div_q[c] = q;
        div_r[c] = r;
        for slot in 0..SLOTS {
            let now_ts = (c * SLOTS + slot + 1) as u32;
            let word = rec.mem_addr & !3;
            let (addr, rval, wval, active): (u32, u32, u32, bool) = match slot {
                0 => (rec.rs1_idx, rec.rs1_val, rec.rs1_val, true),
                1 => (rec.rs2_idx, rec.rs2_val, rec.rs2_val, true),
                2 => {
                    if rec.is_load { (word, rec.mem_prev, rec.mem_prev, true) }
                    else if rec.is_store { (word, rec.mem_prev, rec.mem_val, true) }
                    else { (0, 0, 0, false) }
                }
                _ => {
                    if write_enable && rec.rd_idx != 0 {
                        let old = *last_val.get(&rec.rd_idx).unwrap_or(&0);
                        (rec.rd_idx, old, rec.rd_val, true)
                    } else { (0, 0, 0, false) }
                }
            };
            let _ = now_ts;
            if active {
                let tp = *last_ts.get(&addr).unwrap_or(&0);
                tprev[c][slot] = tp;
                if slot == 2 { vold_c[c] = rval; }
                if slot == 3 { vold_d[c] = rval; }
                let cur = *last_val.get(&addr).unwrap_or(&0);
                assert_eq!(cur, rval, "replay read mismatch c={c} slot={slot} addr={addr:#x}");
                last_ts.insert(addr, now_ts);
                last_val.insert(addr, wval);
            }
        }
    }

    let naddr = NREG + nmem;
    let mut fin_val = vec![0u32; naddr];
    let mut fin_ts = vec![0u32; naddr];
    for r in 0..NREG { fin_val[r] = last_val[&(r as u32)]; fin_ts[r] = last_ts[&(r as u32)]; }
    for k in 0..nmem {
        fin_val[NREG + k] = last_val[&mem_addrs[k]];
        fin_ts[NREG + k] = last_ts[&mem_addrs[k]];
    }
    Hints { tprev, vold_c, vold_d, div_q, div_r, fin_val, fin_ts }
}

fn da_commit(input_vals: &[gf2::GF2x8], num_vars: usize) -> [u8; 32] {
    const NUM_SYMBOLS: usize = 32;
    const ROW_BYTES: usize = 64;
    let k: u32 = 1u32 << (num_vars - 2);
    let n: u32 = k;
    let pack: Vec<[u8; 8]> = input_vals
        .iter()
        .map(|e| { let l = e.unpack(); let mut b = [0u8; 8]; for s in 0..8 { b[s] = l[s].v & 1; } b })
        .collect();
    let mut rows = vec![vec![0u8; ROW_BYTES]; (k + n) as usize];
    for j in 0..(k as usize) {
        for i in 0..NUM_SYMBOLS { let a = j * NUM_SYMBOLS + i; rows[j][i] = pack[a >> 3][a & 7]; }
    }
    let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (c, _h) = rsema1d_sys::commit(k, n, &row_refs).unwrap();
    c
}

/// A tiny opcode case: `prog` runs to a self-loop halt within `ncyc` cycles; the
/// three `out_regs` register indices are exposed as public outputs and checked
/// against `expect` (also cross-checked against the emulator's registers).
struct Case {
    name: &'static str,
    prog: Vec<u32>,
    ncyc: usize,
    out_regs: [usize; NOUT],
    expect: [u32; NOUT],
    /// human-readable per-opcode notes printed on success.
    notes: Vec<String>,
}

fn prove_case(case: &Case) -> bool {
    let prog = case.prog.clone();
    let prog_len = prog.len();
    let ncyc = case.ncyc;

    // Emulate.
    let mut cpu = emulator::Cpu::new(prog.clone(), ckt::BASE);
    let trace = cpu.run(ncyc);

    // Collect touched word addresses.
    let mut set = std::collections::BTreeSet::<u32>::new();
    for r in &trace {
        if r.is_load || r.is_store { set.insert(r.mem_addr & !3); }
    }
    let mem_addrs: Vec<u32> = set.into_iter().collect();
    // initial memory word at each touched address = the earliest mem_prev seen.
    let mut mem_init_map: HashMap<u32, u32> = HashMap::new();
    for r in &trace {
        if r.is_load || r.is_store {
            let w = r.mem_addr & !3;
            mem_init_map.entry(w).or_insert(r.mem_prev);
        }
    }
    let mem_init: Vec<u32> = mem_addrs.iter().map(|a| *mem_init_map.get(a).unwrap()).collect();
    let nmem = mem_addrs.len();

    // Expected outputs = final register values (must equal case.expect).
    let out_vals: [u32; NOUT] = [
        cpu.regs[case.out_regs[0]],
        cpu.regs[case.out_regs[1]],
        cpu.regs[case.out_regs[2]],
    ];
    if out_vals != case.expect {
        println!("  [FAIL] emulator result {out_vals:?} != expected {:?}", case.expect);
        return false;
    }

    unsafe {
        ckt::NCYC = ncyc;
        ckt::NMEM = nmem;
        ckt::PROG_LEN = prog_len;
        ckt::PROGRAM = prog.clone();
        ckt::MEM_ADDRS = mem_addrs.clone();
        ckt::MEM_INIT = mem_init.clone();
        ckt::OUT_IDX = case.out_regs;
    }

    let hints = replay_hints(&trace, &mem_addrs, &mem_init);

    let CompileResult { witness_solver, layered_circuit } =
        compile(&ckt::template(), CompileOptions::default()).unwrap();

    // FS bind: solve with challenges=0 -> commitment -> (alpha,beta).
    let a0 = ckt::build_assignment(&trace, &hints, 0u128, 0u128, &out_vals);
    let w0 = witness_solver.solve_witnesses(&vec![a0; 8]).unwrap();
    let (si0, _sp0) = w0.to_simd::<gf2::GF2x8>();
    let ec0 = layered_circuit.export_to_expander_flatten();
    let num_vars = ec0.log_input_size();
    let c_trace = da_commit(&si0, num_vars);
    let (alpha, beta) = fs_challenges(&c_trace);

    // Real solve.
    let assignment = ckt::build_assignment(&trace, &hints, alpha, beta, &out_vals);
    let witness = witness_solver.solve_witnesses(&vec![assignment.clone(); 8]).unwrap();
    let res = layered_circuit.run(&witness);
    let self_ok = res.iter().all(|x| *x);
    if !self_ok {
        println!("  [FAIL] layered self-eval failed (a constraint is violated by the honest trace)");
        return false;
    }

    let mut ec = layered_circuit.export_to_expander_flatten();
    let (si, sp) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = si.clone();
    ec.public_input = sp.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let gate_entries: usize = ec.layers.iter().map(|l| l.mul.len() + l.add.len()).sum();

    let c_real = da_commit(&input_vals, num_vars);
    if c_real != c_trace {
        println!("  [FAIL] commitment changed with challenges (binding broken)");
        return false;
    }

    let mpi = MPIConfig::prover_new(None, None);
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let ok = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v);
    let verified = ok && claimed_v.is_zero();

    let params = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi.world_size());
    let mut spad = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
    let input_poly = MultiLinearPoly::new(input_vals.clone());
    let c_prover = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &input_poly, &mut spad).unwrap();
    let commit_ok = c_prover.root == c_real;
    let commit_in_proof = proof.bytes.windows(32).any(|w| w == c_prover.root);

    // Reconstruct public outputs (layout: alpha 128 | beta 128 | out).
    let out_off = 256;
    let mut recon = [0u32; NOUT];
    for kk in 0..NOUT {
        let mut w = 0u32;
        for b in 0..XLEN { w |= ((sp[out_off + kk * XLEN + b].unpack()[0].v & 1) as u32) << b; }
        recon[kk] = w;
    }
    let result_match = recon == out_vals;

    println!("  [emulate] cycles={ncyc} nmem={nmem} num_vars={num_vars} gate_entries={gate_entries}");
    println!("  [verify]  Expander ACCEPTED={verified} (claimed==0={})", claimed_v.is_zero());
    println!("  [commit]  GKR PCS root == Go/DA rsema1d = {commit_ok} ({})", &hex_str(&c_real)[..24]);
    println!("  [commit]  embedded in proof transcript  = {commit_in_proof}");
    println!("  [output]  circuit public outputs {recon:?}");
    println!("  [output]  emulator registers     {out_vals:?}  (match={result_match})");
    for note in &case.notes { println!("  [op]      {note}"); }

    let pass = verified && commit_ok && commit_in_proof && result_match;
    println!("  [CASE]    {}", if pass { "PASS" } else { "FAIL" });
    pass
}

fn main() {
    use emulator::*;
    // Registers: x1..x9 scratch. Halt = jal(0,0) self-loop.
    let mut cases: Vec<Case> = Vec::new();

    // ---- Compares: SLT / SLTU / SLTI / SLTIU (signed vs unsigned) ----
    // x1 = -1 (0xFFFFFFFF), x2 = 1.
    // slt  x3,x1,x2 : -1 <s 1  -> 1
    // sltu x4,x1,x2 : 0xFFFFFFFF <u 1 -> 0
    // slti x5,x1,5  : -1 <s 5  -> 1
    // sltiu x6,x1,5 : 0xFFFFFFFF <u 5 -> 0
    cases.push(Case {
        name: "compares SLT/SLTU/SLTI/SLTIU (signed vs unsigned)",
        prog: vec![
            addi(1, 0, -1), addi(2, 0, 1),
            slt(3, 1, 2), sltu(4, 1, 2), slti(5, 1, 5), sltiu(6, 1, 5),
            jal(0, 0),
        ],
        ncyc: 12,
        out_regs: [3, 4, 5],
        expect: [1, 0, 1],
        notes: vec![
            "SLT (-1 <s 1)=1, SLTU (0xffffffff <u 1)=0 -> signed/unsigned differ".into(),
            "SLTI (-1 <s 5)=1, SLTIU (0xffffffff <u 5)=0; x6 also checked below".into(),
        ],
    });

    // ---- Shifts: SRA / SRAI of a negative number vs SRL ----
    // x1 = -16 (0xFFFFFFF0). sra x2,x1,x3(=2) -> -4 ; srai x4,x1,2 -> -4 ; srl x5 -> logical
    cases.push(Case {
        name: "shifts SRA/SRAI (arithmetic, negative) vs SRL (logical)",
        prog: vec![
            addi(1, 0, -16), addi(3, 0, 2),
            sra(2, 1, 3), srai(4, 1, 2), srl(5, 1, 3),
            jal(0, 0),
        ],
        ncyc: 12,
        out_regs: [2, 4, 5],
        expect: [(-4i32) as u32, (-4i32) as u32, 0xFFFFFFF0u32 >> 2],
        notes: vec![
            "SRA/SRAI sign-fill: 0xFFFFFFF0 >>a 2 = 0xFFFFFFFC (-4)".into(),
            "SRL logical: 0xFFFFFFF0 >>l 2 = 0x3FFFFFFC (differs from SRA -> funct7 distinguished)".into(),
        ],
    });

    // ---- RV32M multiply: MUL / MULH / MULHU / MULHSU ----
    // x1 = -1 (0xFFFFFFFF), x2 = -1. products:
    // mul   = 1 (low32 of (-1)*(-1))
    // mulh  = 0 (high32 of signed 1)
    // mulhu = high32 of 0xFFFFFFFF*0xFFFFFFFF = 0xFFFFFFFE
    // mulhsu= high32 of (-1)*(0xFFFFFFFF unsigned) = signed(-0xFFFFFFFF)=... = 0xFFFFFFFF
    cases.push(Case {
        name: "RV32M multiply MUL/MULH/MULHU (a=b=-1)",
        prog: vec![
            addi(1, 0, -1), addi(2, 0, -1),
            mul(3, 1, 2), mulh(4, 1, 2), mulhu(5, 1, 2),
            jal(0, 0),
        ],
        ncyc: 12,
        out_regs: [3, 4, 5],
        expect: [1, 0, 0xFFFFFFFEu32],
        notes: vec![
            "MUL low32((-1)*(-1))=1; MULH high32(signed)=0; MULHU high32(0xffffffff^2)=0xfffffffe".into(),
        ],
    });
    // MULHSU dedicated: x1=-1 signed, x2=2 unsigned -> product signed(-2), high32=0xFFFFFFFF
    cases.push(Case {
        name: "RV32M MULHSU (signed x unsigned)",
        prog: vec![
            addi(1, 0, -1), addi(2, 0, 2),
            mulhsu(3, 1, 2), mul(4, 1, 2), mulhu(5, 1, 2),
            jal(0, 0),
        ],
        ncyc: 12,
        out_regs: [3, 4, 5],
        // (-1)*2 = -2 = 0xFFFFFFFFFFFFFFFE -> hi=0xFFFFFFFF ; MUL low=0xFFFFFFFE ;
        // MULHU: 0xFFFFFFFF*2 = 0x1FFFFFFFE -> hi=1
        expect: [0xFFFFFFFFu32, 0xFFFFFFFEu32, 1],
        notes: vec!["MULHSU high32(signed(-1) * unsigned(2)) = 0xffffffff".into()],
    });

    // ---- RV32M divide: DIVU/REMU + div-by-zero ----
    // x1 = 100, x2 = 7 -> divu=14 remu=2 ; x3=0 -> divu(100,0)=0xffffffff remu=100
    cases.push(Case {
        name: "RV32M DIVU/REMU + divide-by-zero",
        prog: vec![
            addi(1, 0, 100), addi(2, 0, 7),
            divu(4, 1, 2), remu(5, 1, 2), divu(6, 1, 0),
            jal(0, 0),
        ],
        ncyc: 12,
        out_regs: [4, 5, 6],
        expect: [14, 2, 0xFFFFFFFFu32],
        notes: vec![
            "DIVU 100/7=14, REMU 100%7=2".into(),
            "divide-by-zero: DIVU(100,0)=0xffffffff (all-ones) per spec".into(),
        ],
    });
    // ---- signed DIV/REM negative + INT_MIN/-1 overflow ----
    // x1 = -100, x2 = 7 -> div=-14 (trunc toward 0), rem=-2 (sign of dividend)
    // INT_MIN / -1 overflow: div = INT_MIN, rem = 0
    cases.push(Case {
        name: "RV32M signed DIV/REM (negative) + INT_MIN/-1 overflow",
        prog: vec![
            addi(1, 0, -100), addi(2, 0, 7),
            div(3, 1, 2), rem(4, 1, 2),
            lui(5, 0x80000000), // x5 = INT_MIN
            addi(6, 0, -1),     // x6 = -1
            div(7, 5, 6),       // overflow -> INT_MIN
            rem(8, 5, 6),       // overflow -> 0
            jal(0, 0),
        ],
        ncyc: 16,
        out_regs: [3, 4, 7],
        expect: [(-14i32) as u32, (-2i32) as u32, 0x80000000u32],
        notes: vec![
            "signed DIV -100/7 = -14 (trunc), REM = -2 (sign of dividend)".into(),
            "INT_MIN / -1 overflow: DIV = INT_MIN (0x80000000), REM(x8) = 0".into(),
        ],
    });

    // ---- Control: JALR to a computed target + LUI + AUIPC ----
    // lui  x1 = 0x12345000
    // auipc x2 = pc(of auipc) + 0x1000
    // Build a JALR: addi x5 = &target(absolute) ; jalr x6, x5, 0 -> jumps to target,
    // link x6 = pc+4. target sets x7 = 0xBEEF then halts.
    // Layout (pc=4*idx):
    //  0 lui   x1, 0x12345000
    //  1 auipc x2, 0x1000            (x2 = 4 + 0x1000 = 0x1004)
    //  2 addi  x5, x0, 28            (absolute byte addr of idx7 = 28)
    //  3 jalr  x6, x5, 0            (jump to idx7, link x6 = 16)
    //  4 addi  x7, x0, 1            (skipped)
    //  5 addi  x7, x0, 2            (skipped)
    //  6 jal   0, 0                 (halt if fell through - not reached)
    //  7 addi  x7, x0, 0x6EF        (target; small imm) then set x3 marker
    //  8 jal   0, 0                 (halt)
    cases.push(Case {
        name: "control JALR (computed target) + LUI + AUIPC",
        prog: vec![
            lui(1, 0x12345000),
            auipc(2, 0x1000),
            addi(5, 0, 28),
            jalr(6, 5, 0),
            addi(7, 0, 1),
            addi(7, 0, 2),
            jal(0, 0),
            addi(7, 0, 0x6EF),
            jal(0, 0),
        ],
        ncyc: 12,
        out_regs: [1, 6, 7],
        // x1=0x12345000 (LUI), x6=link=pc(idx3)+4=16, x7=0x6EF (reached via JALR, skipping idx4/5)
        expect: [0x12345000u32, 16, 0x6EF],
        notes: vec![
            "LUI x1 = 0x12345000; AUIPC x2 = pc+0x1000 (checked via honest-proof acceptance)".into(),
            "JALR jumped to computed absolute target (skipping 2 insns); link x6 = pc+4 = 16".into(),
        ],
    });

    // ---- Branches: all six, taken and not-taken ----
    // x1=5, x2=5, x3=-3, x4=+4. Sequence of branches; a running counter x9 records
    // which paths were taken by adding distinct powers. We verify final x9 and two
    // more registers.
    // We keep it simple: exercise BEQ(taken), BNE(taken), BLT(taken, signed),
    // BGE(taken), BLTU(nottaken), BGEU(taken). Each taken branch jumps over an
    // "addi marker" so a wrong branch decision changes x9.
    //  x1=5 x2=5 x3=(-3) x4=4 ; x9 accumulates.
    cases.push(Case {
        name: "branches BEQ/BNE/BLT/BGE/BLTU/BGEU (taken & not-taken)",
        prog: vec![
            addi(1, 0, 5),   // 0
            addi(2, 0, 5),   // 1
            addi(3, 0, -3),  // 2
            addi(4, 0, 4),   // 3
            // BEQ taken (5==5): skip the +100
            beq(1, 2, 8),    // 4 -> idx6
            addi(9, 9, 100), // 5 (skipped)
            // BNE taken (5 != -3): skip +200
            bne(1, 3, 8),    // 6 -> idx8
            addi(9, 9, 200), // 7 (skipped)
            // BLT taken signed (-3 <s 5): skip +400
            blt(3, 1, 8),    // 8 -> idx10
            addi(9, 9, 400), // 9 (skipped)
            // BGE taken signed (5 >=s -3): skip +800
            bge(1, 3, 8),    // 10 -> idx12
            addi(9, 9, 800), // 11 (skipped)
            // BLTU NOT taken (5 <u 4 is false): fall through, add +1
            bltu(1, 4, 8),   // 12 -> not taken, go idx13
            addi(9, 9, 1),   // 13 (executed, +1)
            // BGEU taken (5 >=u 4): skip +2000
            bgeu(1, 4, 8),   // 14 -> idx16
            addi(9, 9, 2000),// 15 (skipped)
            jal(0, 0),       // 16 halt
        ],
        ncyc: 20,
        out_regs: [9, 1, 4],
        // Only the BLTU-not-taken path added +1 to x9.
        expect: [1, 5, 4],
        notes: vec![
            "BEQ/BNE/BLT/BGE/BGEU all TAKEN (their skipped markers never fired)".into(),
            "BLTU NOT taken (5 <u 4 = false) -> fell through, x9 += 1; final x9 == 1".into(),
        ],
    });

    // ---- Sub-word memory: SB/SH/LB/LBU/LH/LHU incl. unaligned ----
    // Base x1 = 0x400 (word-aligned). Store bytes/halfwords at various offsets,
    // then load them back with signed/unsigned width. Also LW the whole word.
    //  sb x2(=0xAB) at 0x401  (offset 1)
    //  sb x3(=0xCD) at 0x403  (offset 3)
    //  sh x4(=0xBEEF) at 0x402 (offset 2, unaligned halfword within word)
    //  word now: byte0=00 byte1=AB byte2=EF byte3=BE   (CD at 0x403 overwritten by sh high byte BE)
    // We instead separate: use two words to avoid overwrite interplay.
    cases.push(Case {
        name: "sub-word memory SB/SH + LB/LBU/LH/LHU (unaligned)",
        prog: vec![
            addi(1, 0, 0x400),      // 0: base word A
            addi(2, 0, 0x00AB),     // 1: byte val 0xAB
            sb(2, 1, 1),            // 2: mem[0x401] = 0xAB
            lbu(3, 1, 1),           // 3: x3 = zero-ext byte = 0x00AB
            lb(4, 1, 1),            // 4: x4 = sign-ext byte 0xAB = 0xFFFFFFAB
            addi(5, 0, 0x400),      // 5: base word B (0x420)
            addi(5, 5, 0x20),
            lui(6, 0x0BEEF000),     // load 0xBEEF into low half via lui+srli trick:
            srli(6, 6, 12),         // x6 = 0x0BEEF -> need 0xBEEF; adjust
            sh(6, 5, 2),            // 8: mem[0x422] = low16(x6) (unaligned halfword offset 2)
            lhu(7, 5, 2),           // 9: x7 = zero-ext half
            lh(8, 5, 2),            // 10: x8 = sign-ext half
            jal(0, 0),
        ],
        ncyc: 16,
        out_regs: [3, 4, 7],
        // x3 = 0xAB ; x4 = 0xFFFFFFAB ; x7 = low16 of x6.
        // x6 = 0x0BEEF000 >>12 = 0x0BEEF -> low16 = 0xBEEF ; zero-ext = 0xBEEF.
        expect: [0x00AB, 0xFFFFFFABu32, 0xBEEF],
        notes: vec![
            "SB then LBU (0xAB) / LB (sign-ext 0xFFFFFFAB) at unaligned offset 1".into(),
            "SH at unaligned halfword offset 2 then LHU (0xBEEF); LH sign-extends".into(),
        ],
    });

    // ---- Base R/I ALU AND/OR/XOR/SUB + immediates (funct7-gated selectors) ----
    // x1=0xFF, x2=0x0F. Confirms the restructured base-op selectors (is_op gated by
    // funct7==0x00, OPIMM ungated) still compute correctly -- these are exactly the
    // ops the EVM interpreter emits.
    cases.push(Case {
        name: "base ALU AND/OR/XOR/SUB + ANDI/ORI/XORI (funct7-gated selectors)",
        prog: vec![
            addi(1, 0, 0xFF), addi(2, 0, 0x0F),
            and(3, 1, 2), or(4, 1, 2), xor(5, 1, 2), sub(6, 1, 2),
            andi(7, 1, 0x0F), ori(8, 2, 0x70), xori(9, 1, 0x0F),
            jal(0, 0),
        ],
        ncyc: 13,
        out_regs: [4, 5, 6],
        expect: [0xFF, 0xF0, 0xF0],
        notes: vec![
            "AND=0x0F, OR=0xFF, XOR=0xF0, SUB=0xF0; ANDI/ORI/XORI also checked via acceptance".into(),
        ],
    });

    let sel: Vec<usize> = std::env::var("ISA_CASES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| (0..cases.len()).collect());

    println!("==== RV32IM per-opcode circuit_gp verification ====");
    let mut all_ok = true;
    let mut passed = 0usize;
    let mut total = 0usize;
    for &i in &sel {
        let c = &cases[i];
        println!("\n#### CASE {i}: {} ####", c.name);
        total += 1;
        let ok = prove_case(c);
        if ok { passed += 1; }
        all_ok &= ok;
    }
    println!("\n==== SUMMARY: {passed}/{total} cases PASS ====");
    if !all_ok { std::process::exit(1); }
}
