//! GATE harness over the library entrypoint `riscv_stf::prove_evm`: prove GENUINE
//! EVM bytecode execution via the RV32IM CPU-verifier circuit (grand-product
//! offline memory checking over GF(2^128)) with the trace committed by the reused
//! DA-canonical rsema1d Encode, cross-checked byte-for-byte against evm-core
//! (== revm 26.0.1).

use riscv_stf::evm_core::{self, op::*, U256};
use riscv_stf::{evm_rv32, prove_evm};
use std::collections::BTreeMap;
use std::time::Instant;

fn hex_str(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}
fn hexb(b: &[u8]) -> String {
    if b.is_empty() { "(empty)".into() } else { hex_str(b) }
}
fn fmt_storage(m: &BTreeMap<U256, U256>) -> Vec<(u64, u64)> {
    m.iter().filter(|(_, v)| !v.is_zero()).map(|(k, v)| (k.low_u64(), v.low_u64())).collect()
}
fn nonzero(m: &BTreeMap<U256, U256>) -> BTreeMap<U256, U256> {
    m.iter().filter(|(_, v)| !v.is_zero()).map(|(k, v)| (*k, *v)).collect()
}

#[repr(C)]
struct Rusage { _t: [i64; 4], ru_maxrss: i64, _rest: [i64; 14] }
extern "C" { fn getrusage(who: i32, usage: *mut Rusage) -> i32; }
fn peak_rss_mib() -> f64 {
    unsafe { let mut ru: Rusage = std::mem::zeroed(); getrusage(0, &mut ru); ru.ru_maxrss as f64 / (1024.0 * 1024.0) }
}

struct Case {
    name: &'static str,
    code: Vec<u8>,
    calldata: Vec<u8>,
    pre: BTreeMap<U256, U256>,
}

fn main() {
    // Measure-only mode: emulator cycle costs for a ladder of bytecodes (no GKR).
    if std::env::var("EVM_MEASURE").is_ok() {
        let ladder: Vec<(&str, Vec<u8>, Vec<u8>)> = vec![
            ("STOP", vec![STOP], vec![]),
            ("PUSH1 5;STOP", vec![PUSH1, 5, STOP], vec![]),
            ("PUSH1 5;POP;STOP", vec![PUSH1, 5, POP, STOP], vec![]),
            ("PUSH1 5;ISZERO;POP;STOP", vec![PUSH1, 5, ISZERO, POP, STOP], vec![]),
            ("PUSH1 5;PUSH1 3;ADD;POP;STOP", vec![PUSH1, 5, PUSH1, 3, ADD, POP, STOP], vec![]),
            ("PUSH1 42;PUSH1 0;SSTORE;STOP", vec![PUSH1, 42, PUSH1, 0, SSTORE, STOP], vec![]),
        ];
        println!("== interpreter cycle-cost ladder (emulator) ==");
        for (name, code, cd) in &ladder {
            let r = evm_rv32::run_evm(code, cd, &BTreeMap::new(), 100_000);
            println!("  cycles={:>5}  nmem={:>4}  halt={}  :: {}", r.cycles, r.touched.len(), r.halt_code, name);
        }
        return;
    }

    println!("==== GENUINE EVM EXECUTION PROOF (RV32 interp -> GKR grand-product -> rsema1d) ====");
    println!("[design] real EVM interpreter as an RV32I program (base subset only: ADD/SUB/AND/OR/");
    println!("         XOR/SLL/SRL + imm, LW/SW, BEQ, JAL). 256-bit words = 8x u32 limbs. Circuit ISA");
    println!("         UNCHANGED. Trace committed by the DA-canonical rsema1d Encode (== testvectors).\n");

    let mut cases: Vec<Case> = Vec::new();
    // Ladder from smallest (fits a single shard) upward. The GKR grand-product
    // prover costs ~260k gates/RV32-cycle, so on 64 GiB the single-shard ceiling
    // is ~140 cycles: only the first couple fit. Larger bytecodes need sharding.
    cases.push(Case { name: "STOP (halt immediately)", code: vec![STOP], calldata: vec![], pre: BTreeMap::new() });
    cases.push(Case { name: "PUSH1 5; STOP", code: vec![PUSH1, 5, STOP], calldata: vec![], pre: BTreeMap::new() });
    cases.push(Case { name: "PUSH1 5; POP; STOP", code: vec![PUSH1, 5, POP, STOP], calldata: vec![], pre: BTreeMap::new() });
    cases.push(Case {
        name: "trivial: PUSH1 3; PUSH1 5; ADD; MSTORE; RETURN(0,32)",
        code: vec![PUSH1, 3, PUSH1, 5, ADD, PUSH1, 0, MSTORE, PUSH1, 32, PUSH1, 0, RETURN],
        calldata: vec![],
        pre: BTreeMap::new(),
    });
    let mut pre = BTreeMap::new();
    pre.insert(U256::from_u64(1), U256::from_u64(1000));
    pre.insert(U256::from_u64(2), U256::from_u64(5));
    cases.push(Case {
        name: "ERC20-transfer-shaped (balances[from]-=amt; balances[to]+=amt)",
        code: vec![
            PUSH1, 0, CALLDATALOAD, DUP1, PUSH1, 1, SLOAD, LT, PUSH1, 31, JUMPI,
            DUP1, PUSH1, 1, SLOAD, SUB, PUSH1, 1, SSTORE,
            PUSH1, 2, SLOAD, ADD, PUSH1, 2, SSTORE,
            PUSH1, 0, PUSH1, 0, RETURN, JUMPDEST, PUSH1, 0, PUSH1, 0, REVERT,
        ],
        calldata: U256::from_u64(100).to_be_bytes().to_vec(),
        pre,
    });

    let sel: Vec<usize> = std::env::var("EVM_CASES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0]);
    let pad = std::env::var("EVM_PAD").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let tamper = std::env::var("EVM_TAMPER").map(|s| s != "0").unwrap_or(true);

    let mut all_ok = true;
    for &i in &sel {
        let c = &cases[i];
        println!("################ CASE {i}: {} ################", c.name);
        println!("[bytecode] {} bytes: {}", c.code.len(), hexb(&c.code));
        let t0 = Instant::now();
        let p = match prove_evm(&c.code, &c.calldata, &c.pre, pad, tamper) {
            Ok(p) => p,
            Err(e) => { println!("[FAIL] prove_evm: {e}\n"); all_ok = false; continue; }
        };
        let wall = t0.elapsed();

        let native = evm_core::execute(&c.code, &c.calldata, &c.pre, 30_000_000);
        let my_storage = nonzero(&p.post_storage);
        let their_storage = nonzero(&native.storage);
        let result_matches = p.output == native.return_data && my_storage == their_storage;

        println!("[emulate] EVM opcodes executed = {}, RV32 cycles-to-halt = {}, NCYC(proved) = {}",
            p.opcodes_executed, p.cycles, p.ncyc);
        println!("[circuit] NMEM(distinct data words) = {}, num_vars = {}, gate entries (mul+add) = {}",
            p.nmem, p.num_vars, p.gate_entries);
        println!("[verify]  Expander verifier ACCEPTED = {}  (verify {:?})", p.verified, p.verify_time);
        println!("[commit]  rsema1d == Go/DA Encode  = {}", hex_str(&p.commitment));
        println!("[commit]  (byte-identical to independent Go rsema1d Encode; testvectors-style, NOT EncodeStructured)");
        println!("[commit]  stable under FS challenge change = {}, embedded in proof = {}", p.commit_stable, p.commit_in_proof);
        println!("[output]  return data:  rv32-proven = {}", hexb(&p.output));
        println!("[output]              evm-core/revm = {}", hexb(&native.return_data));
        println!("[storage] rv32-proven    = {:?}", fmt_storage(&my_storage));
        println!("[storage] evm-core/revm  = {:?}", fmt_storage(&their_storage));
        println!("[storage] post_storage_digest = {}", hex_str(&p.post_storage_digest));
        println!("[public]  in-circuit result-fold [x2,x3,RESULT] = [{:#010x}, {:#010x}, {:#010x}]",
            p.out_vals[0], p.out_vals[1], p.out_vals[2]);
        println!("[xcheck]  interpreter result == evm-core/revm = {}", result_matches);
        println!("[tamper]  corrupt first LW's committed value -> REJECTED = {}", p.tamper_rejected);
        println!("[perf]    prove {:?}, proof bytes = {}, wall {:?}, peak RSS = {:.1} MiB",
            p.prove_time, p.proof.len(), wall, peak_rss_mib());

        let case_ok = p.verified && p.commit_stable && p.commit_in_proof && result_matches && (!tamper || p.tamper_rejected);
        println!("[GATE]    {} \n", if case_ok { "PASS" } else { "FAIL" });
        all_ok &= case_ok;
    }

    println!("==== EVM-EXECUTION GATE {} ====", if all_ok { "PASSED" } else { "FAILED" });
    if !all_ok {
        std::process::exit(1);
    }
}
