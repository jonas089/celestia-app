//! Emulator-only validation of the RV32 EVM interpreter (`evm_rv32`) against the
//! native `evm-core` reference (which itself matches revm 26.0.1 byte-for-byte).
//! No GKR here — fast iteration to get the interpreter correct before proving.

mod emulator;
mod evm_asm;
// Drift-free: the RV32 proof's native oracle IS the exact evm-core source that
// evm-xcheck proves byte-identical to revm 26.0.1 (parent workspace). #[path]
// points the module straight at that file — no copy, no divergence.
#[path = "../../evm-core/src/lib.rs"]
mod evm_core;
mod evm_rv32;

use crate::evm_core::U256;
use std::collections::BTreeMap;

fn hex(b: &[u8]) -> String {
    if b.is_empty() {
        return "(empty)".into();
    }
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn fmt_storage(m: &BTreeMap<U256, U256>) -> Vec<(u64, u64)> {
    m.iter()
        .filter(|(_, v)| !v.is_zero())
        .map(|(k, v)| (k.low_u64(), v.low_u64()))
        .collect()
}

fn nonzero(m: &BTreeMap<U256, U256>) -> BTreeMap<U256, U256> {
    m.iter().filter(|(_, v)| !v.is_zero()).map(|(k, v)| (*k, *v)).collect()
}

struct Case {
    name: &'static str,
    code: Vec<u8>,
    calldata: Vec<u8>,
    pre: BTreeMap<U256, U256>,
    max_cycles: usize,
}

fn main() {
    use crate::evm_core::op::*;
    let mut cases: Vec<Case> = Vec::new();

    cases.push(Case {
        name: "PUSH+ADD+MSTORE+RETURN",
        code: vec![PUSH1, 3, PUSH1, 5, ADD, PUSH1, 0, MSTORE, PUSH1, 32, PUSH1, 0, RETURN],
        calldata: vec![],
        pre: BTreeMap::new(),
        max_cycles: 20000,
    });

    cases.push(Case {
        name: "SSTORE/SLOAD/JUMPI branch",
        code: vec![
            PUSH1, 0, CALLDATALOAD, PUSH1, 100, GT,
            PUSH1, 17, JUMPI,
            PUSH1, 2, PUSH1, 0, SSTORE,
            PUSH1, 23, JUMP,
            JUMPDEST,
            PUSH1, 1, PUSH1, 0, SSTORE,
            JUMPDEST,
            PUSH1, 0, SLOAD, PUSH1, 0, MSTORE, PUSH1, 32, PUSH1, 0, RETURN,
        ],
        calldata: U256::from_u64(42).to_be_bytes().to_vec(),
        pre: BTreeMap::new(),
        max_cycles: 40000,
    });

    let mut pre = BTreeMap::new();
    pre.insert(U256::from_u64(1), U256::from_u64(1000));
    pre.insert(U256::from_u64(2), U256::from_u64(5));
    cases.push(Case {
        name: "ERC20-transfer-shaped",
        code: vec![
            PUSH1, 0, CALLDATALOAD,
            DUP1, PUSH1, 1, SLOAD,
            LT,
            PUSH1, 31, JUMPI,
            DUP1, PUSH1, 1, SLOAD, SUB, PUSH1, 1, SSTORE,
            PUSH1, 2, SLOAD, ADD, PUSH1, 2, SSTORE,
            PUSH1, 0, PUSH1, 0, RETURN,
            JUMPDEST,
            PUSH1, 0, PUSH1, 0, REVERT,
        ],
        calldata: U256::from_u64(100).to_be_bytes().to_vec(),
        pre,
        max_cycles: 60000,
    });

    let mut all_ok = true;
    println!("==== RV32 EVM interpreter (emulator)  vs  evm-core reference ====\n");
    for c in &cases {
        let theirs = crate::evm_core::execute(&c.code, &c.calldata, &c.pre, 30_000_000);
        let mine = evm_rv32::run_evm(&c.code, &c.calldata, &c.pre, c.max_cycles);

        let my_storage = nonzero(&mine.storage);
        let their_storage = nonzero(&theirs.storage);
        // halt: 1 STOP / 2 RETURN => success; 3 REVERT / 4 ERR => failure
        let my_success = matches!(mine.halt_code, 1 | 2);
        let ok_success = my_success == theirs.success;
        let ok_return = mine.output == theirs.return_data;
        let ok_storage = my_storage == their_storage;
        let ok = ok_success && ok_return && ok_storage;

        println!("---- {} ----", c.name);
        println!("  RV32 cycles(to halt)={}  halt_code={}  final_pc={:#x} final_sp={}",
            mine.cycles, mine.halt_code, mine.final_pc, mine.final_sp);
        println!("  distinct data-mem words touched = {}", mine.touched.len());
        println!("  success: rv32={} evm-core={}  {}", my_success, theirs.success, if ok_success {"OK"} else {"MISMATCH"});
        println!("  return:  rv32={}", hex(&mine.output));
        println!("           core={}  {}", hex(&theirs.return_data), if ok_return {"OK"} else {"MISMATCH"});
        println!("  storage: rv32={:?}", fmt_storage(&my_storage));
        println!("           core={:?}  {}", fmt_storage(&their_storage), if ok_storage {"OK"} else {"MISMATCH"});
        println!("  => {}\n", if ok {"AGREE"} else {"DRIFT"});
        all_ok &= ok;
    }

    println!("==== interpreter validation {} ====", if all_ok {"PASSED"} else {"FAILED"});
    if !all_ok {
        std::process::exit(1);
    }
}
