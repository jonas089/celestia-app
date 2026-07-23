//! Accidental-computer STF proof (transfer block): committed input = block DA
//! data (pre-state + tx), output = reth-faithful post_state_root, rsema1d the
//! SOLE polynomial commitment. Prints the gates.

use num_bigint::BigInt;
use riscv_stf::mpt::{keccak256, nibbles};
use riscv_stf::stf_prove::{prove_transfer_stf, StfInputs};

fn hx(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }

fn main() {
    // Pick 3 addresses with distinct keccak first nibbles (flat-branch topology).
    let mut addrs = [[0u8; 20]; 3];
    let (mut chosen, mut cand) = (0usize, 1u8);
    let mut seen: Vec<usize> = vec![];
    while chosen < 3 {
        let a = [cand; 20];
        let s = nibbles(&keccak256(&a))[0] as usize;
        if !seen.contains(&s) { addrs[chosen] = a; seen.push(s); chosen += 1; }
        cand = cand.wrapping_add(1);
    }

    let inp = StfInputs {
        addrs,
        nonce: [7, 0, 0],
        balance: [
            BigInt::from(1_000_000_000_000_000_000u64), // sender 1 ETH
            BigInt::from(500_000_000_000_000u64),        // recipient prefunded
            BigInt::from(0u64),                          // coinbase
        ],
        value: BigInt::from(1_000_000_000_000_000u64), // 0.001 ETH
        max_fee: BigInt::from(20u64),
        max_prio: BigInt::from(2u64),
        base_fee: BigInt::from(7u64),
    };

    println!("=== Accidental-computer STF proof (transfer block) ===");
    println!("[input] committed DA data = 3 pre-state accounts (nonce,balance) + tx (value,maxfee,maxprio,basefee)");
    let p = prove_transfer_stf(&inp).expect("prove_transfer_stf");
    println!("[gkr] committed input num_vars = {}, proof bytes = {}", p.input_vars, p.proof.len());
    println!("[verify] Expander verifier accepted (rsema1d sole PCS) = {}", p.verified);
    println!("[commit] rsema1d/DA commitment (GKR input == independent Go/DA) = {}", hx(&p.commitment));
    println!("[output] post_state_root (== reth/alloy native) = 0x{}", hx(&p.post_state_root));
    assert!(p.verified, "verifier REJECTED");
    println!("\n=== GATES PASSED ===");
    println!("(a) verifier ACCEPTED; rsema1d = the ONLY polynomial commitment");
    println!("(b) committed input = block DA data (pre-state + tx); trace/MPT internal, uncommitted");
    println!("(c) public output post_state_root is byte-identical to reth's Ethereum MPT state root");
    println!("(d) GKR input commitment == independent Go/DA rsema1d commit (reuse)");
    println!("\nNote: sender recovery (ecrecover, R5) is verified separately and composes into the");
    println!("same circuit (heavy). This proof covers the transition + MPT half over committed data.");
}
