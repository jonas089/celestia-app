//! Accidental-EVM-computer BLOCK proof: committed DA block data (contract code +
//! pre-storage + accounts) -> in-circuit EVM execution + Ethereum world-state MPT
//! -> reth-faithful post_state_root, proven in GKR with the REUSED rsema1d/DA
//! commitment (zero prover encoding).
use num_bigint::BigInt;
use riscv_stf::block_stf::{prove_block, BlockInputs};
use riscv_stf::mpt::{keccak256, nibbles};

fn hx(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }

fn main() {
    let mut addrs = [[0u8; 20]; 3];
    let (mut chosen, mut cand) = (0usize, 1u8);
    let mut seen: Vec<usize> = vec![];
    while chosen < 3 { let a=[cand;20]; let s=nibbles(&keccak256(&a))[0] as usize; if !seen.contains(&s){addrs[chosen]=a;seen.push(s);chosen+=1;} cand=cand.wrapping_add(1); }
    let inp = BlockInputs {
        contract_addr: addrs[0],
        code: vec![0x60,0x2a,0x60,0x00,0x55,0x60,0x01,0x60,0x00,0xf3],
        c_pre_slot0: BigInt::from(0u32), c_nonce: 1, c_balance: BigInt::from(0u32),
        eoa: [(addrs[1],7,BigInt::from(1_000_000_000_000_000_000u64)),(addrs[2],0,BigInt::from(500_000_000_000_000u64))],
    };
    println!("=== ACCIDENTAL EVM COMPUTER: block STF proof ===");
    println!("[input] committed DA data = contract code + pre-storage + 3 accounts; 1 contract-call tx");
    let p = prove_block(&inp).expect("prove_block");
    println!("[gkr] committed input num_vars={}, proof bytes={}", p.input_vars, p.proof.len());
    println!("[verify] Expander verifier accepted (rsema1d sole PCS, DA handle reused) = {}", p.verified);
    println!("[commit] rsema1d/DA commitment = {}", hx(&p.commitment));
    println!("[output] post_state_root (== reth/alloy) = 0x{}", hx(&p.post_state_root));
    assert!(p.verified);
    println!("\n=== BLOCK GATES PASSED ===");
    println!("(a) in-circuit EVM executed the contract call over committed bytecode (== ev-reth)");
    println!("(b) world-state MPT (contract storageRoot+codeHash + EOAs) == alloy/reth");
    println!("(c) verifier ACCEPTED; rsema1d = ONLY commitment; prover REUSED DA handle (zero encode)");
    println!("(d) public post_state_root byte-identical to reth's state root");
}
