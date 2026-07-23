//! ACCIDENTAL EVM COMPUTER — REAL BLOCK state-root proof. Proves that a real
//! multi-account block's state transition matches reth's GLOBAL post_state_root,
//! verified against the committed parent root via the accounts' MPT witnesses,
//! in a GKR proof whose sole polynomial commitment is the reused rsema1d/DA
//! commitment (zero prover re-encoding). Also round-trips through the ev-reth
//! RealBlockData JSON schema (the real ingestion contract).
use riscv_stf::block_real::{demo_real_block, parse_real_block_data, prove_real_block, to_real_block_data_json};

fn hx(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }

fn main() {
    let rb = demo_real_block();
    // Round-trip through the ev-reth RealBlockData JSON schema (real ingestion).
    let json = to_real_block_data_json(&rb);
    let rb = parse_real_block_data(&json).expect("parse RealBlockData JSON");

    println!("=== ACCIDENTAL EVM COMPUTER: REAL BLOCK state-root proof ===");
    println!("[input] RealBlockData: {} touched accounts, parent_root=0x{}", rb.accounts.len(), hx(&rb.native_parent_root()));
    let p = prove_real_block(&rb).expect("prove_real_block");
    println!("[gkr] committed input num_vars={}, proof bytes={}", p.input_vars, p.proof.len());
    println!("[verify] Expander verifier accepted (rsema1d sole PCS, DA commitment reused) = {}", p.verified);
    println!("[commit] rsema1d/DA commitment = 0x{}", hx(&p.commitment));
    println!("[output] parent_state_root = 0x{}", hx(&p.parent_state_root));
    println!("[output] post_state_root  = 0x{}  (== reth/alloy global root)", hx(&p.post_state_root));
    assert!(p.verified, "verifier rejected");
    assert_eq!(p.post_state_root, rb.native_post_root(), "post root != native/alloy");
    println!("\n=== REAL-BLOCK GATES PASSED ===");
    println!("(a) each touched account's PRE state verified included under the committed parent root");
    println!("(b) all account/storage updates folded -> recomputed root == reth's REAL global post_state_root");
    println!("(c) verifier ACCEPTED; rsema1d = ONLY commitment, reused (zero prover re-encode)");
}
