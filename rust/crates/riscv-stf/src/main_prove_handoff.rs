//! Prover process of the TRUE cross-process accidental-computer hand-off.
//!
//! Reads `(root, extended-rows, num_vars)` emitted by a SEPARATE DA-encoder
//! process (`da_encode_block`), reconstructs the committed square from the
//! already-extended rows WITHOUT re-encoding, then proves the demo block over
//! Expander GKR. It asserts:
//!   * this process performed ZERO RS-encodes (encode_call_count delta == 0);
//!   * verified == true;
//!   * the GKR commitment equals the DA-encoder's root.
//!
//! Usage: prove_block_handoff <in_dir>

use std::path::PathBuf;

fn main() {
    let in_dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".to_string()));

    let root_hex = std::fs::read_to_string(in_dir.join("da_root.hex")).expect("read da_root.hex");
    let root = unhex(root_hex.trim());
    let mut root_arr = [0u8; 32];
    root_arr.copy_from_slice(&root);
    let extended = std::fs::read(in_dir.join("da_extended.bin")).expect("read da_extended.bin");
    let num_vars: usize = std::fs::read_to_string(in_dir.join("da_num_vars.txt"))
        .expect("read da_num_vars.txt").trim().parse().expect("num_vars");

    println!("[prover pid={}]", std::process::id());
    println!("  loaded DA root     = 0x{}", root_hex.trim());
    println!("  extended-rows blob = {} bytes", extended.len());
    println!("  num_vars           = {num_vars}");

    let encodes_at_start = rsema1d_sys::encode_call_count();

    // Reconstruct the committed square from the DA-encoded rows (NO RS-encode),
    // and install it so prove_block reuses it.
    let installed = rsema1d_pcs::install_da_commitment_from_serialized(num_vars, root_arr, &extended);
    assert_eq!(installed, root_arr, "reconstructed commitment != DA root");
    let encodes_after_load = rsema1d_sys::encode_call_count();
    println!("  RS-encodes during load: {} (must be 0)", encodes_after_load - encodes_at_start);

    // Prove the same block. prove_block reuses the installed commitment.
    let inp = riscv_stf::block_stf::demo_block_inputs();
    let p = riscv_stf::block_stf::prove_block(&inp).expect("prove_block");
    let encodes_at_end = rsema1d_sys::encode_call_count();

    let delta = encodes_at_end - encodes_at_start;
    println!("  post_state_root    = 0x{}", hex(&p.post_state_root));
    println!("  GKR commitment     = 0x{}", hex(&p.commitment));
    println!("  verified           = {}", p.verified);
    println!("  proof bytes        = {}", p.proof.len());
    println!("  RS-encodes in prover process (total delta) = {delta}");

    assert_eq!(delta, 0, "PROVER MUST NOT RE-ENCODE: encode_call_count changed by {delta}");
    assert!(p.verified, "GKR verification failed");
    assert_eq!(p.commitment, root_arr, "GKR commitment != DA-encoder root");
    println!("\nOK: prover consumed the DA-side encoding, did ZERO RS-encodes, verified=true, commitment==DA root.");
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}
