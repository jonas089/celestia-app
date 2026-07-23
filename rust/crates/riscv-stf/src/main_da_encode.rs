//! DA-encoder process of the TRUE cross-process accidental-computer hand-off.
//!
//! Derives the demo block's GKR input layer and RS-encodes it ONCE via the Go
//! rsema1d DA encoder, then writes `(root, extended-rows, num_vars)` to an output
//! directory. A SEPARATE prover process (`prove_block_handoff`) reconstructs the
//! committed square from those bytes and proves the block WITHOUT re-encoding.
//!
//! Usage: da_encode_block <out_dir>
//! Emits: <out_dir>/da_root.hex, <out_dir>/da_extended.bin, <out_dir>/da_num_vars.txt

use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".to_string()));
    std::fs::create_dir_all(&out_dir).expect("create out_dir");

    let before = rsema1d_sys::encode_call_count();
    let inp = riscv_stf::block_stf::demo_block_inputs();
    let (root, extended, num_vars) =
        riscv_stf::block_stf::block_da_serialize(&inp).expect("block_da_serialize");
    let after = rsema1d_sys::encode_call_count();

    std::fs::write(out_dir.join("da_root.hex"), hex(&root)).unwrap();
    std::fs::write(out_dir.join("da_extended.bin"), &extended).unwrap();
    std::fs::write(out_dir.join("da_num_vars.txt"), num_vars.to_string()).unwrap();

    println!("[DA-encoder pid={}]", std::process::id());
    println!("  RS-encodes in this process: before={before} after={after} (delta={})", after - before);
    println!("  num_vars           = {num_vars}");
    println!("  DA commitment root = 0x{}", hex(&root));
    println!("  extended-rows blob = {} bytes  -> {}", extended.len(), out_dir.join("da_extended.bin").display());
    assert_eq!(after - before, 1, "DA side must perform exactly one RS-encode");
    println!("  OK: exactly ONE RS-encode happened on the DA side.");
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
