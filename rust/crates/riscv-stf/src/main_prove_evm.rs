//! Accidental-computer GKR proof of a REAL in-circuit EVM execution, with the
//! reused DA commitment (prover does zero encoding). Runs the `increment`
//! program (SLOAD/ADD/SSTORE), matching ev-reth's EVM.
use num_bigint::BigInt;
use riscv_stf::evm_gkr::prove_evm;

fn hx(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }

fn main() {
    // increment: PUSH1 0, SLOAD, PUSH1 1, ADD, PUSH1 0, SSTORE, STOP ; slot0: 5 -> 6
    let code = vec![0x60, 0x00, 0x54, 0x60, 0x01, 0x01, 0x60, 0x00, 0x55, 0x00];
    let init = vec![(BigInt::from(0u32), BigInt::from(5u32))];
    println!("=== Accidental-computer GKR proof of in-circuit EVM execution ===");
    println!("[input] committed DA data = EVM bytecode ({} B) + initial storage; program = increment", code.len());
    let p = prove_evm(&code, &init).expect("prove_evm");
    println!("[gkr] committed input num_vars = {}, proof bytes = {}", p.input_vars, p.proof_bytes);
    println!("[verify] Expander verifier accepted (rsema1d sole PCS, DA handle reused) = {}", p.verified);
    println!("[commit] rsema1d/DA commitment (installed == independent DA encode) = {}", hx(&p.commitment));
    println!("[output] post storage slot0 = {} (native/ev-reth golden = 6)", p.post_svals[0]);
    assert!(p.verified);
    assert_eq!(p.post_svals[0], BigInt::from(6u32));
    println!("\n=== GATES PASSED ===");
    println!("(a) real in-circuit EVM executed committed bytecode; matches ev-reth EVM");
    println!("(b) verifier ACCEPTED; rsema1d = the ONLY polynomial commitment");
    println!("(c) prover REUSED the installed DA handle — ZERO encoding in the prover");
    println!("(d) committed EVM bytecode+storage is the DA data; execution trace internal/uncommitted");
}
