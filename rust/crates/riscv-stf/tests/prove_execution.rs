use riscv_stf::prove_execution;

#[test]
fn real_proof_verifies_and_is_input_dependent() {
    let a = prove_execution(b"hello-accidental").expect("prove a");
    assert!(a.verified, "verifier rejected honest proof");
    assert_eq!(a.commitment.len(), 32);
    assert_eq!(a.input_vars, 12);
    assert!(!a.proof.is_empty());
    println!("A commitment = {}", hex(&a.commitment));
    println!("A public_value = {}", hex(&a.public_value));
    println!("A proof bytes = {}", a.proof.len());

    let b = prove_execution(b"different-input").expect("prove b");
    assert!(b.verified);
    println!("B commitment = {}", hex(&b.commitment));
    assert_ne!(a.commitment, b.commitment, "distinct inputs must give distinct commitments");

    // determinism: same input -> same commitment
    let a2 = prove_execution(b"hello-accidental").expect("prove a2");
    assert_eq!(a.commitment, a2.commitment, "same input must give same commitment");
}

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }
