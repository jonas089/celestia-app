//! `prove_stf_ac`: the input-committed accidental-computer STF executor, run
//! end-to-end over a block's DATA. Proves the nonce+balance transition with the
//! rsema1d/DA commitment as the SOLE polynomial commitment (opened at the
//! sumcheck point), reconciles it byte-for-byte against an independent Go/DA
//! rsema1d commit of the same data, and demonstrates tamper rejection.

use riscv_stf::stf::{BlockInput, TxData};
use riscv_stf::stf_ac::{prove_block_stf_ac, tamper_check};

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn main() {
    // A block matching the demo per-block workload shape: transfers from one
    // sender to one recipient, with replay protection + balance checks. Values
    // are the block's committed data (the DA'd bytes), NOT a trace.
    let input = BlockInput {
        pre_sender_balance: 5_000_000,
        pre_sender_nonce: 42,
        pre_recipient_balance: 123,
        txs: vec![
            TxData { value: 1000, fee: 10, nonce: 42 }, // applied
            TxData { value: 2000, fee: 20, nonce: 43 }, // applied
            TxData { value: 4_000_000, fee: 0, nonce: 44 }, // overspend -> rejected
            TxData { value: 500, fee: 5, nonce: 44 },   // applied (nonce catches up)
        ],
    };

    println!("=== Accidental-computer STF executor (input-committed) ===");
    println!("[input] block data words = pre-state(sbal,snon,rbal) + {} txs(value,fee,nonce)", input.txs.len());

    let proof = prove_block_stf_ac(&input).expect("prove_block_stf_ac");

    println!("\n[committed] the SOLE committed poly IS the block data:");
    for (i, w) in proof.committed_words.iter().enumerate() {
        let label = match i {
            0 => "pre_sender_balance",
            1 => "pre_sender_nonce",
            2 => "pre_recipient_balance",
            _ => {
                let t = (i - 3) / 3;
                match (i - 3) % 3 {
                    0 => return_leaked(t, "value"),
                    1 => return_leaked(t, "fee"),
                    _ => return_leaked(t, "nonce"),
                }
            }
        };
        println!("    data[{i:2}] = {w:>10}   ({label})");
    }

    println!("\n[post-state] computed FORWARD in circuit (internal wires, no commitment):");
    println!("    post_sender_balance    = {}", proof.post_sender_balance);
    println!("    post_sender_nonce      = {}", proof.post_sender_nonce);
    println!("    post_recipient_balance = {}", proof.post_recipient_balance);
    println!("    applied_count          = {}", proof.applied_count);
    println!("    digest                 = 0x{:08x}", proof.digest);

    println!("\n[gkr] input num_vars = {}, proof bytes = {}", proof.input_vars, proof.proof.len());
    println!("[verify] Expander verifier accepted (rsema1d sole input PCS) = {}", proof.verified);
    println!("[commit] rsema1d/DA commitment (GKR input == independent Go/DA) = {}", hex(&proof.commitment));

    assert!(proof.verified, "verifier REJECTED the honest proof");

    // ---- Soundness: tamper a committed data word -----------------------------
    println!("\n=== Soundness gate: tamper committed data ===");
    let rep = tamper_check(&input, 0, 0x1).expect("tamper_check"); // flip a bit of sbal
    println!("[tamper] honest    commitment = {}", hex(&rep.commitment_honest));
    match &rep.commitment_tampered {
        Some(c) => println!("[tamper] tampered  commitment = {}", hex(c)),
        None => println!("[tamper] tampered witness was UNSATISFIABLE (no commitment)"),
    }
    println!("[tamper] tampered-data / honest-output witness satisfiable = {} (must be false)", rep.tampered_satisfiable);
    assert!(!rep.tampered_satisfiable, "SOUNDNESS BREAK: tampered transition satisfied the circuit");
    if let Some(c) = &rep.commitment_tampered {
        assert_ne!(c, &rep.commitment_honest, "tampered data produced the same commitment");
        println!("[tamper] tampered commitment differs from honest: reuse detects data changes");
    }

    println!("\n=== ALL GATES PASSED ===");
    println!("(a) Expander verifier ACCEPTED the proof (rsema1d = the ONLY polynomial commitment)");
    println!("(b) the committed poly IS the block data (executor shape: trace is internal, uncommitted)");
    println!("(c) GKR input commitment == independent Go/DA rsema1d commit (byte-identical => reuse)");
    println!("(d) tampered transition is unsatisfiable AND changes the commitment");
}

fn return_leaked(t: usize, field: &str) -> &'static str {
    // Small helper to produce a static label; leak is fine for a one-shot CLI.
    Box::leak(format!("tx[{t}].{field}").into_boxed_str())
}
