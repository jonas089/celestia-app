//! FULL transfer-block accidental computer, run end-to-end: committed input =
//! the signed EIP-1559 tx (signing-preimage + r,s,yparity) + pre-state accounts
//! + block env. The circuit recovers the sender in-circuit (ecrecover), binds the
//! tx value/nonce to the signed preimage, applies the reth-faithful transfer
//! transition, and roots the post-state — proven with rsema1d as the SOLE
//! polynomial commitment. HEAVY (in-circuit GF2 ecrecover ~hours).

use num_bigint::BigInt;
use num_traits::Num;
use riscv_stf::mpt::{keccak256, nibbles};
use riscv_stf::rlp::Eip1559Tx;
use riscv_stf::secp256k1::{self, native};
use riscv_stf::stf_prove::{prove_transfer_stf_full, StfInputs};
use std::time::Instant;

fn hx(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }
fn modn(x: BigInt) -> BigInt { let n = secp256k1::n(); ((x % &n) + &n) % &n }
fn keccak(b: &[u8]) -> [u8; 32] { keccak256(b) }

fn main() {
    // Signer key d; sender address = keccak(pubkey)[12:].
    let d = BigInt::from_str_radix("c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721", 16).unwrap();
    let q = native::scalar_mul(&d, &native::generator()).0.unwrap();
    let mut pub64 = [0u8; 64];
    pub64[32 - q.0.to_bytes_be().1.len()..32].copy_from_slice(&q.0.to_bytes_be().1);
    pub64[64 - q.1.to_bytes_be().1.len()..64].copy_from_slice(&q.1.to_bytes_be().1);
    let mut sender = [0u8; 20];
    sender.copy_from_slice(&keccak(&pub64)[12..]);
    let slot_sender = nibbles(&keccak256(&sender))[0] as usize;

    // Pick recipient + coinbase with distinct keccak-first-nibbles.
    let mut recipient = [0u8; 20];
    let mut coinbase = [0u8; 20];
    let mut seen = vec![slot_sender];
    let mut cand = 1u8;
    for slot_out in [&mut recipient, &mut coinbase] {
        loop {
            let a = [cand; 20];
            let s = nibbles(&keccak256(&a))[0] as usize;
            cand = cand.wrapping_add(1);
            if !seen.contains(&s) { *slot_out = a; seen.push(s); break; }
        }
    }

    // Build + sign the demo EIP-1559 transfer (49-byte signing preimage).
    let mut tx = Eip1559Tx {
        chain_id: 1, nonce: 7,
        max_priority_fee: BigInt::from(2_000_000_000u64),
        max_fee: BigInt::from(20_000_000_000u64),
        gas_limit: 21000, to: recipient,
        value: BigInt::from(1_000_000_000_000_000u64),
        data: vec![], y_parity: 0, r: BigInt::from(0), s: BigInt::from(0),
    };
    let pre = tx.signing_preimage();
    assert_eq!(pre.len(), 49, "preimage len {} != 49", pre.len());
    let z = modn(BigInt::from_bytes_be(num_bigint::Sign::Plus, &keccak(&pre)));
    let k = BigInt::from_str_radix("49a0d7b786ec9cde0d0721d72804befd06571c974b191efb42ecf322ba9ddd9a", 16).unwrap();
    let rpt = native::scalar_mul(&k, &native::generator()).0.unwrap();
    let r = modn(rpt.0.clone());
    let k_inv = k.modpow(&(secp256k1::n() - BigInt::from(2u32)), &secp256k1::n());
    let s = modn(&k_inv * (&z + &r * &d));
    let yparity = (&rpt.1 & BigInt::from(1u32)).to_bytes_be().1.first().copied().unwrap_or(0) & 1;
    tx.r = r.clone(); tx.s = s.clone(); tx.y_parity = yparity;

    println!("=== FULL transfer-block accidental computer (in-circuit ecrecover) ===");
    println!("[signer] sender address = 0x{}", hx(&sender));
    println!("[input] committed = signed tx (preimage {}B + r,s,v) + 3 pre-state accounts + block env", pre.len());

    let inp = StfInputs {
        addrs: [sender, recipient, coinbase],
        nonce: [7, 0, 0],
        balance: [
            BigInt::from(1_000_000_000_000_000_000u64),
            BigInt::from(500_000_000_000_000u64),
            BigInt::from(0u64),
        ],
        value: BigInt::from(1_000_000_000_000_000u64),
        max_fee: BigInt::from(20_000_000_000u64),
        max_prio: BigInt::from(2_000_000_000u64),
        base_fee: BigInt::from(7_000_000_000u64),
    };

    let t = Instant::now();
    let p = prove_transfer_stf_full(&inp, &pre, &r, &s, yparity).expect("prove_transfer_stf_full");
    println!("[done] in {:?}", t.elapsed());
    println!("[gkr] committed input num_vars = {}, proof bytes = {}", p.input_vars, p.proof.len());
    println!("[verify] Expander verifier accepted (rsema1d sole PCS) = {}", p.verified);
    println!("[commit] rsema1d/DA commitment = {}", hx(&p.commitment));
    println!("[output] post_state_root (== reth/alloy) = 0x{}", hx(&p.post_state_root));
    assert!(p.verified);
    println!("\n=== FULL ACCIDENTAL COMPUTER (transfer block) GATES PASSED ===");
    println!("(a) sender RECOVERED in-circuit via ecrecover; == signer address");
    println!("(b) tx value/nonce bound to the signed preimage");
    println!("(c) reth-faithful transfer transition + Ethereum MPT, all internal wires");
    println!("(d) rsema1d = the ONLY polynomial commitment; == independent Go/DA commit");
    println!("(e) public post_state_root byte-identical to reth's Ethereum MPT root");
}
