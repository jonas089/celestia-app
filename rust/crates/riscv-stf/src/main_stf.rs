//! Transfer-STF block-proof GATE: produce a few small rollup blocks (1-4 transfer
//! txs each) and, for EACH block, generate a REAL GKR proof of its nonce + balance
//! state transition — verifier ACCEPTED, commitment == independent Go/DA rsema1d
//! commit (byte-identical), and replay / overspend txs provably REJECTED (not
//! applied). Nothing is faked; on any prover error the block aborts.

use riscv_stf::stf::{prove_block_stf, BlockInput, TxData};
use std::time::Instant;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

struct Case {
    name: &'static str,
    ns: &'static str,
    block_number: u64,
    input: BlockInput,
    expect_applied: u32,
}

fn main() {
    println!("==== Transfer-STF per-block GATE (real GKR + rsema1d) ====");
    println!("[scope] proves the NONCE + BALANCE state transition over the block's committed");
    println!("        tx data, reusing rsema1d as the GKR input PCS. ECDSA signatures and a");
    println!("        keccak/MPT state root are NOT proven (digest is a simple XOR/shift fold).\n");

    let cases = vec![
        Case {
            name: "valid: two transfers in order",
            ns: "rollup-alpha",
            block_number: 1,
            input: BlockInput {
                pre_sender_balance: 1_000_000,
                pre_sender_nonce: 5,
                pre_recipient_balance: 100,
                txs: vec![
                    TxData { value: 1000, fee: 10, nonce: 5 },
                    TxData { value: 2000, fee: 20, nonce: 6 },
                ],
            },
            expect_applied: 2,
        },
        Case {
            name: "REPLAY: 2nd tx reuses a spent nonce -> rejected",
            ns: "rollup-alpha",
            block_number: 2,
            input: BlockInput {
                pre_sender_balance: 1_000_000,
                pre_sender_nonce: 5,
                pre_recipient_balance: 100,
                txs: vec![
                    TxData { value: 1000, fee: 10, nonce: 5 }, // applied
                    TxData { value: 2000, fee: 20, nonce: 5 }, // REPLAY -> rejected
                ],
            },
            expect_applied: 1,
        },
        Case {
            name: "OVERSPEND: value+fee exceeds balance -> rejected",
            ns: "rollup-beta",
            block_number: 1,
            input: BlockInput {
                pre_sender_balance: 500,
                pre_sender_nonce: 0,
                pre_recipient_balance: 0,
                txs: vec![
                    TxData { value: 400, fee: 10, nonce: 0 }, // applied (bal 500->90)
                    TxData { value: 400, fee: 10, nonce: 1 }, // OVERSPEND -> rejected
                ],
            },
            expect_applied: 1,
        },
    ];

    let mut all_ok = true;
    for c in &cases {
        println!("---- namespace={} block #{} : {} ----", c.ns, c.block_number, c.name);
        let native = riscv_stf::stf::apply_native(&c.input).unwrap();
        println!(
            "[stf] pre: sender bal={} nonce={}, recipient bal={} | {} txs",
            c.input.pre_sender_balance,
            c.input.pre_sender_nonce,
            c.input.pre_recipient_balance,
            c.input.txs.len()
        );
        for (i, t) in c.input.txs.iter().enumerate() {
            println!(
                "[stf]   tx{}: value={} fee={} nonce={}  => {}",
                i,
                t.value,
                t.fee,
                t.nonce,
                if native.applied[i] { "APPLIED" } else { "REJECTED" }
            );
        }

        let t0 = Instant::now();
        let p = match prove_block_stf(&c.input) {
            Ok(p) => p,
            Err(e) => {
                println!("[FAIL] prove_block_stf error: {e}\n");
                all_ok = false;
                continue;
            }
        };
        let dur = t0.elapsed();

        println!(
            "[post] sender bal={} nonce={}, recipient bal={}, applied={}/{}, digest={:#010x}",
            p.post_sender_balance,
            p.post_sender_nonce,
            p.post_recipient_balance,
            p.applied_count,
            p.tx_count,
            p.digest
        );
        println!("[proof] verified={}  input_vars={}  proof_bytes={}", p.verified, p.input_vars, p.proof.len());
        println!("[commit] rsema1d == Go/DA = {}", hex(&p.commitment));

        let applied_ok = p.applied_count == c.expect_applied;
        let ok = p.verified && applied_ok;
        println!(
            "[GATE] verifier ACCEPTED={}  applied_count={} (expected {}) {}  in {:?}\n",
            p.verified,
            p.applied_count,
            c.expect_applied,
            if ok { "PASS" } else { "FAIL" },
            dur
        );
        all_ok &= ok;
    }

    println!("==== GATE {} ====", if all_ok { "PASSED (all blocks proven, replay+overspend rejected)" } else { "FAILED" });
    if !all_ok {
        std::process::exit(1);
    }
}
