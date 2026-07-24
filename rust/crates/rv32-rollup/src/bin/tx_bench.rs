//! Transaction-throughput benchmark for the accidental-computer rv32 rollup.
//!
//! Proves a block of N signed transfers processed by the compiled tx contract
//! (programs/txproc: verify keyed-MAC signature + balances, debit/credit a
//! balance array) against a genesis balance state, and reports prove time. The
//! rollup state root (pre_root -> post_root) is bound in-circuit over the balance
//! state; the DA commitment is reused as the sole PCS (zero prover re-encode).
//!
//! Circuit dims (PROG_LEN/MEM_SLOTS/STATE_SLOTS) are fixed at build time via the
//! RV32_* build-env; STEPS (unroll = cycles for N txs) is set here per N.
//!
//! Usage: tx_bench <N>       (TX_ACCOUNTS env sets the account count, default 16)

use std::time::Instant;

use riscv_stf::rv32_prove::prove_rv32_block;

// Compiled `programs/txproc` (simple no_std Rust -> riscv32im, opt-level=3,
// stack-free): reads N at 0x100, txs at 0x104.., balances at 0x2000+4a, keys at
// 0x3000+4a; applies tx iff mac(key[s],s,r,amt)==sig && bal[s]>=amt.
const TXPROC: &[u32] = &[
    0x10002503, 0x0a050663, 0x11000593, 0x00003637, 0x000026b7, 0x0100006f, 0xfff50513, 0x01058593,
    0x08050863, 0xff45a883, 0xff85a783, 0xffc5a703, 0x0005a283, 0x00289813, 0x0198d313, 0x00c803b3,
    0x0003a383, 0x00789893, 0x0068e8b3, 0x00f75313, 0x0113c8b3, 0x00f888b3, 0x0138d393, 0x00d89893,
    0x0078e8b3, 0x01171393, 0x0063e333, 0x0068c8b3, 0x00589313, 0x011308b3, 0x00b8d313, 0x011348b3,
    0xf8589ce3, 0x00d80833, 0x00082883, 0xf8e8e6e3, 0x40e888b3, 0x00279793, 0x01182023, 0x00d787b3,
    0x0007a803, 0x00e80733, 0x00e7a023, 0xf6dff06f, 0x0000006f,
];

// Must match programs/txproc mac() exactly (multiply-free ARX; RV32I only).
fn mac(key: u32, s: u32, r: u32, amt: u32) -> u32 {
    let mut h = key ^ s.rotate_left(7);
    h = h.wrapping_add(r).rotate_left(13);
    h ^= amt.rotate_left(17);
    h = h.wrapping_add(h << 5);
    h ^= h >> 11;
    h
}

fn key_of(acct: u32) -> u32 {
    0xABCD0000u32 ^ acct
}

fn main() {
    let n: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let a: u32 = std::env::var("TX_ACCOUNTS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);

    // Unroll depth: measured ~38 cycles/tx + ~8 setup; keep a small margin so
    // the program reaches its halt (num_cycles < steps) without much waste.
    let steps = (n as usize) * 40 + 24;
    std::env::set_var("RV32_STEPS", steps.to_string());

    // Genesis state (committed): balance[a]=1e6, key[a]=key_of(a), for a in 0..A.
    let mut pre_mem: Vec<(u32, u32)> = Vec::new();
    for acct in 0..a {
        pre_mem.push((0x2000 + 4 * acct, 1_000_000));
    }
    for acct in 0..a {
        pre_mem.push((0x3000 + 4 * acct, key_of(acct)));
    }

    // Block: N valid signed transfers of 1 unit, round-robin over the accounts.
    let mut input: Vec<u8> = Vec::new();
    input.extend_from_slice(&n.to_le_bytes());
    for i in 0..n {
        let s = i % a;
        let r = (i + 1) % a;
        let amt = 1u32;
        let sig = mac(key_of(s), s, r, amt);
        for v in [s, r, amt, sig] {
            input.extend_from_slice(&v.to_le_bytes());
        }
    }

    let pre_regs = [0u32; 32];
    let t0 = Instant::now();
    match prove_rv32_block(TXPROC, 0, &input, &pre_regs, &pre_mem, steps) {
        Ok(p) => {
            let ms = t0.elapsed().as_millis();
            println!(
                "TXBENCH n={} accounts={} steps={} cycles={} verified={} input_vars={} proof_bytes={} pre_root={} post_root={} prove_ms={}",
                n, a, steps, p.num_cycles, p.verified, p.input_vars, p.proof.len(),
                hexs(&p.pre_root), hexs(&p.post_root), ms
            );
        }
        Err(e) => {
            let ms = t0.elapsed().as_millis();
            eprintln!("TXBENCH n={} accounts={} steps={} ERROR after {}ms: {}", n, a, steps, ms, e);
            std::process::exit(1);
        }
    }
}

fn hexs(b: &[u8]) -> String {
    b.iter().take(8).map(|x| format!("{:02x}", x)).collect()
}
