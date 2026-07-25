//! Throughput benchmark for the accidental-computer rv32 rollup: proves one
//! 8-lane SIMD-batched block (up to 8 distinct chained sub-blocks) and prints the
//! transaction count, verification, and prove time. Combined with the
//! `bench_parallel.sh` harness (N concurrent instances) it measures the TRUE
//! device ceiling: the GKR prover is single-core per proof, so aggregate TPS
//! scales with the number of concurrent workers that fit in RAM.
//!
//! Usage: tx_bench_simd <txs_per_lane> [lanes 1..=8]   (TX_ACCOUNTS env, default 8)

use std::time::Instant;
use riscv_stf::rv32_prove::prove_rv32_blocks_simd;

const TXPROC: &[u32] = &[
    0x10002503, 0x0a050663, 0x11000593, 0x00003637, 0x000026b7, 0x0100006f, 0xfff50513, 0x01058593,
    0x08050863, 0xff45a883, 0xff85a783, 0xffc5a703, 0x0005a283, 0x00289813, 0x0198d313, 0x00c803b3,
    0x0003a383, 0x00789893, 0x0068e8b3, 0x00f75313, 0x0113c8b3, 0x00f888b3, 0x0138d393, 0x00d89893,
    0x0078e8b3, 0x01171393, 0x0063e333, 0x0068c8b3, 0x00589313, 0x011308b3, 0x00b8d313, 0x011348b3,
    0xf8589ce3, 0x00d80833, 0x00082883, 0xf8e8e6e3, 0x40e888b3, 0x00279793, 0x01182023, 0x00d787b3,
    0x0007a803, 0x00e80733, 0x00e7a023, 0xf6dff06f, 0x0000006f,
];
fn mac(key: u32, s: u32, r: u32, amt: u32) -> u32 {
    let mut h = key ^ s.rotate_left(7);
    h = h.wrapping_add(r).rotate_left(13);
    h ^= amt.rotate_left(17);
    h = h.wrapping_add(h << 5);
    h ^= h >> 11;
    h
}
fn key_of(a: u32) -> u32 { 0xABCD0000u32 ^ a }

fn main() {
    let n: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let lanes: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8).clamp(1, 8);
    let a: u32 = std::env::var("TX_ACCOUNTS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    std::env::set_var("RV32_STEPS", ((n as usize) * 40 + 24).to_string());

    let mut pre_mem: Vec<(u32, u32)> = Vec::new();
    for acct in 0..a { pre_mem.push((0x2000 + 4 * acct, 1_000_000)); }
    for acct in 0..a { pre_mem.push((0x3000 + 4 * acct, key_of(acct))); }

    let mut inputs: Vec<Vec<u8>> = Vec::with_capacity(lanes);
    for l in 0..lanes as u32 {
        let mut input = n.to_le_bytes().to_vec();
        for i in 0..n {
            let s = (i + l) % a; let r = (i + l + 1) % a; let amt = 1u32;
            let sig = mac(key_of(s), s, r, amt);
            for v in [s, r, amt, sig] { input.extend_from_slice(&v.to_le_bytes()); }
        }
        inputs.push(input);
    }

    // Prove `reps` blocks in one process: rep 0 pays the one-time compile (cold),
    // later reps are warm (compile amortized) — the steady-state number.
    let reps: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(1).max(1);
    let total_tx = (lanes as u128) * (n as u128);
    for rep in 0..reps {
        let t0 = Instant::now();
        match prove_rv32_blocks_simd(TXPROC, 0, &inputs, &[0u32; 32], &pre_mem, 0) {
            Ok(p) => {
                let ms = t0.elapsed().as_millis().max(1);
                println!(
                    "SIMDBENCH rep={} kind={} lanes={} tx_per_lane={} total_tx={} verified={} prove_ms={} tps={:.3}",
                    rep, if rep == 0 { "cold" } else { "warm" }, lanes, n, total_tx, p.verified, ms,
                    total_tx as f64 * 1000.0 / ms as f64
                );
                if !p.verified { std::process::exit(1); }
            }
            Err(e) => { eprintln!("SIMDBENCH ERROR: {e}"); std::process::exit(1); }
        }
    }
}
