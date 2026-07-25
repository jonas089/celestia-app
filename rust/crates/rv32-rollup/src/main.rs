//! Minimal rv32i sovereign rollup on Celestia — the "accidental rv32i computer".
//!
//! Raw rv32i execution, no permissioning, no tokens. Each submitted program
//! (rv32i machine code) + input is executed over a PERSISTENT global VM state
//! (registers + linear memory) carried block→block. One block per minute.
//!
//! GKR regime (the accidental computer): the ONLY committed data is the INPUT
//! layer — `program + input + pre-state`. The GKR sumcheck reduces the output
//! claim layer-by-layer to a single evaluation on that input layer; every
//! per-cycle register/memory/pc wire is INTERMEDIATE (proven, never committed) —
//! there is deliberately NO committed execution trace. Make that input layer BE
//! the rsema1d/DA-committed data and the DA commitment is the system's sole
//! commitment; its opening discharges the whole reduction ("free"). For each
//! block the rollup:
//!   1. executes the program(s) on the current state (emulator) → output + post-state,
//!   2. posts the committed data `program + input + pre-state` and the resulting
//!      `output + post-state` to Celestia DA (available on-chain),
//!   3. GKR-proves the execution by REUSING the rsema1d/DA commitment of the input
//!      layer as the sole polynomial commitment (the prover encodes nothing itself —
//!      see `prove_rv32_block` / `install_da_commitment`); the trace is intermediate,
//!   4. caches the proof (pruned after 1 day) and serves it over an accProof-style
//!      JSON-RPC the dashboard reads.
//!
//! It never fabricates a proof: on prover/DA error it records the real error.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use riscv_stf::rv32_prove::{
    prove_rv32_blocks_simd, rv32_prepare_batch, rv32_prove_prepared_batch, Rv32BatchProof,
};
use serde_json::{json, Value};

const BLOCK_SECS: u64 = 60; // one block per minute
const PRUNE_SECS: u64 = 86_400; // 1 day retention
const MAX_CYCLES: usize = 1 << 16;
const PROGRAM_BASE: u32 = 0; // rv32_circuit currently fixes base=0

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
fn hexs(b: &[u8]) -> String {
    let mut s = String::with_capacity(2 + b.len() * 2);
    s.push_str("0x");
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}
fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if s.len() % 2 != 0 {
        return Err("odd-length hex".into());
    }
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}
/// Parse a JSON value as u32 — accepts a number or a "0x…"/decimal string.
fn num_u32(v: &Value) -> Option<u32> {
    if let Some(n) = v.as_u64() {
        return u32::try_from(n).ok();
    }
    let s = v.as_str()?;
    if let Some(h) = s.strip_prefix("0x") {
        u32::from_str_radix(h, 16).ok()
    } else {
        s.parse::<u32>().ok()
    }
}
/// Decode a program given as hex of little-endian u32 instruction words.
fn decode_program(hex: &str) -> Result<Vec<u32>, String> {
    let bytes = hex_decode(hex)?;
    if bytes.len() % 4 != 0 {
        return Err("program bytes must be a multiple of 4".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Persistent global VM state carried block→block.
#[derive(Clone, Default)]
struct VmState {
    regs: [u32; 32],
    mem: BTreeMap<u32, u32>, // word address (aligned) -> value
    height: u64,
}
impl VmState {
    fn mem_pairs(&self) -> Vec<(u32, u32)> {
        self.mem.iter().map(|(k, v)| (*k, *v)).collect()
    }
    fn apply(&mut self, post_regs: [u32; 32], post_mem: &[(u32, u32)]) {
        self.regs = post_regs;
        for (a, v) in post_mem {
            self.mem.insert(*a, *v);
        }
    }
}

/// A cached per-block proof record (dashboard-compatible camelCase shape).
#[derive(Clone)]
struct BlockRecord {
    height: u64,
    status: String,
    verified: bool,
    commitment: String,   // reused rsema1d/DA commitment
    da_height: u64,       // Celestia height the block blob landed at (0 if pending)
    program: String,      // rv32i program (hex, LE words) — on DA
    input: String,        // input bytes (hex) — on DA
    output: String,       // output bytes (hex) — on DA
    post_state_root: String, // digest of post-state (regs+touched mem)
    num_cycles: u32,
    input_vars: u32,
    proof_bytes: usize,
    elapsed_ms: u128,
    submitted_unix: u64,
    proved_unix: u64,
    error: String,
    tx_hash: String, // fibre MsgPayForFibre tx hash (fibre mode only)
    blob_id: String, // fibre blob id (fibre mode only)
    program_name: String, // sample program name (e.g. "sum_1_to_n")
    rust_source: String,  // simple no_std Rust source the program was compiled from
    pre_root: String,     // register-state root before execution (bound in-circuit)
    post_root: String,    // register-state root after execution (bound in-circuit)
    num_tx: u32,          // transactions in this block (summed over all SIMD lanes)
    num_lanes: u32,       // GF2x8 SIMD lanes carrying distinct sub-blocks (1..=8)
}
impl BlockRecord {
    fn to_json(&self) -> Value {
        json!({
            "kind": "rv32",
            "blockNumber": self.height,
            "programName": self.program_name,
            "rustSource": self.rust_source,
            "preRoot": self.pre_root,
            "postRoot": self.post_root,
            "status": self.status,
            "verified": self.verified,
            "commitment": self.commitment,
            "daHeight": self.da_height,
            "program": self.program,
            "input": self.input,
            "output": self.output,
            "stfStateRoot": self.post_state_root,
            "numCycles": self.num_cycles,
            "numTx": self.num_tx,
            "numLanes": self.num_lanes,
            "inputVars": self.input_vars,
            "proofBytes": self.proof_bytes,
            "elapsedMs": self.elapsed_ms as u64,
            "submittedUnix": self.submitted_unix,
            "provedUnix": self.proved_unix,
            "error": self.error,
            "txHash": self.tx_hash,
            "blobId": self.blob_id,
        })
    }
}

struct Submission {
    program: Vec<u32>,
    input: Vec<u8>,
    /// Memory slots to declare/seed for this execution (merged over persistent
    /// state). A program's store targets (e.g. an output slot 0x200) must be
    /// declared here since the in-circuit memory is a bounded committed set.
    mem: Vec<(u32, u32)>,
    /// Human name and the simple no_std Rust source the program was compiled
    /// from (both optional; surfaced in the explorer).
    name: String,
    source: String,
}

type Cache = Arc<Mutex<BTreeMap<u64, BlockRecord>>>;

/// keccak-ish state digest (uses the prover crate's keccak so it matches any
/// on-chain use). Kept simple: digest over regs ++ sorted touched memory.
fn state_digest(regs: &[u32; 32], mem: &BTreeMap<u32, u32>) -> [u8; 32] {
    let mut buf = Vec::with_capacity(32 * 4 + mem.len() * 8);
    for r in regs {
        buf.extend_from_slice(&r.to_be_bytes());
    }
    for (a, v) in mem {
        buf.extend_from_slice(&a.to_be_bytes());
        buf.extend_from_slice(&v.to_be_bytes());
    }
    keccak256(&buf)
}

fn keccak256(b: &[u8]) -> [u8; 32] {
    use tiny_keccak::{Hasher, Keccak};
    let mut k = Keccak::v256();
    k.update(b);
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    out
}

// --------------------------------------------------------------------------
// Celestia DA client (celestia-node bridge JSON-RPC: blob.Submit).
// --------------------------------------------------------------------------

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

/// Minimal HTTP JSON POST with optional Bearer auth. Returns the response body.
fn http_post_json(url: &str, bearer: Option<&str>, body: &str) -> Result<String, String> {
    let rest = url.strip_prefix("http://").ok_or("only http:// bridge urls")?;
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match hostport.rfind(':') {
        Some(i) => (&hostport[..i], hostport[i + 1..].parse::<u16>().unwrap_or(26658)),
        None => (hostport, 26658),
    };
    let mut stream = TcpStream::connect((host, port)).map_err(|e| format!("connect {host}:{port}: {e}"))?;
    let mut req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    if let Some(t) = bearer {
        req.push_str(&format!("Authorization: Bearer {t}\r\n"));
    }
    req.push_str("\r\n");
    req.push_str(body);
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut resp = String::new();
    stream.read_to_string(&mut resp).map_err(|e| e.to_string())?;
    // strip HTTP headers
    match resp.find("\r\n\r\n") {
        Some(i) => Ok(resp[i + 4..].to_string()),
        None => Ok(resp),
    }
}

/// Post `data` as a Celestia blob under `namespace` via the bridge; return height.
fn da_submit(bridge: &str, token: &str, namespace_hex: &str, data: &[u8]) -> Result<u64, String> {
    use std::fmt::Write as _;
    // base64 of namespace (from hex) and data.
    let ns = hex_decode(namespace_hex)?;
    let ns_b64 = b64(&ns);
    let data_b64 = b64(data);
    let mut body = String::new();
    let _ = write!(
        body,
        r#"{{"id":1,"jsonrpc":"2.0","method":"blob.Submit","params":[[{{"namespace":"{}","data":"{}","share_version":0,"commitment":""}}],{{"gas_price":0.002}}]}}"#,
        ns_b64, data_b64
    );
    let resp = http_post_json(bridge, Some(token), &body)?;
    let v: Value = serde_json::from_str(&resp).map_err(|e| format!("da resp parse: {e}: {resp}"))?;
    if let Some(err) = v.get("error") {
        return Err(format!("blob.Submit error: {err}"));
    }
    v.get("result").and_then(|r| r.as_u64()).ok_or_else(|| format!("no height in {resp}"))
}

fn b64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut o = String::with_capacity((b.len() + 2) / 3 * 4);
    for c in b.chunks(3) {
        let b0 = c[0] as u32;
        let b1 = *c.get(1).unwrap_or(&0) as u32;
        let b2 = *c.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        o.push(T[(n >> 18 & 63) as usize] as char);
        o.push(T[(n >> 12 & 63) as usize] as char);
        o.push(if c.len() > 1 { T[(n >> 6 & 63) as usize] as char } else { '=' });
        o.push(if c.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    o
}

/// The block blob = program ++ input ++ output ++ post-state digest ++ cycle count.
/// This is the DA data: the committed INPUT (program ++ input, over which the
/// reused rsema1d commitment is taken — pre-state is folded in by the prover) plus
/// the public OUTPUT + post-state. No execution trace is posted or committed.
fn block_blob(program: &[u32], input: &[u8], output: &[u8], post_root: &[u8; 32], cycles: u32) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&(program.len() as u32).to_le_bytes());
    for w in program {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v.extend_from_slice(&(input.len() as u32).to_le_bytes());
    v.extend_from_slice(input);
    v.extend_from_slice(&(output.len() as u32).to_le_bytes());
    v.extend_from_slice(output);
    v.extend_from_slice(post_root);
    v.extend_from_slice(&cycles.to_le_bytes());
    v
}

// --------------------------------------------------------------------------
// Fibre-reuse DA path: prover PREPARES (no encode), the `rv32-fibre-upload` CLI
// encodes the input square ONCE + uploads it to fibre + settles the commitment
// on-chain via MsgPayForFibre, then the prover OPENS the GKR proof against that
// exact on-chain commitment (asserting zero prover-side RS-encode).
// --------------------------------------------------------------------------

/// Config for the fibre-reuse DA mechanism. Present (Some) iff `FIBRE_UPLOAD_BIN`
/// is set; when None, the rollup keeps the internal-encode + bridge `da_submit`
/// path unchanged (smoke-test compatible).
#[derive(Clone)]
struct FibreCfg {
    upload_bin: String,
    grpc_addr: String,
    key_name: String,
    home: String,
    chain_id: String,
    keyring_backend: String,
    namespace: String,
}

/// The DA-side settlement facts + the serialized extended matrix, produced by the
/// fibre upload CLI (encode square ONCE, upload, settle MsgPayForFibre on-chain).
struct FibreUpload {
    commitment: [u8; 32],
    da_height: u64,
    tx_hash: String,
    blob_id: String,
    extended: Vec<u8>,
}

/// Hand the input-layer bytes to the external fibre uploader (encode+upload+settle)
/// and return the settled commitment + extended matrix. Shared by the single-block
/// and 8-lane SIMD-batched prove paths (the input-layer format is identical; the
/// GF2x8 packing already carries all 8 lanes).
fn fibre_upload(cfg: &FibreCfg, input_vals_bytes: &[u8], num_vars: u32) -> Result<FibreUpload, String> {
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_dir = std::env::temp_dir();
    let vals_path = tmp_dir.join(format!("rv32-fibre-vals-{pid}-{ts}.bin"));
    let ext_path = tmp_dir.join(format!("rv32-fibre-ext-{pid}-{ts}.bin"));
    std::fs::write(&vals_path, input_vals_bytes)
        .map_err(|e| format!("write input-vals temp file: {e}"))?;

    let out = std::process::Command::new(&cfg.upload_bin)
        .arg("--input-vals-file").arg(&vals_path)
        .arg("--num-vars").arg(num_vars.to_string())
        .arg("--extended-out").arg(&ext_path)
        .arg("--namespace").arg(&cfg.namespace)
        .arg("--grpc-addr").arg(&cfg.grpc_addr)
        .arg("--chain-id").arg(&cfg.chain_id)
        .arg("--key-name").arg(&cfg.key_name)
        .arg("--keyring-backend").arg(&cfg.keyring_backend)
        .arg("--home").arg(&cfg.home)
        .output()
        .map_err(|e| format!("exec {}: {e}", cfg.upload_bin));
    let out = match out {
        Ok(o) => o,
        Err(e) => { let _ = std::fs::remove_file(&vals_path); return Err(e); }
    };
    if !out.status.success() {
        let _ = std::fs::remove_file(&vals_path);
        return Err(format!("fibre upload exited {}: {}", out.status, String::from_utf8_lossy(&out.stderr).trim()));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json_line = stdout.lines().rev().find(|l| l.trim_start().starts_with('{'))
        .ok_or_else(|| format!("no JSON in fibre upload stdout: {stdout}"))?;
    let v: Value = serde_json::from_str(json_line)
        .map_err(|e| format!("parse fibre upload JSON: {e}: {json_line}"))?;
    let commitment_hex = v.get("commitment").and_then(|x| x.as_str()).ok_or("fibre upload JSON missing commitment")?;
    let commitment_bytes = hex_decode(commitment_hex)?;
    if commitment_bytes.len() != 32 {
        let _ = std::fs::remove_file(&vals_path);
        return Err(format!("fibre commitment not 32 bytes: {commitment_hex}"));
    }
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(&commitment_bytes);
    let da_height = v.get("daHeight").and_then(|x| x.as_u64()).ok_or("fibre upload JSON missing daHeight")?;
    let tx_hash = v.get("txHash").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let blob_id = v.get("blobId").and_then(|x| x.as_str()).unwrap_or("").to_string();
    let extended_file = v.get("extendedFile").and_then(|x| x.as_str()).map(|s| s.to_string())
        .unwrap_or_else(|| ext_path.to_string_lossy().into_owned());
    let extended = std::fs::read(&extended_file).map_err(|e| format!("read extended matrix {extended_file}: {e}"));
    let _ = std::fs::remove_file(&vals_path);
    let extended = match extended {
        Ok(b) => b,
        Err(e) => { let _ = std::fs::remove_file(&ext_path); return Err(e); }
    };
    let _ = std::fs::remove_file(&ext_path);
    Ok(FibreUpload { commitment, da_height, tx_hash, blob_id, extended })
}

/// SIMD-batched fibre round: prepare up to 8 chained sub-blocks (one per GF2x8
/// lane), settle the single (8-lane) input square via the fibre CLI, and open ONE
/// GKR proof against it. ~8x the transactions per proof at ~1x the prover cost.
fn fibre_prove_batch(
    cfg: &FibreCfg,
    program: &[u32],
    inputs: &[Vec<u8>],
    pre_regs: &[u32; 32],
    pre_mem: &[(u32, u32)],
) -> Result<(Rv32BatchProof, u64, String, String), String> {
    let prepared = rv32_prepare_batch(program, PROGRAM_BASE, inputs, pre_regs, pre_mem, MAX_CYCLES)?;
    let up = fibre_upload(cfg, &prepared.input_vals_bytes, prepared.num_vars)?;
    let proof = rv32_prove_prepared_batch(prepared, up.commitment, &up.extended)?;
    Ok((proof, up.da_height, up.tx_hash, up.blob_id))
}

// --------------------------------------------------------------------------
// Block worker: produce one block per minute, execute + prove + DA-post.
// --------------------------------------------------------------------------

fn block_worker(rx: std::sync::mpsc::Receiver<Submission>, cache: Cache) {
    let mut state = VmState::default();
    let bridge = env_or("DA_ADDRESS", "http://localhost:26658");
    let token = env_or("DA_AUTH_TOKEN", "");
    let namespace = env_or("RV32_NAMESPACE", "0000000000000000000000000000000000007276333272757032"); // "rv32rup"
    // Fibre-reuse DA mode is enabled iff FIBRE_UPLOAD_BIN is set. Otherwise the
    // rollup keeps the internal-encode + bridge da_submit path unchanged.
    let fibre_cfg: Option<FibreCfg> = match std::env::var("FIBRE_UPLOAD_BIN") {
        Ok(bin) if !bin.is_empty() => Some(FibreCfg {
            upload_bin: bin,
            grpc_addr: env_or("FIBRE_GRPC_ADDR", "localhost:9090"),
            key_name: env_or("FIBRE_KEY_NAME", "validator"),
            home: env_or("FIBRE_HOME", ""),
            chain_id: env_or("FIBRE_CHAIN_ID", "test"),
            keyring_backend: env_or("FIBRE_KEYRING_BACKEND", "test"),
            namespace: namespace.clone(),
        }),
        _ => None,
    };
    if let Some(c) = &fibre_cfg {
        eprintln!(
            "rv32-rollup: FIBRE-REUSE DA mode: bin={} grpc={} key={} chain={} home={} ns={}",
            c.upload_bin, c.grpc_addr, c.key_name, c.chain_id, c.home, c.namespace
        );
    } else {
        eprintln!("rv32-rollup: internal-encode + bridge da_submit mode (FIBRE_UPLOAD_BIN unset)");
    }
    let mut pending: Vec<Submission> = Vec::new();
    let secs: u64 = env_or("RV32_BLOCK_SECS", &BLOCK_SECS.to_string()).parse().unwrap_or(BLOCK_SECS);
    let deadline_step = std::time::Duration::from_secs(secs);
    loop {
        // Collect submissions for up to BLOCK_SECS, then seal a block.
        let start = Instant::now();
        while start.elapsed() < deadline_step {
            match rx.recv_timeout(std::time::Duration::from_millis(500)) {
                Ok(s) => pending.push(s),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
            }
        }
        if pending.is_empty() {
            continue; // empty minute → no block
        }
        // Seal a block: execute the pending submissions sequentially over the
        // persistent state (one proven record per execution, keyed by height).
        let batch: Vec<Submission> = std::mem::take(&mut pending);
        for sub in batch {
            state.height += 1;
            let h = state.height;
            let submitted = now_unix();
            {
                let mut g = cache.lock().unwrap();
                g.insert(h, BlockRecord {
                    height: h, status: "proving".into(), verified: false, commitment: String::new(),
                    da_height: 0, program: hexs(&words_le(&sub.program)), input: hexs(&sub.input),
                    output: String::new(), post_state_root: String::new(), num_cycles: 0, input_vars: 0,
                    proof_bytes: 0, elapsed_ms: 0, submitted_unix: submitted, proved_unix: 0, error: String::new(),
                    tx_hash: String::new(), blob_id: String::new(),
                    program_name: sub.name.clone(), rust_source: sub.source.clone(),
                    pre_root: String::new(), post_root: String::new(),
                    num_tx: 0, num_lanes: 0,
                });
            }
            let t0 = Instant::now();
            let pre_regs = state.regs;
            // pre_mem = the PERSISTENT ROLLUP STATE only (balance/key region
            // [0x2000, 0x4000)); everything else is transient. Carry the state
            // region, then apply this submission's declared slots (genesis seeding).
            let mut merged: BTreeMap<u32, u32> = state
                .mem
                .iter()
                .filter(|(a, _)| **a >= 0x2000 && **a < 0x4000)
                .map(|(a, v)| (*a, *v))
                .collect();
            for (a, v) in &sub.mem {
                merged.insert(*a, *v);
            }
            let pre_mem: Vec<(u32, u32)> = merged.into_iter().collect();

            // SIMD-BATCH the block: expand a transfer submission into 8 DISTINCT,
            // CHAINED sub-blocks (one per GF2x8 lane), proven in ONE GKR proof at
            // ~1x the cost -> ~8x transactions per proof. Lane l runs a sender-ring
            // rotated by l over lane (l-1)'s post-state; the persistent state
            // advances to the last lane's post. Genesis (no txs) stays a single lane.
            let n = batch_n(&sub.input);
            let lanes: Vec<Vec<u8>> = if sub.name == "transactions" && n > 0 {
                (0..8u32).map(|l| tx_batch_lane(n, l)).collect()
            } else {
                vec![sub.input.clone()]
            };
            let num_lanes = lanes.len() as u32;
            let total_tx: u32 = lanes.iter().map(|inp| batch_n(inp)).sum();
            eprintln!("rv32-rollup: block #{h}: executing + proving ({} instrs, {num_lanes} lanes, {total_tx} tx)", sub.program.len());

            // FIBRE-REUSE mode: prepare (no encode) -> fibre CLI encodes+uploads+
            // settles the (8-lane) commitment on-chain -> open the GKR proof against
            // it (zero prover-side encode). Otherwise: internal single-encode batch.
            let (res, fibre_da): (Result<Rv32BatchProof, String>, Option<(u64, String, String)>) =
                if let Some(cfg) = &fibre_cfg {
                    match fibre_prove_batch(cfg, &sub.program, &lanes, &pre_regs, &pre_mem) {
                        Ok((bp, dh, tx, bid)) => (Ok(bp), Some((dh, tx, bid))),
                        Err(e) => (Err(e), None),
                    }
                } else {
                    (
                        prove_rv32_blocks_simd(&sub.program, PROGRAM_BASE, &lanes, &pre_regs, &pre_mem, MAX_CYCLES),
                        None,
                    )
                };
            let elapsed = t0.elapsed().as_millis();
            let mut g = cache.lock().unwrap();
            let rec = g.get_mut(&h).unwrap();
            rec.elapsed_ms = elapsed;
            rec.proved_unix = now_unix();
            match res {
                Ok(bp) => {
                    let first = bp.blocks.first().expect("batch has >=1 lane");
                    let last = bp.blocks.last().expect("batch has >=1 lane");
                    // advance persistent state via the chained end (last lane)
                    state.apply(last.post_regs, &last.post_mem);
                    let post_root = state_digest(&state.regs, &state.mem);
                    let total_cycles: u32 = bp.blocks.iter().map(|b| b.num_cycles).sum();
                    // DA height: in fibre mode it is the REAL on-chain settlement
                    // height; otherwise post the block DATA to the bridge (best-effort).
                    let da_h = match &fibre_da {
                        Some((h_da, _, _)) => *h_da,
                        None => {
                            let blob = block_blob(&sub.program, &sub.input, &last.output, &post_root, total_cycles);
                            if token.is_empty() { 0 } else {
                                match da_submit(&bridge, &token, &namespace, &blob) {
                                    Ok(hh) => hh,
                                    Err(e) => { eprintln!("rv32-rollup: DA submit failed (block #{h}): {e}"); 0 }
                                }
                            }
                        }
                    };
                    rec.status = "proved".into();
                    rec.verified = bp.verified;
                    rec.commitment = hexs(&bp.commitment);
                    rec.da_height = da_h;
                    rec.output = hexs(&last.output);
                    rec.post_state_root = hexs(&post_root);
                    rec.num_cycles = total_cycles;
                    rec.num_tx = total_tx;
                    rec.num_lanes = num_lanes;
                    rec.input_vars = bp.input_vars;
                    rec.proof_bytes = bp.proof.len();
                    rec.pre_root = hexs(&first.pre_root);
                    rec.post_root = hexs(&last.post_root);
                    if let Some((_, tx, bid)) = &fibre_da {
                        rec.tx_hash = tx.clone();
                        rec.blob_id = bid.clone();
                    }
                    let mode = if fibre_da.is_some() { "fibre" } else { "internal" };
                    eprintln!("rv32-rollup: block #{h} PROVED ({mode}) verified={} lanes={num_lanes} tx={total_tx} cycles={total_cycles} commit={} daHeight={} in {}ms",
                        bp.verified, &rec.commitment[..18.min(rec.commitment.len())], da_h, elapsed);
                }
                Err(e) => {
                    rec.status = "failed".into();
                    rec.error = e.clone();
                    eprintln!("rv32-rollup: block #{h} FAILED: {e}");
                }
            }
        }
    }
}

fn words_le(prog: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(prog.len() * 4);
    for w in prog {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v
}

// --------------------------------------------------------------------------
// JSON-RPC server: rv32_submit + accProof_listBlockProofs / getBlockProof.
// --------------------------------------------------------------------------

fn prune(cache: &Cache) {
    let now = now_unix();
    let mut g = cache.lock().unwrap();
    g.retain(|_, r| {
        if r.status == "proving" || r.status == "queued" {
            return true;
        }
        let ts = if r.proved_unix > 0 { r.proved_unix } else { r.submitted_unix };
        now.saturating_sub(ts) < PRUNE_SECS
    });
}

fn handle_body(body: &[u8], cache: &Cache, tx: &Sender<Submission>) -> String {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return json!({"jsonrpc":"2.0","id":Value::Null,"error":{"code":-32700,"message":format!("parse: {e}")}}).to_string(),
    };
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = req.get("params").cloned().unwrap_or(Value::Null);
    match method {
        // Submit an rv32i program (hex of LE u32 words) + input (hex) for execution.
        "rv32_submit" => {
            let p = params.get(0).cloned().unwrap_or(json!({}));
            let prog = match p.get("program").and_then(|v| v.as_str()).map(decode_program) {
                Some(Ok(pr)) => pr,
                Some(Err(e)) => return err(id, -32602, &format!("bad program: {e}")),
                None => return err(id, -32602, "missing program (hex of LE u32 words)"),
            };
            let input = match p.get("input").and_then(|v| v.as_str()) {
                Some(s) => match hex_decode(s) { Ok(b) => b, Err(e) => return err(id, -32602, &format!("bad input: {e}")) },
                None => vec![],
            };
            // Optional memory declarations: "mem": [[addr, val], ...] (decimal or 0x).
            let mut mem: Vec<(u32, u32)> = vec![];
            if let Some(arr) = p.get("mem").and_then(|v| v.as_array()) {
                for e in arr {
                    let a = e.get(0).and_then(num_u32);
                    let v = e.get(1).and_then(num_u32);
                    match (a, v) {
                        (Some(a), Some(v)) => mem.push((a, v)),
                        _ => return err(id, -32602, "mem entries must be [addr, val]"),
                    }
                }
            }
            let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let source = p.get("source").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let _ = tx.send(Submission { program: prog, input, mem, name, source });
            json!({"jsonrpc":"2.0","id":id,"result":{"status":"queued"}}).to_string()
        }
        "accProof_listBlockProofs" => {
            prune(cache);
            let g = cache.lock().unwrap();
            let list: Vec<Value> = g.values().map(|r| r.to_json()).collect();
            // dashboard shape: namespaces[] with the rv32 rollup's blocks
            let proved = g.values().filter(|r| r.status == "proved").count();
            let verified = g.values().filter(|r| r.verified).count();
            json!({"jsonrpc":"2.0","id":id,"result":{
                "scope":"rv32i accidental computer: each block executes committed rv32i over persistent VM state; program+input+output+trace+state posted to Celestia DA; GKR-proven by reusing the rsema1d/DA commitment as the sole PCS (zero prover re-encode).",
                "rollupName":"rv32i rollup","rollupNS":"rv32-rollup",
                "blockProofs": list.clone(),
                "namespaces":[{"namespace":"rv32-rollup","blocks":list,"blockCount":g.len(),"provedCount":proved,"verifiedCount":verified,"blockStfCount":g.len(),"blockStfVerified":verified}]
            }}).to_string()
        }
        "accProof_getBlockProof" => {
            let h = params.get(1).or_else(|| params.get(0)).and_then(|v| v.as_u64()).unwrap_or(0);
            let g = cache.lock().unwrap();
            match g.get(&h) {
                Some(r) => json!({"jsonrpc":"2.0","id":id,"result":r.to_json()}).to_string(),
                None => json!({"jsonrpc":"2.0","id":id,"result":Value::Null}).to_string(),
            }
        }
        _ => err(id, -32601, &format!("method not found: {method}")),
    }
}

fn err(id: Value, code: i64, msg: &str) -> String {
    json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":msg}}).to_string()
}

fn write_http(stream: &mut TcpStream, status: &str, body: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(resp.as_bytes())
}

// Sample programs are simple #![no_std] Rust, compiled to rv32im
// (riscv32im-unknown-none-elf, opt-level=3) with no stack usage. The compiled
// instruction words are embedded below; the exact source that produced them is
// carried alongside so the explorer can show source + bytecode + proof. See
// programs/*.rs and programs/README.md to regenerate.

// The Rust source of the deployed transaction contract (shown in the explorer).
const SRC_TXPROC: &str = include_str!("../programs/txproc.rs");

// Compiled `programs/txproc` (simple no_std Rust -> riscv32im, opt-level=3;
// stack-free, multiply-free). Reads N at 0x100, txs at 0x104.., balances at
// 0x2000+4a, keys at 0x3000+4a; applies tx iff mac(key[s],s,r,amt)==sig &&
// bal[s]>=amt. This is the rollup's single deployed program.
const TXPROC: &[u32] = &[
    0x10002503, 0x0a050663, 0x11000593, 0x00003637, 0x000026b7, 0x0100006f, 0xfff50513, 0x01058593,
    0x08050863, 0xff45a883, 0xff85a783, 0xffc5a703, 0x0005a283, 0x00289813, 0x0198d313, 0x00c803b3,
    0x0003a383, 0x00789893, 0x0068e8b3, 0x00f75313, 0x0113c8b3, 0x00f888b3, 0x0138d393, 0x00d89893,
    0x0078e8b3, 0x01171393, 0x0063e333, 0x0068c8b3, 0x00589313, 0x011308b3, 0x00b8d313, 0x011348b3,
    0xf8589ce3, 0x00d80833, 0x00082883, 0xf8e8e6e3, 0x40e888b3, 0x00279793, 0x01182023, 0x00d787b3,
    0x0007a803, 0x00e80733, 0x00e7a023, 0xf6dff06f, 0x0000006f,
];

// Demo rollup config: number of accounts in the balance state.
const TX_ACCOUNTS: u32 = 8;

fn key_of(acct: u32) -> u32 {
    0xABCD0000u32 ^ acct
}

// Multiply-free keyed MAC (must match programs/txproc mac()).
fn mac(key: u32, s: u32, r: u32, amt: u32) -> u32 {
    let mut h = key ^ s.rotate_left(7);
    h = h.wrapping_add(r).rotate_left(13);
    h ^= amt.rotate_left(17);
    h = h.wrapping_add(h << 5);
    h ^= h >> 11;
    h
}

/// Genesis balance state: balance[a]=1e6, key[a]=key_of(a), for a in 0..ACCOUNTS.
fn genesis_state() -> Vec<(u32, u32)> {
    let mut mem = Vec::new();
    for a in 0..TX_ACCOUNTS {
        mem.push((0x2000 + 4 * a, 1_000_000));
    }
    for a in 0..TX_ACCOUNTS {
        mem.push((0x3000 + 4 * a, key_of(a)));
    }
    mem
}

/// A block of `n` valid signed transfers (1 unit each, round-robin) for SIMD lane
/// `lane`, encoded as input bytes: [N, (sender,recipient,amount,sig) * N]. The
/// sender/recipient ring is rotated by `lane` so each of the 8 batched lanes is a
/// DISTINCT set of transfers (chained: lane l runs over lane l-1's post-state).
fn tx_batch_lane(n: u32, lane: u32) -> Vec<u8> {
    let mut input = Vec::new();
    input.extend_from_slice(&n.to_le_bytes());
    for i in 0..n {
        let s = (i + lane) % TX_ACCOUNTS;
        let r = (i + lane + 1) % TX_ACCOUNTS;
        let amt = 1u32;
        let sig = mac(key_of(s), s, r, amt);
        for v in [s, r, amt, sig] {
            input.extend_from_slice(&v.to_le_bytes());
        }
    }
    input
}

/// Lane-0 transfer batch (used for `emit-sample`; the rollup expands each block to
/// 8 distinct lanes at prove time).
fn tx_batch(n: u32) -> Vec<u8> {
    tx_batch_lane(n, 0)
}

/// Number of transactions `N` encoded in a batch input (its first LE word).
fn batch_n(input: &[u8]) -> u32 {
    if input.len() >= 4 {
        u32::from_le_bytes([input[0], input[1], input[2], input[3]])
    } else {
        0
    }
}

fn main() {
    // The rollup deploys ONE program: the transaction contract. Block 1 seeds the
    // genesis balance state; subsequent blocks apply signed-transfer batches over
    // the persistent state. `emit-sample <genesis|txbatch [N]>` prints the JSON.
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "emit-sample") {
        let mode = args.iter().skip_while(|a| *a != "emit-sample").nth(1).map(String::as_str).unwrap_or("txbatch");
        let (name, input, mem): (&str, Vec<u8>, Vec<(u32, u32)>) = if mode == "genesis" {
            ("genesis", 0u32.to_le_bytes().to_vec(), genesis_state())
        } else {
            let n: u32 = args.iter().skip_while(|a| *a != "emit-sample").nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
            // Seed the genesis balance state in every batch block so the transfers
            // apply over a funded state (self-contained demo). The declared `mem`
            // overrides the carried state, so each block starts from genesis and
            // proves genesis_root -> post-batch root.
            ("transactions", tx_batch(n), genesis_state())
        };
        let memj: Vec<Value> = mem.iter().map(|(a, v)| json!([a, v])).collect();
        let sub = json!({"name":name,"program":hexs(&words_le(TXPROC)),"input":hexs(&input),"mem":memj,"source":SRC_TXPROC});
        println!("{sub}");
        return;
    }
    let addr = env_or("RV32_ADDR", "0.0.0.0:8545");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    eprintln!("rv32-rollup: accidental rv32i computer on http://{addr}");
    eprintln!("rv32-rollup: methods = rv32_submit, accProof_listBlockProofs, accProof_getBlockProof");
    eprintln!("rv32-rollup: 1 block / {BLOCK_SECS}s, {}-day retention, DA={}", PRUNE_SECS / 86400, env_or("DA_ADDRESS", "http://localhost:26658"));

    let cache: Cache = Arc::new(Mutex::new(BTreeMap::new()));
    let (tx, rx) = channel::<Submission>();
    {
        let cache = cache.clone();
        std::thread::spawn(move || block_worker(rx, cache));
    }
    for stream in listener.incoming() {
        let mut stream = match stream { Ok(s) => s, Err(_) => continue };
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() { continue; }
        let mut clen = 0usize;
        loop {
            let mut h = String::new();
            if reader.read_line(&mut h).unwrap_or(0) == 0 { break; }
            let t = h.trim_end();
            if t.is_empty() { break; }
            if let Some(v) = t.to_ascii_lowercase().strip_prefix("content-length:") {
                clen = v.trim().parse().unwrap_or(0);
            }
        }
        if !line.starts_with("POST") {
            let _ = write_http(&mut stream, "200 OK", "{}");
            continue;
        }
        let mut body = vec![0u8; clen];
        if reader.read_exact(&mut body).is_err() {
            let _ = write_http(&mut stream, "400 Bad Request", r#"{"error":"short body"}"#);
            continue;
        }
        let resp = handle_body(&body, &cache, &tx);
        let _ = write_http(&mut stream, "200 OK", &resp);
    }
}
