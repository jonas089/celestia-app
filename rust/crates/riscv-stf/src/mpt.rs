//! R7 (native reference) — Ethereum world-state Merkle-Patricia-Trie root.
//!
//! The state root is `keccak(rlp(root_node))` of the secure hexary MPT mapping
//! `keccak256(address) -> rlp([nonce, balance, storageRoot, codeHash])`. This is
//! the golden the in-circuit MPT must reproduce; it is validated against the
//! canonical empty-trie root and (once the ev-reth oracle's alloy-trie state_root
//! is available) against authoritative reth vectors.
//!
//! Node encodings (Yellow Paper appendix D): leaf `[HP(path,true), value]`,
//! extension `[HP(path,false), ref(child)]`, branch `[c0..c15, value]`. A child
//! reference is the child's RLP inline if `< 32` bytes, else `keccak(rlp(child))`.

use crate::rlp::{enc_bytes, enc_list};
use num_bigint::BigInt;
use tiny_keccak::Hasher;

pub fn keccak256(b: &[u8]) -> [u8; 32] {
    let mut h = tiny_keccak::Keccak::v256();
    h.update(b);
    let mut o = [0u8; 32];
    h.finalize(&mut o);
    o
}

/// keccak256(rlp("")) — root of an empty trie (computed, not transcribed).
pub fn empty_root() -> [u8; 32] {
    keccak256(&[0x80])
}
/// keccak256("") — EOA code hash.
pub fn keccak_empty() -> [u8; 32] {
    keccak256(&[])
}

/// RLP of an account: [nonce, balance, storageRoot, codeHash].
pub fn account_rlp(nonce: u64, balance: &BigInt, storage_root: &[u8; 32], code_hash: &[u8; 32]) -> Vec<u8> {
    enc_list(&[
        crate::rlp::enc_uint(&BigInt::from(nonce)),
        crate::rlp::enc_uint(balance),
        enc_bytes(storage_root),
        enc_bytes(code_hash),
    ])
}

/// 64 nibbles (high-nibble first) of a 32-byte key.
pub fn nibbles(key: &[u8; 32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(64);
    for &b in key {
        v.push(b >> 4);
        v.push(b & 0x0f);
    }
    v
}

/// Hex-prefix encoding of a nibble path (leaf/extension flag + parity).
pub fn hp(path: &[u8], leaf: bool) -> Vec<u8> {
    let flag = if leaf { 2u8 } else { 0u8 };
    let mut out = Vec::new();
    if path.len() % 2 == 1 {
        out.push(((flag + 1) << 4) | path[0]);
        let mut i = 1;
        while i + 1 < path.len() {
            out.push((path[i] << 4) | path[i + 1]);
            i += 2;
        }
    } else {
        out.push(flag << 4);
        let mut i = 0;
        while i + 1 < path.len() {
            out.push((path[i] << 4) | path[i + 1]);
            i += 2;
        }
    }
    out
}

/// The item to place in a parent: inline child RLP if `< 32` bytes, else the
/// 32-byte hash as an RLP string.
fn node_ref(node_rlp: Vec<u8>) -> Vec<u8> {
    if node_rlp.len() < 32 {
        node_rlp
    } else {
        enc_bytes(&keccak256(&node_rlp))
    }
}

/// Build the RLP of the trie node covering `entries` (each: remaining nibbles +
/// value bytes). `entries` non-empty, sorted by key.
fn build(entries: &[(Vec<u8>, Vec<u8>)]) -> Vec<u8> {
    if entries.len() == 1 {
        let (path, value) = &entries[0];
        return enc_list(&[enc_bytes(&hp(path, true)), enc_bytes(value)]);
    }
    // Longest common prefix of all remaining nibble paths.
    let first = &entries[0].0;
    let mut cp = first.len();
    for (p, _) in entries {
        cp = cp.min(common_len(first, p));
    }
    if cp > 0 {
        let stripped: Vec<(Vec<u8>, Vec<u8>)> =
            entries.iter().map(|(p, v)| (p[cp..].to_vec(), v.clone())).collect();
        let child = build(&stripped);
        return enc_list(&[enc_bytes(&hp(&first[..cp], false)), node_ref(child)]);
    }
    // Branch: group by first nibble.
    let mut slots: Vec<Vec<u8>> = vec![enc_bytes(&[]); 17];
    for nib in 0u8..16 {
        let sub: Vec<(Vec<u8>, Vec<u8>)> = entries
            .iter()
            .filter(|(p, _)| !p.is_empty() && p[0] == nib)
            .map(|(p, v)| (p[1..].to_vec(), v.clone()))
            .collect();
        if !sub.is_empty() {
            slots[nib as usize] = node_ref(build(&sub));
        }
    }
    if let Some((_, v)) = entries.iter().find(|(p, _)| p.is_empty()) {
        slots[16] = enc_bytes(v);
    }
    enc_list(&slots)
}

fn common_len(a: &[u8], b: &[u8]) -> usize {
    let mut i = 0;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i
}

/// State root over `(key32, value_bytes)` pairs (key = keccak(address), value =
/// account RLP). Empty set => EMPTY_ROOT.
pub fn trie_root(mut kv: Vec<([u8; 32], Vec<u8>)>) -> [u8; 32] {
    if kv.is_empty() {
        return keccak256(&[0x80]);
    }
    kv.sort_by(|a, b| a.0.cmp(&b.0));
    let entries: Vec<(Vec<u8>, Vec<u8>)> = kv.iter().map(|(k, v)| (nibbles(k), v.clone())).collect();
    keccak256(&build(&entries))
}

/// A world-state account for the STF.
#[derive(Clone, Debug)]
pub struct Account {
    pub address: [u8; 20],
    pub nonce: u64,
    pub balance: BigInt,
    pub code_hash: [u8; 32],
    pub storage_root: [u8; 32],
}

impl Account {
    pub fn eoa(address: [u8; 20], nonce: u64, balance: BigInt) -> Self {
        Account { address, nonce, balance, code_hash: keccak_empty(), storage_root: empty_root() }
    }
}

/// 32-byte big-endian of a u256 BigInt.
fn be32(x: &BigInt) -> [u8; 32] {
    let mut o = [0u8; 32];
    let b = x.to_bytes_be().1;
    o[32 - b.len()..].copy_from_slice(&b);
    o
}

/// Storage trie root: secure trie over keccak256(slot_be32) -> RLP(value), zero
/// slots pruned. Matches Ethereum/alloy storage_root.
pub fn storage_root(slots: &[(BigInt, BigInt)]) -> [u8; 32] {
    let kv: Vec<([u8; 32], Vec<u8>)> = slots
        .iter()
        .filter(|(_, v)| v != &BigInt::from(0u32))
        .map(|(k, v)| (keccak256(&be32(k)), crate::rlp::enc_uint(v)))
        .collect();
    trie_root(kv)
}

impl Account {
    /// A contract account: codeHash = keccak(code), storageRoot = MPT(storage).
    pub fn contract(address: [u8; 20], nonce: u64, balance: BigInt, code: &[u8], storage: &[(BigInt, BigInt)]) -> Self {
        Account { address, nonce, balance, code_hash: keccak256(code), storage_root: storage_root(storage) }
    }
}

/// World-state root over a set of accounts (secure trie).
pub fn state_root(accounts: &[Account]) -> [u8; 32] {
    let kv: Vec<([u8; 32], Vec<u8>)> = accounts
        .iter()
        .map(|a| {
            let key = keccak256(&a.address);
            let val = account_rlp(a.nonce, &a.balance, &a.storage_root, &a.code_hash);
            (key, val)
        })
        .collect();
    trie_root(kv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hx(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    fn parse_hex32(s: &str) -> [u8; 32] {
        let mut o = [0u8; 32];
        for i in 0..32 {
            o[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
        }
        o
    }

    #[test]
    fn keccak_primitives_and_empty_root() {
        println!("empty_root()   = 0x{}", hx(&empty_root()));
        println!("keccak_empty() = 0x{}", hx(&keccak_empty()));
        // keccak_empty is the canonical EOA code hash (verified vs Go x/crypto).
        assert_eq!(keccak_empty(), parse_hex32("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"));
        // empty_root = keccak256(rlp("")) = keccak256(0x80), verified byte-identical
        // against Go x/crypto Legacy Keccak256 (the Ethereum keccak reth uses).
        assert_eq!(empty_root(), parse_hex32("56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421"));
        assert_eq!(state_root(&[]), empty_root());
    }

    #[test]
    fn contract_account_roots_print() {
        // Print for cross-check vs ev-reth/alloy oracle golden (contract_state_root).
        let code: Vec<u8> = (0..10).map(|i| u8::from_str_radix(&"602a60005560016000f3"[2 * i..2 * i + 2], 16).unwrap()).collect();
        let sr = storage_root(&[(BigInt::from(0u32), BigInt::from(0x2au32))]);
        println!("R8 native contract codeHash    = 0x{}", hx(&keccak256(&code)));
        println!("R8 native contract storageRoot = 0x{}", hx(&sr));
        let one = Account::contract([0u8; 20], 1, BigInt::from(0u32), &code, &[(BigInt::from(0u32), BigInt::from(0x2au32))]);
        println!("R8 native one_contract state_root = 0x{}", hx(&state_root(std::slice::from_ref(&one))));
        let empty = Account::contract([0u8; 20], 1, BigInt::from(0u32), &code, &[]);
        println!("R8 native empty_storage_contract state_root = 0x{}", hx(&state_root(std::slice::from_ref(&empty))));
        assert_eq!(Account::contract([0u8;20],1,BigInt::from(0u32),&code,&[]).storage_root, empty_root());
    }

    #[test]
    fn small_state_root_is_stable_and_structural() {
        // 3 accounts (sender, recipient, coinbase) — prints the root for
        // cross-checking against the ev-reth alloy-trie golden.
        let accs = vec![
            Account::eoa([0x11; 20], 7, BigInt::from(1_000_000_000_000_000_000u64)),
            Account::eoa([0xAB; 20], 0, BigInt::from(500_000_000_000_000u64)),
            Account::eoa([0xCC; 20], 0, BigInt::from(42_000u64)),
        ];
        let root = state_root(&accs);
        println!("R7 native state_root(3 accts) = 0x{}", hx(&root));
        // deterministic
        assert_eq!(state_root(&accs), root);
        // single-account root differs from empty
        let one = state_root(&accs[..1]);
        println!("R7 native state_root(1 acct)  = 0x{}", hx(&one));
        assert_ne!(one, empty_root());
    }
}
