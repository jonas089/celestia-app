//! R7c — in-circuit Ethereum Merkle-Patricia-Trie INCLUSION + UPDATE over GF2.
//!
//! Given a committed PARENT state root and an MPT witness (the RLP-encoded trie
//! nodes along a key's path, root -> leaf) this proves, in-circuit:
//!   (a) INCLUSION: the leaf's value is included under the committed parent root
//!       (`verify_inclusion`), and
//!   (b) UPDATE: after replacing that leaf with a new value and rehashing the
//!       path bottom-up, the recomputed root (`update_and_root`) equals alloy's
//!       post root.
//!
//! This is the state-transition core: reth recomputes a root not by re-parsing a
//! trie but by rehashing the changed path, and the Merkle proof supplies exactly
//! the sibling data (other branch children's hashes, extension paths) needed to
//! do so. Both gadgets operate on committed node byte-buffers plus compile-time
//! STRUCTURAL constants (the byte offset of the child reference inside each
//! parent, and the leaf value offset) that are derived from the trie shape by the
//! native reference below — exactly as `mpt_circuit` treats `hp32`/`slot`.
//!
//! Node encodings match `mpt.rs` (Yellow Paper appendix D) byte-for-byte, and
//! `mpt.rs`'s roots are the ones already cross-checked against reth/alloy vectors,
//! so a circuit result that equals the native reference equals alloy.
//!
//! Scope (single keccak block => every node <= 135 bytes):
//!  * node types: branch (<= 3 hash children, <= 135 B), extension, leaf;
//!  * depth: unbounded for `verify_inclusion` (it only hashes committed buffers
//!    and checks the hash chain); `update_and_root` is unbounded for the path
//!    above the leaf, the leaf itself is rebuilt with the account/storage leaf
//!    gadgets from `mpt_circuit`.
//!  * hash-referenced children only (secure trie: 64-nibble keccak keys => every
//!    node on a path is >= 32 bytes). Embedded (<32 B) children and >135 B
//!    (>= 4-child) branches are out of scope — see the report / module tail.

use crate::mpt::{hp, keccak256, nibbles};
use crate::rlp::{enc_bytes, enc_list};

// ============================================================================
// Native reference: a structured hexary MPT that can emit Merkle proofs.
// ============================================================================

/// A native trie node. Encoding is identical to `mpt::build`.
#[derive(Clone, Debug)]
pub enum Node {
    /// `[HP(path, leaf=true), value]`
    Leaf { path: Vec<u8>, value: Vec<u8> },
    /// `[HP(path, leaf=false), ref(child)]`
    Extension { path: Vec<u8>, child: Box<Node> },
    /// `[c0..c15, value]` — value is `None` for secure tries.
    Branch { children: Vec<Option<Box<Node>>>, value: Option<Vec<u8>> },
}

fn common_len(a: &[u8], b: &[u8]) -> usize {
    let mut i = 0;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Build the structured node covering `entries` (each: remaining nibbles + value
/// bytes). `entries` non-empty, sorted by key. Mirrors `mpt::build`.
fn build_node(entries: &[(Vec<u8>, Vec<u8>)]) -> Node {
    if entries.len() == 1 {
        let (path, value) = &entries[0];
        return Node::Leaf { path: path.clone(), value: value.clone() };
    }
    let first = &entries[0].0;
    let mut cp = first.len();
    for (p, _) in entries {
        cp = cp.min(common_len(first, p));
    }
    if cp > 0 {
        let stripped: Vec<(Vec<u8>, Vec<u8>)> =
            entries.iter().map(|(p, v)| (p[cp..].to_vec(), v.clone())).collect();
        return Node::Extension { path: first[..cp].to_vec(), child: Box::new(build_node(&stripped)) };
    }
    let mut children: Vec<Option<Box<Node>>> = vec![None; 16];
    for nib in 0u8..16 {
        let sub: Vec<(Vec<u8>, Vec<u8>)> = entries
            .iter()
            .filter(|(p, _)| !p.is_empty() && p[0] == nib)
            .map(|(p, v)| (p[1..].to_vec(), v.clone()))
            .collect();
        if !sub.is_empty() {
            children[nib as usize] = Some(Box::new(build_node(&sub)));
        }
    }
    let value = entries.iter().find(|(p, _)| p.is_empty()).map(|(_, v)| v.clone());
    Node::Branch { children, value }
}

/// The item placed in a parent slot: inline child RLP if `< 32` bytes, else the
/// 32-byte hash as an RLP string. Matches `mpt::node_ref`.
fn node_ref(node_rlp: &[u8]) -> Vec<u8> {
    if node_rlp.len() < 32 {
        node_rlp.to_vec()
    } else {
        enc_bytes(&keccak256(node_rlp))
    }
}

/// RLP encoding of a node — byte-identical to `mpt::build`'s output.
pub fn node_rlp(n: &Node) -> Vec<u8> {
    match n {
        Node::Leaf { path, value } => enc_list(&[enc_bytes(&hp(path, true)), enc_bytes(value)]),
        Node::Extension { path, child } => {
            let c = node_rlp(child);
            enc_list(&[enc_bytes(&hp(path, false)), node_ref(&c)])
        }
        Node::Branch { children, value } => {
            let mut slots: Vec<Vec<u8>> = Vec::with_capacity(17);
            for c in children.iter() {
                match c {
                    Some(child) => slots.push(node_ref(&node_rlp(child))),
                    None => slots.push(enc_bytes(&[])),
                }
            }
            match value {
                Some(v) => slots.push(enc_bytes(v)),
                None => slots.push(enc_bytes(&[])),
            }
            enc_list(&slots)
        }
    }
}

/// keccak256(rlp(node)) — the node hash / trie root of the subtree.
pub fn node_hash(n: &Node) -> [u8; 32] {
    keccak256(&node_rlp(n))
}

/// Build the root node from `(key32, value_bytes)` pairs (secure trie). Panics on
/// empty input (an empty trie has no path to prove).
pub fn root_node(mut kv: Vec<([u8; 32], Vec<u8>)>) -> Node {
    assert!(!kv.is_empty(), "empty trie has no inclusion path");
    kv.sort_by(|a, b| a.0.cmp(&b.0));
    let entries: Vec<(Vec<u8>, Vec<u8>)> =
        kv.iter().map(|(k, v)| (nibbles(k), v.clone())).collect();
    build_node(&entries)
}

/// A Merkle-Patricia inclusion proof for one key, in the shape reth's
/// `eth_getProof` returns: `nodes` are the RLP-encoded trie nodes from the root
/// down to the leaf (each hash-referenced by its parent), plus the structural
/// offsets the circuit needs.
#[derive(Clone, Debug)]
pub struct PathProof {
    /// RLP node bytes, `nodes[0]` = root node, `nodes[last]` = leaf. Each node
    /// is <= 135 bytes (single keccak block) in this scope.
    pub nodes: Vec<Vec<u8>>,
    /// `child_off[d]` = byte offset in `nodes[d]` where `nodes[d+1]`'s 32-byte
    /// hash reference sits (len = nodes.len() - 1).
    pub child_off: Vec<usize>,
    /// Byte offset in the leaf node where the value CONTENT begins.
    pub value_off: usize,
    /// The leaf value bytes (RLP(account) for the account trie, RLP(slot value)
    /// for a storage trie).
    pub value: Vec<u8>,
    /// HP-encoded nibble path stored in the leaf (for cross-checking the key).
    pub leaf_hp: Vec<u8>,
}

/// Walk `root` along `key`'s nibble path, emitting the inclusion proof.
pub fn prove(root: &Node, key: &[u8; 32]) -> ([u8; 32], PathProof) {
    let path = nibbles(key);
    let mut rem: &[u8] = &path;
    let mut nodes: Vec<Vec<u8>> = Vec::new();
    let mut cur: &Node = root;
    let (value, leaf_hp);
    loop {
        nodes.push(node_rlp(cur));
        match cur {
            Node::Leaf { path: p, value: v } => {
                assert_eq!(rem, &p[..], "key does not terminate at this leaf");
                value = v.clone();
                leaf_hp = hp(p, true);
                break;
            }
            Node::Extension { path: p, child } => {
                assert!(rem.starts_with(p), "extension path mismatch");
                rem = &rem[p.len()..];
                cur = child;
            }
            Node::Branch { children, .. } => {
                let nib = rem[0] as usize;
                rem = &rem[1..];
                cur = children[nib].as_ref().expect("key routes to an empty branch slot");
            }
        }
    }
    let root_hash = keccak256(&nodes[0]);
    // child offsets: locate nodes[d+1]'s hash inside nodes[d].
    let mut child_off = Vec::with_capacity(nodes.len().saturating_sub(1));
    for d in 0..nodes.len() - 1 {
        let child_hash = keccak256(&nodes[d + 1]);
        let off = find_subslice(&nodes[d], &child_hash)
            .expect("child hash not found in parent (inline child? out of scope)");
        child_off.push(off);
    }
    // value content is the tail of the leaf RLP.
    let leaf = nodes.last().unwrap();
    let value_off = leaf.len() - value.len();
    // Nodes may exceed one keccak block now (dense branches are hashed multi-block
    // via `PathShape::node_lens`); leaves stay <= 135 B for reasonable balances.
    (root_hash, PathProof { nodes, child_off, value_off, value, leaf_hp })
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

// ============================================================================
// In-circuit gadgets (GF2).
// ============================================================================

use expander_compiler::frontend::*;

/// One committed proof node: `bits` = the 135-byte buffer (LSB-first per byte,
/// tail beyond the node length may be anything), `len_bits` = the 8-bit LE node
/// byte-length (<= 135).
pub struct NodeWitness {
    pub bits: Vec<Variable>,
    pub len_bits: Vec<Variable>,
}

/// Compile-time structural descriptor of a path (derived from the native
/// `PathProof`): the child-reference offset at each internal node, the leaf value
/// offset/length, and each node's byte length. `node_lens[d]` selects single-
/// block hashing (<= 135 B) or the multi-block keccak (dense branches > 135 B).
pub struct PathShape {
    pub child_off: Vec<usize>,
    pub value_off: usize,
    pub value_len: usize,
    /// Byte length of each node (root-first). Nodes > 135 B are hashed with the
    /// multi-block keccak; <= 135 B with the single-block variable-length keccak.
    pub node_lens: Vec<usize>,
}

/// Hash one committed node with the RIGHT keccak: single-block variable-length
/// for nodes <= 135 B (a dense-branch path may mix small leaves with a large
/// branch), multi-block fixed-length for dense nodes > 135 B. `node_len` is the
/// compile-time byte length from `PathShape::node_lens`.
fn keccak_node<C: Config>(api: &mut impl RootAPI<C>, n: &NodeWitness, node_len: usize) -> Vec<Variable> {
    if node_len <= 135 {
        crate::batch_keccak::keccak256_varlen(api, &n.bits[..135 * 8], &n.len_bits)
    } else {
        crate::batch_keccak::keccak256_fixed(api, &n.bits, node_len)
    }
}

/// Byte-slice `[off .. off+32]` of a node buffer as a 256-bit vector in keccak
/// output order (bit `b*8+j` = byte `off+b`, bit `j`).
fn node_word(node_bits: &[Variable], off: usize) -> Vec<Variable> {
    node_bits[off * 8..(off + 32) * 8].to_vec()
}

/// INCLUSION: keccak-hash each committed node, check the root of the first node
/// equals the committed `root_bits`, that each parent's child reference equals
/// keccak(next node), and that the leaf holds `value_bytes` at the value offset.
pub fn verify_inclusion<C: Config>(
    api: &mut impl RootAPI<C>,
    root_bits: &[Variable],
    value_bytes: &[Vec<Variable>],
    nodes: &[NodeWitness],
    shape: &PathShape,
) {
    let depth = nodes.len();
    assert!(depth >= 1);
    // hash every node (single- or multi-block per its compile-time length).
    assert_eq!(shape.node_lens.len(), depth, "node_lens must cover every node");
    let hashes: Vec<Vec<Variable>> = nodes
        .iter()
        .enumerate()
        .map(|(d, n)| keccak_node(api, n, shape.node_lens[d]))
        .collect();
    // root of first node == committed root.
    for i in 0..256 {
        api.assert_is_equal(hashes[0][i], root_bits[i]);
    }
    // each child reference == keccak(next node).
    for d in 0..depth - 1 {
        let off = shape.child_off[d];
        let refw = node_word(&nodes[d].bits, off);
        for i in 0..256 {
            api.assert_is_equal(refw[i], hashes[d + 1][i]);
        }
    }
    // leaf holds the value at the value offset.
    let leaf = &nodes[depth - 1].bits;
    assert_eq!(value_bytes.len(), shape.value_len);
    for k in 0..shape.value_len {
        let base = (shape.value_off + k) * 8;
        for j in 0..8 {
            api.assert_is_equal(leaf[base + j], value_bytes[k][j]);
        }
    }
}

/// UPDATE: recompute the root bottom-up with the leaf replaced. `new_leaf_hash`
/// is keccak(new leaf node) (256 bits, keccak output order) — computed by the
/// caller with `mpt_circuit::account_leaf_hash` / `storage_leaf_root`. For each
/// parent (leaf's parent up to the root) the committed node buffer is reused with
/// the child reference spliced to the freshly recomputed child hash (only 32
/// bytes change; the node length is invariant), and rehashed. Returns the new
/// 256-bit root.
pub fn update_and_root<C: Config>(
    api: &mut impl RootAPI<C>,
    new_leaf_hash: &[Variable],
    nodes: &[NodeWitness],
    shape: &PathShape,
) -> Vec<Variable> {
    let depth = nodes.len();
    let mut cur = new_leaf_hash.to_vec();
    // walk parents from the leaf's parent (depth-2) up to the root (0).
    for d in (0..depth.saturating_sub(1)).rev() {
        let off = shape.child_off[d];
        let mut buf = nodes[d].bits.clone();
        for b in 0..32 {
            for j in 0..8 {
                buf[(off + b) * 8 + j] = cur[b * 8 + j];
            }
        }
        let spliced = NodeWitness { bits: buf, len_bits: nodes[d].len_bits.clone() };
        cur = keccak_node(api, &spliced, shape.node_lens[d]);
    }
    cur
}

// ============================================================================
// VARIABLE-PATH leaf builder: builds the account leaf RLP + keccak for the ACTUAL
// remaining nibble path taken from the proof (any HP length), rather than the
// hardcoded 63/64-nibble builders in `mpt_circuit`. The HP encoding `hp_enc` =
// `enc_bytes(hp(remaining_path, leaf))` is a compile-time constant derived from
// the proof; only the account fields are circuit variables.
// ============================================================================

use crate::u256::add as u256_add;

fn byte_const_v<C: Config>(api: &mut impl RootAPI<C>, v: u8) -> Vec<Variable> {
    (0..8).map(|b| api.constant(((v >> b) & 1) as u32)).collect()
}
fn const_bits8<C: Config>(api: &mut impl RootAPI<C>, v: u32) -> Vec<Variable> {
    (0..8).map(|b| api.constant((v >> b) & 1)).collect()
}
fn zeros_bytes<C: Config>(api: &mut impl RootAPI<C>, n: usize) -> Vec<Vec<Variable>> {
    let zero = api.constant(0);
    (0..n).map(|_| vec![zero; 8]).collect()
}
/// Add two <=8-bit LE numbers -> low 8 bits.
fn add8<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let s = u256_add(api, a, b);
    s[0..8].to_vec()
}
fn zext<C: Config>(api: &mut impl RootAPI<C>, src: &[Vec<Variable>], w: usize) -> Vec<Vec<Variable>> {
    let zero = api.constant(0);
    (0..w).map(|i| if i < src.len() { src[i].clone() } else { vec![zero; 8] }).collect()
}
/// Right-shift a byte buffer by a variable offset into a `w`-byte buffer.
fn barrel_shr<C: Config>(api: &mut impl RootAPI<C>, src: &[Vec<Variable>], shift_bits: &[Variable], w: usize) -> Vec<Vec<Variable>> {
    let zero = api.constant(0);
    let mut cur = zext(api, src, w);
    for (k, &sbit) in shift_bits.iter().enumerate() {
        let sh = 1usize << k;
        if sh >= w {
            break;
        }
        let shifted: Vec<Vec<Variable>> =
            (0..w).map(|i| if i >= sh { cur[i - sh].clone() } else { vec![zero; 8] }).collect();
        cur = (0..w)
            .map(|i| (0..8).map(|j| { let d = api.add(shifted[i][j], cur[i][j]); let t = api.mul(sbit, d); api.add(cur[i][j], t) }).collect())
            .collect();
    }
    cur
}
fn xor_bytes<C: Config>(api: &mut impl RootAPI<C>, a: &[Vec<Variable>], b: &[Vec<Variable>]) -> Vec<Vec<Variable>> {
    (0..a.len()).map(|i| (0..8).map(|j| api.add(a[i][j], b[i][j])).collect()).collect()
}
/// enc_bytes of a 32-byte word given as a constant: [0xa0, b0..b31].
fn enc_word_const<C: Config>(api: &mut impl RootAPI<C>, b: &[u8; 32]) -> Vec<Vec<Variable>> {
    let mut out = vec![byte_const_v(api, 0xa0)];
    for &x in b.iter() { out.push(byte_const_v(api, x)); }
    out
}
/// enc_bytes of a 32-byte word given as circuit bits (byte b = word[b*8..b*8+8]).
fn enc_word_var<C: Config>(api: &mut impl RootAPI<C>, word: &[Variable]) -> Vec<Vec<Variable>> {
    let mut out = vec![byte_const_v(api, 0xa0)];
    for b in 0..32 { out.push(word[b * 8..b * 8 + 8].to_vec()); }
    out
}

/// Assemble + keccak an account leaf `[hp_enc, enc(account_rlp)]` for an ARBITRARY
/// remaining path (`hp_enc` = compile-time `enc_bytes(hp(path,leaf))`). `sr_enc` /
/// `ch_enc` are the (possibly variable) `enc_bytes` of storageRoot / codeHash.
/// Single-block (leaf <= 135 B; holds for balances < ~2^128, matching the existing
/// account-leaf scope). Byte-identical to `mpt::node_rlp` of the leaf, so its
/// keccak equals the native/alloy leaf hash.
fn account_leaf_path_core<C: Config>(
    api: &mut impl RootAPI<C>,
    hp_enc: &[u8],
    nonce: &[Variable],
    balance: &[Variable],
    sr_enc: &[Vec<Variable>],
    ch_enc: &[Vec<Variable>],
) -> Vec<Variable> {
    use crate::mpt_circuit::min_rlp_uint;
    let (nb, n1) = min_rlp_uint(api, nonce);
    let (bb, n2) = min_rlp_uint(api, balance);
    // account body = enc(nonce) ++ enc(balance) ++ enc(sr) ++ enc(ch)
    let mut n1_8 = n1.clone(); while n1_8.len() < 8 { n1_8.push(api.constant(0)); }
    let mut n2_8 = n2.clone(); while n2_8.len() < 8 { n2_8.push(api.constant(0)); }
    let c33 = const_bits8(api, 33);
    let o2 = n1_8.clone();
    let o3 = add8(api, &n1_8, &n2_8);
    let o4 = add8(api, &o3, &c33);
    const WBODY: usize = 108;
    let p1 = zext(api, &nb, WBODY);
    let p2 = barrel_shr(api, &bb, &o2, WBODY);
    let p3 = barrel_shr(api, sr_enc, &o3, WBODY);
    let p4 = barrel_shr(api, ch_enc, &o4, WBODY);
    let b12 = xor_bytes(api, &p1, &p2);
    let b123 = xor_bytes(api, &b12, &p3);
    let body = xor_bytes(api, &b123, &p4);
    let body_len = add8(api, &o4, &c33); // n1 + n2 + 66

    // account_inner = [0xf8, body_len, body]  (body >= 66 => long-list header)
    const WACCT: usize = WBODY + 2;
    let mut acct = zeros_bytes(api, WACCT);
    acct[0] = byte_const_v(api, 0xf8);
    acct[1] = body_len.clone();
    for i in 0..WBODY { acct[2 + i] = body[i].clone(); }
    let c2 = const_bits8(api, 2);
    let account_len = add8(api, &body_len, &c2);

    // leaf = [0xf8, leaf_body_len, hp_enc.., 0xb8, account_len, account_inner]
    let l_enc = hp_enc.len();
    let mut leaf = zeros_bytes(api, 200);
    leaf[0] = byte_const_v(api, 0xf8);
    let c_extra = const_bits8(api, (l_enc + 2) as u32); // hp_enc + [0xb8, account_len]
    let leaf_body_len = add8(api, &c_extra, &account_len);
    leaf[1] = leaf_body_len.clone();
    for (i, &x) in hp_enc.iter().enumerate() { leaf[2 + i] = byte_const_v(api, x); }
    leaf[2 + l_enc] = byte_const_v(api, 0xb8);
    leaf[3 + l_enc] = account_len.clone();
    for i in 0..WACCT {
        if 4 + l_enc + i < 200 { leaf[4 + l_enc + i] = acct[i].clone(); }
    }
    let leaf_len = add8(api, &c2, &leaf_body_len); // 2 + leaf_body_len

    let flat: Vec<Variable> = leaf[0..135].iter().flatten().cloned().collect();
    crate::batch_keccak::keccak256_varlen(api, &flat, &leaf_len)
}

/// Variable-path account leaf hash with CONSTANT storageRoot / codeHash.
pub fn account_leaf_hash_path<C: Config>(
    api: &mut impl RootAPI<C>,
    hp_enc: &[u8],
    nonce: &[Variable],
    balance: &[Variable],
    storage_root: &[u8; 32],
    code_hash: &[u8; 32],
) -> Vec<Variable> {
    let sr = enc_word_const(api, storage_root);
    let ch = enc_word_const(api, code_hash);
    account_leaf_path_core(api, hp_enc, nonce, balance, &sr, &ch)
}

/// Variable-path account leaf hash with a VARIABLE storageRoot word (for accounts
/// whose storage changed) and constant codeHash.
pub fn account_leaf_hash_path_var<C: Config>(
    api: &mut impl RootAPI<C>,
    hp_enc: &[u8],
    nonce: &[Variable],
    balance: &[Variable],
    storage_root: &[Variable],
    code_hash: &[u8; 32],
) -> Vec<Variable> {
    let sr = enc_word_var(api, storage_root);
    let ch = enc_word_const(api, code_hash);
    account_leaf_path_core(api, hp_enc, nonce, balance, &sr, &ch)
}

// ============================================================================
// Tests: every in-circuit result is checked against the native reference
// (mpt.rs / this module), whose roots are the alloy/reth-validated goldens.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpt::{account_rlp, empty_root, keccak_empty, state_root, storage_root, Account};
    use crate::mpt_circuit::{account_leaf_hash, storage_leaf_root};
    use crate::u256::{bigint_to_bits, BITS};
    use expander_compiler::frontend::*;
    use num_bigint::BigInt;

    fn hx(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    /// Fill `dst` (fixed 135*8) with node bytes, tail zeroed; return the node len.
    fn set_node(dst: &mut [GF2], bytes: &[u8]) {
        for i in 0..135 {
            let byte = if i < bytes.len() { bytes[i] } else { 0 };
            for j in 0..8 {
                dst[i * 8 + j] = (((byte >> j) & 1) as u32).into();
            }
        }
    }
    fn set_len(dst: &mut [GF2], len: usize) {
        for b in 0..8 {
            dst[b] = (((len as u32) >> b) & 1).into();
        }
    }
    fn set_word(dst: &mut [GF2], w: &[u8; 32]) {
        for i in 0..32 {
            for j in 0..8 {
                dst[i * 8 + j] = (((w[i] >> j) & 1) as u32).into();
            }
        }
    }

    /// Pick `n` addresses whose keccak keys have DISTINCT first nibbles (flat
    /// single-branch trie), returning (addresses, first-nibble slots).
    fn distinct_first_nibble_addrs(n: usize) -> (Vec<[u8; 20]>, Vec<usize>) {
        let (mut addrs, mut slots, mut seen, mut cand) = (vec![], vec![], vec![], 1u8);
        while addrs.len() < n {
            let a = [cand; 20];
            let s = nibbles(&keccak256(&a))[0] as usize;
            if !seen.contains(&s) {
                addrs.push(a);
                slots.push(s);
                seen.push(s);
            }
            cand = cand.wrapping_add(1);
            assert!(cand != 0, "ran out of candidates");
        }
        (addrs, slots)
    }

    // --------------------------------------------------------------------
    // 1) Native proof shape (reth eth_getProof shape): branch(17)/leaf, HP,
    //    keccak node hashes chain to the root, leaf decodes to the value.
    // --------------------------------------------------------------------
    #[test]
    fn native_proof_shape_account_trie() {
        let (addrs, _slots) = distinct_first_nibble_addrs(3);
        let accs: Vec<Account> = vec![
            Account::eoa(addrs[0], 7, BigInt::from(1_000_000_000_000_000_000u64)),
            Account::eoa(addrs[1], 0, BigInt::from(500_000_000_000_000u64)),
            Account::eoa(addrs[2], 0, BigInt::from(42_000u64)),
        ];
        let kv: Vec<([u8; 32], Vec<u8>)> = accs
            .iter()
            .map(|a| (keccak256(&a.address), account_rlp(a.nonce, &a.balance, &a.storage_root, &a.code_hash)))
            .collect();
        let root = root_node(kv.clone());
        let root_hash = node_hash(&root);
        // native golden: mpt::state_root (alloy/reth-validated).
        assert_eq!(root_hash, state_root(&accs), "proof-trie root != mpt::state_root");

        // prove each key: verify node-hash chain + leaf value.
        for a in &accs {
            let key = keccak256(&a.address);
            let (rh, proof) = prove(&root, &key);
            assert_eq!(rh, root_hash);
            // root node is a 17-item branch.
            assert_eq!(proof.nodes.len(), 2, "distinct-nibble trie => branch -> leaf");
            let (top, _) = crate::rlp::decode_at(&proof.nodes[0], 0);
            assert_eq!(top.as_list().len(), 17, "root must be a 17-slot branch");
            // hash chain: keccak(leaf) sits in the branch at child_off[0].
            let leaf_hash = keccak256(&proof.nodes[1]);
            assert_eq!(&proof.nodes[0][proof.child_off[0]..proof.child_off[0] + 32], &leaf_hash[..]);
            // leaf value is the account RLP.
            let want = account_rlp(a.nonce, &a.balance, &a.storage_root, &a.code_hash);
            assert_eq!(proof.value, want);
            assert_eq!(&proof.nodes[1][proof.value_off..proof.value_off + proof.value.len()], &want[..]);
        }
        println!("R7c native account proof: root=0x{} (branch->leaf, 17-slot, HP, keccak chain OK)", hx(&root_hash));
    }

    #[test]
    fn native_proof_shape_extension() {
        // Two addresses whose keccak keys share >= 1 leading nibble => the root
        // is an EXTENSION over the shared prefix, then a branch, then leaves.
        let (a, b, shared) = find_shared_prefix_pair();
        let accs = vec![
            Account::eoa(a, 3, BigInt::from(111u64)),
            Account::eoa(b, 5, BigInt::from(222u64)),
        ];
        let kv: Vec<([u8; 32], Vec<u8>)> = accs
            .iter()
            .map(|x| (keccak256(&x.address), account_rlp(x.nonce, &x.balance, &x.storage_root, &x.code_hash)))
            .collect();
        let root = root_node(kv);
        assert_eq!(node_hash(&root), state_root(&accs));
        // root must be an extension (2 items: HP, ref).
        match &root {
            Node::Extension { path, .. } => assert_eq!(path.len(), shared),
            _ => panic!("expected extension root"),
        }
        let (_rh, proof) = prove(&root, &keccak256(&a));
        assert_eq!(proof.nodes.len(), 3, "extension -> branch -> leaf");
        let (ext, _) = crate::rlp::decode_at(&proof.nodes[0], 0);
        assert_eq!(ext.as_list().len(), 2, "extension node is [HP, ref]");
        let (br, _) = crate::rlp::decode_at(&proof.nodes[1], 0);
        assert_eq!(br.as_list().len(), 17, "middle node is a branch");
        println!("R7c native extension proof: shared_prefix={} nibbles, ext->branch->leaf OK", shared);
    }

    /// Find two addresses whose keccak keys share >= 1 leading nibble (and no key
    /// is a prefix of the other). Returns (addr_a, addr_b, shared_prefix_len).
    fn find_shared_prefix_pair() -> ([u8; 20], [u8; 20], usize) {
        let mut map: std::collections::HashMap<u8, [u8; 20]> = std::collections::HashMap::new();
        for c in 1u16..=1000 {
            let a = [c as u8; 20];
            let key = keccak256(&a);
            let n0 = nibbles(&key)[0];
            if let Some(&prev) = map.get(&n0) {
                let na = nibbles(&keccak256(&prev));
                let nb = nibbles(&key);
                let shared = common_len(&na, &nb);
                return (prev, a, shared);
            }
            map.insert(n0, a);
        }
        panic!("no shared-prefix pair found");
    }

    // --------------------------------------------------------------------
    // 2) In-circuit INCLUSION — account trie (branch -> leaf, depth 2).
    // --------------------------------------------------------------------
    static mut ACC_SHAPE: Option<PathShape> = None;

    // 3 accounts: two nodes (branch, leaf). value = account RLP (<= 80 bytes).
    const ACC_VLEN: usize = 80;
    declare_circuit!(AcctInclCircuit {
        n0: [Variable; 135 * 8],
        l0: [Variable; 8],
        n1: [Variable; 135 * 8],
        l1: [Variable; 8],
        value: [Variable; ACC_VLEN * 8],
        root: [PublicVariable; 256],
    });
    impl Define<GF2Config> for AcctInclCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let shape = unsafe { (*std::ptr::addr_of!(ACC_SHAPE)).as_ref().unwrap() };
            let nodes = vec![
                NodeWitness { bits: self.n0.to_vec(), len_bits: self.l0.to_vec() },
                NodeWitness { bits: self.n1.to_vec(), len_bits: self.l1.to_vec() },
            ];
            let value: Vec<Vec<Variable>> =
                (0..shape.value_len).map(|k| self.value[k * 8..k * 8 + 8].to_vec()).collect();
            verify_inclusion(api, &self.root.to_vec(), &value, &nodes, shape);
        }
    }

    #[test]
    fn incircuit_account_inclusion_depth2() {
        let (addrs, _slots) = distinct_first_nibble_addrs(3);
        let accs: Vec<Account> = vec![
            Account::eoa(addrs[0], 7, BigInt::from(1_000_000_000_000_000_000u64)),
            Account::eoa(addrs[1], 0, BigInt::from(500_000_000_000_000u64)),
            Account::eoa(addrs[2], 0, BigInt::from(42_000u64)),
        ];
        let kv: Vec<([u8; 32], Vec<u8>)> = accs
            .iter()
            .map(|a| (keccak256(&a.address), account_rlp(a.nonce, &a.balance, &a.storage_root, &a.code_hash)))
            .collect();
        let root = root_node(kv);
        let root_hash = node_hash(&root);
        // prove account 0.
        let (_rh, proof) = prove(&root, &keccak256(&accs[0].address));
        assert_eq!(proof.nodes.len(), 2);
        assert!(proof.value.len() <= ACC_VLEN);

        unsafe {
            ACC_SHAPE = Some(PathShape {
                child_off: proof.child_off.clone(),
                value_off: proof.value_off,
                value_len: proof.value.len(),
                node_lens: proof.nodes.iter().map(|n| n.len()).collect(),
            });
        }
        let CompileResult { witness_solver, layered_circuit } =
            compile(&AcctInclCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = AcctInclCircuit::<GF2>::default();
        set_node(&mut asg.n0, &proof.nodes[0]);
        set_len(&mut asg.l0, proof.nodes[0].len());
        set_node(&mut asg.n1, &proof.nodes[1]);
        set_len(&mut asg.l1, proof.nodes[1].len());
        for k in 0..proof.value.len() {
            for j in 0..8 {
                asg.value[k * 8 + j] = (((proof.value[k] >> j) & 1) as u32).into();
            }
        }
        set_word_pub(&mut asg.root, &root_hash);
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit inclusion != native root");
        println!("R7c in-circuit account INCLUSION OK, root=0x{}", hx(&root_hash));
    }

    fn set_word_pub(dst: &mut [GF2], w: &[u8; 32]) {
        set_word(dst, w);
    }

    // --------------------------------------------------------------------
    // 3) In-circuit UPDATE — account trie (change one balance, depth 2). New
    //    leaf rebuilt with account_leaf_hash; parent (branch) spliced + rehashed.
    // --------------------------------------------------------------------
    static mut UPD_HP32: [u8; 32] = [0u8; 32];
    static mut UPD_SHAPE: Option<PathShape> = None;

    declare_circuit!(AcctUpdateCircuit {
        // committed proof node: the branch (parent of the changed leaf).
        n0: [Variable; 135 * 8],
        l0: [Variable; 8],
        // new leaf account fields.
        new_nonce: [Variable; BITS],
        new_balance: [Variable; BITS],
        new_root: [PublicVariable; 256],
    });
    impl Define<GF2Config> for AcctUpdateCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let hp32 = unsafe { UPD_HP32 };
            let shape = unsafe { (*std::ptr::addr_of!(UPD_SHAPE)).as_ref().unwrap() };
            let sr = empty_root();
            let ch = keccak_empty();
            let new_leaf_hash =
                account_leaf_hash(api, &self.new_nonce.to_vec(), &self.new_balance.to_vec(), &hp32, &sr, &ch);
            // nodes[0] = branch (parent); nodes[1] placeholder (leaf, rebuilt).
            let zero = api.constant(0);
            let nodes = vec![
                NodeWitness { bits: self.n0.to_vec(), len_bits: self.l0.to_vec() },
                NodeWitness { bits: vec![zero; 135 * 8], len_bits: vec![zero; 8] },
            ];
            let nr = update_and_root(api, &new_leaf_hash, &nodes, shape);
            for i in 0..256 {
                api.assert_is_equal(nr[i], self.new_root[i]);
            }
        }
    }

    #[test]
    fn incircuit_account_update_depth2() {
        let (addrs, _slots) = distinct_first_nibble_addrs(3);
        let mut accs: Vec<Account> = vec![
            Account::eoa(addrs[0], 7, BigInt::from(1_000_000_000_000_000_000u64)),
            Account::eoa(addrs[1], 0, BigInt::from(500_000_000_000_000u64)),
            Account::eoa(addrs[2], 0, BigInt::from(42_000u64)),
        ];
        let kv: Vec<([u8; 32], Vec<u8>)> = accs
            .iter()
            .map(|a| (keccak256(&a.address), account_rlp(a.nonce, &a.balance, &a.storage_root, &a.code_hash)))
            .collect();
        let root = root_node(kv);
        let (_rh, proof) = prove(&root, &keccak256(&accs[0].address));

        // new value for account 0.
        let (new_nonce, new_balance) = (8u64, BigInt::from(999_000_000_000_000_000u64));
        // native post root: rebuild the trie with account 0 changed.
        accs[0] = Account::eoa(addrs[0], new_nonce, new_balance.clone());
        let native_new = state_root(&accs);

        // 63-nibble leaf HP for account 0 (branch consumed the first nibble).
        let nibs = nibbles(&keccak256(&addrs[0]));
        let hpv = hp(&nibs[1..64], true);
        assert_eq!(hpv.len(), 32);
        let mut hp32 = [0u8; 32];
        hp32.copy_from_slice(&hpv);

        unsafe {
            UPD_HP32 = hp32;
            UPD_SHAPE = Some(PathShape {
                child_off: proof.child_off.clone(),
                value_off: proof.value_off,
                value_len: proof.value.len(),
                node_lens: proof.nodes.iter().map(|n| n.len()).collect(),
            });
        }
        let CompileResult { witness_solver, layered_circuit } =
            compile(&AcctUpdateCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = AcctUpdateCircuit::<GF2>::default();
        set_node(&mut asg.n0, &proof.nodes[0]);
        set_len(&mut asg.l0, proof.nodes[0].len());
        asg.new_nonce.copy_from_slice(
            &bigint_to_bits(&BigInt::from(new_nonce), BITS).into_iter().map(|b| (b as u32).into()).collect::<Vec<GF2>>(),
        );
        asg.new_balance.copy_from_slice(
            &bigint_to_bits(&new_balance, BITS).into_iter().map(|b| (b as u32).into()).collect::<Vec<GF2>>(),
        );
        set_word(&mut asg.new_root, &native_new);
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit update root != native new root");
        println!("R7c in-circuit account UPDATE OK, new_root=0x{}", hx(&native_new));
    }

    // --------------------------------------------------------------------
    // 4) In-circuit INCLUSION — extension node (ext -> branch -> leaf, depth 3).
    // --------------------------------------------------------------------
    static mut EXT_SHAPE: Option<PathShape> = None;
    const EXT_VLEN: usize = 80;
    declare_circuit!(ExtInclCircuit {
        n0: [Variable; 135 * 8],
        l0: [Variable; 8],
        n1: [Variable; 135 * 8],
        l1: [Variable; 8],
        n2: [Variable; 135 * 8],
        l2: [Variable; 8],
        value: [Variable; EXT_VLEN * 8],
        root: [PublicVariable; 256],
    });
    impl Define<GF2Config> for ExtInclCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let shape = unsafe { (*std::ptr::addr_of!(EXT_SHAPE)).as_ref().unwrap() };
            let nodes = vec![
                NodeWitness { bits: self.n0.to_vec(), len_bits: self.l0.to_vec() },
                NodeWitness { bits: self.n1.to_vec(), len_bits: self.l1.to_vec() },
                NodeWitness { bits: self.n2.to_vec(), len_bits: self.l2.to_vec() },
            ];
            let value: Vec<Vec<Variable>> =
                (0..shape.value_len).map(|k| self.value[k * 8..k * 8 + 8].to_vec()).collect();
            verify_inclusion(api, &self.root.to_vec(), &value, &nodes, shape);
        }
    }

    #[test]
    fn incircuit_account_extension_inclusion_depth3() {
        let (a, b, _shared) = find_shared_prefix_pair();
        let accs = vec![
            Account::eoa(a, 3, BigInt::from(111u64)),
            Account::eoa(b, 5, BigInt::from(222u64)),
        ];
        let kv: Vec<([u8; 32], Vec<u8>)> = accs
            .iter()
            .map(|x| (keccak256(&x.address), account_rlp(x.nonce, &x.balance, &x.storage_root, &x.code_hash)))
            .collect();
        let root = root_node(kv);
        let root_hash = node_hash(&root);
        let (_rh, proof) = prove(&root, &keccak256(&a));
        assert_eq!(proof.nodes.len(), 3, "ext -> branch -> leaf");
        assert!(proof.value.len() <= EXT_VLEN);

        unsafe {
            EXT_SHAPE = Some(PathShape {
                child_off: proof.child_off.clone(),
                value_off: proof.value_off,
                value_len: proof.value.len(),
                node_lens: proof.nodes.iter().map(|n| n.len()).collect(),
            });
        }
        let CompileResult { witness_solver, layered_circuit } =
            compile(&ExtInclCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = ExtInclCircuit::<GF2>::default();
        set_node(&mut asg.n0, &proof.nodes[0]);
        set_len(&mut asg.l0, proof.nodes[0].len());
        set_node(&mut asg.n1, &proof.nodes[1]);
        set_len(&mut asg.l1, proof.nodes[1].len());
        set_node(&mut asg.n2, &proof.nodes[2]);
        set_len(&mut asg.l2, proof.nodes[2].len());
        for k in 0..proof.value.len() {
            for j in 0..8 {
                asg.value[k * 8 + j] = (((proof.value[k] >> j) & 1) as u32).into();
            }
        }
        set_word(&mut asg.root, &root_hash);
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit extension inclusion != native root");
        println!("R7c in-circuit EXTENSION INCLUSION (depth 3) OK, root=0x{}", hx(&root_hash));
    }

    // --------------------------------------------------------------------
    // 5) Storage trie — single slot (root IS the leaf, depth 1): INCLUSION +
    //    UPDATE against mpt::storage_root.
    // --------------------------------------------------------------------
    static mut ST_HP33: [u8; 33] = [0u8; 33];
    static mut ST_SHAPE: Option<PathShape> = None;
    const ST_VLEN: usize = 4;
    declare_circuit!(StorageInclCircuit {
        n0: [Variable; 135 * 8],
        l0: [Variable; 8],
        value: [Variable; ST_VLEN * 8],
        root: [PublicVariable; 256],
    });
    impl Define<GF2Config> for StorageInclCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let shape = unsafe { (*std::ptr::addr_of!(ST_SHAPE)).as_ref().unwrap() };
            let nodes = vec![NodeWitness { bits: self.n0.to_vec(), len_bits: self.l0.to_vec() }];
            let value: Vec<Vec<Variable>> =
                (0..shape.value_len).map(|k| self.value[k * 8..k * 8 + 8].to_vec()).collect();
            verify_inclusion(api, &self.root.to_vec(), &value, &nodes, shape);
        }
    }

    declare_circuit!(StorageUpdateCircuit {
        new_val: [Variable; BITS],
        new_root: [PublicVariable; 256],
    });
    impl Define<GF2Config> for StorageUpdateCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let hp33 = unsafe { ST_HP33 };
            // depth 1: new root == keccak(new leaf).
            let nr = storage_leaf_root(api, &hp33, &self.new_val.to_vec());
            for i in 0..256 {
                api.assert_is_equal(nr[i], self.new_root[i]);
            }
        }
    }

    #[test]
    fn incircuit_storage_single_slot_inclusion_and_update() {
        let slot = BigInt::from(0u32);
        let val = BigInt::from(0x2au32);
        let native = storage_root(&[(slot.clone(), val.clone())]);
        // single-slot storage trie: root node is the leaf.
        let mut be = [0u8; 32];
        let vb = slot.to_bytes_be().1;
        be[32 - vb.len()..].copy_from_slice(&vb);
        let key = keccak256(&be);
        let kv = vec![(key, crate::rlp::enc_uint(&val))];
        let root = root_node(kv);
        assert_eq!(node_hash(&root), native);
        let (_rh, proof) = prove(&root, &key);
        assert_eq!(proof.nodes.len(), 1, "single-slot storage trie => leaf only");
        assert!(proof.value.len() <= ST_VLEN);

        // hp33 for the full 64-nibble storage key.
        let nibs = nibbles(&key);
        let hpv = hp(&nibs, true);
        assert_eq!(hpv.len(), 33);
        let mut hp33 = [0u8; 33];
        hp33.copy_from_slice(&hpv);

        // --- inclusion ---
        unsafe {
            ST_HP33 = hp33;
            ST_SHAPE = Some(PathShape {
                child_off: proof.child_off.clone(),
                value_off: proof.value_off,
                value_len: proof.value.len(),
                node_lens: proof.nodes.iter().map(|n| n.len()).collect(),
            });
        }
        let CompileResult { witness_solver, layered_circuit } =
            compile(&StorageInclCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = StorageInclCircuit::<GF2>::default();
        set_node(&mut asg.n0, &proof.nodes[0]);
        set_len(&mut asg.l0, proof.nodes[0].len());
        for k in 0..proof.value.len() {
            for j in 0..8 {
                asg.value[k * 8 + j] = (((proof.value[k] >> j) & 1) as u32).into();
            }
        }
        set_word(&mut asg.root, &native);
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit storage inclusion != native");
        println!("R7c in-circuit storage INCLUSION OK, root=0x{}", hx(&native));

        // --- update: change slot 0's value 0x2a -> 0x1234 ---
        let new_val = BigInt::from(0x1234u32);
        let native_new = storage_root(&[(slot.clone(), new_val.clone())]);
        let CompileResult { witness_solver: ws2, layered_circuit: lc2 } =
            compile(&StorageUpdateCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg2 = StorageUpdateCircuit::<GF2>::default();
        asg2.new_val.copy_from_slice(
            &bigint_to_bits(&new_val, BITS).into_iter().map(|b| (b as u32).into()).collect::<Vec<GF2>>(),
        );
        set_word(&mut asg2.new_root, &native_new);
        let w2 = ws2.solve_witnesses(&vec![asg2; 1]).unwrap();
        assert!(lc2.run(&w2).iter().all(|x| *x), "in-circuit storage update != native new root");
        println!("R7c in-circuit storage UPDATE OK, new_root=0x{}", hx(&native_new));
    }

    // --------------------------------------------------------------------
    // 6) DENSE branch (12 accounts, distinct first nibbles => a single ~404-byte
    //    root branch that EXCEEDS one keccak block) — INCLUSION + variable-path
    //    UPDATE in one circuit, both matched to native/alloy roots. Exercises the
    //    multi-block keccak node hashing and the variable-path leaf builder
    //    (`account_leaf_hash_path`, HP taken from the proof — not hardcoded).
    // --------------------------------------------------------------------
    const DENSE_N: usize = 12;
    const DENSE_BRANCH_CAP: usize = 448; // >= 404 B dense branch
    const DENSE_VLEN: usize = 80;
    static mut DENSE_SHAPE: Option<PathShape> = None;
    static mut DENSE_HP_ENC: Option<Vec<u8>> = None;

    declare_circuit!(DenseCircuit {
        branch: [Variable; DENSE_BRANCH_CAP * 8],
        bl: [Variable; 8],
        leaf: [Variable; 135 * 8],
        ll: [Variable; 8],
        value: [Variable; DENSE_VLEN * 8],
        new_nonce: [Variable; BITS],
        new_balance: [Variable; BITS],
        root: [PublicVariable; 256],
        new_root: [PublicVariable; 256],
    });
    impl Define<GF2Config> for DenseCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let shape = unsafe { (*std::ptr::addr_of!(DENSE_SHAPE)).as_ref().unwrap() };
            let hp_enc = unsafe { (*std::ptr::addr_of!(DENSE_HP_ENC)).as_ref().unwrap() };
            let nodes = vec![
                NodeWitness { bits: self.branch.to_vec(), len_bits: self.bl.to_vec() },
                NodeWitness { bits: self.leaf.to_vec(), len_bits: self.ll.to_vec() },
            ];
            // (a) INCLUSION under the committed root (dense branch hashed multi-block).
            let value: Vec<Vec<Variable>> =
                (0..shape.value_len).map(|k| self.value[k * 8..k * 8 + 8].to_vec()).collect();
            verify_inclusion(api, &self.root.to_vec(), &value, &nodes, shape);
            // (b) UPDATE: rebuild the leaf for its ACTUAL path (from the proof),
            //     splice into the dense branch, rehash multi-block -> new root.
            let sr = empty_root();
            let ch = keccak_empty();
            let new_leaf_hash = account_leaf_hash_path(
                api, hp_enc, &self.new_nonce.to_vec(), &self.new_balance.to_vec(), &sr, &ch);
            let zero = api.constant(0);
            let upd_nodes = vec![
                NodeWitness { bits: self.branch.to_vec(), len_bits: self.bl.to_vec() },
                NodeWitness { bits: vec![zero; 135 * 8], len_bits: vec![zero; 8] },
            ];
            let nr = update_and_root(api, &new_leaf_hash, &upd_nodes, shape);
            for i in 0..256 {
                api.assert_is_equal(nr[i], self.new_root[i]);
            }
        }
    }

    #[test]
    fn incircuit_dense_branch_inclusion_and_update() {
        let (addrs, _slots) = distinct_first_nibble_addrs(DENSE_N);
        let mut accs: Vec<Account> = (0..DENSE_N)
            .map(|i| Account::eoa(addrs[i], (i as u64) + 1, BigInt::from(1_000_000_000_000_000u64 * (i as u64 + 1))))
            .collect();
        let kv: Vec<([u8; 32], Vec<u8>)> = accs
            .iter()
            .map(|a| (keccak256(&a.address), account_rlp(a.nonce, &a.balance, &a.storage_root, &a.code_hash)))
            .collect();
        let root = root_node(kv);
        let root_hash = node_hash(&root);
        assert_eq!(root_hash, state_root(&accs), "dense proof-trie root != mpt::state_root");
        // prove account 0.
        let (_rh, proof) = prove(&root, &keccak256(&accs[0].address));
        assert_eq!(proof.nodes.len(), 2, "dense single-branch => branch -> leaf");
        assert!(proof.nodes[0].len() > 135, "root branch must exceed one keccak block (got {})", proof.nodes[0].len());
        assert!(proof.nodes[0].len() <= DENSE_BRANCH_CAP);
        assert!(proof.value.len() <= DENSE_VLEN);
        println!("R7c dense branch node = {} bytes (multi-block), leaf = {} bytes", proof.nodes[0].len(), proof.nodes[1].len());

        // new value for account 0 -> native post root.
        let (new_nonce, new_balance) = (99u64, BigInt::from(777_000_000_000_000_000u64));
        accs[0] = Account::eoa(addrs[0], new_nonce, new_balance.clone());
        let native_new = state_root(&accs);

        // hp_enc = enc_bytes(HP(actual remaining path from the proof)).
        let hp_enc = enc_bytes(&proof.leaf_hp);

        unsafe {
            DENSE_SHAPE = Some(PathShape {
                child_off: proof.child_off.clone(),
                value_off: proof.value_off,
                value_len: proof.value.len(),
                node_lens: proof.nodes.iter().map(|n| n.len()).collect(),
            });
            DENSE_HP_ENC = Some(hp_enc);
        }
        let CompileResult { witness_solver, layered_circuit } =
            compile(&DenseCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = DenseCircuit::<GF2>::default();
        // dense branch buffer (>135 B): fill full length, tail zeroed.
        for i in 0..DENSE_BRANCH_CAP {
            let byte = if i < proof.nodes[0].len() { proof.nodes[0][i] } else { 0 };
            for j in 0..8 { asg.branch[i * 8 + j] = (((byte >> j) & 1) as u32).into(); }
        }
        set_len(&mut asg.bl, proof.nodes[0].len());
        set_node(&mut asg.leaf, &proof.nodes[1]);
        set_len(&mut asg.ll, proof.nodes[1].len());
        for k in 0..proof.value.len() {
            for j in 0..8 { asg.value[k * 8 + j] = (((proof.value[k] >> j) & 1) as u32).into(); }
        }
        asg.new_nonce.copy_from_slice(
            &bigint_to_bits(&BigInt::from(new_nonce), BITS).into_iter().map(|b| (b as u32).into()).collect::<Vec<GF2>>());
        asg.new_balance.copy_from_slice(
            &bigint_to_bits(&new_balance, BITS).into_iter().map(|b| (b as u32).into()).collect::<Vec<GF2>>());
        set_word(&mut asg.root, &root_hash);
        set_word(&mut asg.new_root, &native_new);
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "dense in-circuit inclusion/update != native root");
        println!("R7c in-circuit DENSE branch INCLUSION+UPDATE OK: root=0x{} new_root=0x{}", hx(&root_hash), hx(&native_new));
    }
}
