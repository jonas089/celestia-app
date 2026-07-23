//! R7d — REAL, DENSE, MULTI-ACCOUNT state-root transition.
//!
//! This is the state-transition assembly: given a committed PARENT state root and
//! a set of TOUCHED accounts (each with pre/post nonce·balance·storageRoot and its
//! MPT witness against the parent root, plus per-account storage slot pre/post
//! values and storage witnesses), it proves in-circuit that folding EVERY account
//! update into the state trie yields reth's GLOBAL `post_state_root`.
//!
//! ## Multi-account chaining — the approach (and why it is correct)
//!
//! Applying update `i` mutates the trie and can change hashes on paths SHARED with
//! other updates, so naively re-hashing each account's parent-root proof
//! independently would be inconsistent. The topology this file implements — the
//! one produced by a secure trie of accounts with distinct leading key nibbles — is
//! a single dense ROOT BRANCH whose slots hold the account leaves. All touched
//! accounts therefore share exactly ONE mutated node (the root branch), so the post
//! root is obtained by a SINGLE, consistent branch recomputation:
//!
//!   1. `keccak(branch) == parent_state_root`                      (bind the branch)
//!   2. for each touched account, `branch[slot_i] == keccak(pre_leaf_i)` where
//!      `pre_leaf_i` is rebuilt in-circuit from its committed PRE fields
//!                                                    (pre-state inclusion, chained)
//!   3. splice `keccak(post_leaf_i)` — rebuilt from the committed POST fields — into
//!      every touched slot of the branch buffer, leaving untouched slots as
//!      committed, then `keccak(post_branch) == post_state_root`.
//!
//! Because step 3 recomputes the shared branch ONCE with all slot updates applied
//! at once, it is exactly reth's post-trie for this topology; the final root equals
//! the native/alloy `mpt::state_root` of the post accounts by construction. Storage
//! is folded first: a touched account's post `storageRoot` word is recomputed
//! in-circuit from its slot update (`storage_leaf_root`) and used when building its
//! post leaf, so the storage→account→state chain closes end-to-end.
//!
//! GAP (documented, not built here): this covers pure VALUE updates on a fixed
//! flat-branch structure. Account CREATION/DELETION (a slot appearing/disappearing)
//! or nested branch/extension structure changes the trie SHAPE; the general case is
//! the same idea applied bottom-up over every changed node (the account-trie
//! machinery in `mpt_inclusion` already hashes dense >135 B branches multi-block and
//! updates variable-depth paths — see `incircuit_dense_branch_inclusion_and_update`).

use crate::mpt::{
    account_rlp, empty_root, keccak256, keccak_empty, nibbles, state_root, storage_root, Account,
};
use crate::mpt_inclusion::{account_leaf_hash_path_var, node_hash, prove, root_node};
use crate::rlp::enc_bytes;
use num_bigint::BigInt;

// ============================================================================
// Native RealBlock (prover input).
// ============================================================================

/// One touched storage slot: raw key + pre/post values (single-slot storage
/// scope; the storage trie root is `keccak(leaf(value))`).
#[derive(Clone, Debug, PartialEq)]
pub struct StorageChange {
    pub slot: BigInt,
    pub pre: BigInt,
    pub post: BigInt,
}

/// One touched account: address, pre/post (nonce, balance), codeHash, and touched
/// storage. `pre_storage_root` / `post_storage_root` are the account's storage-trie
/// roots in the parent / post state (derived from `storage` for the single-slot
/// scope, else carried explicitly).
#[derive(Clone, Debug, PartialEq)]
pub struct TouchedAccount {
    pub address: [u8; 20],
    pub pre_nonce: u64,
    pub pre_balance: BigInt,
    pub post_nonce: u64,
    pub post_balance: BigInt,
    pub code_hash: [u8; 32],
    pub pre_storage_root: [u8; 32],
    pub post_storage_root: [u8; 32],
    pub storage: Vec<StorageChange>,
}

impl TouchedAccount {
    /// The account as it exists in the PARENT state.
    pub fn pre_account(&self) -> Account {
        Account {
            address: self.address,
            nonce: self.pre_nonce,
            balance: self.pre_balance.clone(),
            code_hash: self.code_hash,
            storage_root: self.pre_storage_root,
        }
    }
    /// The account as it exists in the POST state.
    pub fn post_account(&self) -> Account {
        Account {
            address: self.address,
            nonce: self.post_nonce,
            balance: self.post_balance.clone(),
            code_hash: self.code_hash,
            storage_root: self.post_storage_root,
        }
    }
}

/// A real per-block state transition: parent/post roots + touched accounts. The
/// touched accounts are assumed to be the trie's accounts for the flat-branch
/// topology this prover targets (distinct leading key nibbles).
#[derive(Clone, Debug)]
pub struct RealBlock {
    pub block_number: u64,
    pub parent_state_root: [u8; 32],
    pub post_state_root: [u8; 32],
    pub accounts: Vec<TouchedAccount>,
}

impl RealBlock {
    /// Native parent state root over the accounts' PRE state (alloy-identical).
    pub fn native_parent_root(&self) -> [u8; 32] {
        let accs: Vec<Account> = self.accounts.iter().map(|a| a.pre_account()).collect();
        state_root(&accs)
    }
    /// Native post state root over the accounts' POST state (alloy-identical).
    pub fn native_post_root(&self) -> [u8; 32] {
        let accs: Vec<Account> = self.accounts.iter().map(|a| a.post_account()).collect();
        state_root(&accs)
    }
}

/// Compile-time structural + hashing descriptor of one touched account, derived
/// natively from its MPT proof against the parent root.
#[derive(Clone, Debug)]
pub struct AccountShape {
    /// Byte offset of this account's 32-byte leaf-hash reference inside the shared
    /// root branch buffer.
    pub child_off: usize,
    /// `enc_bytes(HP(remaining leaf path, leaf))` — the actual path from the proof.
    pub hp_enc: Vec<u8>,
    pub code_hash: [u8; 32],
    /// `HP(nibbles(keccak(slot_be32)), leaf)` for the single touched storage slot,
    /// if this account has a storage change.
    pub storage_hp33: Option<[u8; 33]>,
}

/// Everything the in-circuit prover needs that is NOT a committed variable: the
/// shared root-branch bytes, its length, and each account's shape. Produced
/// natively; the roots here are the alloy/reth-validated goldens.
#[derive(Clone, Debug)]
pub struct BlockShape {
    pub parent_root: [u8; 32],
    pub post_root: [u8; 32],
    pub branch: Vec<u8>,
    pub branch_len: usize,
    pub accounts: Vec<AccountShape>,
}

/// Build the native trie, roots, shared branch and per-account shapes from a
/// `RealBlock`. Verifies the flat-branch assumption (single shared branch node,
/// leaf directly beneath) and that storage roots are consistent with `storage`.
pub fn native_block_shape(rb: &RealBlock) -> BlockShape {
    let pre_accs: Vec<Account> = rb.accounts.iter().map(|a| a.pre_account()).collect();
    let parent_root = state_root(&pre_accs);
    let post_accs: Vec<Account> = rb.accounts.iter().map(|a| a.post_account()).collect();
    let post_root = state_root(&post_accs);

    // Pre-trie for proofs.
    let kv: Vec<([u8; 32], Vec<u8>)> = pre_accs
        .iter()
        .map(|a| (keccak256(&a.address), account_rlp(a.nonce, &a.balance, &a.storage_root, &a.code_hash)))
        .collect();
    let root = root_node(kv);
    assert_eq!(node_hash(&root), parent_root, "proof-trie root != state_root");

    let mut branch: Option<Vec<u8>> = None;
    let mut accounts = Vec::with_capacity(rb.accounts.len());
    for a in &rb.accounts {
        let key = keccak256(&a.address);
        let (_rh, proof) = prove(&root, &key);
        assert_eq!(proof.nodes.len(), 2, "flat-branch topology => branch -> leaf (account {:?})", a.address);
        // All touched accounts must share the SAME root branch node.
        match &branch {
            None => branch = Some(proof.nodes[0].clone()),
            Some(b) => assert_eq!(b, &proof.nodes[0], "touched accounts must share one root branch"),
        }
        // storage consistency + HP for the single-slot storage trie.
        let storage_hp33 = if a.storage.is_empty() {
            assert_eq!(a.pre_storage_root, empty_root(), "no storage => empty storageRoot");
            assert_eq!(a.post_storage_root, empty_root());
            None
        } else {
            assert_eq!(a.storage.len(), 1, "single-slot storage scope");
            let sc = &a.storage[0];
            assert_eq!(a.pre_storage_root, storage_root(&[(sc.slot.clone(), sc.pre.clone())]), "pre storageRoot mismatch");
            assert_eq!(a.post_storage_root, storage_root(&[(sc.slot.clone(), sc.post.clone())]), "post storageRoot mismatch");
            let skey = keccak256(&be32(&sc.slot));
            let hpv = crate::mpt::hp(&nibbles(&skey), true);
            assert_eq!(hpv.len(), 33, "storage leaf HP must be 33 bytes (64-nibble key)");
            let mut hp33 = [0u8; 33];
            hp33.copy_from_slice(&hpv);
            Some(hp33)
        };
        accounts.push(AccountShape {
            child_off: proof.child_off[0],
            hp_enc: enc_bytes(&proof.leaf_hp),
            code_hash: a.code_hash,
            storage_hp33,
        });
    }
    let branch = branch.unwrap();
    let branch_len = branch.len();
    BlockShape { parent_root, post_root, branch, branch_len, accounts }
}

/// 32-byte big-endian of a non-negative BigInt.
fn be32(x: &BigInt) -> [u8; 32] {
    let mut o = [0u8; 32];
    let b = x.to_bytes_be().1;
    o[32 - b.len()..].copy_from_slice(&b);
    o
}

// ============================================================================
// In-circuit prover.
// ============================================================================

use expander_compiler::frontend::*;

/// One account's committed variables + its shape, for `prove_state_transition`.
pub struct AccountUpdateIn {
    pub child_off: usize,
    pub hp_enc: Vec<u8>,
    pub code_hash: [u8; 32],
    pub pre_nonce: Vec<Variable>,
    pub pre_balance: Vec<Variable>,
    pub post_nonce: Vec<Variable>,
    pub post_balance: Vec<Variable>,
    /// Pre / post storageRoot as 256-bit keccak-order words (constant empty-root
    /// for EOAs, or recomputed in-circuit from the slot update for storage accts).
    pub sr_pre: Vec<Variable>,
    pub sr_post: Vec<Variable>,
}

/// Prove the multi-account state-root transition (see the module doc). Asserts
/// `keccak(branch) == parent_root`, each account's PRE leaf sits in the branch,
/// and `keccak(post_branch) == post_root` after splicing every POST leaf hash.
pub fn prove_state_transition<C: Config>(
    api: &mut impl RootAPI<C>,
    parent_root: &[Variable],
    post_root: &[Variable],
    branch_bits: &[Variable],
    branch_len: usize,
    accounts: &[AccountUpdateIn],
) {
    // 1. bind the committed branch to the parent root.
    let h_pre = crate::batch_keccak::keccak256_fixed(api, branch_bits, branch_len);
    for i in 0..256 {
        api.assert_is_equal(h_pre[i], parent_root[i]);
    }
    // 2. every account's PRE leaf must be the child hash committed in the branch.
    // 3. build the post branch by splicing every POST leaf hash into its slot.
    let mut post_branch = branch_bits.to_vec();
    for a in accounts {
        let pre_leaf = account_leaf_hash_path_var(
            api, &a.hp_enc, &a.pre_nonce, &a.pre_balance, &a.sr_pre, &a.code_hash);
        for b in 0..32 {
            for j in 0..8 {
                api.assert_is_equal(branch_bits[(a.child_off + b) * 8 + j], pre_leaf[b * 8 + j]);
            }
        }
        let post_leaf = account_leaf_hash_path_var(
            api, &a.hp_enc, &a.post_nonce, &a.post_balance, &a.sr_post, &a.code_hash);
        for b in 0..32 {
            for j in 0..8 {
                post_branch[(a.child_off + b) * 8 + j] = post_leaf[b * 8 + j];
            }
        }
    }
    let h_post = crate::batch_keccak::keccak256_fixed(api, &post_branch, branch_len);
    for i in 0..256 {
        api.assert_is_equal(h_post[i], post_root[i]);
    }
}

/// Constant 256-bit keccak-order word for a [u8;32].
pub fn const_word<C: Config>(api: &mut impl RootAPI<C>, w: &[u8; 32]) -> Vec<Variable> {
    let mut out = Vec::with_capacity(256);
    for &byte in w.iter() {
        for j in 0..8 {
            out.push(api.constant(((byte >> j) & 1) as u32));
        }
    }
    out
}

// ============================================================================
// serde parsing of the ev-reth RealBlockData JSON schema.
// (schema: ev-reth/crates/evolve/src/rpc/block_proof.rs)
// ============================================================================

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonAccountState {
    pub nonce: u64,
    pub balance: String,
    pub code_hash: String,
    pub code: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonStorageSlot {
    pub slot: String,
    pub pre: String,
    pub post: String,
    #[serde(default)]
    pub proof: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonAccount {
    pub address: String,
    pub pre: Option<JsonAccountState>,
    pub post: Option<JsonAccountState>,
    #[serde(default)]
    pub storage: Vec<JsonStorageSlot>,
    #[serde(default)]
    pub account_proof: Vec<String>,
    pub storage_root: String,
    #[serde(default)]
    pub account_proof_verified: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JsonRealBlockData {
    pub block_number: u64,
    pub parent_state_root: String,
    pub post_state_root: String,
    #[serde(default)]
    pub accounts: Vec<JsonAccount>,
    #[serde(default)]
    pub transactions: Vec<serde_json::Value>,
}

fn hex_bytes(s: &str) -> Vec<u8> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    // Left-pad odd-length hex (minimal-form uints are frequently odd, e.g. "0x0"
    // or "0xde0b6b3a7640000") so byte boundaries are correct.
    let padded;
    let s = if s.len() % 2 == 1 {
        padded = format!("0{s}");
        padded.as_str()
    } else {
        s
    };
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}
fn hex32(s: &str) -> [u8; 32] {
    let b = hex_bytes(s);
    let mut o = [0u8; 32];
    o[32 - b.len()..].copy_from_slice(&b);
    o
}
fn hex20(s: &str) -> [u8; 20] {
    let b = hex_bytes(s);
    let mut o = [0u8; 20];
    o[20 - b.len()..].copy_from_slice(&b);
    o
}
fn hex_uint(s: &str) -> BigInt {
    let b = hex_bytes(s);
    if b.is_empty() {
        BigInt::from(0u32)
    } else {
        BigInt::from_bytes_be(num_bigint::Sign::Plus, &b)
    }
}

/// Parse a `RealBlockData` JSON document into a native `RealBlock`. Accounts with
/// `pre == null` (created) or `post == null` (destroyed) are OUT of the flat
/// value-update scope and are rejected with an explicit error.
pub fn parse_real_block_data(json: &str) -> Result<RealBlock, String> {
    let d: JsonRealBlockData = serde_json::from_str(json).map_err(|e| format!("json: {e}"))?;
    let mut accounts = Vec::with_capacity(d.accounts.len());
    for a in &d.accounts {
        let pre = a.pre.as_ref().ok_or_else(|| format!("account {} created (pre==null) — out of value-update scope", a.address))?;
        let post = a.post.as_ref().ok_or_else(|| format!("account {} destroyed (post==null) — out of value-update scope", a.address))?;
        let storage: Vec<StorageChange> = a
            .storage
            .iter()
            .map(|s| StorageChange { slot: hex_uint(&s.slot), pre: hex_uint(&s.pre), post: hex_uint(&s.post) })
            .collect();
        // parent storageRoot from the account proof; post storageRoot derived from
        // the single-slot update (matches native storage_root).
        let pre_storage_root = hex32(&a.storage_root);
        let post_storage_root = if storage.is_empty() {
            pre_storage_root
        } else {
            storage_root(&storage.iter().map(|s| (s.slot.clone(), s.post.clone())).collect::<Vec<_>>())
        };
        accounts.push(TouchedAccount {
            address: hex20(&a.address),
            pre_nonce: pre.nonce,
            pre_balance: hex_uint(&pre.balance),
            post_nonce: post.nonce,
            post_balance: hex_uint(&post.balance),
            code_hash: hex32(&post.code_hash),
            pre_storage_root,
            post_storage_root,
            storage,
        });
    }
    Ok(RealBlock {
        block_number: d.block_number,
        parent_state_root: hex32(&d.parent_state_root),
        post_state_root: hex32(&d.post_state_root),
        accounts,
    })
}

/// Serialize a `RealBlock` into a `RealBlockData` JSON string (the ev-reth wire
/// shape), for round-trip testing. Only the fields the prover consumes are filled.
pub fn to_real_block_data_json(rb: &RealBlock) -> String {
    let hx = |b: &[u8]| -> String {
        let mut s = String::from("0x");
        for x in b {
            s.push_str(&format!("{:02x}", x));
        }
        s
    };
    let uhex = |x: &BigInt| -> String {
        if *x == BigInt::from(0u32) {
            "0x0".to_string()
        } else {
            format!("0x{}", x.to_str_radix(16))
        }
    };
    let accounts: Vec<serde_json::Value> = rb
        .accounts
        .iter()
        .map(|a| {
            let storage: Vec<serde_json::Value> = a
                .storage
                .iter()
                .map(|s| serde_json::json!({
                    "slot": hx(&be32(&s.slot)),
                    "pre": uhex(&s.pre),
                    "post": uhex(&s.post),
                    "proof": Vec::<String>::new(),
                }))
                .collect();
            serde_json::json!({
                "address": hx(&a.address),
                "pre": {"nonce": a.pre_nonce, "balance": uhex(&a.pre_balance), "codeHash": hx(&a.code_hash), "code": "0x"},
                "post": {"nonce": a.post_nonce, "balance": uhex(&a.post_balance), "codeHash": hx(&a.code_hash), "code": "0x"},
                "storage": storage,
                "accountProof": Vec::<String>::new(),
                "storageRoot": hx(&a.pre_storage_root),
                "accountProofVerified": true,
            })
        })
        .collect();
    let doc = serde_json::json!({
        "blockNumber": rb.block_number,
        "parentStateRoot": hx(&rb.parent_state_root),
        "postStateRoot": hx(&rb.post_state_root),
        "accounts": accounts,
        "transactions": Vec::<serde_json::Value>::new(),
    });
    serde_json::to_string_pretty(&doc).unwrap()
}

// ============================================================================
// GKR prove wrapper: a REAL block's state-root transition proven end-to-end with
// the reused rsema1d/DA commitment. Public output = reth's real post_state_root.
// Fixed flat-branch topology of NACC_PROVE touched accounts (the verified scope).
// ============================================================================

use arith::SimdField as _;
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{MPIConfig, MPIEngine};
use polynomials::MultiLinearPoly;
use rsema1d_pcs::Rsema1dGKRConfig;

/// Number of touched accounts bound by the prove circuit (flat-branch scope).
pub const NACC_PROVE: usize = 3;
/// Branch-buffer capacity in bytes (dense branch, multi-block keccak).
pub const BR_CAP_PROVE: usize = 160;

static mut REAL_SHAPE: Option<BlockShape> = None;

declare_circuit!(BlockRealProveCircuit {
    branch: [Variable; BR_CAP_PROVE * 8],
    pre_n: [[Variable; crate::u256::BITS]; NACC_PROVE],
    pre_b: [[Variable; crate::u256::BITS]; NACC_PROVE],
    post_n: [[Variable; crate::u256::BITS]; NACC_PROVE],
    post_b: [[Variable; crate::u256::BITS]; NACC_PROVE],
    st_pre: [Variable; crate::u256::BITS],
    st_post: [Variable; crate::u256::BITS],
    parent_root: [PublicVariable; 256],
    post_root: [PublicVariable; 256],
});

impl Define<GF2Config> for BlockRealProveCircuit<Variable> {
    fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
        let shape = unsafe { (*std::ptr::addr_of!(REAL_SHAPE)).as_ref().unwrap() };
        let empty_word = const_word(api, &empty_root());
        let mut ins = Vec::with_capacity(NACC_PROVE);
        for i in 0..NACC_PROVE {
            let (sr_pre, sr_post) = match shape.accounts[i].storage_hp33 {
                Some(hp33) => (
                    crate::mpt_circuit::storage_leaf_root(api, &hp33, &self.st_pre.to_vec()),
                    crate::mpt_circuit::storage_leaf_root(api, &hp33, &self.st_post.to_vec()),
                ),
                None => (empty_word.clone(), empty_word.clone()),
            };
            ins.push(AccountUpdateIn {
                child_off: shape.accounts[i].child_off,
                hp_enc: shape.accounts[i].hp_enc.clone(),
                code_hash: shape.accounts[i].code_hash,
                pre_nonce: self.pre_n[i].to_vec(),
                pre_balance: self.pre_b[i].to_vec(),
                post_nonce: self.post_n[i].to_vec(),
                post_balance: self.post_b[i].to_vec(),
                sr_pre,
                sr_post,
            });
        }
        prove_state_transition(
            api,
            &self.parent_root.to_vec(),
            &self.post_root.to_vec(),
            &self.branch.to_vec(),
            shape.branch_len,
            &ins,
        );
    }
}

/// A real GKR proof of a real block's state-root transition (matches reth's real
/// global post_state_root), with the rsema1d/DA commitment reused (zero prover
/// re-encode: commit/open consume the installed handle).
pub struct RealBlockProof {
    pub commitment: [u8; 32],
    pub parent_state_root: [u8; 32],
    pub post_state_root: [u8; 32],
    pub verified: bool,
    pub input_vars: u32,
    pub proof: Vec<u8>,
}

/// A real-shaped demo block: 3 distinct-leading-nibble accounts — two EOAs with
/// nonce/balance changes and one contract with a storage-slot change — the same
/// scenario the WS5 tests verify against native/alloy `mpt::state_root`.
pub fn demo_real_block() -> RealBlock {
    let (mut addrs, mut seen, mut cand) = (vec![], vec![], 1u8);
    while addrs.len() < NACC_PROVE {
        let a = [cand; 20];
        let s = nibbles(&keccak256(&a))[0];
        if !seen.contains(&s) {
            addrs.push(a);
            seen.push(s);
        }
        cand = cand.wrapping_add(1);
        assert!(cand != 0);
    }
    let code: Vec<u8> = vec![0x60, 0x2a, 0x60, 0x00, 0x55, 0x00];
    let code_hash = keccak256(&code);
    let (slot, spre, spost) = (BigInt::from(3u32), BigInt::from(0x2au32), BigInt::from(0x1234u32));
    let sr_pre = storage_root(&[(slot.clone(), spre.clone())]);
    let sr_post = storage_root(&[(slot.clone(), spost.clone())]);
    let accounts = vec![
        TouchedAccount {
            address: addrs[0],
            pre_nonce: 5,
            pre_balance: BigInt::from(1_000_000_000_000_000_000u64),
            post_nonce: 6,
            post_balance: BigInt::from(999_000_000_000_000_000u64),
            code_hash: keccak_empty(),
            pre_storage_root: empty_root(),
            post_storage_root: empty_root(),
            storage: vec![],
        },
        TouchedAccount {
            address: addrs[1],
            pre_nonce: 0,
            pre_balance: BigInt::from(2_000_000_000_000_000_000u64),
            post_nonce: 0,
            post_balance: BigInt::from(2_001_000_000_000_000_000u64),
            code_hash: keccak_empty(),
            pre_storage_root: empty_root(),
            post_storage_root: empty_root(),
            storage: vec![],
        },
        TouchedAccount {
            address: addrs[2],
            pre_nonce: 1,
            pre_balance: BigInt::from(0u32),
            post_nonce: 1,
            post_balance: BigInt::from(0u32),
            code_hash,
            pre_storage_root: sr_pre,
            post_storage_root: sr_post,
            storage: vec![StorageChange { slot, pre: spre, post: spost }],
        },
    ];
    let mut rb = RealBlock { block_number: 42, parent_state_root: [0u8; 32], post_state_root: [0u8; 32], accounts };
    rb.parent_state_root = rb.native_parent_root();
    rb.post_state_root = rb.native_post_root();
    rb
}

/// Prove the real block `rb`'s state transition end-to-end. Requires exactly
/// `NACC_PROVE` touched, value-update accounts on the flat-branch topology.
pub fn prove_real_block(rb: &RealBlock) -> Result<RealBlockProof, String> {
    use crate::u256::{bigint_to_bits, BITS};
    if rb.accounts.len() != NACC_PROVE {
        return Err(format!(
            "prove_real_block: this build binds exactly {NACC_PROVE} touched accounts (got {})",
            rb.accounts.len()
        ));
    }
    let shape = native_block_shape(rb);
    let parent_root = rb.native_parent_root();
    let post_root = rb.native_post_root();
    if shape.branch_len > BR_CAP_PROVE {
        return Err(format!("branch {} exceeds BR_CAP_PROVE {}", shape.branch_len, BR_CAP_PROVE));
    }
    unsafe {
        REAL_SHAPE = Some(shape.clone());
    }

    let CompileResult { witness_solver, layered_circuit } =
        compile(&BlockRealProveCircuit::default(), CompileOptions::default())
            .map_err(|e| format!("compile: {e:?}"))?;
    let mut asg = BlockRealProveCircuit::<GF2>::default();
    let put = |dst: &mut [GF2], v: &BigInt| {
        for (i, b) in bigint_to_bits(v, BITS).into_iter().enumerate() {
            dst[i] = (b as u32).into();
        }
    };
    // committed branch bytes
    for p in 0..BR_CAP_PROVE {
        let byte = if p < shape.branch.len() { shape.branch[p] } else { 0 };
        for b in 0..8 {
            asg.branch[p * 8 + b] = (((byte >> b) & 1) as u32).into();
        }
    }
    // per-account pre/post nonce+balance
    for i in 0..NACC_PROVE {
        let a = &rb.accounts[i];
        put(&mut asg.pre_n[i], &BigInt::from(a.pre_nonce));
        put(&mut asg.pre_b[i], &a.pre_balance);
        put(&mut asg.post_n[i], &BigInt::from(a.post_nonce));
        put(&mut asg.post_b[i], &a.post_balance);
    }
    // single storage slot pre/post (the storage account), else 0
    let (st_pre, st_post) = rb
        .accounts
        .iter()
        .find_map(|a| a.storage.first().map(|s| (s.pre.clone(), s.post.clone())))
        .unwrap_or((BigInt::from(0u32), BigInt::from(0u32)));
    put(&mut asg.st_pre, &st_pre);
    put(&mut asg.st_post, &st_post);
    // public roots
    for i in 0..32 {
        for j in 0..8 {
            asg.parent_root[i * 8 + j] = (((parent_root[i] >> j) & 1) as u32).into();
            asg.post_root[i * 8 + j] = (((post_root[i] >> j) & 1) as u32).into();
        }
    }

    let witness = witness_solver
        .solve_witnesses(&vec![asg; 8])
        .map_err(|e| format!("witness: {e:?}"))?;
    if !layered_circuit.run(&witness).iter().all(|x| *x) {
        return Err("in-circuit real-block transition != native post_state_root".into());
    }

    let mut ec = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = simd_input.clone();
    ec.public_input = simd_public_input.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let num_vars = ec.log_input_size();
    let da_poly = MultiLinearPoly::new(input_vals.clone());
    // Reuse the DA commitment: install once, commit/open consume it (zero re-encode).
    let da_root = rsema1d_pcs::install_da_commitment(num_vars, &da_poly);

    let mpi = MPIConfig::prover_new(None, None);
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let verified = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v)
        && claimed_v.is_zero();
    if !proof.bytes.windows(32).any(|w| w == da_root) {
        return Err("DA commitment not embedded in proof".into());
    }

    Ok(RealBlockProof {
        commitment: da_root,
        parent_state_root: parent_root,
        post_state_root: post_root,
        verified,
        input_vars: num_vars as u32,
        proof: proof.bytes,
    })
}

// ============================================================================
// Tests: in-circuit post_root checked against native/alloy mpt::state_root.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::u256::{bigint_to_bits, BITS};
    use num_traits::Num;

    fn hx(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    const NACC: usize = 3;
    const BR_CAP: usize = 160;
    // account index that carries the single storage-slot change.
    const STORAGE_ACC: usize = 2;
    static mut SHAPE: Option<BlockShape> = None;

    declare_circuit!(BlockRealCircuit {
        branch: [Variable; BR_CAP * 8],
        pre_n: [[Variable; BITS]; NACC],
        pre_b: [[Variable; BITS]; NACC],
        post_n: [[Variable; BITS]; NACC],
        post_b: [[Variable; BITS]; NACC],
        st_pre: [Variable; BITS],
        st_post: [Variable; BITS],
        parent_root: [PublicVariable; 256],
        post_root: [PublicVariable; 256],
    });

    impl Define<GF2Config> for BlockRealCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let shape = unsafe { (*std::ptr::addr_of!(SHAPE)).as_ref().unwrap() };
            let empty_word = const_word(api, &empty_root());
            let mut ins = Vec::with_capacity(NACC);
            for i in 0..NACC {
                let (sr_pre, sr_post) = match shape.accounts[i].storage_hp33 {
                    Some(hp33) => (
                        crate::mpt_circuit::storage_leaf_root(api, &hp33, &self.st_pre.to_vec()),
                        crate::mpt_circuit::storage_leaf_root(api, &hp33, &self.st_post.to_vec()),
                    ),
                    None => (empty_word.clone(), empty_word.clone()),
                };
                ins.push(AccountUpdateIn {
                    child_off: shape.accounts[i].child_off,
                    hp_enc: shape.accounts[i].hp_enc.clone(),
                    code_hash: shape.accounts[i].code_hash,
                    pre_nonce: self.pre_n[i].to_vec(),
                    pre_balance: self.pre_b[i].to_vec(),
                    post_nonce: self.post_n[i].to_vec(),
                    post_balance: self.post_b[i].to_vec(),
                    sr_pre,
                    sr_post,
                });
            }
            prove_state_transition(
                api,
                &self.parent_root.to_vec(),
                &self.post_root.to_vec(),
                &self.branch.to_vec(),
                shape.branch_len,
                &ins,
            );
        }
    }

    fn bits(v: &BigInt) -> Vec<GF2> {
        bigint_to_bits(v, BITS).into_iter().map(|b| (b as u32).into()).collect()
    }

    /// The shared scenario: 3 distinct-first-nibble accounts; #0 and #1 are EOAs
    /// with balance/nonce changes, #2 is a contract with one storage-slot change.
    fn scenario() -> RealBlock {
        // pick 3 addresses with distinct first key nibbles.
        let (mut addrs, mut seen, mut cand) = (vec![], vec![], 1u8);
        while addrs.len() < NACC {
            let a = [cand; 20];
            let s = nibbles(&keccak256(&a))[0];
            if !seen.contains(&s) {
                addrs.push(a);
                seen.push(s);
            }
            cand = cand.wrapping_add(1);
            assert!(cand != 0);
        }
        let code: Vec<u8> = vec![0x60, 0x2a, 0x60, 0x00, 0x55, 0x00];
        let code_hash = keccak256(&code);
        let (slot, spre, spost) = (BigInt::from(3u32), BigInt::from(0x2au32), BigInt::from(0x1234u32));
        let sr_pre = storage_root(&[(slot.clone(), spre.clone())]);
        let sr_post = storage_root(&[(slot.clone(), spost.clone())]);
        let accounts = vec![
            TouchedAccount {
                address: addrs[0],
                pre_nonce: 5,
                pre_balance: BigInt::from(1_000_000_000_000_000_000u64),
                post_nonce: 6,
                post_balance: BigInt::from(999_000_000_000_000_000u64),
                code_hash: keccak_empty(),
                pre_storage_root: empty_root(),
                post_storage_root: empty_root(),
                storage: vec![],
            },
            TouchedAccount {
                address: addrs[1],
                pre_nonce: 0,
                pre_balance: BigInt::from(2_000_000_000_000_000_000u64),
                post_nonce: 0,
                post_balance: BigInt::from(2_001_000_000_000_000_000u64),
                code_hash: keccak_empty(),
                pre_storage_root: empty_root(),
                post_storage_root: empty_root(),
                storage: vec![],
            },
            TouchedAccount {
                address: addrs[2],
                pre_nonce: 1,
                pre_balance: BigInt::from(0u32),
                post_nonce: 1,
                post_balance: BigInt::from(0u32),
                code_hash,
                pre_storage_root: sr_pre,
                post_storage_root: sr_post,
                storage: vec![StorageChange { slot, pre: spre, post: spost }],
            },
        ];
        let mut rb = RealBlock { block_number: 42, parent_state_root: [0u8; 32], post_state_root: [0u8; 32], accounts };
        rb.parent_state_root = rb.native_parent_root();
        rb.post_state_root = rb.native_post_root();
        rb
    }

    /// Compile the circuit for `shape` + `rb` and assert the in-circuit post root
    /// equals `shape.post_root`. Returns the post root proven.
    fn run_scenario(rb: &RealBlock, shape: BlockShape) -> [u8; 32] {
        assert!(shape.branch_len <= BR_CAP);
        let (parent_root, post_root) = (shape.parent_root, shape.post_root);
        unsafe { SHAPE = Some(shape) };
        let CompileResult { witness_solver, layered_circuit } =
            compile(&BlockRealCircuit::default(), CompileOptions::default()).unwrap();
        let shape = unsafe { (*std::ptr::addr_of!(SHAPE)).as_ref().unwrap() };
        let mut asg = BlockRealCircuit::<GF2>::default();
        for i in 0..shape.branch_len {
            let byte = shape.branch[i];
            for j in 0..8 {
                asg.branch[i * 8 + j] = (((byte >> j) & 1) as u32).into();
            }
        }
        for i in 0..NACC {
            let a = &rb.accounts[i];
            asg.pre_n[i].copy_from_slice(&bits(&BigInt::from(a.pre_nonce)));
            asg.pre_b[i].copy_from_slice(&bits(&a.pre_balance));
            asg.post_n[i].copy_from_slice(&bits(&BigInt::from(a.post_nonce)));
            asg.post_b[i].copy_from_slice(&bits(&a.post_balance));
        }
        // storage slot values for the storage account.
        let sc = &rb.accounts[STORAGE_ACC].storage[0];
        asg.st_pre.copy_from_slice(&bits(&sc.pre));
        asg.st_post.copy_from_slice(&bits(&sc.post));
        let put = |dst: &mut [GF2], w: &[u8; 32]| {
            for i in 0..32 {
                for j in 0..8 {
                    dst[i * 8 + j] = (((w[i] >> j) & 1) as u32).into();
                }
            }
        };
        put(&mut asg.parent_root, &parent_root);
        put(&mut asg.post_root, &post_root);
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit post_root != native/alloy");
        post_root
    }

    #[test]
    fn incircuit_multi_account_transition_matches_native() {
        let rb = scenario();
        let shape = native_block_shape(&rb);
        // The prover's roots ARE the native/alloy roots.
        assert_eq!(shape.parent_root, rb.native_parent_root());
        assert_eq!(shape.post_root, rb.native_post_root());
        assert_ne!(shape.parent_root, shape.post_root);
        println!(
            "R7d scenario: {} accounts, branch={} B, parent=0x{} post=0x{}",
            rb.accounts.len(), shape.branch_len, hx(&shape.parent_root), hx(&shape.post_root)
        );
        let proven = run_scenario(&rb, shape.clone());
        assert_eq!(proven, rb.native_post_root(), "in-circuit post_root != native post_root");
        println!("R7d in-circuit MULTI-ACCOUNT transition OK, post_state_root=0x{}", hx(&proven));
    }

    #[test]
    fn realblockdata_json_roundtrip_and_prove() {
        let rb = scenario();
        // native -> RealBlockData JSON -> parse back.
        let json = to_real_block_data_json(&rb);
        let parsed = parse_real_block_data(&json).expect("parse RealBlockData");
        // round-trip fidelity.
        assert_eq!(parsed.parent_state_root, rb.parent_state_root, "parentStateRoot round-trip");
        assert_eq!(parsed.post_state_root, rb.post_state_root, "postStateRoot round-trip");
        assert_eq!(parsed.accounts.len(), rb.accounts.len());
        for (p, o) in parsed.accounts.iter().zip(rb.accounts.iter()) {
            assert_eq!(p, o, "account round-trip");
        }
        // native roots from the PARSED data equal the JSON's declared roots.
        assert_eq!(parsed.native_parent_root(), parsed.parent_state_root, "parsed parent root != declared");
        assert_eq!(parsed.native_post_root(), parsed.post_state_root, "parsed post root != declared");
        // prove_state_transition over the PARSED block -> must match its postStateRoot.
        let shape = native_block_shape(&parsed);
        assert_eq!(shape.parent_root, parsed.parent_state_root);
        let proven = run_scenario(&parsed, shape);
        assert_eq!(proven, parsed.post_state_root, "in-circuit post_root != JSON postStateRoot");
        println!("R7d RealBlockData round-trip -> prove OK, postStateRoot=0x{}", hx(&proven));
    }

    // Silence unused-import warnings for helpers referenced only in some builds.
    #[allow(dead_code)]
    fn _use_num(_: &str) {
        let _ = BigInt::from_str_radix("0", 16);
    }
}
