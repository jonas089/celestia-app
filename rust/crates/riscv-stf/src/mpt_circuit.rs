//! R7 (circuit) — in-circuit Ethereum world-state MPT root, deterministic /
//! hint-free, built on the verified `u256` arithmetic and `batch_keccak`
//! variable-length keccak. Foundation atom: minimal-length RLP encoding of an
//! integer (Ethereum RLP strips leading zero bytes), which every account field
//! and node needs. Matches the native `rlp::enc_uint` byte-for-byte.

use crate::u256::{eq, lt, select, BITS};
use expander_compiler::frontend::*;

/// 32 big-endian bytes of a u256 LE bit vector (index 0 = most significant).
/// Returns Vec of 32 bytes, each an 8-bit LSB-first slice.
fn be_bytes<C: Config>(_api: &mut impl RootAPI<C>, u: &[Variable]) -> Vec<Vec<Variable>> {
    assert_eq!(u.len(), BITS);
    (0..32)
        .map(|k| {
            // BE byte k = LE byte (31-k); its bit j = u[(31-k)*8 + j].
            let le = 31 - k;
            (0..8).map(|j| u[le * 8 + j]).collect::<Vec<_>>()
        })
        .collect()
}

fn byte_is_zero<C: Config>(api: &mut impl RootAPI<C>, byte: &[Variable]) -> Variable {
    let mut acc = api.constant(1);
    for &b in byte {
        let nb = api.sub(1, b);
        acc = api.mul(acc, nb);
    }
    acc
}

/// Left-shift a 32-byte array by `shift` bytes (shift toward index 0), zero-fill.
/// `shift_bits` is the 5-bit LE shift amount.
fn barrel_shl_bytes<C: Config>(api: &mut impl RootAPI<C>, bytes: &[Vec<Variable>], shift_bits: &[Variable]) -> Vec<Vec<Variable>> {
    let zero = api.constant(0);
    let mut cur = bytes.to_vec();
    for (k, &sbit) in shift_bits.iter().enumerate().take(5) {
        let sh = 1usize << k;
        let shifted: Vec<Vec<Variable>> = (0..32)
            .map(|i| {
                if i + sh < 32 {
                    cur[i + sh].clone()
                } else {
                    vec![zero; 8]
                }
            })
            .collect();
        // cur = sbit ? shifted : cur
        cur = (0..32)
            .map(|i| {
                (0..8)
                    .map(|j| {
                        let d = api.add(shifted[i][j], cur[i][j]);
                        let t = api.mul(sbit, d);
                        api.add(cur[i][j], t)
                    })
                    .collect()
            })
            .collect();
    }
    cur
}

/// Minimal RLP encoding of a u256 (Ethereum strips leading zeros). Returns the
/// encoding as up to 33 bytes (flattened LSB-first bits, 33*8 long; only the
/// first `len` bytes are meaningful) and a 6-bit length `len` (1..=33).
pub fn min_rlp_uint<C: Config>(api: &mut impl RootAPI<C>, u: &[Variable]) -> (Vec<Vec<Variable>>, Vec<Variable>) {
    let zero = api.constant(0);
    let be = be_bytes(api, u);
    // significant-byte detection (MSB-first).
    let mut seen = zero;
    let mut seen_flags = Vec::with_capacity(32);
    for k in 0..32 {
        let z = byte_is_zero(api, &be[k]);
        let nz = api.sub(1, z);
        // seen |= nz
        let and = api.mul(seen, nz);
        seen = { let s = api.add(seen, nz); api.sub(s, and) }; // OR
        seen_flags.push(seen);
    }
    // sig_count = popcount(seen_flags) (monotonic 0..0,1..1). shift = 32 - sig.
    // sig_count in 0..=32 -> 6 bits. Build via ripple addition of the 32 bits.
    let sig_bits = popcount32(api, &seen_flags); // 6 bits
    // shift = 32 - sig_count. Compute (32 - sig) with 6-bit sub.
    let c32: Vec<Variable> = (0..6).map(|b| api.constant((32u32 >> b) & 1)).collect();
    let shift_bits = sub6(api, &c32, &sig_bits); // 6 bits; value = leading zero bytes (0..32)

    // Left-align significant bytes.
    let la = barrel_shl_bytes(api, &be, &shift_bits[..5]);

    // value == 0 ? (all bytes zero)
    let is_zero = {
        let mut acc = api.constant(1);
        for k in 0..32 {
            let z = byte_is_zero(api, &be[k]);
            acc = api.mul(acc, z);
        }
        acc
    };
    // single_small = (sig_count == 1) AND (la[0] < 0x80)  (i.e. top bit of la[0] == 0)
    let one_c = const_bits(api, 1, 6);
    let sig_is_one = eq(api, &sig_bits, &one_c);
    let la0_high = la[0][7];
    let la0_lt_80 = api.sub(1, la0_high);
    let single_small = api.mul(sig_is_one, la0_lt_80);

    // out byte 0:
    //   zero        -> 0x80
    //   single_small-> la[0]
    //   else        -> 0x80 | sig_count
    let byte_0x80: Vec<Variable> = const_byte(api, 0x80);
    // 0x80 | sig  == sig_bits (<=32 -> bits0..5) with bit7 set.
    let mut prefix_else = vec![zero; 8];
    for b in 0..6 {
        prefix_else[b] = sig_bits[b];
    }
    prefix_else[7] = api.constant(1);
    // select else vs single_small vs zero
    let sel1 = select(api, single_small, &la[0], &prefix_else); // single? la0 : elsePrefix
    let out0 = select(api, is_zero, &byte_0x80, &sel1); // zero? 0x80 : sel1

    // is_short = zero OR single_small (=> len 1); else-case has the sig bytes.
    let is_short = { let s = api.add(is_zero, single_small); let a = api.mul(is_zero, single_small); api.sub(s, a) };
    let else_case = api.sub(1, is_short);

    // Assemble output buffer (33 bytes): [out0, then sig bytes only in else-case;
    // tail masked to zero elsewhere so the buffer is canonical and the length-
    // aware consumer (and tests) can compare byte-for-byte].
    let mut out: Vec<Vec<Variable>> = Vec::with_capacity(33);
    out.push(out0);
    for k in 0..32 {
        let masked: Vec<Variable> = (0..8).map(|j| api.mul(else_case, la[k][j])).collect();
        out.push(masked);
    }

    // len: short -> 1 ; else -> 1 + sig_count.
    let one6 = const_bits(api, 1, 6);
    let len_else = add6(api, &one6, &sig_bits); // 1 + sig
    let len = select(api, is_short, &one6, &len_else);

    (out, len)
}

// -------- small (<=6-bit) integer helpers over GF2 --------

fn const_bits<C: Config>(api: &mut impl RootAPI<C>, v: u32, n: usize) -> Vec<Variable> {
    (0..n).map(|b| api.constant((v >> b) & 1)).collect()
}
fn const_byte<C: Config>(api: &mut impl RootAPI<C>, v: u32) -> Vec<Variable> {
    (0..8).map(|b| api.constant((v >> b) & 1)).collect()
}
fn full_adder<C: Config>(api: &mut impl RootAPI<C>, a: Variable, b: Variable, c: Variable) -> (Variable, Variable) {
    let ab = api.add(a, b);
    let s = api.add(ab, c);
    let and_ab = api.mul(a, b);
    let cc = api.mul(c, ab);
    let cout = api.add(and_ab, cc);
    (s, cout)
}
fn add6<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut out = Vec::with_capacity(6);
    let mut carry = zero;
    for i in 0..6 {
        let ai = *a.get(i).unwrap_or(&zero);
        let bi = *b.get(i).unwrap_or(&zero);
        let (s, c) = full_adder(api, ai, bi, carry);
        out.push(s);
        carry = c;
    }
    out
}
fn sub6<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let one = api.constant(1);
    let nb: Vec<Variable> = (0..6).map(|i| api.sub(1, b[i])).collect();
    let mut out = Vec::with_capacity(6);
    let mut carry = one;
    for i in 0..6 {
        let (s, c) = full_adder(api, a[i], nb[i], carry);
        out.push(s);
        carry = c;
    }
    out
}
/// popcount of 32 bits -> 6-bit count.
fn popcount32<C: Config>(api: &mut impl RootAPI<C>, bits: &[Variable]) -> Vec<Variable> {
    let mut acc = const_bits(api, 0, 6);
    for &b in bits {
        let mut term = vec![b];
        acc = add6(api, &acc, &term);
        term.clear();
    }
    acc
}

// -------- byte-buffer helpers for node assembly --------

fn zeros_bytes<C: Config>(api: &mut impl RootAPI<C>, n: usize) -> Vec<Vec<Variable>> {
    let zero = api.constant(0);
    (0..n).map(|_| vec![zero; 8]).collect()
}
fn byte_const_v<C: Config>(api: &mut impl RootAPI<C>, v: u8) -> Vec<Variable> {
    (0..8).map(|b| api.constant(((v >> b) & 1) as u32)).collect()
}
/// enc of a 32-byte string: [0xa0, b0..b31] (fixed).
fn enc_bytes32<C: Config>(api: &mut impl RootAPI<C>, b: &[u8; 32]) -> Vec<Vec<Variable>> {
    let mut out = vec![byte_const_v(api, 0xa0)];
    for &x in b.iter() {
        out.push(byte_const_v(api, x));
    }
    out
}
/// Zero-extend a byte buffer to `w` bytes.
fn zext<C: Config>(api: &mut impl RootAPI<C>, src: &[Vec<Variable>], w: usize) -> Vec<Vec<Variable>> {
    let zero = api.constant(0);
    (0..w).map(|i| if i < src.len() { src[i].clone() } else { vec![zero; 8] }).collect()
}
/// Right-shift bytes by a variable offset `shift_bits` into a `w`-byte buffer.
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
/// XOR two equal-width byte buffers (used to OR non-overlapping placements).
fn xor_bytes<C: Config>(api: &mut impl RootAPI<C>, a: &[Vec<Variable>], b: &[Vec<Variable>]) -> Vec<Vec<Variable>> {
    (0..a.len()).map(|i| (0..8).map(|j| api.add(a[i][j], b[i][j])).collect()).collect()
}
/// Add two <=8-bit LE numbers -> 8-bit result (low 8 bits).
fn add8<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let s = crate::u256::add(api, a, b);
    s[0..8].to_vec()
}

/// In-circuit keccak(leaf node) for an EOA account at a FIXED 63-nibble path
/// (`hp32` = HP(path,leaf), a compile-time constant for a known address). nonce
/// and balance are committed u256 bit vectors. Matches native keccak(leaf_rlp).
/// Single-block (leaf <= 135 B) — holds for EOA balances <= ~2^64.
pub fn account_leaf_hash<C: Config>(
    api: &mut impl RootAPI<C>,
    nonce: &[Variable],
    balance: &[Variable],
    hp32: &[u8; 32],
    storage_root: &[u8; 32],
    code_hash: &[u8; 32],
) -> Vec<Variable> {
    let sr_enc = enc_bytes32(api, storage_root);
    let ch_enc = enc_bytes32(api, code_hash);
    account_leaf_core(api, nonce, balance, hp32, &sr_enc, &ch_enc)
}

/// [0xa0] ++ the 32 bytes of a keccak-output word (byte b = bits[b*8..b*8+8]).
fn enc_bytes32_word<C: Config>(api: &mut impl RootAPI<C>, hash: &[Variable]) -> Vec<Vec<Variable>> {
    let mut out = vec![byte_const_v(api, 0xa0)];
    for b in 0..32 { out.push(hash[b * 8..b * 8 + 8].to_vec()); }
    out
}

/// Account leaf where storageRoot/codeHash are COMPUTED words (contract account).
pub fn account_leaf_hash_var<C: Config>(
    api: &mut impl RootAPI<C>,
    nonce: &[Variable],
    balance: &[Variable],
    hp32: &[u8; 32],
    storage_root: &[Variable],
    code_hash: &[Variable],
) -> Vec<Variable> {
    let sr_enc = enc_bytes32_word(api, storage_root);
    let ch_enc = enc_bytes32_word(api, code_hash);
    account_leaf_core(api, nonce, balance, hp32, &sr_enc, &ch_enc)
}

fn account_leaf_core<C: Config>(
    api: &mut impl RootAPI<C>,
    nonce: &[Variable],
    balance: &[Variable],
    hp32: &[u8; 32],
    sr_enc: &[Vec<Variable>],
    ch_enc: &[Vec<Variable>],
) -> Vec<Variable> {
    let (nb, n1) = min_rlp_uint(api, nonce);
    let (bb, n2) = min_rlp_uint(api, balance);

    // account body = enc(nonce) ++ enc(balance) ++ enc(storageRoot) ++ enc(codeHash)
    let n1_8 = { let mut v = n1.clone(); while v.len() < 8 { v.push(api.constant(0)); } v };
    let n2_8 = { let mut v = n2.clone(); while v.len() < 8 { v.push(api.constant(0)); } v };
    let c33 = const_bits(api, 33, 8);
    let o2 = n1_8.clone();
    let o3 = add8(api, &n1_8, &n2_8);
    let o4 = add8(api, &o3, &c33);
    const WBODY: usize = 96;
    let p1 = zext(api, &nb, WBODY);
    let p2 = barrel_shr(api, &bb, &o2, WBODY);
    let p3 = barrel_shr(api, &sr_enc, &o3, WBODY);
    let p4 = barrel_shr(api, &ch_enc, &o4, WBODY);
    let b12 = xor_bytes(api, &p1, &p2);
    let b123 = xor_bytes(api, &b12, &p3);
    let body = xor_bytes(api, &b123, &p4);
    let body_len = add8(api, &o4, &c33); // n1+n2+66

    // account_inner = [0xf8, body_len, body]  (EOA body >= 66 => long-list header)
    let mut acct = zeros_bytes(api, 98);
    acct[0] = byte_const_v(api, 0xf8);
    acct[1] = body_len.clone();
    for i in 0..WBODY {
        acct[2 + i] = body[i].clone();
    }
    let c2 = const_bits(api, 2, 8);
    let account_len = add8(api, &body_len, &c2);

    // leaf = [0xf8, leaf_body_len, 0xa0, hp32(32), 0xb8, account_len, account_inner]
    let mut leaf = zeros_bytes(api, 135);
    leaf[0] = byte_const_v(api, 0xf8);
    let c35 = const_bits(api, 35, 8);
    let leaf_body_len = add8(api, &c35, &account_len); // 33(hp_enc)+2+account_len
    leaf[1] = leaf_body_len.clone();
    leaf[2] = byte_const_v(api, 0xa0);
    for i in 0..32 {
        leaf[3 + i] = byte_const_v(api, hp32[i]);
    }
    leaf[35] = byte_const_v(api, 0xb8);
    leaf[36] = account_len.clone();
    for i in 0..98 {
        if 37 + i < 135 {
            leaf[37 + i] = acct[i].clone();
        }
    }
    let leaf_len = add8(api, &c2, &leaf_body_len); // 2 + leaf_body_len

    let flat: Vec<Variable> = leaf[0..135].iter().flatten().cloned().collect();
    crate::batch_keccak::keccak256_varlen(api, &flat, &leaf_len)
}

/// One account in the fixed-topology 3-account state trie.
pub struct AcctIn<'a> {
    pub nonce: &'a [Variable],
    pub balance: &'a [Variable],
    pub hp32: [u8; 32], // HP(63-nibble leaf path, leaf) — fixed per address
    pub slot: usize,     // first nibble of keccak(address) — the root-branch slot
}

/// In-circuit world-state root for a fixed 3-account trie whose addresses have
/// DISTINCT first nibbles (root = one branch of 3 leaves). All EOAs. Returns the
/// 256-bit root; matches native/alloy `state_root`.
pub fn state_root_3<C: Config>(api: &mut impl RootAPI<C>, accts: &[AcctIn; 3], sr: &[u8; 32], ch: &[u8; 32]) -> Vec<Variable> {
    // leaf hashes (each 32 bytes, LSB-first per byte).
    let mut leaf_bytes: Vec<Vec<Vec<Variable>>> = Vec::new();
    for a in accts.iter() {
        let h = account_leaf_hash(api, a.nonce, a.balance, &a.hp32, sr, ch);
        let bytes: Vec<Vec<Variable>> = (0..32).map(|b| h[b * 8..b * 8 + 8].to_vec()).collect();
        leaf_bytes.push(bytes);
    }
    // branch = [0xf8, body_len=113, slot0..slot15, value=0x80].
    let mut branch: Vec<Vec<Variable>> = Vec::with_capacity(115);
    branch.push(byte_const_v(api, 0xf8));
    branch.push(byte_const_v(api, 113));
    for nib in 0..16usize {
        if let Some(idx) = accts.iter().position(|a| a.slot == nib) {
            branch.push(byte_const_v(api, 0xa0));
            for b in 0..32 {
                branch.push(leaf_bytes[idx][b].clone());
            }
        } else {
            branch.push(byte_const_v(api, 0x80));
        }
    }
    branch.push(byte_const_v(api, 0x80)); // empty value slot
    assert_eq!(branch.len(), 115);
    let flat: Vec<Variable> = branch.iter().flatten().cloned().collect();
    crate::batch_keccak::keccak256_single_block(api, &flat)
}

/// Branch root over 3 leaf-hash words (32 bytes each) placed at their `slot`
/// (distinct first nibbles). Lets callers mix EOA and contract account leaves.
pub fn branch_root_3<C: Config>(api: &mut impl RootAPI<C>, leaves: &[(Vec<Variable>, usize); 3]) -> Vec<Variable> {
    let mut branch: Vec<Vec<Variable>> = Vec::with_capacity(115);
    branch.push(byte_const_v(api, 0xf8));
    branch.push(byte_const_v(api, 113));
    for nib in 0..16usize {
        if let Some(idx) = leaves.iter().position(|(_, s)| *s == nib) {
            branch.push(byte_const_v(api, 0xa0));
            for b in 0..32 { branch.push(leaves[idx].0[b * 8..b * 8 + 8].to_vec()); }
        } else {
            branch.push(byte_const_v(api, 0x80));
        }
    }
    branch.push(byte_const_v(api, 0x80));
    let flat: Vec<Variable> = branch.iter().flatten().cloned().collect();
    crate::batch_keccak::keccak256_single_block(api, &flat)
}

/// In-circuit value-transfer STF -> post-state root. Committed inputs: the 3
/// accounts' pre (nonce,balance) [sender=0, recipient=1, coinbase=2], and the tx
/// (value, max_fee, max_priority_fee) + block base_fee, all u256 LE bit vectors.
/// Applies ev-reth 1559 mechanics (gas 21000) with plain integer arithmetic, then
/// roots the post-state. Matches native `stf_transfer::apply`.
pub fn transfer_stf_root<C: Config>(
    api: &mut impl RootAPI<C>,
    pre_nonce: &[Vec<Variable>; 3],
    pre_bal: &[Vec<Variable>; 3],
    value: &[Variable],
    max_fee: &[Variable],
    max_prio: &[Variable],
    base_fee: &[Variable],
    hp3: &[[u8; 32]; 3],
    slots: &[usize; 3],
    sr: &[u8; 32],
    ch: &[u8; 32],
) -> Vec<Variable> {
    use crate::u256::{add, lt, mul_by_const, select, sub};
    let trunc = |v: Vec<Variable>| v[0..256].to_vec();
    // priority = min(max_prio, max_fee - base_fee); effective = base + priority.
    let (headroom, _b) = sub(api, max_fee, base_fee);
    let is_lt = lt(api, max_prio, &headroom);
    let priority = select(api, is_lt, max_prio, &headroom);
    let effective = trunc(add(api, base_fee, &priority));
    let gas_cost = trunc(mul_by_const(api, &effective, 21000));
    let tip = trunc(mul_by_const(api, &priority, 21000));
    // sender: balance -= value + gas_cost ; nonce += 1
    let debit = add(api, value, &gas_cost);
    let (post_bal0, _b0) = sub(api, &pre_bal[0], &debit);
    let one = { let mut v = vec![api.constant(0); 256]; v[0] = api.constant(1); v };
    let post_nonce0 = trunc(add(api, &pre_nonce[0], &one));
    // recipient: balance += value ; coinbase: balance += tip
    let post_bal1 = trunc(add(api, &pre_bal[1], value));
    let post_bal2 = trunc(add(api, &pre_bal[2], &tip));

    let post_bal0 = post_bal0[0..256].to_vec();
    let n0 = post_nonce0; let n1 = pre_nonce[1].clone(); let n2 = pre_nonce[2].clone();
    let accts = [
        AcctIn { nonce: &n0, balance: &post_bal0, hp32: hp3[0], slot: slots[0] },
        AcctIn { nonce: &n1, balance: &post_bal1, hp32: hp3[1], slot: slots[1] },
        AcctIn { nonce: &n2, balance: &post_bal2, hp32: hp3[2], slot: slots[2] },
    ];
    state_root_3(api, &accts, sr, ch)
}

/// In-circuit single-slot storage-trie root: keccak(rlp([HP(64-nibble key),
/// RLP(value)])). `hp33` = HP(nibbles(keccak(slot_be32)), leaf) — 33 const bytes.
/// Handles small leaves (body < 56 => 1-byte list header); the demo's slot
/// values fit. Matches native `mpt::storage_root` for a single non-zero slot.
pub fn storage_leaf_root<C: Config>(api: &mut impl RootAPI<C>, hp33: &[u8; 33], value: &[Variable]) -> Vec<Variable> {
    // RLP(value) = min_rlp_uint(value) => venc buffer (33B) + vlen.
    let (venc, vlen) = min_rlp_uint(api, value);
    // storage-trie leaf value field = enc_bytes(RLP(value)):
    //   if vlen==1 && venc[0]<0x80  -> [venc[0]]            (len 1)
    //   else                        -> [0x80+vlen, venc...] (len 1+vlen)
    let one = const_bits(api, 1, 6);
    let one8 = const_bits(api, 1, 8);
    let vlen8 = { let mut v = vlen.clone(); while v.len() < 8 { v.push(api.constant(0)); } v };
    let vlen_is_1 = eq(api, &vlen, &one);
    let hi = venc[0][7]; // top bit of first RLP byte
    let single = { let nhi = api.sub(1, hi); api.mul(vlen_is_1, nhi) };
    // value-field header byte = single ? venc[0] : (0x80 + vlen)
    let prefix = { let mut p = vec![api.constant(0); 8]; for b in 0..6 { p[b] = vlen[b]; } p[7] = api.constant(1); p };
    let vf0 = select(api, single, &venc[0], &prefix);
    // value-field length = single ? 1 : 1+vlen
    let vf_len = { let l = add8(api, &one8, &vlen8); select(api, single, &one8, &l) };
    // value-field bytes buffer: [vf0, (if !single) venc[0..vlen] ...]
    // When single: [venc[0]]; else: [0x80+vlen, venc[0], venc[1], ...].
    // Build a 34-byte value-field buffer.
    let notsingle = api.sub(1, single);
    let mut vfield: Vec<Vec<Variable>> = Vec::with_capacity(34);
    vfield.push(vf0);
    for k in 0..33 {
        // in the non-single case, byte at position 1+k = venc[k]; single case: 0 (beyond len 1)
        let b: Vec<Variable> = (0..8).map(|j| api.mul(notsingle, venc[k][j])).collect();
        vfield.push(b);
    }
    // hp_enc = enc_bytes(hp33) = [0xa1, hp33...] (34 bytes, const)
    let mut hp_enc: Vec<Vec<Variable>> = vec![byte_const_v(api, 0xa1)];
    for &x in hp33.iter() { hp_enc.push(byte_const_v(api, x)); }
    // body = hp_enc(34) ++ vfield(vf_len). body_len = 34 + vf_len.
    let c34 = const_bits(api, 34, 8);
    let cc0 = const_bits(api, 0xc0, 8);
    let body_len = add8(api, &c34, &vf_len);
    // leaf header = 0xc0 + body_len (small case: body < 56).
    let hdr = add8(api, &cc0, &body_len);
    let zbyte = vec![api.constant(0); 8];
    let mut leaf: Vec<Vec<Variable>> = Vec::with_capacity(135);
    leaf.push(hdr);
    for b in hp_enc.iter() { leaf.push(b.clone()); }
    for b in vfield.iter() { leaf.push(b.clone()); }
    while leaf.len() < 135 { leaf.push(zbyte.clone()); }
    let leaf_len = add8(api, &one8, &body_len); // 1 + body
    let flat: Vec<Variable> = leaf[0..135].iter().flatten().cloned().collect();
    crate::batch_keccak::keccak256_varlen(api, &flat, &leaf_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rlp::enc_uint;
    use crate::u256::bigint_to_bits;
    use num_bigint::BigInt;
    use num_traits::Num;

    declare_circuit!(MinRlpCircuit {
        v: [Variable; BITS],
        // public expected: 33 bytes buffer + len (only [0..len] asserted).
        out: [PublicVariable; 33 * 8],
        len: [PublicVariable; 6],
    });
    impl Define<GF2Config> for MinRlpCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let (buf, len) = min_rlp_uint(api, &self.v.to_vec());
            for b in 0..6 {
                api.assert_is_equal(len[b], self.len[b]);
            }
            // Buffer is canonical (tail masked to zero), so compare all 33 bytes.
            for i in 0..33 {
                for j in 0..8 {
                    api.assert_is_equal(buf[i][j], self.out[i * 8 + j]);
                }
            }
        }
    }

    fn run(v: &BigInt) {
        let enc = enc_uint(v);
        let CompileResult { witness_solver, layered_circuit } =
            compile(&MinRlpCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = MinRlpCircuit::<GF2>::default();
        asg.v.copy_from_slice(&bigint_to_bits(v, BITS).into_iter().map(|b| (b as u32).into()).collect::<Vec<GF2>>());
        // expected buffer: enc bytes then zero pad to 33.
        let mut buf = vec![0u8; 33];
        buf[..enc.len()].copy_from_slice(&enc);
        for i in 0..33 {
            for j in 0..8 {
                asg.out[i * 8 + j] = (((buf[i] >> j) & 1) as u32).into();
            }
        }
        for b in 0..6 {
            asg.len[b] = (((enc.len() as u32) >> b) & 1).into();
        }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "min_rlp_uint wrong for {v}");
    }

    use crate::mpt::{account_rlp, empty_root, hp, keccak256, keccak_empty, nibbles};
    use crate::rlp::{enc_bytes, enc_list};

    declare_circuit!(LeafCircuit {
        nonce: [Variable; BITS],
        balance: [Variable; BITS],
        out: [PublicVariable; 256],
    });
    // hp32 path constant installed via a process-global (single-threaded test).
    static mut HP32: [u8; 32] = [0u8; 32];
    impl Define<GF2Config> for LeafCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let hp32 = unsafe { HP32 };
            let sr = crate::mpt::empty_root();
            let ch = crate::mpt::keccak_empty();
            let h = account_leaf_hash(api, &self.nonce.to_vec(), &self.balance.to_vec(), &hp32, &sr, &ch);
            for i in 0..256 {
                api.assert_is_equal(h[i], self.out[i]);
            }
        }
    }

    use crate::mpt::{state_root, Account};

    static mut HP3: [[u8; 32]; 3] = [[0u8; 32]; 3];
    static mut SLOT3: [usize; 3] = [0; 3];
    declare_circuit!(RootCircuit {
        n: [[Variable; BITS]; 3],
        b: [[Variable; BITS]; 3],
        out: [PublicVariable; 256],
    });
    impl Define<GF2Config> for RootCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let sr = crate::mpt::empty_root();
            let ch = crate::mpt::keccak_empty();
            let (hp, slot) = unsafe { (HP3, SLOT3) };
            let n0 = self.n[0].to_vec(); let b0 = self.b[0].to_vec();
            let n1 = self.n[1].to_vec(); let b1 = self.b[1].to_vec();
            let n2 = self.n[2].to_vec(); let b2 = self.b[2].to_vec();
            let accts = [
                AcctIn { nonce: &n0, balance: &b0, hp32: hp[0], slot: slot[0] },
                AcctIn { nonce: &n1, balance: &b1, hp32: hp[1], slot: slot[1] },
                AcctIn { nonce: &n2, balance: &b2, hp32: hp[2], slot: slot[2] },
            ];
            let root = state_root_3(api, &accts, &sr, &ch);
            for i in 0..256 {
                api.assert_is_equal(root[i], self.out[i]);
            }
        }
    }

    #[test]
    fn circuit_state_root_matches_native_and_alloy() {
        // Pick 3 addresses whose keccak(address) have DISTINCT first nibbles
        // (flat single-branch topology; nested topology is the generalization).
        let mut addrs = [[0u8; 20]; 3];
        let mut slots = [0usize; 3];
        let mut hp3 = [[0u8; 32]; 3];
        let mut chosen = 0usize;
        let mut seen: Vec<usize> = vec![];
        let mut cand = 1u8;
        while chosen < 3 {
            let a = [cand; 20];
            let key = keccak256(&a);
            let nibs = nibbles(&key);
            let s = nibs[0] as usize;
            if !seen.contains(&s) {
                addrs[chosen] = a;
                slots[chosen] = s;
                hp3[chosen].copy_from_slice(&hp(&nibs[1..64], true));
                seen.push(s);
                chosen += 1;
            }
            cand = cand.wrapping_add(1);
            assert!(cand != 0, "ran out of candidate addresses");
        }
        let vals: [(u64, BigInt); 3] = [
            (7, BigInt::from(1_000_000_000_000_000_000u64)),
            (0, BigInt::from(500_000_000_000_000u64)),
            (0, BigInt::from(42_000u64)),
        ];
        unsafe { HP3 = hp3; SLOT3 = slots; }

        // native/alloy root (== 0xfae227113d5d…708b057, cross-checked earlier).
        let accs: Vec<Account> = (0..3).map(|i| Account::eoa(addrs[i], vals[i].0, vals[i].1.clone())).collect();
        let native = state_root(&accs);

        let CompileResult { witness_solver, layered_circuit } =
            compile(&RootCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = RootCircuit::<GF2>::default();
        for i in 0..3 {
            asg.n[i].copy_from_slice(&bigint_to_bits(&BigInt::from(vals[i].0), BITS).into_iter().map(|x| (x as u32).into()).collect::<Vec<GF2>>());
            asg.b[i].copy_from_slice(&bigint_to_bits(&vals[i].1, BITS).into_iter().map(|x| (x as u32).into()).collect::<Vec<GF2>>());
        }
        for i in 0..32 {
            for j in 0..8 {
                asg.out[i * 8 + j] = (((native[i] >> j) & 1) as u32).into();
            }
        }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit state root != native/alloy");
        println!("R7 circuit state_root == native/alloy 0x{}", native.iter().map(|x| format!("{:02x}", x)).collect::<String>());
    }

    use crate::rlp::Eip1559Tx;
    use crate::stf_transfer::{apply, BlockEnv, TransferBlock};

    declare_circuit!(StfCircuit {
        n: [[Variable; BITS]; 3],
        b: [[Variable; BITS]; 3],
        value: [Variable; BITS],
        maxfee: [Variable; BITS],
        maxprio: [Variable; BITS],
        basefee: [Variable; BITS],
        out: [PublicVariable; 256],
    });
    impl Define<GF2Config> for StfCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let sr = crate::mpt::empty_root();
            let ch = crate::mpt::keccak_empty();
            let (hp, slot) = unsafe { (HP3, SLOT3) };
            let pn = [self.n[0].to_vec(), self.n[1].to_vec(), self.n[2].to_vec()];
            let pb = [self.b[0].to_vec(), self.b[1].to_vec(), self.b[2].to_vec()];
            let root = transfer_stf_root(api, &pn, &pb, &self.value.to_vec(), &self.maxfee.to_vec(), &self.maxprio.to_vec(), &self.basefee.to_vec(), &hp, &slot, &sr, &ch);
            for i in 0..256 {
                api.assert_is_equal(root[i], self.out[i]);
            }
        }
    }

    #[test]
    fn circuit_transfer_stf_root_matches_native() {
        // 3 distinct-first-nibble addresses: sender=0, recipient=1, coinbase=2.
        let mut addrs = [[0u8; 20]; 3];
        let mut slots = [0usize; 3];
        let mut hp3 = [[0u8; 32]; 3];
        let (mut chosen, mut cand) = (0usize, 1u8);
        let mut seen: Vec<usize> = vec![];
        while chosen < 3 {
            let a = [cand; 20];
            let nibs = nibbles(&keccak256(&a));
            let s = nibs[0] as usize;
            if !seen.contains(&s) {
                addrs[chosen] = a; slots[chosen] = s;
                hp3[chosen].copy_from_slice(&hp(&nibs[1..64], true));
                seen.push(s); chosen += 1;
            }
            cand = cand.wrapping_add(1);
        }
        unsafe { HP3 = hp3; SLOT3 = slots; }

        let (nonce, bal0) = (7u64, BigInt::from(1_000_000_000_000_000_000u64));
        let bal1 = BigInt::from(500_000_000_000_000u64); // recipient prefunded
        let value = BigInt::from(1_000_000_000_000_000u64);
        let (maxfee, maxprio, basefee) = (BigInt::from(20u64), BigInt::from(2u64), BigInt::from(7u64));

        let pre = vec![
            crate::mpt::Account::eoa(addrs[0], nonce, bal0.clone()),
            crate::mpt::Account::eoa(addrs[1], 0, bal1.clone()),
            crate::mpt::Account::eoa(addrs[2], 0, BigInt::from(0u64)),
        ];
        let tx = Eip1559Tx {
            chain_id: 1, nonce, max_priority_fee: maxprio.clone(), max_fee: maxfee.clone(),
            gas_limit: 21000, to: addrs[1], value: value.clone(), data: vec![], y_parity: 0, r: BigInt::from(1), s: BigInt::from(1),
        };
        let env = BlockEnv { base_fee: basefee.clone(), coinbase: addrs[2] };
        let block = TransferBlock { pre, tx, sender: addrs[0], env };
        let (_post, native_root) = apply(&block).unwrap();

        let CompileResult { witness_solver, layered_circuit } =
            compile(&StfCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = StfCircuit::<GF2>::default();
        let put = |dst: &mut [GF2], v: &BigInt| dst.copy_from_slice(&bigint_to_bits(v, BITS).into_iter().map(|x| (x as u32).into()).collect::<Vec<GF2>>());
        put(&mut asg.n[0], &BigInt::from(nonce)); put(&mut asg.n[1], &BigInt::from(0u64)); put(&mut asg.n[2], &BigInt::from(0u64));
        put(&mut asg.b[0], &bal0); put(&mut asg.b[1], &bal1); put(&mut asg.b[2], &BigInt::from(0u64));
        put(&mut asg.value, &value); put(&mut asg.maxfee, &maxfee); put(&mut asg.maxprio, &maxprio); put(&mut asg.basefee, &basefee);
        for i in 0..32 { for j in 0..8 { asg.out[i * 8 + j] = (((native_root[i] >> j) & 1) as u32).into(); } }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit STF post_root != native");
        println!("R7 in-circuit STF post_state_root == native 0x{}", native_root.iter().map(|x| format!("{:02x}", x)).collect::<String>());
    }

    #[test]
    fn account_leaf_hash_matches_native() {
        // Demo sender account at its 63-nibble leaf path (as inside the branch).
        let addr = [0x11u8; 20];
        let key = keccak256(&addr);
        let nibs = nibbles(&key);
        let hp32_vec = hp(&nibs[1..64], true); // 63 nibbles -> 32 bytes
        assert_eq!(hp32_vec.len(), 32);
        let mut hp32 = [0u8; 32];
        hp32.copy_from_slice(&hp32_vec);
        unsafe { HP32 = hp32 };

        let nonce = 7u64;
        let balance = BigInt::from(1_000_000_000_000_000_000u64);
        // native leaf hash
        let acct = account_rlp(nonce, &balance, &empty_root(), &keccak_empty());
        let leaf_rlp = enc_list(&[enc_bytes(&hp32_vec), enc_bytes(&acct)]);
        let native = keccak256(&leaf_rlp);

        let CompileResult { witness_solver, layered_circuit } =
            compile(&LeafCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = LeafCircuit::<GF2>::default();
        asg.nonce.copy_from_slice(&bigint_to_bits(&BigInt::from(nonce), BITS).into_iter().map(|b| (b as u32).into()).collect::<Vec<GF2>>());
        asg.balance.copy_from_slice(&bigint_to_bits(&balance, BITS).into_iter().map(|b| (b as u32).into()).collect::<Vec<GF2>>());
        for i in 0..32 {
            for j in 0..8 {
                asg.out[i * 8 + j] = (((native[i] >> j) & 1) as u32).into();
            }
        }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit leaf hash != native");
    }

    static mut HP33S: [u8; 33] = [0u8; 33];
    declare_circuit!(StorageRootCircuit {
        val: [Variable; BITS],
        out: [PublicVariable; 256],
    });
    impl Define<GF2Config> for StorageRootCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let hp = unsafe { HP33S };
            let r = super::storage_leaf_root(api, &hp, &self.val.to_vec());
            for i in 0..256 { api.assert_is_equal(r[i], self.out[i]); }
        }
    }

    #[test]
    fn incircuit_storage_root_matches_native() {
        use crate::mpt::{hp, keccak256, nibbles, storage_root};
        let slot = BigInt::from(0u32);
        let val = BigInt::from(0x2au32);
        let native = storage_root(&[(slot.clone(), val.clone())]);
        // hp33 = HP(nibbles(keccak(slot_be32=zeros)), leaf).
        let key = keccak256(&[0u8; 32]);
        let nibs = nibbles(&key);
        let hpv = hp(&nibs, true);
        assert_eq!(hpv.len(), 33);
        let mut hp33 = [0u8; 33];
        hp33.copy_from_slice(&hpv);
        unsafe { HP33S = hp33; }
        let CompileResult { witness_solver, layered_circuit } = compile(&StorageRootCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = StorageRootCircuit::<GF2>::default();
        asg.val.copy_from_slice(&bigint_to_bits(&val, BITS).into_iter().map(|b| (b as u32).into()).collect::<Vec<GF2>>());
        for i in 0..32 { for j in 0..8 { asg.out[i * 8 + j] = (((native[i] >> j) & 1) as u32).into(); } }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit storage_root != native (0x81d1fa69…)");
    }

    #[test]
    fn min_rlp_uint_matches_native() {
        for s in ["0", "7", "127", "128", "255", "256", "1000000000000000000", "500000000000000"] {
            run(&BigInt::from_str_radix(s, 10).unwrap());
        }
        // full 32-byte value
        run(&BigInt::from_str_radix("a1b2c3d4e5f60718293a4b5c6d7e8f90112233445566778899aabbccddeeff00", 16).unwrap());
    }
}
