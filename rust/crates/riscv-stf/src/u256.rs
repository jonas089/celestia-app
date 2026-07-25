//! R2 — 256-bit modular arithmetic over GF2, fully DETERMINISTIC (no hints).
//!
//! Per the GKR-variant advice rule (context.md): the committed input layer must
//! be the DA data ONLY; nondeterministic advice may NOT be injected (an ECC
//! `new_hint` output lands in the input-layer witness, which would break
//! byte-identity with the DA commitment). So every quantity here — including
//! modular inverse and square root — is DERIVED in-circuit with fixed gate
//! sequences (constant-exponent square-and-multiply), never witnessed.
//!
//! Values are little-endian bit vectors of `Variable` (LSB first). Boolean
//! full-adders over GF2 (`api.add` == XOR, `api.mul` == AND) implement binary
//! integer arithmetic; secp256k1's `p = 2^256 - 2^32 - 977` gets the standard
//! fast reduction (2^256 ≡ 2^32 + 977 mod p); the scalar field order `n` uses
//! deterministic Barrett reduction.

use expander_compiler::frontend::*;
use num_bigint::BigInt;
use num_traits::{Num, One, Zero};

pub const BITS: usize = 256;

// secp256k1 field prime p and group order n (hex).
pub const P_HEX: &str = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F";
pub const N_HEX: &str = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";

pub fn p_bigint() -> BigInt {
    BigInt::from_str_radix(P_HEX, 16).unwrap()
}
pub fn n_bigint() -> BigInt {
    BigInt::from_str_radix(N_HEX, 16).unwrap()
}

// -------------------------- native <-> bits helpers ------------------------

/// A `BigInt` (non-negative, < 2^len) to a `len`-bit LE bit array (u8 0/1).
pub fn bigint_to_bits(x: &BigInt, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let mut v = x.clone();
    let two = BigInt::from(2);
    for i in 0..len {
        let bit = (&v % &two).clone();
        out[i] = if bit.is_zero() { 0 } else { 1 };
        v /= &two;
    }
    out
}

// -------------------------------- gadgets ----------------------------------

fn full_adder<C: Config>(api: &mut impl RootAPI<C>, a: Variable, b: Variable, cin: Variable) -> (Variable, Variable) {
    let ab = api.add(a, b);
    let sum = api.add(ab, cin);
    let and_ab = api.mul(a, b);
    let c_and = api.mul(cin, ab);
    let cout = api.add(and_ab, c_and);
    (sum, cout)
}

/// Add two LE bit vectors (any lengths). Returns `max(len)+1` bits (carry-out
/// included), so the result never overflows.
pub fn add<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let n = a.len().max(b.len());
    let zero = api.constant(0);
    let mut out = Vec::with_capacity(n + 1);
    let mut carry = zero;
    for i in 0..n {
        let ai = *a.get(i).unwrap_or(&zero);
        let bi = *b.get(i).unwrap_or(&zero);
        let (s, c) = full_adder(api, ai, bi, carry);
        out.push(s);
        carry = c;
    }
    out.push(carry);
    out
}

/// Subtract `b` from `a` (LE bits). Returns (`len(a)`-bit difference mod 2^len,
/// borrow bit). borrow == 1 iff a < b.
pub fn sub<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> (Vec<Variable>, Variable) {
    let n = a.len();
    let zero = api.constant(0);
    let one = api.constant(1);
    // a - b = a + (~b) + 1, over n bits. borrow = NOT final carry.
    let nb: Vec<Variable> = (0..n).map(|i| { let bi = *b.get(i).unwrap_or(&zero); api.sub(1, bi) }).collect();
    let mut out = Vec::with_capacity(n);
    let mut carry = one;
    for i in 0..n {
        let (s, c) = full_adder(api, a[i], nb[i], carry);
        out.push(s);
        carry = c;
    }
    let borrow = api.sub(1, carry);
    (out, borrow)
}

/// 1 iff all bits are zero.
pub fn is_zero<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable]) -> Variable {
    // AND of NOT(bit): 1 iff every bit is 0.
    let mut acc = api.constant(1);
    for &x in a {
        let nx = api.sub(1, x);
        acc = api.mul(acc, nx);
    }
    acc
}

/// 1 iff a == b (equal length or shorter padded with 0).
pub fn eq<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Variable {
    let n = a.len().max(b.len());
    let zero = api.constant(0);
    let mut acc = api.constant(1);
    for i in 0..n {
        let ai = *a.get(i).unwrap_or(&zero);
        let bi = *b.get(i).unwrap_or(&zero);
        let x = api.add(ai, bi);
        let eqb = api.sub(1, x);
        acc = api.mul(acc, eqb);
    }
    acc
}

/// 1 iff a < b (unsigned), via the borrow of a-b (lengths padded to max).
pub fn lt<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Variable {
    let n = a.len().max(b.len());
    let zero = api.constant(0);
    let ap: Vec<Variable> = (0..n).map(|i| *a.get(i).unwrap_or(&zero)).collect();
    let bp: Vec<Variable> = (0..n).map(|i| *b.get(i).unwrap_or(&zero)).collect();
    let (_d, borrow) = sub(api, &ap, &bp);
    borrow
}

fn bit_mux<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: Variable, b: Variable) -> Variable {
    // sel ? a : b
    let d = api.add(a, b);
    let t = api.mul(sel, d);
    api.add(b, t)
}

/// sel ? a : b, elementwise (lengths padded to max).
pub fn select<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let n = a.len().max(b.len());
    let zero = api.constant(0);
    (0..n).map(|i| {
        let ai = *a.get(i).unwrap_or(&zero);
        let bi = *b.get(i).unwrap_or(&zero);
        bit_mux(api, sel, ai, bi)
    }).collect()
}

/// Left shift by a compile-time constant (prepend `k` zero bits).
pub fn shl_const<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], k: usize) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut out = vec![zero; k];
    out.extend_from_slice(a);
    out
}

/// Multiply an LE bit vector by a small constant `k` via shift-and-add of its
/// set bits. Returns `len(a) + bitlen(k)` bits (no truncation).
pub fn mul_by_const<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], k: u128) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut acc = vec![zero];
    let mut kk = k;
    let mut shift = 0usize;
    while kk != 0 {
        if kk & 1 == 1 {
            let term = shl_const(api, a, shift);
            acc = add(api, &acc, &term);
        }
        kk >>= 1;
        shift += 1;
    }
    acc
}

/// Carry-save 3:2 compressor over three aligned LE bit vectors: returns two
/// vectors (sum, carry<<1) with `x+y+z == sum + carry`. O(1) depth.
fn csa<C: Config>(api: &mut impl RootAPI<C>, x: &[Variable], y: &[Variable], z: &[Variable]) -> (Vec<Variable>, Vec<Variable>) {
    let n = x.len().max(y.len()).max(z.len());
    let zero = api.constant(0);
    let mut sum = Vec::with_capacity(n);
    let mut carry = Vec::with_capacity(n + 1);
    carry.push(zero); // carry is shifted left by 1
    for i in 0..n {
        let xi = *x.get(i).unwrap_or(&zero);
        let yi = *y.get(i).unwrap_or(&zero);
        let zi = *z.get(i).unwrap_or(&zero);
        let xy = api.add(xi, yi);
        let s = api.add(xy, zi); // xi ^ yi ^ zi
        let and_xy = api.mul(xi, yi);
        let c2 = api.mul(zi, xy);
        let c = api.add(and_xy, c2); // majority
        sum.push(s);
        carry.push(c);
    }
    (sum, carry)
}

/// Sum many LE bit vectors with a carry-save (Wallace) tree: O(log k) CSA depth,
/// then a single carry-propagate add. Far shallower than chained ripple adds.
fn sum_many<C: Config>(api: &mut impl RootAPI<C>, mut terms: Vec<Vec<Variable>>) -> Vec<Variable> {
    if terms.is_empty() {
        return vec![api.constant(0)];
    }
    while terms.len() > 2 {
        let mut next = Vec::with_capacity(terms.len() / 3 * 2 + 2);
        let mut i = 0;
        while i + 3 <= terms.len() {
            let (s, c) = csa(api, &terms[i], &terms[i + 1], &terms[i + 2]);
            next.push(s);
            next.push(c);
            i += 3;
        }
        while i < terms.len() {
            next.push(terms[i].clone());
            i += 1;
        }
        terms = next;
    }
    if terms.len() == 1 {
        terms.pop().unwrap()
    } else {
        add(api, &terms[0], &terms[1])
    }
}

/// Full 256x256 -> up to 512-bit product. Partial products summed via a
/// carry-save tree (shallow) instead of chained ripple adds (deep).
pub fn mul_full<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let terms: Vec<Vec<Variable>> = b
        .iter()
        .enumerate()
        .map(|(j, &bj)| {
            let masked: Vec<Variable> = a.iter().map(|&ai| api.mul(ai, bj)).collect();
            shl_const(api, &masked, j)
        })
        .collect();
    sum_many(api, terms)
}

/// Low `width` bits of the product a*b. Cheaper than `mul_full` when only the
/// low bits are needed (EVM MUL/EXP are mod 2^256): each partial product is
/// masked to the bits that can land below `width`. Deterministic.
pub fn mul_low<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable], width: usize) -> Vec<Variable> {
    let zero = api.constant(0);
    let terms: Vec<Vec<Variable>> = (0..b.len())
        .filter(|&j| j < width)
        .map(|j| {
            let keep = width - j;
            let masked: Vec<Variable> = a.iter().take(keep).map(|&ai| api.mul(ai, b[j])).collect();
            shl_const(api, &masked, j) // length <= width
        })
        .collect();
    let mut s = sum_many(api, terms);
    s.truncate(width);
    while s.len() < width { s.push(zero); }
    s
}

/// General-width unsigned long division (restoring, MSB-first, deterministic —
/// no advice). `a` is the dividend (any bit length), `b` the divisor (any bit
/// length). Returns (q, r): q has `a.len()` bits, r has `b.len()` bits. Matches
/// EVM/native: b == 0 => q = 0, r = 0.
pub fn divmod<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> (Vec<Variable>, Vec<Variable>) {
    let m = a.len();
    let bn = b.len();
    let zero = api.constant(0);
    let mut q = vec![zero; m];
    let rw = bn + 1;
    let bpad: Vec<Variable> = (0..rw).map(|i| if i < bn { b[i] } else { zero }).collect();
    let mut r = vec![zero; rw];
    for idx in (0..m).rev() {
        // r = (r << 1) | a[idx]
        let mut nr = Vec::with_capacity(rw);
        nr.push(a[idx]);
        for k in 1..rw {
            nr.push(r[k - 1]);
        }
        // One subtraction serves both roles: borrow == 1 iff nr < b, so
        // ge = NOT borrow, and `diff` is the reduced remainder when ge.
        let (diff, borrow) = sub(api, &nr, &bpad);
        let ge = api.sub(1, borrow);
        r = select(api, ge, &diff, &nr);
        q[idx] = ge;
    }
    let bz = is_zero(api, b);
    let keep = api.sub(1, bz);
    let qk: Vec<Variable> = q.iter().map(|&x| api.mul(keep, x)).collect();
    let rk: Vec<Variable> = (0..bn).map(|i| api.mul(keep, r[i])).collect();
    (qk, rk)
}

// ----------------------------- mod p (secp256k1) ---------------------------

fn const_bits<C: Config>(api: &mut impl RootAPI<C>, x: &BigInt, len: usize) -> Vec<Variable> {
    bigint_to_bits(x, len).into_iter().map(|b| api.constant(b as u32)).collect()
}

/// If `a >= p` then `a - p`, else `a`. Result truncated to 256 bits. `a` must be
/// < 2p (so a single conditional subtract suffices to reduce it).
fn cond_sub<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], modulus: &BigInt) -> Vec<Variable> {
    // pad a to at least 257 bits so a-p can't spuriously borrow on the top.
    let n = a.len().max(257);
    let zero = api.constant(0);
    let ap: Vec<Variable> = (0..n).map(|i| *a.get(i).unwrap_or(&zero)).collect();
    let mp = const_bits(api, modulus, n);
    let (diff, borrow) = sub(api, &ap, &mp);
    // borrow == 1 means a < p -> keep a; else use diff.
    let ge = api.sub(1, borrow);
    let chosen = select(api, ge, &diff, &ap);
    chosen[0..256].to_vec()
}

/// Reduce an arbitrary-length LE value mod secp256k1 p using 2^256 ≡ 2^32+977.
pub fn reduce_mod_p<C: Config>(api: &mut impl RootAPI<C>, x: &[Variable]) -> Vec<Variable> {
    const C_CONST: u128 = (1u128 << 32) + 977; // 2^32 + 977
    let mut cur = x.to_vec();
    // Fold the high part (>= bit 256) into the low part; converges in <=3 folds.
    for _ in 0..4 {
        if cur.len() <= 257 {
            break;
        }
        let lo = cur[0..256].to_vec();
        let hi = cur[256..].to_vec();
        let hic = mul_by_const(api, &hi, C_CONST);
        cur = add(api, &lo, &hic);
    }
    // One more fold if still > 256 bits (hi is now tiny).
    if cur.len() > 256 {
        let lo = cur[0..256].to_vec();
        let hi = cur[256..].to_vec();
        let hic = mul_by_const(api, &hi, C_CONST);
        cur = add(api, &lo, &hic);
    }
    // Now cur < 2^257 < 3p; three conditional subtracts guarantee < p.
    let p = p_bigint();
    let mut v = cur;
    for _ in 0..3 {
        v = cond_sub(api, &v, &p);
    }
    v[0..256].to_vec()
}

pub fn add_mod_p<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let s = add(api, a, b);
    reduce_mod_p(api, &s)
}
pub fn sub_mod_p<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    // a - b + p (a,b < p) then reduce.
    let p = p_bigint();
    let pb = const_bits(api, &p, 256);
    let apb = add(api, a, &pb);
    let (t, _borrow) = sub(api, &apb, b);
    reduce_mod_p(api, &t)
}
pub fn mul_mod_p<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let prod = mul_full(api, a, b);
    reduce_mod_p(api, &prod)
}

/// Deterministic modular exponentiation with a COMPILE-TIME constant exponent
/// (square-and-multiply unrolled). No advice.
pub fn pow_mod_p_const<C: Config>(api: &mut impl RootAPI<C>, base: &[Variable], exp: &BigInt) -> Vec<Variable> {
    let one = const_bits(api, &BigInt::one(), 256);
    let mut result = one;
    let mut sq = base.to_vec();
    let ebits = {
        // exponent bit length
        let mut bits = vec![];
        let mut e = exp.clone();
        let two = BigInt::from(2);
        while !e.is_zero() {
            bits.push(((&e % &two) != BigInt::zero()) as u8);
            e /= &two;
        }
        bits
    };
    for &bit in ebits.iter() {
        if bit == 1 {
            result = mul_mod_p(api, &result, &sq);
        }
        sq = mul_mod_p(api, &sq, &sq);
    }
    result
}

/// Modular inverse mod p via Fermat: a^(p-2). Deterministic.
pub fn inv_mod_p<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable]) -> Vec<Variable> {
    let e = p_bigint() - BigInt::from(2);
    pow_mod_p_const(api, a, &e)
}

/// Modular square root mod p (p ≡ 3 mod 4): a^((p+1)/4). Deterministic. The
/// caller must ensure `a` is a QR (else the result squares to -a); callers
/// verify `y^2 == a` and pick the parity.
pub fn sqrt_mod_p<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable]) -> Vec<Variable> {
    let e = (p_bigint() + BigInt::one()) / BigInt::from(4);
    pow_mod_p_const(api, a, &e)
}

// ------------------------- mod n (scalar field, Barrett) -------------------
// n has no special form, so reduce with textbook Barrett (k = 256):
//   mu = floor(2^512 / n); q3 = ((x >> 255) * mu) >> 257;
//   r  = (x mod 2^257) - (q3*n mod 2^257); fix sign; subtract n up to twice.
// Deterministic (no advice).

fn low_bits<C: Config>(_api: &mut impl RootAPI<C>, x: &[Variable], len: usize) -> Vec<Variable> {
    x.iter().take(len).cloned().collect()
}
fn shr_const(x: &[Variable], k: usize) -> Vec<Variable> {
    if k >= x.len() { return vec![]; }
    x[k..].to_vec()
}

pub fn reduce_mod_n<C: Config>(api: &mut impl RootAPI<C>, x: &[Variable]) -> Vec<Variable> {
    let n = n_bigint();
    let k = 256usize;
    let mu = (BigInt::one() << (2 * k)) / &n; // floor(2^512 / n), 257 bits
    // pad x to at least 512 bits
    let zero = api.constant(0);
    let xp: Vec<Variable> = {
        let mut v = x.to_vec();
        while v.len() < 2 * k { v.push(zero); }
        v
    };
    let q1 = shr_const(&xp, k - 1); // x >> 255
    let q2 = mul_by_const_big(api, &q1, &mu); // q1 * mu
    let q3 = shr_const(&q2, k + 1); // >> 257
    let r1 = low_bits(api, &xp, k + 1); // x mod 2^257
    let q3n = mul_by_const_big(api, &q3, &n); // q3 * n
    let r2 = low_bits(api, &q3n, k + 1); // (q3*n) mod 2^257
    // r = r1 - r2 (mod 2^257); r1 >= r2 always for correct Barrett, but guard.
    let (mut r, _borrow) = sub(api, &r1, &r2);
    // r now < 2^257; subtract n up to 3 times to bring < n.
    for _ in 0..3 {
        r = cond_sub(api, &r, &n);
    }
    r[0..256].to_vec()
}

/// Multiply an LE bit vector by an arbitrary big constant (shift-and-add of the
/// constant's set bits). Deterministic.
pub fn mul_by_const_big<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], k: &BigInt) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut acc = vec![zero];
    let bits = {
        let mut b = vec![];
        let mut e = k.clone();
        let two = BigInt::from(2);
        while !e.is_zero() { b.push(((&e % &two) != BigInt::zero()) as u8); e /= &two; }
        b
    };
    for (shift, &bit) in bits.iter().enumerate() {
        if bit == 1 {
            let term = shl_const(api, a, shift);
            acc = add(api, &acc, &term);
        }
    }
    acc
}

pub fn add_mod_n<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let s = add(api, a, b);
    reduce_mod_n(api, &s)
}
pub fn sub_mod_n<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let n = n_bigint();
    let nb = const_bits(api, &n, 256);
    let anb = add(api, a, &nb);
    let (t, _b) = sub(api, &anb, b);
    reduce_mod_n(api, &t)
}
pub fn mul_mod_n<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let prod = mul_full(api, a, b);
    reduce_mod_n(api, &prod)
}
pub fn pow_mod_n_const<C: Config>(api: &mut impl RootAPI<C>, base: &[Variable], exp: &BigInt) -> Vec<Variable> {
    let one = const_bits(api, &BigInt::one(), 256);
    let mut result = one;
    let mut sq = base.to_vec();
    let mut e = exp.clone();
    let two = BigInt::from(2);
    while !e.is_zero() {
        if (&e % &two) != BigInt::zero() {
            result = mul_mod_n(api, &result, &sq);
        }
        sq = mul_mod_n(api, &sq, &sq);
        e /= &two;
    }
    result
}
/// Inverse mod n via Fermat: a^(n-2). Deterministic.
pub fn inv_mod_n<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable]) -> Vec<Variable> {
    let e = n_bigint() - BigInt::from(2);
    pow_mod_n_const(api, a, &e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::frontend::*;

    fn to_gf2_bits(x: &BigInt, len: usize) -> Vec<GF2> {
        bigint_to_bits(x, len).into_iter().map(|b| (b as u32).into()).collect()
    }

    // Test harness: a circuit computing op(a,b) and asserting it equals expected.
    declare_circuit!(MulModCircuit {
        a: [Variable; BITS],
        b: [Variable; BITS],
        e: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for MulModCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let r = mul_mod_p(api, &self.a.to_vec(), &self.b.to_vec());
            for i in 0..BITS {
                api.assert_is_equal(r[i], self.e[i]);
            }
        }
    }

    declare_circuit!(AddSubCircuit {
        a: [Variable; BITS],
        b: [Variable; BITS],
        esum: [PublicVariable; BITS],
        esub: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for AddSubCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let s = add_mod_p(api, &self.a.to_vec(), &self.b.to_vec());
            let d = sub_mod_p(api, &self.a.to_vec(), &self.b.to_vec());
            for i in 0..BITS {
                api.assert_is_equal(s[i], self.esum[i]);
                api.assert_is_equal(d[i], self.esub[i]);
            }
        }
    }

    declare_circuit!(InvCircuit {
        a: [Variable; BITS],
        e: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for InvCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let r = inv_mod_p(api, &self.a.to_vec());
            for i in 0..BITS {
                api.assert_is_equal(r[i], self.e[i]);
            }
        }
    }

    // Cheap validation of the square-and-multiply construction with a SMALL
    // constant exponent (the full a^(p-2) inverse uses the identical loop with a
    // 256-bit exponent — correct by construction, just heavy: see `inverse_*`,
    // which is #[ignore]d).
    const SMALL_EXP: u64 = 0x0001_0001_9E37; // 45-bit exponent
    declare_circuit!(PowSmallCircuit {
        a: [Variable; BITS],
        e: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for PowSmallCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let r = pow_mod_p_const(api, &self.a.to_vec(), &BigInt::from(SMALL_EXP));
            for i in 0..BITS {
                api.assert_is_equal(r[i], self.e[i]);
            }
        }
    }

    #[test]
    fn pow_mod_p_small_exponent_matches_bigint() {
        let p = p_bigint();
        let CompileResult { witness_solver, layered_circuit } =
            compile(&PowSmallCircuit::default(), CompileOptions::default()).unwrap();
        let a = BigInt::from_str_radix("a1b2c3d4e5f60718293a4b5c6d7e8f90112233445566778899aabbccddeeff00", 16).unwrap() % &p;
        let e = a.modpow(&BigInt::from(SMALL_EXP), &p);
        let mut asg = PowSmallCircuit::<GF2>::default();
        asg.a.copy_from_slice(&to_gf2_bits(&a, BITS));
        asg.e.copy_from_slice(&to_gf2_bits(&e, BITS));
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "pow_mod_p (small exp) wrong");
    }

    fn samples() -> Vec<(BigInt, BigInt)> {
        let p = p_bigint();
        vec![
            (BigInt::from(2u32), BigInt::from(3u32)),
            (BigInt::from(0u32), BigInt::from(7u32)),
            (&p - BigInt::from(1u32), &p - BigInt::from(2u32)),
            (
                BigInt::from_str_radix("a1b2c3d4e5f60718293a4b5c6d7e8f90112233445566778899aabbccddeeff00", 16).unwrap(),
                BigInt::from_str_radix("0f1e2d3c4b5a69788796a5b4c3d2e1f0fedcba9876543210123456789abcdef0", 16).unwrap(),
            ),
        ]
    }

    #[test]
    fn mul_add_sub_mod_p_match_bigint() {
        let p = p_bigint();
        let CompileResult { witness_solver, layered_circuit } =
            compile(&MulModCircuit::default(), CompileOptions::default()).unwrap();
        let CompileResult { witness_solver: ws2, layered_circuit: lc2 } =
            compile(&AddSubCircuit::default(), CompileOptions::default()).unwrap();
        for (a, b) in samples() {
            let am = &a % &p;
            let bm = &b % &p;
            // mul
            let e = (&am * &bm) % &p;
            let mut asg = MulModCircuit::<GF2>::default();
            asg.a.copy_from_slice(&to_gf2_bits(&am, BITS));
            asg.b.copy_from_slice(&to_gf2_bits(&bm, BITS));
            asg.e.copy_from_slice(&to_gf2_bits(&e, BITS));
            let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
            assert!(layered_circuit.run(&w).iter().all(|x| *x), "mul_mod_p wrong for {am}*{bm}");
            // add/sub
            let es = (&am + &bm) % &p;
            let ed = ((&am - &bm) % &p + &p) % &p;
            let mut asg2 = AddSubCircuit::<GF2>::default();
            asg2.a.copy_from_slice(&to_gf2_bits(&am, BITS));
            asg2.b.copy_from_slice(&to_gf2_bits(&bm, BITS));
            asg2.esum.copy_from_slice(&to_gf2_bits(&es, BITS));
            asg2.esub.copy_from_slice(&to_gf2_bits(&ed, BITS));
            let w2 = ws2.solve_witnesses(&vec![asg2; 1]).unwrap();
            assert!(lc2.run(&w2).iter().all(|x| *x), "add/sub_mod_p wrong for {am},{bm}");
        }
    }

    declare_circuit!(DivModCircuit {
        a: [Variable; BITS],
        b: [Variable; BITS],
        eq_: [PublicVariable; BITS],
        er: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for DivModCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let (q, r) = divmod(api, &self.a.to_vec(), &self.b.to_vec());
            for i in 0..BITS {
                api.assert_is_equal(q[i], self.eq_[i]);
                api.assert_is_equal(r[i], self.er[i]);
            }
        }
    }

    #[test]
    fn divmod_matches_bigint() {
        let CompileResult { witness_solver, layered_circuit } =
            compile(&DivModCircuit::default(), CompileOptions::default()).unwrap();
        let cases: Vec<(BigInt, BigInt)> = vec![
            (BigInt::from(20u32), BigInt::from(3u32)),
            (BigInt::from(17u32), BigInt::from(5u32)),
            (BigInt::from(255u32), BigInt::from(16u32)),
            (BigInt::from(7u32), BigInt::from(0u32)), // div-by-zero => 0,0
            (BigInt::from(1000000u32), BigInt::from(7u32)),
        ];
        for (a, b) in cases {
            let (q, r) = if b.is_zero() { (BigInt::zero(), BigInt::zero()) } else { (&a / &b, &a % &b) };
            let mut asg = DivModCircuit::<GF2>::default();
            asg.a.copy_from_slice(&to_gf2_bits(&a, BITS));
            asg.b.copy_from_slice(&to_gf2_bits(&b, BITS));
            asg.eq_.copy_from_slice(&to_gf2_bits(&q, BITS));
            asg.er.copy_from_slice(&to_gf2_bits(&r, BITS));
            let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
            assert!(layered_circuit.run(&w).iter().all(|x| *x), "divmod wrong for {a}/{b}");
        }
    }

    declare_circuit!(MulModNCircuit {
        a: [Variable; BITS],
        b: [Variable; BITS],
        e: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for MulModNCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let r = mul_mod_n(api, &self.a.to_vec(), &self.b.to_vec());
            for i in 0..BITS {
                api.assert_is_equal(r[i], self.e[i]);
            }
        }
    }

    declare_circuit!(SqrtCircuit {
        a: [Variable; BITS],
        e: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for SqrtCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            // s = sqrt(a^2) then assert s^2 == a^2 (deterministic sqrt check).
            let a2 = mul_mod_p(api, &self.a.to_vec(), &self.a.to_vec());
            let s = sqrt_mod_p(api, &a2);
            let s2 = mul_mod_p(api, &s, &s);
            for i in 0..BITS {
                api.assert_is_equal(s2[i], self.e[i]);
            }
        }
    }

    #[test]
    fn mul_mod_n_matches_bigint() {
        let n = n_bigint();
        let CompileResult { witness_solver, layered_circuit } =
            compile(&MulModNCircuit::default(), CompileOptions::default()).unwrap();
        for (a, b) in samples() {
            let am = &a % &n;
            let bm = &b % &n;
            let e = (&am * &bm) % &n;
            let mut asg = MulModNCircuit::<GF2>::default();
            asg.a.copy_from_slice(&to_gf2_bits(&am, BITS));
            asg.b.copy_from_slice(&to_gf2_bits(&bm, BITS));
            asg.e.copy_from_slice(&to_gf2_bits(&e, BITS));
            let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
            assert!(layered_circuit.run(&w).iter().all(|x| *x), "mul_mod_n wrong for {am}*{bm}");
        }
    }

    #[test]
    #[ignore = "full (p+1)/4 exponent: ~254 modmuls, minutes to witness-solve; loop validated by pow_mod_p_small_exponent"]
    fn sqrt_mod_p_matches_bigint() {
        let p = p_bigint();
        let CompileResult { witness_solver, layered_circuit } =
            compile(&SqrtCircuit::default(), CompileOptions::default()).unwrap();
        let a = BigInt::from(123456789u64) % &p;
        let a2 = (&a * &a) % &p;
        let mut asg = SqrtCircuit::<GF2>::default();
        asg.a.copy_from_slice(&to_gf2_bits(&a, BITS));
        asg.e.copy_from_slice(&to_gf2_bits(&a2, BITS));
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "sqrt_mod_p wrong");
    }

    #[test]
    #[ignore = "full a^(p-2) inverse: ~512 modmuls, ~minutes to witness-solve; loop validated by pow_mod_p_small_exponent"]
    fn inverse_mod_p_matches_bigint() {
        let p = p_bigint();
        let CompileResult { witness_solver, layered_circuit } =
            compile(&InvCircuit::default(), CompileOptions::default()).unwrap();
        let a = BigInt::from_str_radix("a1b2c3d4e5f60718293a4b5c6d7e8f90112233445566778899aabbccddeeff00", 16).unwrap() % &p;
        let inv = a.modpow(&(&p - BigInt::from(2u32)), &p);
        assert_eq!((&a * &inv) % &p, BigInt::one());
        let mut asg = InvCircuit::<GF2>::default();
        asg.a.copy_from_slice(&to_gf2_bits(&a, BITS));
        asg.e.copy_from_slice(&to_gf2_bits(&inv, BITS));
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "inv_mod_p wrong");
    }
}
