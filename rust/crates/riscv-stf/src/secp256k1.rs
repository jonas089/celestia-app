//! R5 — secp256k1 + ECDSA `ecrecover`.
//!
//! Two layers:
//!  * `native` — a pure `num-bigint` reference implementation (affine point ops,
//!    deterministic recover). Fast; it is the golden the in-circuit version is
//!    differential-tested against (and matches ev-reth's `recover_sender`).
//!  * (circuit layer, built next) — the same recovery expressed over GF2 with the
//!    hint-free `u256` modular arithmetic, Jacobian point ops (one inverse per
//!    scalar-mul), reused data-parallel across the block's txs.
//!
//! Recovery follows the deterministic path required by the advice rule (no
//! witnessed pubkey): x = r; y = sqrt(x^3+7) with parity from `v`; then
//! Q = r^{-1}(s·R − z·G); address = keccak256(Q.x‖Q.y)[12:].

use num_bigint::BigInt;
use num_traits::{One, Zero};

/// secp256k1 field prime, group order, and generator.
pub fn p() -> BigInt { crate::u256::p_bigint() }
pub fn n() -> BigInt { crate::u256::n_bigint() }
pub fn gx() -> BigInt {
    BigInt::parse_bytes(b"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16).unwrap()
}
pub fn gy() -> BigInt {
    BigInt::parse_bytes(b"483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16).unwrap()
}

fn modp(x: BigInt) -> BigInt { let p = p(); ((x % &p) + &p) % &p }
fn modn(x: BigInt) -> BigInt { let n = n(); ((x % &n) + &n) % &n }
fn inv_modp(a: &BigInt) -> BigInt { a.modpow(&(p() - BigInt::from(2u32)), &p()) }
fn inv_modn(a: &BigInt) -> BigInt { a.modpow(&(n() - BigInt::from(2u32)), &n()) }

pub mod native {
    use super::*;

    /// Affine point; `None` == point at infinity.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Point(pub Option<(BigInt, BigInt)>);

    pub fn infinity() -> Point { Point(None) }
    pub fn generator() -> Point { Point(Some((gx(), gy()))) }

    pub fn add(a: &Point, b: &Point) -> Point {
        match (&a.0, &b.0) {
            (None, _) => b.clone(),
            (_, None) => a.clone(),
            (Some((x1, y1)), Some((x2, y2))) => {
                if x1 == x2 && modp(y1 + y2).is_zero() {
                    return infinity();
                }
                let lam = if x1 == x2 && y1 == y2 {
                    // doubling: (3 x1^2) / (2 y1)
                    let num = modp(BigInt::from(3u32) * x1 * x1);
                    let den = inv_modp(&modp(BigInt::from(2u32) * y1));
                    modp(num * den)
                } else {
                    let num = modp(y2 - y1);
                    let den = inv_modp(&modp(x2 - x1));
                    modp(num * den)
                };
                let x3 = modp(&lam * &lam - x1 - x2);
                let y3 = modp(&lam * (x1 - &x3) - y1);
                Point(Some((x3, y3)))
            }
        }
    }

    pub fn scalar_mul(k: &BigInt, p0: &Point) -> Point {
        let mut acc = infinity();
        let mut base = p0.clone();
        let mut kk = k.clone();
        while !kk.is_zero() {
            if (&kk & BigInt::one()) == BigInt::one() {
                acc = add(&acc, &base);
            }
            base = add(&base, &base);
            kk >>= 1;
        }
        acc
    }

    /// Deterministic ECDSA public-key recovery. `z` is the 256-bit message hash,
    /// `v` in {0,1} is the parity of R.y. Returns the 64-byte pubkey (x‖y) or None.
    pub fn recover(r: &BigInt, s: &BigInt, v: u8, z: &BigInt) -> Option<[u8; 64]> {
        if r.is_zero() || s.is_zero() || r >= &n() || s >= &n() {
            return None;
        }
        // R.x = r ; y^2 = x^3 + 7 ; y = (x^3+7)^((p+1)/4)
        let x = r.clone();
        let alpha = modp(&x * &x * &x + BigInt::from(7u32));
        let beta = alpha.modpow(&((p() + BigInt::one()) / BigInt::from(4u32)), &p());
        // pick parity
        let y = if (&beta & BigInt::one()) == BigInt::from(v & 1) { beta.clone() } else { modp(-beta) };
        // verify on curve
        if modp(&y * &y) != alpha {
            return None;
        }
        let rpoint = Point(Some((x, y)));
        let r_inv = inv_modn(r);
        let u1 = modn(-(z.clone()) * &r_inv);
        let u2 = modn(s * &r_inv);
        let q = add(&scalar_mul(&u1, &generator()), &scalar_mul(&u2, &rpoint));
        match q.0 {
            None => None,
            Some((qx, qy)) => {
                let mut out = [0u8; 64];
                let xb = qx.to_bytes_be().1;
                let yb = qy.to_bytes_be().1;
                out[32 - xb.len()..32].copy_from_slice(&xb);
                out[64 - yb.len()..64].copy_from_slice(&yb);
                Some(out)
            }
        }
    }
}

/// In-circuit secp256k1 over GF2, on the hint-free `u256` modular arithmetic.
/// Jacobian coordinates (X : Y : Z) => affine (X/Z², Y/Z³); infinity iff Z == 0.
/// One modular inverse per scalar-mul (at the final affine conversion), never
/// per step. All deterministic — no advice (advice rule).
pub mod circuit {
    use crate::u256::*;
    use expander_compiler::frontend::*;

    #[derive(Clone)]
    pub struct JacPoint {
        pub x: Vec<Variable>,
        pub y: Vec<Variable>,
        pub z: Vec<Variable>,
    }

    fn dbl_word<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable]) -> Vec<Variable> {
        add_mod_p(api, a, a)
    }
    fn mul_small<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], k: u32) -> Vec<Variable> {
        // a*k by add chains (k small: 2,3,4,8).
        let mut acc = a.to_vec();
        for _ in 1..k {
            acc = add_mod_p(api, &acc, a);
        }
        acc
    }

    /// Jacobian point doubling for a=0 (secp256k1). Correct for Z==0 (stays 0).
    pub fn jdouble<C: Config>(api: &mut impl RootAPI<C>, p: &JacPoint) -> JacPoint {
        let a = mul_mod_p(api, &p.x, &p.x); // A = X^2
        let b = mul_mod_p(api, &p.y, &p.y); // B = Y^2
        let c = mul_mod_p(api, &b, &b); // C = B^2
        // D = 2*((X+B)^2 - A - C)
        let xb = add_mod_p(api, &p.x, &b);
        let xb2 = mul_mod_p(api, &xb, &xb);
        let t = sub_mod_p(api, &xb2, &a);
        let t = sub_mod_p(api, &t, &c);
        let d = dbl_word(api, &t);
        let e = mul_small(api, &a, 3); // E = 3A
        let f = mul_mod_p(api, &e, &e); // F = E^2
        let two_d = dbl_word(api, &d);
        let x3 = sub_mod_p(api, &f, &two_d); // X3 = F - 2D
        let d_x3 = sub_mod_p(api, &d, &x3);
        let e_dx3 = mul_mod_p(api, &e, &d_x3);
        let eight_c = mul_small(api, &c, 8);
        let y3 = sub_mod_p(api, &e_dx3, &eight_c); // Y3 = E(D-X3) - 8C
        let yz = mul_mod_p(api, &p.y, &p.z);
        let z3 = dbl_word(api, &yz); // Z3 = 2 Y Z
        JacPoint { x: x3, y: y3, z: z3 }
    }

    /// Jacobian + affine (x2,y2) mixed addition, with the three exception cases
    /// (P=∞, P=Q, P=-Q) resolved by select. `base` is a non-infinity affine point.
    pub fn jadd_mixed<C: Config>(api: &mut impl RootAPI<C>, p: &JacPoint, x2: &[Variable], y2: &[Variable]) -> JacPoint {
        let z1z1 = mul_mod_p(api, &p.z, &p.z);
        let u2 = mul_mod_p(api, x2, &z1z1); // U2 = x2 Z1^2
        let z1z1z1 = mul_mod_p(api, &z1z1, &p.z);
        let s2 = mul_mod_p(api, y2, &z1z1z1); // S2 = y2 Z1^3
        let h = sub_mod_p(api, &u2, &p.x); // H = U2 - X1
        let rr = sub_mod_p(api, &s2, &p.y); // r0 = S2 - Y1
        let hh = mul_mod_p(api, &h, &h);
        let i = mul_small(api, &hh, 4); // I = 4 HH
        let j = mul_mod_p(api, &h, &i); // J = H I
        let r = dbl_word(api, &rr); // r = 2 r0
        let v = mul_mod_p(api, &p.x, &i); // V = X1 I
        let r2 = mul_mod_p(api, &r, &r);
        let two_v = dbl_word(api, &v);
        let x3 = sub_mod_p(api, &r2, &j);
        let x3 = sub_mod_p(api, &x3, &two_v); // X3 = r^2 - J - 2V
        let v_x3 = sub_mod_p(api, &v, &x3);
        let r_vx3 = mul_mod_p(api, &r, &v_x3);
        let y1j = mul_mod_p(api, &p.y, &j);
        let two_y1j = dbl_word(api, &y1j);
        let y3 = sub_mod_p(api, &r_vx3, &two_y1j); // Y3 = r(V-X3) - 2 Y1 J
        let z1h = mul_mod_p(api, &p.z, &h);
        let z3 = dbl_word(api, &z1h); // Z3 = 2 Z1 H  ( = (Z1+H)^2 - Z1Z1 - HH )
        let normal = JacPoint { x: x3, y: y3, z: z3 };

        // Exception handling. In a valid double-and-add scalar multiplication the
        // accumulator is k*base for 0 < k < scalar < n, and every add follows a
        // double, so acc != base and acc != -base — the P==Q and P==-Q cases
        // NEVER arise (they would need k ≡ ±1 mod n mid-loop). The only real
        // exception is acc == infinity before the first set bit. So we keep just
        // the Z==0 -> base select and drop the (expensive, per-add) jdouble that
        // handled P==Q. This is sound for scalar_mul; callers that add arbitrary
        // points (where P==Q is possible) must use `jadd_jac`, which keeps the
        // full exception handling.
        let z_is_zero = is_zero(api, &p.z);
        let one = { let mut v = vec![api.constant(0); BITS]; v[0] = api.constant(1); v };
        let base = JacPoint { x: x2.to_vec(), y: y2.to_vec(), z: one };
        select_point(api, z_is_zero, &base, &normal)
    }

    fn select_point<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: &JacPoint, b: &JacPoint) -> JacPoint {
        JacPoint {
            x: select(api, sel, &a.x, &b.x),
            y: select(api, sel, &a.y, &b.y),
            z: select(api, sel, &a.z, &b.z),
        }
    }

    /// Precomputed fixed-base window table for G: table[wi][j] = j * 2^(4*wi) * G
    /// (affine (x,y) big-endian bytes), wi in 0..64, j in 0..16 (j=0 = infinity,
    /// unused). Constants — computed natively at build time.
    pub fn fixed_base_table_g() -> Vec<Vec<([u8; 32], [u8; 32])>> {
        use crate::secp256k1::native;
        use num_bigint::BigInt;
        let mut table = Vec::with_capacity(64);
        for wi in 0..64usize {
            let mut row = Vec::with_capacity(16);
            for j in 0..16u32 {
                if j == 0 {
                    row.push(([0u8; 32], [0u8; 32]));
                    continue;
                }
                let scalar = BigInt::from(j) << (4 * wi);
                let p = native::scalar_mul(&scalar, &native::generator()).0.unwrap();
                let mut xb = [0u8; 32];
                let mut yb = [0u8; 32];
                let x = p.0.to_bytes_be().1;
                let y = p.1.to_bytes_be().1;
                xb[32 - x.len()..].copy_from_slice(&x);
                yb[32 - y.len()..].copy_from_slice(&y);
                row.push((xb, yb));
            }
            table.push(row);
        }
        table
    }

    /// Select the affine point table_wi[j] (j = value of the 4 window bits,
    /// LSB-first) into (x,y) u256 LE bit vectors; also returns is_zero (j==0).
    /// Table entries are constants, so this costs no field muls.
    fn select_window_point<C: Config>(api: &mut impl RootAPI<C>, jbits: &[Variable], row: &[([u8; 32], [u8; 32])]) -> (Vec<Variable>, Vec<Variable>, Variable) {
        // onehot[j] = (jbits == j).
        let mut onehot = Vec::with_capacity(16);
        for j in 0..16u32 {
            let mut acc = api.constant(1);
            for b in 0..4 {
                let want = (j >> b) & 1;
                let t = if want == 1 { jbits[b] } else { api.sub(1, jbits[b]) };
                acc = api.mul(acc, t);
            }
            onehot.push(acc);
        }
        // x[i] = XOR over j in 1..16 of onehot[j] where table byte-bit is 1.
        let zero = api.constant(0);
        let mut xo = vec![zero; BITS];
        let mut yo = vec![zero; BITS];
        for j in 1..16usize {
            let (xb, yb) = &row[j];
            for i in 0..BITS {
                // big-endian byte (31 - i/8), bit (i%8) -> u256 LE bit i.
                let byte = 31 - i / 8;
                let bit = (i % 8) as u32;
                if (xb[byte] >> bit) & 1 == 1 { xo[i] = api.add(xo[i], onehot[j]); }
                if (yb[byte] >> bit) & 1 == 1 { yo[i] = api.add(yo[i], onehot[j]); }
            }
        }
        (xo, yo, onehot[0])
    }

    /// Fixed-base windowed scalar multiplication: u*G via 64 4-bit windows over
    /// the precomputed constant table — no in-circuit doublings, 64 mixed-adds.
    /// `k_bits` LSB-first (256). Much cheaper than double-and-add.
    pub fn scalar_mul_fixed_base<C: Config>(api: &mut impl RootAPI<C>, k_bits: &[Variable], table: &[Vec<([u8; 32], [u8; 32])>]) -> JacPoint {
        let mut acc = JacPoint {
            x: vec![api.constant(0); BITS],
            y: vec![api.constant(0); BITS],
            z: vec![api.constant(0); BITS], // infinity
        };
        for wi in 0..64usize {
            let jbits = &k_bits[4 * wi..4 * wi + 4];
            let (px, py, jzero) = select_window_point(api, jbits, &table[wi]);
            let added = jadd_mixed(api, &acc, &px, &py);
            // j==0 => add nothing.
            acc = select_point(api, jzero, &acc, &added);
        }
        acc
    }

    /// scalar_mul(k, base_affine): double-and-add, MSB first. `k_bits` LSB-first.
    pub fn scalar_mul<C: Config>(api: &mut impl RootAPI<C>, k_bits: &[Variable], x2: &[Variable], y2: &[Variable]) -> JacPoint {
        let mut acc = JacPoint {
            x: vec![api.constant(0); BITS],
            y: vec![api.constant(0); BITS],
            z: vec![api.constant(0); BITS], // infinity
        };
        for i in (0..k_bits.len()).rev() {
            acc = jdouble(api, &acc);
            let added = jadd_mixed(api, &acc, x2, y2);
            acc = select_point(api, k_bits[i], &added, &acc);
        }
        acc
    }

    /// Full Jacobian + Jacobian addition (add-2007-bl) with exceptions resolved
    /// by select (P1=∞ -> P2; P2=∞ -> P1; H==0 -> double or ∞).
    pub fn jadd_jac<C: Config>(api: &mut impl RootAPI<C>, p1: &JacPoint, p2: &JacPoint) -> JacPoint {
        let z1z1 = mul_mod_p(api, &p1.z, &p1.z);
        let z2z2 = mul_mod_p(api, &p2.z, &p2.z);
        let u1 = mul_mod_p(api, &p1.x, &z2z2);
        let u2 = mul_mod_p(api, &p2.x, &z1z1);
        let z2c = mul_mod_p(api, &z2z2, &p2.z);
        let s1 = mul_mod_p(api, &p1.y, &z2c);
        let z1c = mul_mod_p(api, &z1z1, &p1.z);
        let s2 = mul_mod_p(api, &p2.y, &z1c);
        let h = sub_mod_p(api, &u2, &u1);
        let rr = sub_mod_p(api, &s2, &s1);
        let r = dbl_word(api, &rr);
        let two_h = dbl_word(api, &h);
        let i = mul_mod_p(api, &two_h, &two_h); // (2H)^2
        let j = mul_mod_p(api, &h, &i);
        let v = mul_mod_p(api, &u1, &i);
        let r2 = mul_mod_p(api, &r, &r);
        let two_v = dbl_word(api, &v);
        let x3 = sub_mod_p(api, &r2, &j);
        let x3 = sub_mod_p(api, &x3, &two_v);
        let v_x3 = sub_mod_p(api, &v, &x3);
        let r_vx3 = mul_mod_p(api, &r, &v_x3);
        let s1j = mul_mod_p(api, &s1, &j);
        let two_s1j = dbl_word(api, &s1j);
        let y3 = sub_mod_p(api, &r_vx3, &two_s1j);
        // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * H
        let z1z2 = add_mod_p(api, &p1.z, &p2.z);
        let z1z2sq = mul_mod_p(api, &z1z2, &z1z2);
        let t = sub_mod_p(api, &z1z2sq, &z1z1);
        let t = sub_mod_p(api, &t, &z2z2);
        let z3 = mul_mod_p(api, &t, &h);
        let normal = JacPoint { x: x3, y: y3, z: z3 };

        let inf = JacPoint { x: vec![api.constant(0); BITS], y: vec![api.constant(0); BITS], z: vec![api.constant(0); BITS] };
        let dbl1 = jdouble(api, p1);
        let h_zero = is_zero(api, &h);
        let r_zero = is_zero(api, &r);
        let hz = select_point(api, r_zero, &dbl1, &inf);
        let h_case = select_point(api, h_zero, &hz, &normal);
        let z1_zero = is_zero(api, &p1.z);
        let z2_zero = is_zero(api, &p2.z);
        let pick = select_point(api, z1_zero, p2, &h_case); // P1=∞ -> P2
        select_point(api, z2_zero, p1, &pick) // P2=∞ -> P1
    }

    /// Deterministic in-circuit ECDSA public-key recovery. Inputs (all 256-bit LE
    /// bit vectors): r, s (< n), the parity bit `v`, and z (msg hash mod n).
    /// Returns the recovered affine pubkey (qx, qy). No advice.
    pub fn ecrecover_pubkey<C: Config>(
        api: &mut impl RootAPI<C>,
        r: &[Variable],
        s: &[Variable],
        v: Variable,
        z: &[Variable],
    ) -> (Vec<Variable>, Vec<Variable>) {
        // alpha = r^3 + 7
        let r2 = mul_mod_p(api, r, r);
        let r3 = mul_mod_p(api, &r2, r);
        let seven = { let mut c = vec![api.constant(0); BITS]; c[0] = api.constant(1); c[1] = api.constant(1); c[2] = api.constant(1); c };
        let alpha = add_mod_p(api, &r3, &seven);
        // beta = sqrt(alpha); y = parity(beta)==v ? beta : p-beta
        let beta = sqrt_mod_p(api, &alpha);
        let zero = vec![api.constant(0); BITS];
        let neg_beta = sub_mod_p(api, &zero, &beta);
        let parity = beta[0];
        let match_v = { let x = api.add(parity, v); api.sub(1, x) }; // parity==v
        let y = select(api, match_v, &beta, &neg_beta);
        // u1 = (-z) * r^{-1} mod n ; u2 = s * r^{-1} mod n
        let rinv = inv_mod_n(api, r);
        let neg_z = sub_mod_n(api, &zero, z);
        let u1 = mul_mod_n(api, &neg_z, &rinv);
        let u2 = mul_mod_n(api, s, &rinv);
        // Q = u1*G + u2*R. u1*G via windowed fixed-base (constant table, cheap);
        // u2*R via double-and-add (variable base).
        let table = fixed_base_table_g();
        let j1 = scalar_mul_fixed_base(api, &u1, &table);
        let j2 = scalar_mul(api, &u2, r, &y); // R = (r, y)
        let jsum = jadd_jac(api, &j1, &j2);
        to_affine(api, &jsum)
    }

    /// Convert Jacobian to affine (x = X/Z^2, y = Y/Z^3). One inverse.
    pub fn to_affine<C: Config>(api: &mut impl RootAPI<C>, p: &JacPoint) -> (Vec<Variable>, Vec<Variable>) {
        let zinv = inv_mod_p(api, &p.z);
        let zinv2 = mul_mod_p(api, &zinv, &zinv);
        let zinv3 = mul_mod_p(api, &zinv2, &zinv);
        let x = mul_mod_p(api, &p.x, &zinv2);
        let y = mul_mod_p(api, &p.y, &zinv3);
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiny_keccak::Hasher;

    fn keccak(b: &[u8]) -> [u8; 32] {
        let mut h = tiny_keccak::Keccak::v256();
        h.update(b);
        let mut o = [0u8; 32];
        h.finalize(&mut o);
        o
    }

    use crate::u256::{bigint_to_bits, mul_mod_p, BITS};
    use expander_compiler::frontend::*;

    fn gf2_bits(x: &BigInt, len: usize) -> Vec<GF2> {
        bigint_to_bits(x, len).into_iter().map(|b| (b as u32).into()).collect()
    }

    // Validate the in-circuit scalar_mul (jdouble/jadd_mixed) against the native
    // reference PROJECTIVELY: assert x_native*Z^2 == X and y_native*Z^3 == Y, so
    // no expensive modular inverse is needed for the check.
    const KBITS: usize = 3;
    declare_circuit!(SmulCircuit {
        kbits: [Variable; KBITS],
        xn: [PublicVariable; BITS],
        yn: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for SmulCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let gx_bits: Vec<Variable> = bigint_to_bits(&gx(), BITS).into_iter().map(|b| api.constant(b as u32)).collect();
            let gy_bits: Vec<Variable> = bigint_to_bits(&gy(), BITS).into_iter().map(|b| api.constant(b as u32)).collect();
            let j = super::circuit::scalar_mul(api, &self.kbits.to_vec(), &gx_bits, &gy_bits);
            let z2 = mul_mod_p(api, &j.z, &j.z);
            let z3 = mul_mod_p(api, &z2, &j.z);
            let xz2 = mul_mod_p(api, &self.xn.to_vec(), &z2);
            let yz3 = mul_mod_p(api, &self.yn.to_vec(), &z3);
            for i in 0..BITS {
                api.assert_is_equal(xz2[i], j.x[i]);
                api.assert_is_equal(yz3[i], j.y[i]);
            }
        }
    }

    // Validate jadd_jac: G + 2G == 3G, projectively vs native.
    declare_circuit!(JaddCircuit {
        dummy: Variable, // ECC requires >=1 committed input; unused in the result
        xn: [PublicVariable; BITS],
        yn: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for JaddCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            api.assert_is_bool(self.dummy);
            let gx_bits: Vec<Variable> = bigint_to_bits(&gx(), BITS).into_iter().map(|b| api.constant(b as u32)).collect();
            let gy_bits: Vec<Variable> = bigint_to_bits(&gy(), BITS).into_iter().map(|b| api.constant(b as u32)).collect();
            let one = { let mut v = vec![api.constant(0); BITS]; v[0] = api.constant(1); v };
            let g = super::circuit::JacPoint { x: gx_bits, y: gy_bits, z: one };
            let g2 = super::circuit::jdouble(api, &g);
            let g3 = super::circuit::jadd_jac(api, &g, &g2);
            let z2 = mul_mod_p(api, &g3.z, &g3.z);
            let z3 = mul_mod_p(api, &z2, &g3.z);
            let xz2 = mul_mod_p(api, &self.xn.to_vec(), &z2);
            let yz3 = mul_mod_p(api, &self.yn.to_vec(), &z3);
            for i in 0..BITS {
                api.assert_is_equal(xz2[i], g3.x[i]);
                api.assert_is_equal(yz3[i], g3.y[i]);
            }
        }
    }

    #[test]
    fn circuit_jadd_jac_matches_native() {
        let g3 = native::scalar_mul(&BigInt::from(3u32), &native::generator());
        let (x, y) = g3.0.clone().unwrap();
        let CompileResult { witness_solver, layered_circuit } =
            compile(&JaddCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = JaddCircuit::<GF2>::default();
        asg.xn.copy_from_slice(&gf2_bits(&x, BITS));
        asg.yn.copy_from_slice(&gf2_bits(&y, BITS));
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "circuit G+2G != 3G");
    }

    // Full end-to-end in-circuit ecrecover vs native (HEAVY: ~hours). #[ignore].
    declare_circuit!(EcrecoverCircuit {
        r: [Variable; BITS],
        s: [Variable; BITS],
        v: Variable,
        z: [Variable; BITS],
        qx: [PublicVariable; BITS],
        qy: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for EcrecoverCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let (qx, qy) = super::circuit::ecrecover_pubkey(api, &self.r.to_vec(), &self.s.to_vec(), self.v, &self.z.to_vec());
            for i in 0..BITS {
                api.assert_is_equal(qx[i], self.qx[i]);
                api.assert_is_equal(qy[i], self.qy[i]);
            }
        }
    }

    #[test]
    #[ignore = "full in-circuit ecrecover: ~8000 modmuls, hours to compile+solve; run on demand"]
    fn circuit_ecrecover_matches_native() {
        // Same keypair/signature as native_recover_roundtrip.
        let d = BigInt::parse_bytes(b"c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721", 16).unwrap();
        let q = native::scalar_mul(&d, &native::generator());
        let (qx, qy) = q.0.clone().unwrap();
        let z = BigInt::from_bytes_be(num_bigint::Sign::Plus, &keccak(b"accidental computer"));
        let zmod = super::modn(z);
        let k = BigInt::parse_bytes(b"49a0d7b786ec9cde0d0721d72804befd06571c974b191efb42ecf322ba9ddd9a", 16).unwrap();
        let rpt = native::scalar_mul(&k, &native::generator());
        let (rx, ry) = rpt.0.clone().unwrap();
        let r = super::modn(rx);
        let k_inv = super::inv_modn(&k);
        let s = super::modn(&k_inv * (&zmod + &r * &d));
        let v: u8 = (&ry & BigInt::from(1u32)).try_into().unwrap_or(0);

        let CompileResult { witness_solver, layered_circuit } =
            compile(&EcrecoverCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = EcrecoverCircuit::<GF2>::default();
        asg.r.copy_from_slice(&gf2_bits(&r, BITS));
        asg.s.copy_from_slice(&gf2_bits(&s, BITS));
        asg.v = (v as u32).into();
        asg.z.copy_from_slice(&gf2_bits(&zmod, BITS));
        asg.qx.copy_from_slice(&gf2_bits(&qx, BITS));
        asg.qy.copy_from_slice(&gf2_bits(&qy, BITS));
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit ecrecover != native pubkey");
    }

    #[test]
    fn circuit_scalar_mul_matches_native_small() {
        let k = 7u32;
        let kb = BigInt::from(k);
        let q = native::scalar_mul(&kb, &native::generator());
        let (qx, qy) = q.0.clone().unwrap();
        let CompileResult { witness_solver, layered_circuit } =
            compile(&SmulCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = SmulCircuit::<GF2>::default();
        for i in 0..KBITS {
            asg.kbits[i] = (((k >> i) & 1) as u32).into();
        }
        asg.xn.copy_from_slice(&gf2_bits(&qx, BITS));
        asg.yn.copy_from_slice(&gf2_bits(&qy, BITS));
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "circuit {k}*G != native");
    }

    #[test]
    fn native_scalar_mul_matches_generator_small() {
        // 2G computed two ways.
        let g = native::generator();
        let g2a = native::add(&g, &g);
        let g2b = native::scalar_mul(&BigInt::from(2u32), &g);
        assert_eq!(g2a, g2b);
        // (n)G == infinity
        assert_eq!(native::scalar_mul(&n(), &g), native::infinity());
    }

    #[test]
    fn native_recover_roundtrip() {
        // Deterministic keypair: priv d, pub Q = d*G. Sign a message, recover.
        let d = BigInt::parse_bytes(b"c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721", 16).unwrap();
        let q = native::scalar_mul(&d, &native::generator());
        let (qx, qy) = q.0.clone().unwrap();

        // message hash z
        let z = BigInt::from_bytes_be(num_bigint::Sign::Plus, &keccak(b"accidental computer"));
        let zmod = modn(z.clone());

        // Deterministic nonce k (test only), R = k*G, r = R.x mod n.
        let k = BigInt::parse_bytes(b"49a0d7b786ec9cde0d0721d72804befd06571c974b191efb42ecf322ba9ddd9a", 16).unwrap();
        let rpt = native::scalar_mul(&k, &native::generator());
        let (rx, ry) = rpt.0.clone().unwrap();
        let r = modn(rx.clone());
        let k_inv = inv_modn(&k);
        let s = modn(&k_inv * (&zmod + &r * &d));
        // recovery id: parity of R.y (assuming rx < n so no reduction)
        let v = (&ry & BigInt::one()).try_into().unwrap_or(0u8);

        let rec = native::recover(&r, &s, v, &zmod).expect("recover");
        let mut want = [0u8; 64];
        let xb = qx.to_bytes_be().1;
        let yb = qy.to_bytes_be().1;
        want[32 - xb.len()..32].copy_from_slice(&xb);
        want[64 - yb.len()..64].copy_from_slice(&yb);
        assert_eq!(rec, want, "recovered pubkey != d*G");

        // Ethereum address = keccak(pubkey)[12:].
        let addr = &keccak(&rec)[12..];
        println!("native ecrecover address = 0x{}", addr.iter().map(|b| format!("{:02x}", b)).collect::<String>());
    }
}
