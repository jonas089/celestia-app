//! In-circuit GF(2^128) arithmetic over the boolean ECC frontend (GF2 config:
//! `api.add` == XOR, `api.mul` == AND). A field element is 128 `Variable` bits,
//! LSB at index 0, interpreted in the polynomial basis with reduction polynomial
//!
//!     f(x) = x^128 + x^7 + x^2 + x + 1        (the GCM / AES-GCM GF(2^128) poly)
//!
//! Multiplication is a 128x128 carryless (XOR-of-ANDs) product giving a degree
//! <= 254 result, then reduced mod f by folding the high bits (bits 128..254)
//! down using x^128 == x^7 + x^2 + x + 1. Reduction is pure XOR (no AND gates),
//! processed from the highest bit downward so every fold targets a lower index.
//!
//! A `native` mirror computes the identical convention on `u128`s so the circuit
//! output can be asserted bit-for-bit against a host-side reference.

use expander_compiler::frontend::*;

pub const W: usize = 128;

// -------------------------------- native mirror ----------------------------

/// Native GF(2^128) multiply, EXACT same convention/poly as the circuit gadget.
/// Computed via a 255-bit carryless product + fold reduction on a bit array, so
/// there is zero chance of a convention mismatch with the in-circuit version.
pub fn native_mul(a: u128, b: u128) -> u128 {
    let mut prod = [0u8; 2 * W - 1]; // indices 0..=254
    for i in 0..W {
        if (a >> i) & 1 == 1 {
            for j in 0..W {
                if (b >> j) & 1 == 1 {
                    prod[i + j] ^= 1;
                }
            }
        }
    }
    // reduce: x^k (k>=128) == x^(k-121) + x^(k-126) + x^(k-127) + x^(k-128)
    for k in (W..(2 * W - 1)).rev() {
        if prod[k] == 1 {
            prod[k] = 0;
            prod[k - W] ^= 1; // + 1
            prod[k - W + 1] ^= 1; // + x
            prod[k - W + 2] ^= 1; // + x^2
            prod[k - W + 7] ^= 1; // + x^7
        }
    }
    let mut out = 0u128;
    for i in 0..W {
        if prod[i] == 1 {
            out |= 1u128 << i;
        }
    }
    out
}

/// Split a u128 into its 128 bits (LSB first).
pub fn to_bits(x: u128) -> [u8; W] {
    let mut b = [0u8; W];
    for i in 0..W {
        b[i] = ((x >> i) & 1) as u8;
    }
    b
}

// ----------------------------- in-circuit gadgets ---------------------------

/// Constant GF(2^128) element as 128 constant Variables.
pub fn const_gf<C: Config>(api: &mut impl RootAPI<C>, x: u128) -> Vec<Variable> {
    (0..W).map(|i| api.constant(((x >> i) & 1) as u32)).collect()
}

/// XOR (field addition) of two 128-bit elements.
pub fn add<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..W).map(|i| api.add(a[i], b[i])).collect()
}

/// Full GF(2^128) multiply of two 128-bit elements: 128x128 carryless product
/// (XOR of ANDs) then reduction by folding high bits with the GCM poly.
pub fn mul<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    mul_bounded(api, a, b, W)
}

/// Multiply where `a` is known to have at most `abits` significant bits (the rest
/// are constant zero). This skips the all-zero partial-product rows, cutting the
/// AND count from 128*128 to abits*128 -- used when a factor is a small addr/ts.
pub fn mul_bounded<C: Config>(
    api: &mut impl RootAPI<C>,
    a: &[Variable],
    b: &[Variable],
    abits: usize,
) -> Vec<Variable> {
    let zero = api.constant(0);
    // carryless product, degree <= (abits-1)+127
    let deg = abits + W - 1; // number of output positions 0..deg-1
    let mut prod = vec![zero; deg.max(W)];
    for i in 0..abits {
        for j in 0..W {
            let t = api.mul(a[i], b[j]);
            prod[i + j] = api.add(prod[i + j], t);
        }
    }
    // reduction: fold bits >= 128 down. Process high -> low so each fold lands
    // on a strictly lower index (and any still-high target is folded later).
    for k in (W..prod.len()).rev() {
        let hi = prod[k];
        prod[k - W] = api.add(prod[k - W], hi); // + 1
        prod[k - W + 1] = api.add(prod[k - W + 1], hi); // + x
        prod[k - W + 2] = api.add(prod[k - W + 2], hi); // + x^2
        prod[k - W + 7] = api.add(prod[k - W + 7], hi); // + x^7
    }
    prod[0..W].to_vec()
}
