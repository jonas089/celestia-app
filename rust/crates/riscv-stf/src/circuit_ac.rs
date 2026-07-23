//! Input-committed STF **executor** circuit — the CORRECT accidental-computer
//! shape.
//!
//! Unlike `circuit_stf` / `circuit_full` (which commit the execution *trace* as
//! the GKR input layer), this circuit commits ONLY the block's **data**
//! (pre-state + transactions). The state transition is computed FORWARD, in
//! circuit, as ordinary internal wires — ECC lowers it to layered GKR, so every
//! wire except the committed input layer is proven by sumcheck with NO
//! commitment. This is exactly the paper's win: the sole polynomial commitment
//! is the DA (rsema1d) commitment of the block data, reused as the GKR input
//! commitment and opened at the sumcheck point.
//!
//! The "program" (the transfer-STF semantics) is the fixed circuit structure
//! itself — public/constant, not committed. The machine threads state
//! (sender balance/nonce, recipient balance, applied count) across the MAXTX tx
//! steps as internal wires. Output = the post-state, exposed as PUBLIC outputs.
//!
//! Committed input layout (little-endian u32 words, bit-decomposed LSB-first):
//!   word 0 : pre sender balance
//!   word 1 : pre sender nonce
//!   word 2 : pre recipient balance
//!   word 3 + 3*i + {0,1,2} : tx[i].{value, fee, nonce}   (i in 0..MAXTX)
//! => NWORDS = 3 + 3*MAXTX words. These bits ARE the committed poly / DA rows.
//!
//! HONEST SCOPE (same rung as `circuit_stf`, but the right commitment shape):
//! this proves the nonce + balance transfer transition over the committed block
//! data. ECDSA sender recovery and a keccak/MPT state root are the rungs above
//! (built on this same input-committed foundation). Balances/values < 2^31.

use expander_compiler::frontend::*;

pub const XLEN: usize = 32;
/// Max transactions per block (unused slots carry NULL_NONCE => rejected).
pub const MAXTX: usize = 4;
/// Committed data words: 3 pre-state + 3 per tx.
pub const NWORDS: usize = 3 + 3 * MAXTX;
/// Public outputs: post sbal, post snon, post rbal, applied count, digest.
pub const NOUT: usize = 5;
/// Guaranteed-invalid nonce for padding tx slots.
pub const NULL_NONCE: u32 = 0xFFFF_FFFF;

declare_circuit!(StfAcCircuit {
    // Committed GKR input layer = the block's data words (bit-decomposed).
    data: [[Variable; XLEN]; NWORDS],
    // Public post-state outputs (NOT part of the committed input layer).
    out: [[PublicVariable; XLEN]; NOUT],
});

fn set_word(dst: &mut [GF2], word: u32) {
    for i in 0..XLEN {
        dst[i] = ((word >> i) & 1).into();
    }
}

/// Build the witness assignment from the committed data words + public outputs.
pub fn build_assignment(words: &[u32; NWORDS], out_vals: &[u32; NOUT]) -> StfAcCircuit<GF2> {
    let mut a = StfAcCircuit::<GF2>::default();
    for i in 0..NWORDS {
        set_word(&mut a.data[i], words[i]);
    }
    for k in 0..NOUT {
        set_word(&mut a.out[k], out_vals[k]);
    }
    a
}

// ------------------------------- gadgets -----------------------------------
// Bit-level over GF2: api.add == XOR, api.mul == AND.

fn not1<C: Config>(api: &mut impl RootAPI<C>, x: Variable) -> Variable {
    api.sub(1, x)
}
fn mux1<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: Variable, b: Variable) -> Variable {
    // sel ? a : b == b ^ (sel & (a ^ b))
    let d = api.add(a, b);
    let t = api.mul(sel, d);
    api.add(b, t)
}
fn mux32<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| mux1(api, sel, a[i], b[i])).collect()
}
fn xor32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| api.add(a[i], b[i])).collect()
}
fn add32_cin<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable], cin: Variable) -> Vec<Variable> {
    let mut sum = vec![api.constant(0); XLEN];
    let mut carry = cin;
    for i in 0..XLEN {
        let ab = api.add(a[i], b[i]);
        sum[i] = api.add(ab, carry);
        let and_ab = api.mul(a[i], b[i]);
        let c_and = api.mul(carry, ab);
        carry = api.add(and_ab, c_and);
    }
    sum
}
fn add32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    add32_cin(api, a, b, zero)
}
/// a - b = a + (~b) + 1 (mod 2^32).
fn sub32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let nb: Vec<Variable> = b.iter().map(|&x| not1(api, x)).collect();
    let one = api.constant(1);
    add32_cin(api, a, &nb, one)
}
/// 1 iff the two 32-bit words are equal.
fn eq32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Variable {
    let mut acc = api.constant(1);
    for i in 0..XLEN {
        let x = api.add(a[i], b[i]); // 0 iff equal bit
        let eqb = not1(api, x);
        acc = api.mul(acc, eqb);
    }
    acc
}
fn const_word<C: Config>(api: &mut impl RootAPI<C>, word: u32) -> Vec<Variable> {
    (0..XLEN).map(|i| api.constant((word >> i) & 1)).collect()
}
/// Logical left shift by a COMPILE-TIME constant `k` (pure wire reindex).
fn shl_const(word: &[Variable], k: usize, zero: Variable) -> Vec<Variable> {
    (0..XLEN).map(|i| if i >= k { word[i - k] } else { zero }).collect()
}

// ------------------------------ the circuit --------------------------------

impl Define<GF2Config> for StfAcCircuit<Variable> {
    fn define<Builder: RootAPI<GF2Config>>(&self, api: &mut Builder) {
        let zero = api.constant(0);
        let one_word = const_word(api, 1);

        // Load pre-state from the committed data words.
        let mut sbal = self.data[0].to_vec();
        let mut snon = self.data[1].to_vec();
        let mut rbal = self.data[2].to_vec();
        let mut applied = vec![zero; XLEN];

        // Apply each tx IN ORDER (state threaded as internal wires).
        for i in 0..MAXTX {
            let value = self.data[3 + 3 * i].to_vec();
            let fee = self.data[3 + 3 * i + 1].to_vec();
            let nonce = self.data[3 + 3 * i + 2].to_vec();

            let need = add32(api, &value, &fee); // value + fee
            let nonce_ok = eq32(api, &nonce, &snon); // replay protection
            let diff = sub32(api, &sbal, &need);
            let overspend = diff[31]; // sign bit (valid balances < 2^31)
            let not_over = not1(api, overspend);
            let valid = api.mul(nonce_ok, not_over);

            // Candidate post-state if this tx is applied.
            let new_sbal = sub32(api, &sbal, &need);
            let new_snon = add32(api, &snon, &one_word);
            let new_rbal = add32(api, &rbal, &value);

            sbal = mux32(api, valid, &new_sbal, &sbal);
            snon = mux32(api, valid, &new_snon, &snon);
            rbal = mux32(api, valid, &new_rbal, &rbal);

            // applied += valid  (valid is a single bit in {0,1}).
            let mut valid_word = vec![zero; XLEN];
            valid_word[0] = valid;
            applied = add32(api, &applied, &valid_word);
        }

        // Post-state digest: sbal ^ (snon<<8) ^ (rbal<<16).
        let snon8 = shl_const(&snon, 8, zero);
        let rbal16 = shl_const(&rbal, 16, zero);
        let d1 = xor32(api, &sbal, &snon8);
        let digest = xor32(api, &d1, &rbal16);

        // Assert the PUBLIC outputs equal the computed post-state.
        let outs = [&sbal, &snon, &rbal, &applied, &digest];
        for (k, w) in outs.iter().enumerate() {
            for b in 0..XLEN {
                api.assert_is_equal(w[b], self.out[k][b]);
            }
        }
    }
}
