//! GATES for the rsema1d-backed Expander input PCS.
//!
//! GATE A (adapter-oracle): for several random polynomials and `num_vars`,
//! `Rsema1dPCS` commit -> open -> verify ACCEPTS, and the verified value (after
//! the field isomorphism) EQUALS Expander's ground-truth oracle at the SAME
//! challenge. The oracle is
//! `GF2ExtConfig::single_core_eval_circuit_vals_at_expander_challenge`, which is
//! byte-for-byte the evaluation `RawExpanderGKR::verify` checks against
//! (poly_commit/src/raw.rs line 206-207) — so accepting the oracle value proves
//! agreement with `RawExpanderGKR`.
//!
//! GATE B (tamper): a tampered commitment, opening, value, or point is REJECTED
//! at the Rust layer. (The Go layer's tamper rejection is covered by
//! `TestFullPointEvaluationRejectsTampering` in pkg/rsema1d/pcs_full_test.go.)

use arith::Field;
use gf2::GF2x8;
use gf2_128::GF2_128;
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, GF2ExtConfig, MPIConfig, Transcript,
};
use gkr_hashers::SHA256hasher;
use polynomials::{MultiLinearPoly, MultilinearExtension};
use rand::{rngs::StdRng, SeedableRng};
use rsema1d_pcs::{Rsema1dCommitment, Rsema1dOpening, Rsema1dPCS};
use serdes::ExpSerde;
use transcript::BytesHashTranscript;

type Ts = BytesHashTranscript<SHA256hasher>;

fn challenge(num_vars: usize, rng: &mut StdRng) -> ExpanderSingleVarChallenge<GF2ExtConfig> {
    let rz: Vec<GF2_128> = (0..num_vars).map(|_| GF2_128::random_unsafe(&mut *rng)).collect();
    let r_simd: Vec<GF2_128> = (0..3).map(|_| GF2_128::random_unsafe(&mut *rng)).collect();
    ExpanderSingleVarChallenge::new(rz, r_simd, vec![])
}

/// Oracle ground truth = the exact evaluation `RawExpanderGKR::verify` recomputes.
fn oracle(poly: &MultiLinearPoly<GF2x8>, x: &ExpanderSingleVarChallenge<GF2ExtConfig>) -> GF2_128 {
    GF2ExtConfig::single_core_eval_circuit_vals_at_expander_challenge(&poly.hypercube_basis(), x)
}

fn commit(poly: &MultiLinearPoly<GF2x8>, num_vars: usize) -> Rsema1dCommitment {
    let mpi = MPIConfig::default();
    let mut sp = ();
    Rsema1dPCS::commit(&num_vars, &mpi, &(), poly, &mut sp).expect("commit")
}

fn open(
    poly: &MultiLinearPoly<GF2x8>,
    num_vars: usize,
    x: &ExpanderSingleVarChallenge<GF2ExtConfig>,
) -> Rsema1dOpening {
    let mpi = MPIConfig::default();
    let mut ts = Ts::new();
    Rsema1dPCS::open(&num_vars, &mpi, &(), poly, x, &mut ts, &()).expect("open")
}

fn verify(
    num_vars: usize,
    c: &Rsema1dCommitment,
    x: &ExpanderSingleVarChallenge<GF2ExtConfig>,
    v: GF2_128,
    o: &Rsema1dOpening,
) -> bool {
    let mut ts = Ts::new();
    Rsema1dPCS::verify(&num_vars, &(), c, x, v, &mut ts, o)
}

#[test]
fn gate_a_adapter_oracle() {
    let mut rng = StdRng::seed_from_u64(0xACC1_DE47);
    for &num_vars in &[5usize, 6, 7] {
        for trial in 0..3 {
            let poly = MultiLinearPoly::<GF2x8>::random(num_vars, &mut rng);
            let x = challenge(num_vars, &mut rng);

            let c = commit(&poly, num_vars);
            let o = open(&poly, num_vars, &x);
            let v_oracle = oracle(&poly, &x);

            // (1) verify ACCEPTS the oracle value.
            assert!(
                verify(num_vars, &c, &x, v_oracle, &o),
                "num_vars={num_vars} trial={trial}: verify rejected the oracle value"
            );

            // (2) verify REJECTS a value that is not the oracle (guards against a
            // verifier that accepts everything).
            assert!(
                !verify(num_vars, &c, &x, v_oracle + GF2_128::ONE, &o),
                "num_vars={num_vars} trial={trial}: verify accepted a non-oracle value"
            );

            println!(
                "GATE A ok: num_vars={num_vars} K={} trial={trial} oracle=0x{:032x}",
                1u32 << (num_vars - 2),
                {
                    let mut b = [0u8; 16];
                    v_oracle.serialize_into(&mut b[..]).unwrap();
                    u128::from_le_bytes(b)
                }
            );
        }
    }
}

#[test]
fn gate_b_tamper() {
    let mut rng = StdRng::seed_from_u64(0x7A11_9E44);
    let num_vars = 6usize;
    let poly = MultiLinearPoly::<GF2x8>::random(num_vars, &mut rng);
    let x = challenge(num_vars, &mut rng);

    let c = commit(&poly, num_vars);
    let o = open(&poly, num_vars, &x);
    let v = oracle(&poly, &x);

    // Honest proof verifies.
    assert!(verify(num_vars, &c, &x, v, &o), "honest proof must verify");

    // (a) tampered commitment.
    let mut c_bad = c.clone();
    c_bad.root[0] ^= 0x01;
    assert!(!verify(num_vars, &c_bad, &x, v, &o), "tampered commitment must be rejected");

    // (b) tampered opening bytes.
    let mut o_bad = o.clone();
    o_bad.proof[0] ^= 0x01;
    assert!(!verify(num_vars, &c, &x, v, &o_bad), "tampered opening must be rejected");

    // (c) tampered value.
    assert!(!verify(num_vars, &c, &x, v + GF2_128::ONE, &o), "tampered value must be rejected");

    // (d) tampered point (evaluation challenge).
    let mut x_bad = x.clone();
    x_bad.rz[0] += GF2_128::ONE;
    assert!(!verify(num_vars, &c, &x_bad, v, &o), "tampered point must be rejected");

    // (e) tampered SIMD coordinate of the point.
    let mut x_bad2 = x.clone();
    x_bad2.r_simd[0] += GF2_128::ONE;
    assert!(!verify(num_vars, &c, &x_bad2, v, &o), "tampered simd point must be rejected");

    println!("GATE B ok: commitment/opening/value/point/simd tampering all rejected");
}
