//! Empirical feasibility probe for the in-circuit GF(2^128) arithmetic that the
//! grand-product memory-checking argument is built on. Builds a circuit that
//! computes a product tree of N committed GF(2^128) elements and asserts it
//! equals a committed expected product, then proves/verifies it over the real
//! Expander GKR + rsema1d spine and reports num_vars / prove / verify / RSS.
//!
//! Also directly checks the in-circuit gadget's convention against the native
//! `gf128::native_mul` mirror (the circuit asserts equality, so a passing proof
//! IS the cross-check).

mod gf128;

use arith::Field;
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{MPIConfig, MPIEngine};
use rsema1d_pcs::Rsema1dGKRConfig;
use std::time::Instant;

const N: usize = 16; // number of leaves in the product tree

declare_circuit!(GfTreeCircuit {
    vals: [[Variable; 128]; N],
    exp: [Variable; 128],
    z: PublicVariable, // dummy public output (0)
});

impl Define<GF2Config> for GfTreeCircuit<Variable> {
    fn define<Builder: RootAPI<GF2Config>>(&self, api: &mut Builder) {
        // Balanced product tree of the N leaves.
        let mut layer: Vec<Vec<Variable>> = (0..N).map(|i| self.vals[i].to_vec()).collect();
        while layer.len() > 1 {
            let mut next = Vec::new();
            let mut i = 0;
            while i < layer.len() {
                if i + 1 < layer.len() {
                    next.push(gf128::mul(api, &layer[i], &layer[i + 1]));
                    i += 2;
                } else {
                    next.push(layer[i].clone());
                    i += 1;
                }
            }
            layer = next;
        }
        for b in 0..128 {
            api.assert_is_equal(layer[0][b], self.exp[b]);
        }
        api.assert_is_equal(self.z, 0);
    }
}

fn set128(dst: &mut [GF2], x: u128) {
    for i in 0..128 {
        dst[i] = (((x >> i) & 1) as u32).into();
    }
}

#[repr(C)]
struct Rusage {
    _t: [i64; 4],
    ru_maxrss: i64,
    _rest: [i64; 14],
}
extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
}
fn peak_rss_mib() -> f64 {
    unsafe {
        let mut ru: Rusage = std::mem::zeroed();
        getrusage(0, &mut ru);
        ru.ru_maxrss as f64 / (1024.0 * 1024.0)
    }
}

fn main() {
    println!("==== GF(2^128) in-circuit multiply / product-tree feasibility probe (N={N}) ====");

    // Random-ish leaves (deterministic), native expected product.
    let mut vals = [0u128; N];
    let mut x = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
    for v in vals.iter_mut() {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = x | 1; // avoid zero
    }
    let mut exp = vals[0];
    for &v in vals.iter().skip(1) {
        exp = gf128::native_mul(exp, v);
    }
    // (product tree associates differently than the linear fold, but GF mul is
    // associative/commutative, so the result matches.)

    // Sanity: native mul identity and a known small case.
    assert_eq!(gf128::native_mul(1, vals[0]), vals[0], "1*x != x");
    assert_eq!(gf128::native_mul(vals[0], 1), vals[0], "x*1 != x");

    let mut a = GfTreeCircuit::<GF2>::default();
    for i in 0..N {
        set128(&mut a.vals[i], vals[i]);
    }
    set128(&mut a.exp, exp);
    a.z = GF2::zero();

    let t = Instant::now();
    let CompileResult { witness_solver, layered_circuit } =
        compile(&GfTreeCircuit::default(), CompileOptions::default()).unwrap();
    println!("[compile] in {:?}", t.elapsed());

    let assignments = vec![a; 8];
    let tw = Instant::now();
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();
    println!(
        "[witness] solved {} in {:?}; inputs/witness = {}",
        witness.num_witnesses,
        tw.elapsed(),
        witness.num_inputs_per_witness
    );
    let res = layered_circuit.run(&witness);
    assert!(res.iter().all(|x| *x), "self-eval FAILED (gadget wrong): {res:?}");
    println!("[witness] layered_circuit.run() = all true (gadget matches native)");

    let mut ec = layered_circuit.export_to_expander_flatten();
    let (si, sp) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = si;
    ec.public_input = sp;
    ec.evaluate();
    let num_vars = ec.log_input_size();
    let total_gates: usize = ec.layers.iter().map(|l| l.mul.len() + l.add.len()).sum();
    println!(
        "[export] layers = {}, committed input bits = {}, num_vars = {}, sum(mul+add gate entries) = {}",
        ec.layers.len(),
        ec.layers[0].input_vals.len() * 8,
        num_vars,
        total_gates
    );

    let mpi = MPIConfig::prover_new(None, None);
    let tp = Instant::now();
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let pd = tp.elapsed();
    let tv = Instant::now();
    let ok = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v);
    let vd = tv.elapsed();
    println!(
        "[prove] {:?}, proof bytes = {}; [verify] {} in {:?}; claimed==0: {}",
        pd,
        proof.bytes.len(),
        ok,
        vd,
        claimed_v.is_zero()
    );
    assert!(ok && claimed_v.is_zero(), "verifier rejected honest GF128 tree proof");
    println!(
        "[RESULT] N={N} num_vars={num_vars} gate_entries={total_gates} prove={pd:?} verify={vd:?} RSS={:.1}MiB",
        peak_rss_mib()
    );
}
