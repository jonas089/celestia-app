//! R8 assembly — a REAL accidental-computer GKR proof of in-circuit EVM
//! execution: committed input = the EVM bytecode + initial storage (the DA'd
//! data), computation = the in-circuit EVM step function (`evm_circuit`), public
//! output = post-execution storage. Proven over Expander GKR with the reused
//! rsema1d/DA commitment (install_da_commitment → prover does ZERO encoding).
//! This is the EVM half of the full ev-reth STF, in GKR, with true DA reuse.

use crate::evm_circuit::{run, Cfg, Evm};
use crate::u256::{bigint_to_bits, BITS};
use arith::SimdField;
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use num_bigint::BigInt;
use polynomials::MultiLinearPoly;
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};

pub const CODE_LEN: usize = 16;
pub const SD: usize = 8;
pub const SS: usize = 4;
pub const MEM_BYTES: usize = 8;
pub const N_STEPS: usize = 16;
pub const SP_BITS: usize = 4;
pub const PC_BITS: usize = 6;

declare_circuit!(EvmExecCircuit {
    // committed DA data: bytecode + initial storage (keys/vals).
    code: [[Variable; 8]; CODE_LEN],
    skeys: [[Variable; BITS]; SS],
    svals: [[Variable; BITS]; SS],
    // public output: post-execution storage values (same key order).
    out_svals: [[PublicVariable; BITS]; SS],
});

impl Define<GF2Config> for EvmExecCircuit<Variable> {
    fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
        let cfg = Cfg { sd: SD, ss: SS, code_len: CODE_LEN, mem_bytes: MEM_BYTES, sp_bits: SP_BITS, pc_bits: PC_BITS };
        let code: Vec<Vec<Variable>> = (0..CODE_LEN).map(|i| self.code[i].to_vec()).collect();
        let st = Evm {
            stack: vec![vec![api.constant(0); BITS]; SD],
            sp: vec![api.constant(0); SP_BITS],
            pc: vec![api.constant(0); PC_BITS],
            skeys: (0..SS).map(|i| self.skeys[i].to_vec()).collect(),
            svals: (0..SS).map(|i| self.svals[i].to_vec()).collect(),
            mem: vec![vec![api.constant(0); 8]; MEM_BYTES],
            ret: vec![vec![api.constant(0); 8]; MEM_BYTES],
            ret_len: vec![api.constant(0); PC_BITS],
            halted: api.constant(0),
        };
        let fin = run(api, st, &code, &cfg, N_STEPS);
        for i in 0..SS {
            for b in 0..BITS {
                api.assert_is_equal(fin.svals[i][b], self.out_svals[i][b]);
            }
        }
    }
}

pub struct EvmProof {
    pub commitment: [u8; 32],
    pub verified: bool,
    pub input_vars: u32,
    pub post_svals: Vec<BigInt>,
    pub proof_bytes: usize,
}

fn da_commit(input_vals: &[gf2::GF2x8], num_vars: usize) -> [u8; 32] {
    const NUM_SYMBOLS: usize = 32;
    const ROW_BYTES: usize = 64;
    let k: u32 = 1u32 << (num_vars - 2);
    let n: u32 = k;
    let bits: Vec<[u8; 8]> = input_vals.iter().map(|e| { let l = e.unpack(); let mut b = [0u8; 8]; for s in 0..8 { b[s] = l[s].v & 1; } b }).collect();
    let mut rows = vec![vec![0u8; ROW_BYTES]; (k + n) as usize];
    for j in 0..(k as usize) { for i in 0..NUM_SYMBOLS { let a = j * NUM_SYMBOLS + i; rows[j][i] = bits[a >> 3][a & 7]; } }
    let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    rsema1d_sys::commit(k, n, &row_refs).unwrap().0
}

/// Prove an EVM execution with the reused DA commitment. `code` bytes, initial
/// storage `(key,val)` pairs (padded to SS). Runs the native EVM to get the
/// golden post-storage, proves the in-circuit execution equals it.
pub fn prove_evm(code: &[u8], init_storage: &[(BigInt, BigInt)]) -> Result<EvmProof, String> {
    assert!(code.len() <= CODE_LEN);
    assert!(init_storage.len() <= SS);
    // native golden
    let ctx = crate::evm::CallCtx { code: code.to_vec(), ..Default::default() };
    let mut storage = std::collections::BTreeMap::new();
    for (k, v) in init_storage { storage.insert(k.clone(), v.clone()); }
    let r = crate::evm::execute(&ctx, &mut storage, 10_000_000);
    if !r.success { return Err("native execution reverted".into()); }
    // post values in the committed key order
    let mut post = Vec::with_capacity(SS);
    for i in 0..SS {
        let key = init_storage.get(i).map(|(k, _)| k.clone()).unwrap_or_else(BigInt::zero_stub);
        post.push(storage.get(&key).cloned().unwrap_or_else(BigInt::zero_stub));
    }

    let CompileResult { witness_solver, layered_circuit } =
        compile(&EvmExecCircuit::default(), CompileOptions::default()).map_err(|e| format!("compile: {e:?}"))?;
    let mut asg = EvmExecCircuit::<GF2>::default();
    let put = |dst: &mut [GF2], v: &BigInt| { for (i, b) in bigint_to_bits(v, BITS).into_iter().enumerate() { dst[i] = (b as u32).into(); } };
    for i in 0..CODE_LEN { let byte = if i < code.len() { code[i] } else { 0 }; for b in 0..8 { asg.code[i][b] = (((byte >> b) & 1) as u32).into(); } }
    for i in 0..SS {
        let (k, v) = init_storage.get(i).cloned().unwrap_or((BigInt::zero_stub(), BigInt::zero_stub()));
        put(&mut asg.skeys[i], &k);
        put(&mut asg.svals[i], &v);
        put(&mut asg.out_svals[i], &post[i]);
    }
    let witness = witness_solver.solve_witnesses(&vec![asg; 8]).map_err(|e| format!("witness: {e:?}"))?;
    if !layered_circuit.run(&witness).iter().all(|x| *x) { return Err("layered self-eval failed (in-circuit EVM != native)".into()); }

    let mut ec = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = simd_input.clone();
    ec.public_input = simd_public_input.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let num_vars = ec.log_input_size();

    // DA encoder step (once); prover reuses the handle (zero encoding).
    let da_poly = MultiLinearPoly::new(input_vals.clone());
    let da_root = rsema1d_pcs::install_da_commitment(num_vars, &da_poly);

    let mpi = MPIConfig::prover_new(None, None);
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let verified = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v) && claimed_v.is_zero();
    // reused commitment must equal an independent DA encode of the same rows.
    let c_da = da_commit(&input_vals, num_vars);
    if da_root != c_da { return Err("installed DA root != independent DA encode".into()); }
    if !proof.bytes.windows(32).any(|w| w == da_root) { return Err("DA commitment not embedded in proof".into()); }

    // reconstruct public post-storage
    let mut recon = Vec::with_capacity(SS);
    for i in 0..SS {
        let mut v = BigInt::zero_stub();
        for b in 0..BITS { if simd_public_input[i * BITS + b].unpack()[0].v & 1 == 1 { v |= BigInt::one_shl(b); } }
        recon.push(v);
    }
    Ok(EvmProof { commitment: c_da, verified, input_vars: num_vars as u32, post_svals: recon, proof_bytes: proof.bytes.len() })
}

// tiny helpers to avoid importing num_traits everywhere
trait BigIntExt { fn zero_stub() -> BigInt; fn one_shl(b: usize) -> BigInt; }
impl BigIntExt for BigInt {
    fn zero_stub() -> BigInt { BigInt::from(0u32) }
    fn one_shl(b: usize) -> BigInt { BigInt::from(1u32) << b }
}
