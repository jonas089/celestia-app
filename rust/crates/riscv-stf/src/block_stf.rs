//! R8 capstone — the accidental-EVM-computer BLOCK state transition, in-circuit:
//! committed input = the block's DA data (a contract account's code + pre-storage
//! + the touched accounts' nonce/balance); computation = in-circuit EVM execution
//! of the contract call + Ethereum world-state MPT (contract storageRoot +
//! codeHash + EOA leaves); public output = the reth-faithful post_state_root.
//! Proven over Expander GKR with the reused rsema1d/DA commitment (zero prover
//! encoding). Small block; ~few GB.
//!
//! This ties together every verified component: EVM interpreter (evm_circuit),
//! storage trie + account trie (mpt_circuit, ==alloy), keccak, and DA reuse.

use crate::evm_circuit::{run, Cfg, Evm};
use crate::mpt::{empty_root, hp, keccak256, keccak_empty, nibbles, state_root, storage_root, Account};
use crate::mpt_circuit::{account_leaf_hash, account_leaf_hash_var, branch_root_3, storage_leaf_root};
use crate::u256::{bigint_to_bits, BITS};
use arith::SimdField;
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use num_bigint::BigInt;
use polynomials::MultiLinearPoly;
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};

pub const CODE: usize = 32;
pub const CODE_ACTUAL: usize = 10; // demo runtime length (for codeHash keccak)
pub const SD: usize = 8;
pub const SS: usize = 2;
pub const MEM: usize = 8;
pub const N: usize = 8;
pub const SP_BITS: usize = 4;
pub const PC_BITS: usize = 6;

// Fixed topology (set by the prover from the chosen addresses + storage slot).
pub static mut HP_CONTRACT: [u8; 32] = [0u8; 32];
pub static mut HP_EOA0: [u8; 32] = [0u8; 32];
pub static mut HP_EOA1: [u8; 32] = [0u8; 32];
pub static mut SLOT_C: usize = 0;
pub static mut SLOT_E0: usize = 0;
pub static mut SLOT_E1: usize = 0;
pub static mut HP_STORAGE0: [u8; 33] = [0u8; 33]; // HP of the single storage slot (slot 0)

declare_circuit!(BlockCircuit {
    code: [[Variable; 8]; CODE],
    c_pre_slot0: [Variable; BITS],
    c_nonce: [Variable; BITS],
    c_balance: [Variable; BITS],
    e0_nonce: [Variable; BITS],
    e0_balance: [Variable; BITS],
    e1_nonce: [Variable; BITS],
    e1_balance: [Variable; BITS],
    out_root: [PublicVariable; 256],
});

impl Define<GF2Config> for BlockCircuit<Variable> {
    fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
        let zw = vec![api.constant(0); BITS];
        // (1) Execute the contract call in-circuit over committed code.
        let cfg = Cfg { sd: SD, ss: SS, code_len: CODE, mem_bytes: MEM, sp_bits: SP_BITS, pc_bits: PC_BITS };
        let code: Vec<Vec<Variable>> = (0..CODE).map(|i| self.code[i].to_vec()).collect();
        let st = Evm {
            stack: vec![zw.clone(); SD], sp: vec![api.constant(0); SP_BITS], pc: vec![api.constant(0); PC_BITS],
            skeys: vec![zw.clone(), zw.clone()],           // slot 0 key = 0
            svals: vec![self.c_pre_slot0.to_vec(), zw.clone()],
            mem: vec![vec![api.constant(0); 8]; MEM], ret: vec![vec![api.constant(0); 8]; MEM],
            ret_len: vec![api.constant(0); PC_BITS], halted: api.constant(0),
        };
        let fin = run(api, st, &code, &cfg, N);
        let post_slot0 = fin.svals[0].clone();

        // (2) codeHash = keccak(code[0..CODE_ACTUAL]) via var-len keccak (135-pad).
        let mut msg = Vec::with_capacity(135 * 8);
        for i in 0..135 { for b in 0..8 { msg.push(if i < CODE { self.code[i][b] } else { api.constant(0) }); } }
        let len_bits: Vec<Variable> = (0..8).map(|b| api.constant(((CODE_ACTUAL as u32) >> b) & 1)).collect();
        let code_hash = crate::batch_keccak::keccak256_varlen(api, &msg, &len_bits);

        // (3) storageRoot = MPT over the single (slot0 -> post_slot0).
        let (hp_s,) = unsafe { (HP_STORAGE0,) };
        let storage_root_w = storage_leaf_root(api, &hp_s, &post_slot0);

        // (4) account leaves: contract (computed sr/ch) + 2 EOAs (empty sr/ch).
        let sr_e = empty_root();
        let ch_e = keccak_empty();
        let (hpc, hpe0, hpe1, sc, se0, se1) = unsafe { (HP_CONTRACT, HP_EOA0, HP_EOA1, SLOT_C, SLOT_E0, SLOT_E1) };
        let cleaf = account_leaf_hash_var(api, &self.c_nonce.to_vec(), &self.c_balance.to_vec(), &hpc, &storage_root_w, &code_hash);
        let e0leaf = account_leaf_hash(api, &self.e0_nonce.to_vec(), &self.e0_balance.to_vec(), &hpe0, &sr_e, &ch_e);
        let e1leaf = account_leaf_hash(api, &self.e1_nonce.to_vec(), &self.e1_balance.to_vec(), &hpe1, &sr_e, &ch_e);
        let root = branch_root_3(api, &[(cleaf, sc), (e0leaf, se0), (e1leaf, se1)]);
        for i in 0..256 { api.assert_is_equal(root[i], self.out_root[i]); }
    }
}

pub struct BlockInputs {
    pub contract_addr: [u8; 20],
    pub code: Vec<u8>,       // runtime code (CODE_ACTUAL bytes)
    pub c_pre_slot0: BigInt, // pre-state storage slot 0 value
    pub c_nonce: u64,
    pub c_balance: BigInt,
    pub eoa: [([u8; 20], u64, BigInt); 2],
}

pub struct BlockProof {
    pub commitment: [u8; 32],
    pub verified: bool,
    pub post_state_root: [u8; 32],
    pub input_vars: u32,
    pub proof: Vec<u8>,
}

fn da_commit(input_vals: &[gf2::GF2x8], num_vars: usize) -> [u8; 32] {
    const NUM_SYMBOLS: usize = 32; const ROW_BYTES: usize = 64;
    let k: u32 = 1u32 << (num_vars - 2); let n = k;
    let bits: Vec<[u8; 8]> = input_vals.iter().map(|e| { let l = e.unpack(); let mut b = [0u8; 8]; for s in 0..8 { b[s] = l[s].v & 1; } b }).collect();
    let mut rows = vec![vec![0u8; ROW_BYTES]; (k + n) as usize];
    for j in 0..(k as usize) { for i in 0..NUM_SYMBOLS { let a = j * NUM_SYMBOLS + i; rows[j][i] = bits[a >> 3][a & 7]; } }
    let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    rsema1d_sys::commit(k, n, &row_refs).unwrap().0
}

/// Native golden: execute the contract call and compute the post-state root.
pub fn native_post_root(inp: &BlockInputs) -> [u8; 32] {
    let ctx = crate::evm::CallCtx { code: inp.code.clone(), ..Default::default() };
    let mut storage = std::collections::BTreeMap::new();
    storage.insert(BigInt::from(0u32), inp.c_pre_slot0.clone());
    let r = crate::evm::execute(&ctx, &mut storage, 10_000_000);
    assert!(r.success, "native contract exec reverted");
    let post_slots: Vec<(BigInt, BigInt)> = storage.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let accts = vec![
        Account::contract(inp.contract_addr, inp.c_nonce, inp.c_balance.clone(), &inp.code, &post_slots),
        Account::eoa(inp.eoa[0].0, inp.eoa[0].1, inp.eoa[0].2.clone()),
        Account::eoa(inp.eoa[1].0, inp.eoa[1].1, inp.eoa[1].2.clone()),
    ];
    state_root(&accts)
}

/// Deterministic demo block used by the cross-process hand-off binaries so the
/// DA-encoder process and the prover process build the byte-identical block.
pub fn demo_block_inputs() -> BlockInputs {
    let mut addrs = [[0u8; 20]; 3];
    let (mut chosen, mut cand) = (0usize, 1u8);
    let mut seen: Vec<usize> = vec![];
    while chosen < 3 {
        let a = [cand; 20];
        let s = nibbles(&keccak256(&a))[0] as usize;
        if !seen.contains(&s) { addrs[chosen] = a; seen.push(s); chosen += 1; }
        cand = cand.wrapping_add(1);
    }
    BlockInputs {
        contract_addr: addrs[0],
        code: vec![0x60, 0x2a, 0x60, 0x00, 0x55, 0x60, 0x01, 0x60, 0x00, 0xf3],
        c_pre_slot0: BigInt::from(0u32),
        c_nonce: 1, c_balance: BigInt::from(0u32),
        eoa: [(addrs[1], 7, BigInt::from(1_000_000_000_000_000_000u64)), (addrs[2], 0, BigInt::from(500_000_000_000_000u64))],
    }
}

fn install_topology(inp: &BlockInputs) {
    let set = |addr: &[u8; 20]| -> ([u8; 32], usize) {
        let nibs = nibbles(&keccak256(addr));
        let mut h = [0u8; 32]; h.copy_from_slice(&hp(&nibs[1..64], true));
        (h, nibs[0] as usize)
    };
    let (hc, sc) = set(&inp.contract_addr);
    let (h0, s0) = set(&inp.eoa[0].0);
    let (h1, s1) = set(&inp.eoa[1].0);
    // storage slot 0 -> hp of full 64-nibble key keccak(be32(0))
    let snibs = nibbles(&keccak256(&[0u8; 32]));
    let mut hs = [0u8; 33]; hs.copy_from_slice(&hp(&snibs, true));
    unsafe {
        HP_CONTRACT = hc; SLOT_C = sc; HP_EOA0 = h0; SLOT_E0 = s0; HP_EOA1 = h1; SLOT_E1 = s1; HP_STORAGE0 = hs;
    }
}

/// Compile the block circuit, solve the witness, and evaluate it to obtain the
/// GKR input-layer polynomial (`input_vals`) and its `num_vars`. This is the
/// deterministic "block DA data" of the accidental computer — the same input
/// layer the prover commits. Factored so the DA-encoder side
/// ([`block_da_serialize`]) can derive the exact input the prover will open,
/// without proving.
fn block_input_layer(inp: &BlockInputs) -> Result<(Vec<gf2::GF2x8>, usize), String> {
    install_topology(inp);
    let native = native_post_root(inp);
    let CompileResult { witness_solver, layered_circuit } =
        compile(&BlockCircuit::default(), CompileOptions::default()).map_err(|e| format!("compile: {e:?}"))?;
    let mut asg = BlockCircuit::<GF2>::default();
    let put = |dst: &mut [GF2], v: &BigInt| { for (i, b) in bigint_to_bits(v, BITS).into_iter().enumerate() { dst[i] = (b as u32).into(); } };
    for i in 0..CODE { let byte = if i < inp.code.len() { inp.code[i] } else { 0 }; for b in 0..8 { asg.code[i][b] = (((byte >> b) & 1) as u32).into(); } }
    put(&mut asg.c_pre_slot0, &inp.c_pre_slot0);
    put(&mut asg.c_nonce, &BigInt::from(inp.c_nonce));
    put(&mut asg.c_balance, &inp.c_balance);
    put(&mut asg.e0_nonce, &BigInt::from(inp.eoa[0].1)); put(&mut asg.e0_balance, &inp.eoa[0].2);
    put(&mut asg.e1_nonce, &BigInt::from(inp.eoa[1].1)); put(&mut asg.e1_balance, &inp.eoa[1].2);
    for i in 0..32 { for j in 0..8 { asg.out_root[i * 8 + j] = (((native[i] >> j) & 1) as u32).into(); } }
    let witness = witness_solver.solve_witnesses(&vec![asg; 8]).map_err(|e| format!("witness: {e:?}"))?;
    if !layered_circuit.run(&witness).iter().all(|x| *x) { return Err("in-circuit block STF != native post_state_root".into()); }
    let mut ec = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = simd_input;
    ec.public_input = simd_public_input;
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let num_vars = ec.log_input_size();
    Ok((input_vals, num_vars))
}

/// **DA-encoder entry point (cross-process hand-off).** Derives the block's GKR
/// input layer and RS-encodes it ONCE via the Go DA encoder, returning the DA
/// commitment root, the serialized extended-row matrix, and `num_vars`. Emit the
/// `(root, extended, num_vars)` to another process, which reconstructs the
/// committed square with [`rsema1d_pcs::install_da_commitment_from_serialized`]
/// and then calls [`prove_block`] — the prover performs ZERO RS-encoding.
pub fn block_da_serialize(inp: &BlockInputs) -> Result<([u8; 32], Vec<u8>, u32), String> {
    let (input_vals, num_vars) = block_input_layer(inp)?;
    let da_poly = MultiLinearPoly::new(input_vals);
    let (root, extended) = rsema1d_pcs::da_encode_serialized(num_vars, &da_poly);
    Ok((root, extended, num_vars as u32))
}

/// Prove the contract-block STF with the reused DA commitment.
///
/// If a DA commitment for this block's `num_vars` is already installed (e.g. by
/// [`rsema1d_pcs::install_da_commitment_from_serialized`], the cross-process
/// hand-off), it is REUSED and this process does NO RS-encoding. Otherwise the
/// commitment is encoded in-process (the original single-process behavior).
pub fn prove_block(inp: &BlockInputs) -> Result<BlockProof, String> {
    let sc = { let nibs = nibbles(&keccak256(&inp.contract_addr)); nibs[0] as usize };
    let s0 = { let nibs = nibbles(&keccak256(&inp.eoa[0].0)); nibs[0] as usize };
    let s1 = { let nibs = nibbles(&keccak256(&inp.eoa[1].0)); nibs[0] as usize };
    if sc == s0 || sc == s1 || s0 == s1 { return Err(format!("addr first-nibbles collide: {sc},{s0},{s1}")); }
    install_topology(inp);
    let native = native_post_root(inp);

    let CompileResult { witness_solver, layered_circuit } =
        compile(&BlockCircuit::default(), CompileOptions::default()).map_err(|e| format!("compile: {e:?}"))?;
    let mut asg = BlockCircuit::<GF2>::default();
    let put = |dst: &mut [GF2], v: &BigInt| { for (i, b) in bigint_to_bits(v, BITS).into_iter().enumerate() { dst[i] = (b as u32).into(); } };
    for i in 0..CODE { let byte = if i < inp.code.len() { inp.code[i] } else { 0 }; for b in 0..8 { asg.code[i][b] = (((byte >> b) & 1) as u32).into(); } }
    put(&mut asg.c_pre_slot0, &inp.c_pre_slot0);
    put(&mut asg.c_nonce, &BigInt::from(inp.c_nonce));
    put(&mut asg.c_balance, &inp.c_balance);
    put(&mut asg.e0_nonce, &BigInt::from(inp.eoa[0].1)); put(&mut asg.e0_balance, &inp.eoa[0].2);
    put(&mut asg.e1_nonce, &BigInt::from(inp.eoa[1].1)); put(&mut asg.e1_balance, &inp.eoa[1].2);
    for i in 0..32 { for j in 0..8 { asg.out_root[i * 8 + j] = (((native[i] >> j) & 1) as u32).into(); } }

    let witness = witness_solver.solve_witnesses(&vec![asg; 8]).map_err(|e| format!("witness: {e:?}"))?;
    if !layered_circuit.run(&witness).iter().all(|x| *x) { return Err("in-circuit block STF != native post_state_root".into()); }

    let mut ec = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = simd_input.clone();
    ec.public_input = simd_public_input.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let num_vars = ec.log_input_size();
    let da_poly = MultiLinearPoly::new(input_vals.clone());
    // Reuse a pre-installed DA commitment (cross-process hand-off) if present;
    // otherwise encode it in-process. When reused, this process runs no RS-encode.
    let (da_root, reused) = match rsema1d_pcs::installed_da_root(num_vars) {
        Some(root) => (root, true),
        None => (rsema1d_pcs::install_da_commitment(num_vars, &da_poly), false),
    };

    let mpi = MPIConfig::prover_new(None, None);
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let verified = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v) && claimed_v.is_zero();
    // Independent in-process re-encode as a sanity cross-check ONLY on the
    // single-process path. On the cross-process (reused) path we deliberately do
    // NOT re-encode — the DA side already produced (and we reconstructed) the
    // commitment, and re-encoding here would defeat the zero-prover-encode point.
    let c_da = if reused {
        da_root
    } else {
        let c = da_commit(&input_vals, num_vars);
        if da_root != c { return Err("installed DA root != independent DA encode".into()); }
        c
    };
    if !proof.bytes.windows(32).any(|w| w == da_root) { return Err("DA commitment not embedded".into()); }

    Ok(BlockProof { commitment: c_da, verified, post_state_root: native, input_vars: num_vars as u32, proof: proof.bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo() -> BlockInputs {
        // pick 3 distinct-first-nibble addresses
        let mut addrs = [[0u8; 20]; 3];
        let (mut chosen, mut cand) = (0usize, 1u8);
        let mut seen: Vec<usize> = vec![];
        while chosen < 3 {
            let a = [cand; 20];
            let s = nibbles(&keccak256(&a))[0] as usize;
            if !seen.contains(&s) { addrs[chosen] = a; seen.push(s); chosen += 1; }
            cand = cand.wrapping_add(1);
        }
        BlockInputs {
            contract_addr: addrs[0],
            code: vec![0x60, 0x2a, 0x60, 0x00, 0x55, 0x60, 0x01, 0x60, 0x00, 0xf3], // SSTORE 0x2a @ slot0; RETURN
            c_pre_slot0: BigInt::from(0u32),
            c_nonce: 1, c_balance: BigInt::from(0u32),
            eoa: [(addrs[1], 7, BigInt::from(1_000_000_000_000_000_000u64)), (addrs[2], 0, BigInt::from(500_000_000_000_000u64))],
        }
    }

    #[test]
    fn incircuit_block_stf_matches_native() {
        let inp = demo();
        install_topology(&inp);
        let native = native_post_root(&inp);
        let CompileResult { witness_solver, layered_circuit } = compile(&BlockCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = BlockCircuit::<GF2>::default();
        let put = |dst: &mut [GF2], v: &BigInt| { for (i, b) in bigint_to_bits(v, BITS).into_iter().enumerate() { dst[i] = (b as u32).into(); } };
        for i in 0..CODE { let byte = if i < inp.code.len() { inp.code[i] } else { 0 }; for b in 0..8 { asg.code[i][b] = (((byte >> b) & 1) as u32).into(); } }
        put(&mut asg.c_pre_slot0, &inp.c_pre_slot0);
        put(&mut asg.c_nonce, &BigInt::from(inp.c_nonce)); put(&mut asg.c_balance, &inp.c_balance);
        put(&mut asg.e0_nonce, &BigInt::from(inp.eoa[0].1)); put(&mut asg.e0_balance, &inp.eoa[0].2);
        put(&mut asg.e1_nonce, &BigInt::from(inp.eoa[1].1)); put(&mut asg.e1_balance, &inp.eoa[1].2);
        for i in 0..32 { for j in 0..8 { asg.out_root[i * 8 + j] = (((native[i] >> j) & 1) as u32).into(); } }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit block STF post_state_root != native/alloy");
        println!("R8 block STF post_state_root = 0x{}", native.iter().map(|b| format!("{:02x}", b)).collect::<String>());
    }
}
