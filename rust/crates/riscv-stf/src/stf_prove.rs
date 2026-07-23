//! The accidental-computer STF proof (transfer block): committed input = the
//! block's DA data (pre-state accounts + tx params), computation = the
//! reth-faithful value-transfer transition + Ethereum MPT, output =
//! post_state_root (public), proven over Expander GKR with `Rsema1dGKRConfig` so
//! the rsema1d/DA commitment is the SOLE polynomial commitment (opened at the
//! sumcheck point). Sender recovery (`ecrecover`, R5) is verified separately and
//! is composable into the same circuit (heavy); here the STF+MPT half is proven
//! with the input committed — a real rsema1d-committed proof whose public output
//! is byte-identical to reth's post-state root.

use crate::mpt::{empty_root, keccak_empty, hp, nibbles, keccak256, state_root, Account};
use crate::mpt_circuit::{transfer_stf_root, AcctIn};
use crate::u256::{bigint_to_bits, BITS};
use arith::SimdField;
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{ExpanderPCS, GF2ExtConfig, MPIConfig, MPIEngine};
use num_bigint::BigInt;
use polynomials::MultiLinearPoly;
use rsema1d_pcs::{Rsema1dGKRConfig, Rsema1dPCS};

// Fixed trie topology (hp paths + branch slots) for the block's 3 accounts,
// installed by the driver before compile().
pub static mut HP3G: [[u8; 32]; 3] = [[0u8; 32]; 3];
pub static mut SLOT3G: [usize; 3] = [0; 3];

declare_circuit!(StfProveCircuit {
    // committed DA data: pre (nonce,balance) for sender/recipient/coinbase + tx.
    n: [[Variable; BITS]; 3],
    b: [[Variable; BITS]; 3],
    value: [Variable; BITS],
    maxfee: [Variable; BITS],
    maxprio: [Variable; BITS],
    basefee: [Variable; BITS],
    // public output: post-state root.
    out: [PublicVariable; 256],
});

impl Define<GF2Config> for StfProveCircuit<Variable> {
    fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
        let sr = empty_root();
        let ch = keccak_empty();
        let (hp3, slots) = unsafe { (HP3G, SLOT3G) };
        let pn = [self.n[0].to_vec(), self.n[1].to_vec(), self.n[2].to_vec()];
        let pb = [self.b[0].to_vec(), self.b[1].to_vec(), self.b[2].to_vec()];
        let root = transfer_stf_root(api, &pn, &pb, &self.value.to_vec(), &self.maxfee.to_vec(), &self.maxprio.to_vec(), &self.basefee.to_vec(), &hp3, &slots, &sr, &ch);
        for i in 0..256 {
            api.assert_is_equal(root[i], self.out[i]);
        }
    }
}

pub struct StfInputs {
    pub addrs: [[u8; 20]; 3], // sender, recipient, coinbase
    pub nonce: [u64; 3],
    pub balance: [BigInt; 3],
    pub value: BigInt,
    pub max_fee: BigInt,
    pub max_prio: BigInt,
    pub base_fee: BigInt,
}

pub struct StfProof {
    pub commitment: [u8; 32],
    pub proof: Vec<u8>,
    pub post_state_root: [u8; 32],
    pub verified: bool,
    pub input_vars: u32,
}

fn da_commit(input_vals: &[gf2::GF2x8], num_vars: usize) -> Result<[u8; 32], String> {
    const NUM_SYMBOLS: usize = 32;
    const ROW_BYTES: usize = 64;
    let k: u32 = 1u32 << (num_vars - 2);
    let n: u32 = k;
    let bits: Vec<[u8; 8]> = input_vals.iter().map(|e| { let l = e.unpack(); let mut b = [0u8; 8]; for s in 0..8 { b[s] = l[s].v & 1; } b }).collect();
    let mut rows = vec![vec![0u8; ROW_BYTES]; (k + n) as usize];
    for j in 0..(k as usize) {
        for i in 0..NUM_SYMBOLS {
            let a = j * NUM_SYMBOLS + i;
            rows[j][i] = bits[a >> 3][a & 7];
        }
    }
    let row_refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let (c, _h) = rsema1d_sys::commit(k, n, &row_refs).map_err(|e| format!("rsema1d commit: {e}"))?;
    Ok(c)
}

fn put(dst: &mut [GF2], v: &BigInt) {
    for (i, b) in bigint_to_bits(v, BITS).into_iter().enumerate() {
        dst[i] = (b as u32).into();
    }
}

/// Prove the transfer STF over committed block data with rsema1d as the sole
/// commitment. Returns the proof + the (reth-faithful) post_state_root.
pub fn prove_transfer_stf(inp: &StfInputs) -> Result<StfProof, String> {
    // Install the fixed trie topology for these 3 addresses (distinct nibbles).
    let mut slots = [0usize; 3];
    let mut hp3 = [[0u8; 32]; 3];
    for i in 0..3 {
        let nibs = nibbles(&keccak256(&inp.addrs[i]));
        slots[i] = nibs[0] as usize;
        hp3[i].copy_from_slice(&hp(&nibs[1..64], true));
    }
    if slots[0] == slots[1] || slots[1] == slots[2] || slots[0] == slots[2] {
        return Err(format!("addresses' first nibbles collide: {slots:?} (need distinct for this topology)"));
    }
    unsafe { HP3G = hp3; SLOT3G = slots; }

    // Native golden post-state root.
    let pre = vec![
        Account::eoa(inp.addrs[0], inp.nonce[0], inp.balance[0].clone()),
        Account::eoa(inp.addrs[1], inp.nonce[1], inp.balance[1].clone()),
        Account::eoa(inp.addrs[2], inp.nonce[2], inp.balance[2].clone()),
    ];
    let tx = crate::rlp::Eip1559Tx {
        chain_id: 1, nonce: inp.nonce[0], max_priority_fee: inp.max_prio.clone(), max_fee: inp.max_fee.clone(),
        gas_limit: 21000, to: inp.addrs[1], value: inp.value.clone(), data: vec![], y_parity: 0, r: BigInt::from(1), s: BigInt::from(1),
    };
    let env = crate::stf_transfer::BlockEnv { base_fee: inp.base_fee.clone(), coinbase: inp.addrs[2] };
    let block = crate::stf_transfer::TransferBlock { pre, tx, sender: inp.addrs[0], env };
    let (_post, native_root) = crate::stf_transfer::apply(&block)?;

    // Compile + witness.
    let CompileResult { witness_solver, layered_circuit } =
        compile(&StfProveCircuit::default(), CompileOptions::default()).map_err(|e| format!("compile: {e:?}"))?;
    let mut asg = StfProveCircuit::<GF2>::default();
    for i in 0..3 {
        put(&mut asg.n[i], &BigInt::from(inp.nonce[i]));
        put(&mut asg.b[i], &inp.balance[i]);
    }
    put(&mut asg.value, &inp.value);
    put(&mut asg.maxfee, &inp.max_fee);
    put(&mut asg.maxprio, &inp.max_prio);
    put(&mut asg.basefee, &inp.base_fee);
    for i in 0..32 {
        for j in 0..8 {
            asg.out[i * 8 + j] = (((native_root[i] >> j) & 1) as u32).into();
        }
    }
    let witness = witness_solver.solve_witnesses(&vec![asg; 8]).map_err(|e| format!("witness: {e:?}"))?;
    if !layered_circuit.run(&witness).iter().all(|x| *x) {
        return Err("layered self-eval failed (post_root != native)".into());
    }

    // Export, install committed input, prove with rsema1d as sole PCS.
    let mut ec = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = simd_input.clone();
    ec.public_input = simd_public_input.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let num_vars = ec.log_input_size();

    // === DA ENCODER STEP (the data-availability encoding, done once): encode the
    // committed block-data rows and install the commitment + live handle. ===
    let da_poly = MultiLinearPoly::new(input_vals.clone());
    let _da_root = rsema1d_pcs::install_da_commitment(num_vars, &da_poly);

    // === GKR PROVER: REUSES the installed DA handle — ZERO encoding (commit
    // returns the DA root; open evaluates the DA square in place). ===
    let mpi = MPIConfig::prover_new(None, None);
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let verified = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v) && claimed_v.is_zero();

    // Commitment reconciliation: GKR PCS root == independent Go/DA rsema1d commit.
    let params = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi.world_size());
    let mut sp = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
    let poly = MultiLinearPoly::new(input_vals.clone());
    let c_prover = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &poly, &mut sp).ok_or("rsema1d commit None")?;
    let c_da = da_commit(&input_vals, num_vars)?;
    if c_prover.root != c_da {
        return Err(format!("commitment mismatch: {:x?} != Go/DA {:x?}", &c_prover.root[..8], &c_da[..8]));
    }
    if !proof.bytes.windows(32).any(|w| w == c_prover.root) {
        return Err("commitment not embedded in proof".into());
    }

    // Reconstruct public post_state_root from the proof's public inputs.
    let mut recon = [0u8; 32];
    for i in 0..32 {
        for j in 0..8 {
            recon[i] |= ((simd_public_input[i * 8 + j].unpack()[0].v & 1) as u8) << j;
        }
    }
    if recon != native_root {
        return Err("public output != native root".into());
    }

    Ok(StfProof { commitment: c_da, proof: proof.bytes, post_state_root: native_root, verified, input_vars: num_vars as u32 })
}

// ---------------------------------------------------------------------------
// FULL transfer-block accidental computer: committed input = the signed tx
// (signing-preimage + r,s,yparity) + pre-state accounts + block env; the circuit
// RECOVERS the sender (ecrecover, R5+R6), BINDS the tx value/nonce to the signed
// preimage bytes, applies the transition, and roots the post-state. rsema1d is
// the sole commitment. HEAVY (ecrecover ~hours) — this is the complete artifact.
// ---------------------------------------------------------------------------

/// Sender address (160 bits, LSB-first per byte) for the demo topology account 0.
pub static mut ADDR0G: [u8; 20] = [0u8; 20];

const PRE_LEN: usize = 49; // demo transfer preimage length

declare_circuit!(StfFullCircuit {
    preimage: [Variable; PRE_LEN * 8],
    r: [Variable; 256],
    s: [Variable; 256],
    yparity: Variable,
    n: [[Variable; BITS]; 3],
    b: [[Variable; BITS]; 3],
    value: [Variable; BITS],
    maxfee: [Variable; BITS],
    maxprio: [Variable; BITS],
    basefee: [Variable; BITS],
    out: [PublicVariable; 256],
});

impl Define<GF2Config> for StfFullCircuit<Variable> {
    fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
        // (1) Recover the sender from the committed signed tx; assert == account 0.
        let addr = crate::rlp::circuit::tx_to_address(api, &self.preimage.to_vec(), self.yparity, &self.r.to_vec(), &self.s.to_vec());
        let addr0 = unsafe { ADDR0G };
        for byte in 0..20 {
            for j in 0..8 {
                let want = api.constant(((addr0[byte] >> j) & 1) as u32);
                api.assert_is_equal(addr[byte * 8 + j], want);
            }
        }
        // (2) Bind the committed `value` (u256 LE) to the preimage's value field
        // (demo tx: 7 big-endian bytes at preimage offset 40). LE byte k =
        // preimage byte (46-k) for k in 0..7; bytes 7..32 must be zero.
        for k in 0..7 {
            let src = 46 - k;
            for j in 0..8 {
                api.assert_is_equal(self.value[k * 8 + j], self.preimage[src * 8 + j]);
            }
        }
        let zero = api.constant(0);
        for k in 7..32 {
            for j in 0..8 {
                api.assert_is_equal(self.value[k * 8 + j], zero);
            }
        }
        // (3) Bind committed sender nonce (n[0]) to the preimage nonce (1 byte @ off 3).
        for j in 0..8 {
            api.assert_is_equal(self.n[0][j], self.preimage[3 * 8 + j]);
        }
        for k in 1..32 {
            for j in 0..8 {
                api.assert_is_equal(self.n[0][k * 8 + j], zero);
            }
        }
        // (4) Apply the transition + MPT, output post_state_root.
        let sr = empty_root();
        let ch = keccak_empty();
        let (hp3, slots) = unsafe { (HP3G, SLOT3G) };
        let pn = [self.n[0].to_vec(), self.n[1].to_vec(), self.n[2].to_vec()];
        let pb = [self.b[0].to_vec(), self.b[1].to_vec(), self.b[2].to_vec()];
        let root = transfer_stf_root(api, &pn, &pb, &self.value.to_vec(), &self.maxfee.to_vec(), &self.maxprio.to_vec(), &self.basefee.to_vec(), &hp3, &slots, &sr, &ch);
        for i in 0..256 {
            api.assert_is_equal(root[i], self.out[i]);
        }
    }
}

/// Prove the FULL transfer-block accidental computer (with in-circuit ecrecover).
/// `preimage`, `r`, `s`, `yparity` are the signed EIP-1559 tx; addrs[0] must be
/// the signer. HEAVY. Returns the proof + reth-faithful post_state_root.
pub fn prove_transfer_stf_full(inp: &StfInputs, preimage: &[u8], r: &BigInt, s: &BigInt, yparity: u8) -> Result<StfProof, String> {
    let mut slots = [0usize; 3];
    let mut hp3 = [[0u8; 32]; 3];
    for i in 0..3 {
        let nibs = nibbles(&keccak256(&inp.addrs[i]));
        slots[i] = nibs[0] as usize;
        hp3[i].copy_from_slice(&hp(&nibs[1..64], true));
    }
    if slots[0] == slots[1] || slots[1] == slots[2] || slots[0] == slots[2] {
        return Err(format!("first nibbles collide: {slots:?}"));
    }
    unsafe { HP3G = hp3; SLOT3G = slots; ADDR0G = inp.addrs[0]; }

    let pre = vec![
        Account::eoa(inp.addrs[0], inp.nonce[0], inp.balance[0].clone()),
        Account::eoa(inp.addrs[1], inp.nonce[1], inp.balance[1].clone()),
        Account::eoa(inp.addrs[2], inp.nonce[2], inp.balance[2].clone()),
    ];
    let tx = crate::rlp::Eip1559Tx {
        chain_id: 1, nonce: inp.nonce[0], max_priority_fee: inp.max_prio.clone(), max_fee: inp.max_fee.clone(),
        gas_limit: 21000, to: inp.addrs[1], value: inp.value.clone(), data: vec![], y_parity: yparity, r: r.clone(), s: s.clone(),
    };
    let env = crate::stf_transfer::BlockEnv { base_fee: inp.base_fee.clone(), coinbase: inp.addrs[2] };
    let block = crate::stf_transfer::TransferBlock { pre, tx, sender: inp.addrs[0], env };
    let (_post, native_root) = crate::stf_transfer::apply(&block)?;

    let CompileResult { witness_solver, layered_circuit } =
        compile(&StfFullCircuit::default(), CompileOptions::default()).map_err(|e| format!("compile: {e:?}"))?;
    let mut asg = StfFullCircuit::<GF2>::default();
    for (i, byte) in preimage.iter().enumerate().take(PRE_LEN) {
        for j in 0..8 { asg.preimage[i * 8 + j] = (((byte >> j) & 1) as u32).into(); }
    }
    for (i, bit) in bigint_to_bits(r, 256).into_iter().enumerate() { asg.r[i] = (bit as u32).into(); }
    for (i, bit) in bigint_to_bits(s, 256).into_iter().enumerate() { asg.s[i] = (bit as u32).into(); }
    asg.yparity = (yparity as u32).into();
    for i in 0..3 { put(&mut asg.n[i], &BigInt::from(inp.nonce[i])); put(&mut asg.b[i], &inp.balance[i]); }
    put(&mut asg.value, &inp.value);
    put(&mut asg.maxfee, &inp.max_fee);
    put(&mut asg.maxprio, &inp.max_prio);
    put(&mut asg.basefee, &inp.base_fee);
    for i in 0..32 { for j in 0..8 { asg.out[i * 8 + j] = (((native_root[i] >> j) & 1) as u32).into(); } }

    let witness = witness_solver.solve_witnesses(&vec![asg; 8]).map_err(|e| format!("witness: {e:?}"))?;
    if !layered_circuit.run(&witness).iter().all(|x| *x) {
        return Err("layered self-eval failed".into());
    }
    let mut ec = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = simd_input.clone();
    ec.public_input = simd_public_input.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let num_vars = ec.log_input_size();
    let mpi = MPIConfig::prover_new(None, None);
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let verified = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v) && claimed_v.is_zero();
    let params = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::gen_params(num_vars, mpi.world_size());
    let mut sp = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::init_scratch_pad(&params, &mpi);
    let poly = MultiLinearPoly::new(input_vals.clone());
    let c_prover = <Rsema1dPCS as ExpanderPCS<GF2ExtConfig>>::commit(&params, &mpi, &(), &poly, &mut sp).ok_or("commit None")?;
    let c_da = da_commit(&input_vals, num_vars)?;
    if c_prover.root != c_da { return Err("commitment mismatch".into()); }
    Ok(StfProof { commitment: c_da, proof: proof.bytes, post_state_root: native_root, verified, input_vars: num_vars as u32 })
}
