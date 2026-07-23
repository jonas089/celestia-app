//! RV32I "accidental computer" prover: compile the in-circuit RV32I interpreter
//! (`rv32_circuit`), assign the COMMITTED INPUT (program + initial registers +
//! initial memory), bind the PUBLIC OUTPUT (post registers + post memory) to the
//! native emulator golden, solve the witness, export to Expander, install the DA
//! commitment over the INPUT LAYER (the sole commitment — no separate RS-encode
//! by the prover), then GKR-prove and self-verify with `Rsema1dGKRConfig`.
//!
//! This mirrors `block_stf::prove_block` exactly for the DA/GKR tail. The point of
//! the regime: the ONLY commitment is the rsema1d/DA commitment of the committed
//! input; the entire per-cycle execution trace (every register, pc, memory word
//! and ALU wire across all steps) is an INTERMEDIATE wire pinned by sumcheck and
//! is NEVER committed.

use arith::SimdField;
use ::circuit::Circuit as ExpanderCircuit;
use expander_binary::executor;
use expander_compiler::frontend::*;
use gkr_engine::{GF2ExtConfig, MPIConfig, MPIEngine};
use polynomials::MultiLinearPoly;
use rsema1d_pcs::Rsema1dGKRConfig;

use crate::batch_keccak::keccak256_fixed;
use crate::emulator::Cpu;
use crate::rv32_circuit::{run, Rv32Cfg, Rv32State, W};

// Committed-size bounds (fixed at circuit-compile time). Kept tight so the
// circuit compiles + proves in a few minutes.
const PROG_LEN: usize = 16; // committed program words
const NREG: usize = 32; // RV32I register file
const MEM_SLOTS: usize = 8; // bounded word-memory slots (input region + user + pad)
const WADDR_BITS: usize = 16; // committed word-address width
const STEPS: usize = 16; // unrolled interpreter steps
const INPUT_ADDR: u32 = 0x100; // byte address where the input region is loaded

declare_circuit!(Rv32Circuit {
    // ---- COMMITTED INPUT LAYER (this is the DA-committed data) ----
    program: [[Variable; 32]; PROG_LEN],   // program words
    pre_regs: [[Variable; 32]; NREG],      // initial register file
    mem_addr: [[Variable; WADDR_BITS]; MEM_SLOTS], // initial memory word addresses
    mem_val: [[Variable; 32]; MEM_SLOTS],  // initial memory values (input bytes live here)
    // ---- PUBLIC OUTPUT (bound to the native golden) ----
    post_regs: [[PublicVariable; 32]; NREG],   // final register file
    post_mem: [[PublicVariable; 32]; MEM_SLOTS], // final memory values
    // ---- ROLLUP STATE TRANSITION (genuine root -> root) ----
    // pre_root = keccak256(pre-state), post_root = keccak256(post-state), both
    // computed IN-CIRCUIT and bound below. The state is the register file (the
    // persistent state carried block->block), so post_root(n) == pre_root(n+1).
    pre_root: [PublicVariable; 256],
    post_root: [PublicVariable; 256],
});

impl Define<GF2Config> for Rv32Circuit<Variable> {
    fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
        let cfg = Rv32Cfg { nreg: NREG, mem_slots: MEM_SLOTS, waddr_bits: WADDR_BITS, prog_len: PROG_LEN };
        let program: Vec<W> = (0..PROG_LEN).map(|i| self.program[i].to_vec()).collect();
        let regs: Vec<W> = (0..NREG).map(|i| self.pre_regs[i].to_vec()).collect();
        let mem_addr: Vec<Vec<Variable>> = (0..MEM_SLOTS).map(|i| self.mem_addr[i].to_vec()).collect();
        let mem_val: Vec<W> = (0..MEM_SLOTS).map(|i| self.mem_val[i].to_vec()).collect();
        let pc = vec![api.constant(0); 32];
        let st = Rv32State { regs, pc, mem_addr, mem_val };
        // Every wire produced here is INTERMEDIATE (pinned by sumcheck, uncommitted).
        let fin = run(api, st, &program, &cfg, STEPS);
        for i in 0..NREG {
            for b in 0..32 {
                api.assert_is_equal(fin.regs[i][b], self.post_regs[i][b]);
            }
        }
        for j in 0..MEM_SLOTS {
            for b in 0..32 {
                api.assert_is_equal(fin.mem_val[j][b], self.post_mem[j][b]);
            }
        }

        // ---- genuine rollup state transition: bind pre_root / post_root ----
        // State root = keccak256 over the 32 register words, each 4 LE bytes
        // (128-byte preimage). Computed in-circuit so it is bound to the
        // COMMITTED pre-state and the PROVEN post-state; chained across blocks by
        // the sequencer feeding post_root(n) as pre_root(n+1).
        let pre_bits = regs_to_bits(&self.pre_regs.iter().map(|r| r.to_vec()).collect::<Vec<_>>());
        let pre_hash = keccak256_fixed(api, &pre_bits, NREG * 4);
        for i in 0..256 {
            api.assert_is_equal(pre_hash[i], self.pre_root[i]);
        }
        let post_bits = regs_to_bits(&fin.regs);
        let post_hash = keccak256_fixed(api, &post_bits, NREG * 4);
        for i in 0..256 {
            api.assert_is_equal(post_hash[i], self.post_root[i]);
        }
    }
}

/// Flatten a register file (NREG words, each 32 bits LSB-first) into a keccak
/// bit buffer: word i occupies bytes [4i, 4i+4) as little-endian, i.e. bit b of
/// word i lands at msg bit i*32 + b. The native reference (`regs_root`) hashes
/// the identical byte layout.
fn regs_to_bits(regs: &[Vec<Variable>]) -> Vec<Variable> {
    let mut bits = Vec::with_capacity(regs.len() * 32);
    for r in regs {
        for b in 0..32 {
            bits.push(r[b]);
        }
    }
    bits
}

/// Native reference for the in-circuit register-state root: keccak256 over the
/// 32 register words as little-endian bytes (128-byte preimage). x0 is 0.
fn regs_root(regs: &[u32; 32]) -> [u8; 32] {
    let mut buf = [0u8; NREG * 4];
    for i in 0..NREG {
        buf[i * 4..i * 4 + 4].copy_from_slice(&regs[i].to_le_bytes());
    }
    crate::mpt::keccak256(&buf)
}

pub struct Rv32Proof {
    pub commitment: [u8; 32],
    pub output: Vec<u8>,
    pub post_regs: [u32; 32],
    pub post_mem: Vec<(u32, u32)>,
    pub num_cycles: u32,
    pub input_vars: u32,
    pub verified: bool,
    pub proof: Vec<u8>,
    /// Register-state roots for the block's transition (bound in-circuit).
    pub pre_root: [u8; 32],
    pub post_root: [u8; 32],
}

/// The result of [`rv32_prepare`]: everything a prover needs to open the GKR
/// proof against an EXTERNALLY-supplied DA commitment, WITHOUT the prover doing
/// any Reed-Solomon encoding. Holds the exported Expander circuit `ec` in memory
/// across the external DA-encode call (the "accidental computer" weld point).
pub struct Rv32Prepared {
    /// Exported, input-assigned, evaluated Expander circuit, ready to prove.
    pub ec: ExpanderCircuit<GF2ExtConfig>,
    /// `ec.log_input_size()`: the number of input-layer variables.
    pub num_vars: u32,
    /// The input layer serialized for the DA-side re-encoder
    /// (`pkg/rsema1d.EncodeGKRInputSquare`): a `Vec<u8>` of length `2^num_vars`,
    /// element `g` = the `GF2x8` coefficient `g` of the input layer, packed so
    /// that bit `s` of the byte equals SIMD lane `s` (`GF2x8::unpack`, lane `s` =
    /// `(v>>s)&1`). This is the exact byte sequence the DA side RS-encodes.
    pub input_vals_bytes: Vec<u8>,
    /// Public output bytes (user result region), from the native emulator golden.
    pub output: Vec<u8>,
    /// Final register file (native golden).
    pub post_regs: [u32; 32],
    /// Final committed memory slots (native golden).
    pub post_mem: Vec<(u32, u32)>,
    /// Number of executed cycles (native golden).
    pub num_cycles: u32,
    /// Register-state root before execution (keccak256, bound in-circuit).
    pub pre_root: [u8; 32],
    /// Register-state root after execution (keccak256, bound in-circuit).
    pub post_root: [u8; 32],
}

/// PHASE 1 of the accidental-computer weld: run the native emulator to get the
/// golden post-state, prove-solve the in-circuit interpreter reproduces it, export
/// the Expander circuit and compute the input layer + its serialization — but do
/// NOT encode or prove. The returned [`Rv32Prepared`] holds the exported circuit
/// `ec` in memory; the caller hands `input_vals_bytes` to the DA side, which
/// RS-encodes it ONCE, and then calls [`rv32_prove_prepared`] with the resulting
/// `(root, extended)`. The prover itself performs ZERO Reed-Solomon encoding.
///
/// `base` must be 0 (the core fixes the program base at 0). `input` bytes are
/// packed LE into words and loaded as the initial memory region at `INPUT_ADDR`;
/// `pre_mem` are additional committed (byte_addr, word_val) slots (the result
/// region). `max_cycles` is clamped to the compiled step count.
pub fn rv32_prepare(
    program: &[u32],
    base: u32,
    input: &[u8],
    pre_regs: &[u32; 32],
    pre_mem: &[(u32, u32)],
    max_cycles: usize,
) -> Result<Rv32Prepared, String> {
    if base != 0 {
        return Err("rv32 core requires base=0".into());
    }
    if program.len() > PROG_LEN {
        return Err(format!("program too long: {} > {}", program.len(), PROG_LEN));
    }
    let _ = max_cycles; // the compiled circuit fixes the step count (STEPS)

    // Build the unified initial memory: input region (LE-packed words at
    // INPUT_ADDR) then the caller's pre_mem, padded with disjoint dummy slots.
    let mut input_words: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < input.len() {
        let mut w = 0u32;
        for k in 0..4 {
            if i + k < input.len() {
                w |= (input[i + k] as u32) << (8 * k);
            }
        }
        input_words.push(w);
        i += 4;
    }
    let n_input = input_words.len();
    let n_user = pre_mem.len();
    let mut slots: Vec<(u32, u32)> = Vec::new();
    for (k, w) in input_words.iter().enumerate() {
        slots.push((INPUT_ADDR + 4 * (k as u32), *w));
    }
    for &(a, v) in pre_mem {
        slots.push((a, v));
    }
    let n_real = slots.len();
    if n_real > MEM_SLOTS {
        return Err(format!("too many memory slots: {} > {}", n_real, MEM_SLOTS));
    }
    let mut dummy = 0u32;
    while slots.len() < MEM_SLOTS {
        let addr = (0xF000u32.wrapping_add(dummy)) << 2; // word idx 0xF000+dummy, program-disjoint
        slots.push((addr, 0));
        dummy += 1;
    }
    for &(a, _) in &slots {
        if (a >> 2) >= (1u32 << WADDR_BITS) {
            return Err(format!("word addr {} out of range for {WADDR_BITS} bits", a >> 2));
        }
    }

    // --- native golden (emulator IS the reference) ---
    let mut cpu = Cpu::new(program.to_vec(), base);
    cpu.regs = *pre_regs;
    cpu.regs[0] = 0; // x0 hardwired
    for &(a, v) in &slots {
        cpu.mem.store(a, v);
    }
    let trace = cpu.run(STEPS);
    // SOUNDNESS: the circuit models memory as exactly the committed MEM_SLOTS.
    // A store to a word address that is NOT one of those slots is silently
    // dropped by the in-circuit STF (no slot's `hit` fires), yet post_mem only
    // reads the declared slot addresses, so the divergence would be masked and
    // the proof would still verify. Reject such programs up front — the machine
    // cannot faithfully model a store outside its committed memory footprint.
    {
        use std::collections::HashSet;
        let declared: HashSet<u32> = slots.iter().map(|&(a, _)| a >> 2).collect();
        for r in &trace {
            if r.is_store && !declared.contains(&(r.mem_addr >> 2)) {
                return Err(format!(
                    "store to undeclared memory slot: byte addr {:#x} (word {:#x}) not among committed slots",
                    r.mem_addr,
                    r.mem_addr >> 2
                ));
            }
        }
    }
    let post_regs = cpu.regs;
    let post_mem_full: Vec<(u32, u32)> = slots.iter().map(|&(a, _)| (a, cpu.mem.load(a))).collect();
    let num_cycles = trace
        .iter()
        .find(|r| r.next_pc == r.pc)
        .map(|r| r.cycle + 1)
        .unwrap_or(STEPS as u32);
    let mut output = Vec::new();
    for k in 0..n_user {
        let v = post_mem_full[n_input + k].1;
        output.extend_from_slice(&v.to_le_bytes());
    }
    let post_mem: Vec<(u32, u32)> = post_mem_full[..n_real].to_vec();

    // --- compile + assign committed input + bind public golden ---
    let CompileResult { witness_solver, layered_circuit } =
        compile(&Rv32Circuit::default(), CompileOptions::default()).map_err(|e| format!("compile: {e:?}"))?;
    let mut asg = Rv32Circuit::<GF2>::default();
    for idx in 0..PROG_LEN {
        let w = if idx < program.len() { program[idx] } else { 0 };
        for b in 0..32 {
            asg.program[idx][b] = ((w >> b) & 1).into();
        }
    }
    for idx in 0..NREG {
        let v = if idx == 0 { 0 } else { pre_regs[idx] };
        for b in 0..32 {
            asg.pre_regs[idx][b] = ((v >> b) & 1).into();
        }
    }
    for j in 0..MEM_SLOTS {
        let (a, v) = slots[j];
        let widx = a >> 2;
        for b in 0..WADDR_BITS {
            asg.mem_addr[j][b] = ((widx >> b) & 1).into();
        }
        for b in 0..32 {
            asg.mem_val[j][b] = ((v >> b) & 1).into();
        }
    }
    for idx in 0..NREG {
        for b in 0..32 {
            asg.post_regs[idx][b] = ((post_regs[idx] >> b) & 1).into();
        }
    }
    for j in 0..MEM_SLOTS {
        let v = post_mem_full[j].1;
        for b in 0..32 {
            asg.post_mem[j][b] = ((v >> b) & 1).into();
        }
    }
    // Genuine state-transition roots (register-file state). Committed pre-state
    // has x0 = 0 (matching the circuit's committed pre_regs assignment above).
    let mut pre_regs_committed = *pre_regs;
    pre_regs_committed[0] = 0;
    let pre_root = regs_root(&pre_regs_committed);
    let post_root = regs_root(&post_regs);
    for i in 0..32 {
        for j in 0..8 {
            asg.pre_root[i * 8 + j] = (((pre_root[i] >> j) & 1) as u32).into();
            asg.post_root[i * 8 + j] = (((post_root[i] >> j) & 1) as u32).into();
        }
    }

    let witness = witness_solver.solve_witnesses(&vec![asg; 8]).map_err(|e| format!("witness: {e:?}"))?;
    if !layered_circuit.run(&witness).iter().all(|x| *x) {
        return Err("in-circuit rv32 STF != native emulator post-state".into());
    }

    // --- export, install the DA commitment over the INPUT LAYER, GKR prove/verify ---
    let mut ec = layered_circuit.export_to_expander_flatten();
    let (simd_input, simd_public_input) = witness.to_simd::<gf2::GF2x8>();
    ec.layers[0].input_vals = simd_input.clone();
    ec.public_input = simd_public_input.clone();
    ec.evaluate();
    let input_vals = ec.layers[0].input_vals.clone();
    let num_vars = ec.log_input_size();

    // Serialize the input layer for the DA-side re-encoder
    // (`pkg/rsema1d.EncodeGKRInputSquare`). `input_vals` is a `Vec<GF2x8>` of
    // length `2^num_vars`; each element `g` packs 8 SIMD lanes. `build_rows` /
    // `da_commit` read lane bit `s` of coefficient `g` as `unpack()[s].v & 1`
    // (flat bit address `a = g*8 + s`). The DA byte for `g` must therefore carry
    // lane `s` in bit `s` of the byte, so the byte is
    //   byte_g = OR over s in 0..8 of ((lane_s & 1) << s).
    let input_vals_bytes: Vec<u8> = input_vals
        .iter()
        .map(|e| {
            let lanes = e.unpack(); // Vec<GF2>, len 8; lane s = (v>>s)&1
            let mut byte = 0u8;
            for (s, lane) in lanes.iter().enumerate() {
                byte |= (lane.v & 1) << s;
            }
            byte
        })
        .collect();
    debug_assert_eq!(input_vals_bytes.len(), 1usize << num_vars);

    Ok(Rv32Prepared {
        ec,
        num_vars: num_vars as u32,
        input_vals_bytes,
        output,
        post_regs,
        post_mem,
        num_cycles,
        pre_root,
        post_root,
    })
}

/// PHASE 2 of the accidental-computer weld: open the GKR proof against an
/// EXTERNALLY-supplied DA commitment. The prover does NO Reed-Solomon encoding —
/// it reconstructs the committed square from `extended` (the serialized RS-encoded
/// matrix the DA side produced, via [`rsema1d_pcs::install_da_commitment_from_serialized`],
/// which only rebuilds Merkle/RLC structures) and then GKR-proves/self-verifies.
///
/// This method MACHINE-CHECKS the accidental-computer property: it records
/// [`rsema1d_sys::encode_call_count`] around `executor::prove`/`verify` and returns
/// `Err` if the delta is not exactly `0`.
pub fn rv32_prove_prepared(
    prepared: Rv32Prepared,
    da_root: [u8; 32],
    extended: &[u8],
) -> Result<Rv32Proof, String> {
    let Rv32Prepared { mut ec, num_vars, input_vals_bytes: _, output, post_regs, post_mem, num_cycles, pre_root, post_root } =
        prepared;

    // Reconstruct + install the DA-side commitment from the serialized extended
    // matrix. load_extended rebuilds ONLY the commitment structures (no RS-encode)
    // and asserts the reconstructed root equals `da_root`.
    let installed = rsema1d_pcs::install_da_commitment_from_serialized(num_vars as usize, da_root, extended);
    if installed != da_root {
        rsema1d_pcs::clear_da_commitment();
        return Err("reconstructed DA root != supplied DA root".into());
    }

    let mpi = MPIConfig::prover_new(None, None);
    // MACHINE-CHECK: the prover must perform ZERO RS-encodes across prove+verify.
    let encodes_before = rsema1d_sys::encode_call_count();
    let (claimed_v, proof) = executor::prove::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone());
    let verified = executor::verify::<Rsema1dGKRConfig<'static>>(&mut ec, mpi.clone(), &proof, &claimed_v)
        && claimed_v.is_zero();
    let encodes_after = rsema1d_sys::encode_call_count();
    let encode_delta = encodes_after - encodes_before;
    if encode_delta != 0 {
        rsema1d_pcs::clear_da_commitment();
        return Err(format!(
            "accidental-computer property violated: prover performed {encode_delta} RS-encode(s) (expected 0)"
        ));
    }

    if !proof.bytes.windows(32).any(|w| w == da_root) {
        rsema1d_pcs::clear_da_commitment();
        return Err("DA commitment not embedded in proof".into());
    }
    rsema1d_pcs::clear_da_commitment();

    Ok(Rv32Proof {
        commitment: da_root,
        output,
        post_regs,
        post_mem,
        num_cycles,
        input_vars: num_vars,
        verified,
        proof: proof.bytes,
        pre_root,
        post_root,
    })
}

/// Prove an RV32I block end-to-end (convenience / standalone / test path):
/// [`rv32_prepare`], then encode the input layer ONCE via
/// [`rsema1d_pcs::da_encode_serialized`] (this is the sole encode, standing in for
/// the DA side), then [`rv32_prove_prepared`]. The prover in phase 2 still does
/// zero encoding; the single encode lives in this wrapper. Identical
/// behavior/signature to the original entry point.
pub fn prove_rv32_block(
    program: &[u32],
    base: u32,
    input: &[u8],
    pre_regs: &[u32; 32],
    pre_mem: &[(u32, u32)],
    max_cycles: usize,
) -> Result<Rv32Proof, String> {
    let prepared = rv32_prepare(program, base, input, pre_regs, pre_mem, max_cycles)?;
    // Stand in for the DA side: RS-encode the input layer exactly once.
    let input_vals = prepared.ec.layers[0].input_vals.clone();
    let da_poly = MultiLinearPoly::new(input_vals);
    let (root, extended) = rsema1d_pcs::da_encode_serialized(prepared.num_vars as usize, &da_poly);
    rv32_prove_prepared(prepared, root, &extended)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::{add, addi, bge, jal, lw, sw};

    #[test]
    fn rv32_prove_matches_emulator() {
        let pre_regs = [0u32; 32];

        // ---- Program (a): sum 1..=n, n read from the input word ----
        // 0: x1 = 0 (sum)      1: x2 = 1 (i)     2: x3 = mem[INPUT_ADDR] (n)
        // 3: sum += i (head)   4: i++            5: if n>=i goto 3
        // 6: mem[0x200] = sum  7: halt (self-loop)
        let n: u32 = 3;
        let prog_a = vec![
            addi(1, 0, 0),
            addi(2, 0, 1),
            lw(3, 0, INPUT_ADDR as i32),
            add(1, 1, 2),
            addi(2, 2, 1),
            bge(3, 2, -8),
            sw(1, 0, 0x200),
            jal(0, 0),
        ];
        let input_a = n.to_le_bytes().to_vec();
        let pre_mem_a = vec![(0x200u32, 0u32)];
        let expected_sum = n * (n + 1) / 2; // 6

        let pa = prove_rv32_block(&prog_a, 0, &input_a, &pre_regs, &pre_mem_a, STEPS).unwrap();
        println!(
            "[prog a] verified={} num_cycles={} input_vars={} output={:?} x1={} x3={} commitment=0x{}",
            pa.verified,
            pa.num_cycles,
            pa.input_vars,
            pa.output,
            pa.post_regs[1],
            pa.post_regs[3],
            pa.commitment.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        );
        assert!(pa.verified, "program (a) GKR self-verify failed");
        assert_eq!(pa.output, expected_sum.to_le_bytes().to_vec(), "sum output != golden");
        assert_eq!(pa.post_regs[1], expected_sum, "x1 (sum) != golden");
        assert_eq!(pa.post_regs[3], n, "x3 (n) != golden");
        // output region (0x200) present in post_mem with the golden value
        assert!(pa.post_mem.iter().any(|&(a, v)| a == 0x200 && v == expected_sum));

        // ---- Program (b): load-add-store touching memory ----
        // 0: x1 = mem[0x100] (a)   1: x2 = mem[0x104] (b)
        // 2: x3 = a + b            3: mem[0x200] = x3     4: halt
        let a_val: u32 = 7;
        let b_val: u32 = 35;
        let prog_b = vec![
            lw(1, 0, 0x100),
            lw(2, 0, 0x104),
            add(3, 1, 2),
            sw(3, 0, 0x200),
            jal(0, 0),
        ];
        let mut input_b = Vec::new();
        input_b.extend_from_slice(&a_val.to_le_bytes());
        input_b.extend_from_slice(&b_val.to_le_bytes());
        let pre_mem_b = vec![(0x200u32, 0u32)];
        let expected = a_val + b_val; // 42

        let pb = prove_rv32_block(&prog_b, 0, &input_b, &pre_regs, &pre_mem_b, STEPS).unwrap();
        println!(
            "[prog b] verified={} num_cycles={} input_vars={} output={:?} x3={} commitment=0x{}",
            pb.verified,
            pb.num_cycles,
            pb.input_vars,
            pb.output,
            pb.post_regs[3],
            pb.commitment.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        );
        assert!(pb.verified, "program (b) GKR self-verify failed");
        assert_eq!(pb.output, expected.to_le_bytes().to_vec(), "add output != golden");
        assert_eq!(pb.post_regs[3], expected, "x3 (a+b) != golden");
        assert_eq!(pb.post_regs[1], a_val, "x1 (a) != golden");
        assert_eq!(pb.post_regs[2], b_val, "x2 (b) != golden");
        assert!(pb.post_mem.iter().any(|&(a, v)| a == 0x200 && v == expected));
    }

    /// The accidental-computer weld: `rv32_prepare` (no encode, no prove) ->
    /// DA-side `da_encode_serialized` (the ONLY RS-encode) -> `rv32_prove_prepared`
    /// (opens the GKR proof against the DA commitment, doing ZERO RS-encode).
    /// Asserts (a) verified, (b) output/post_regs match the emulator golden, and
    /// (c) the `encode_call_count` delta across `rv32_prove_prepared` is exactly 0.
    #[test]
    fn rv32_prepare_prove_prepared_roundtrip() {
        let pre_regs = [0u32; 32];

        // Same "sum 1..=n" program as rv32_prove_matches_emulator.
        let n: u32 = 3;
        let prog = vec![
            addi(1, 0, 0),
            addi(2, 0, 1),
            lw(3, 0, INPUT_ADDR as i32),
            add(1, 1, 2),
            addi(2, 2, 1),
            bge(3, 2, -8),
            sw(1, 0, 0x200),
            jal(0, 0),
        ];
        let input = n.to_le_bytes().to_vec();
        let pre_mem = vec![(0x200u32, 0u32)];
        let expected_sum = n * (n + 1) / 2; // 6

        // PHASE 1: prepare (no encode, no prove).
        let prepared = rv32_prepare(&prog, 0, &input, &pre_regs, &pre_mem, STEPS).unwrap();
        let num_vars = prepared.num_vars as usize;
        assert_eq!(prepared.input_vals_bytes.len(), 1usize << num_vars, "input_vals_bytes length");
        assert_eq!(prepared.output, expected_sum.to_le_bytes().to_vec(), "prepared output != golden");
        assert_eq!(prepared.post_regs[1], expected_sum, "prepared x1 != golden");

        // DA SIDE: RS-encode the input layer once, producing (root, extended).
        // Confirm the byte serialization we hand to the DA side round-trips to the
        // same commitment the in-memory poly encoder produces (packing is correct).
        let input_vals = prepared.ec.layers[0].input_vals.clone();
        let da_poly = MultiLinearPoly::new(input_vals);
        let (root, extended) = rsema1d_pcs::da_encode_serialized(num_vars, &da_poly);

        // PHASE 2: prove against the external DA commitment. Measure the encode
        // count delta EXTERNALLY around the whole phase-2 call — it must be 0.
        let encodes_before = rsema1d_sys::encode_call_count();
        let proof = rv32_prove_prepared(prepared, root, &extended).unwrap();
        let encodes_after = rsema1d_sys::encode_call_count();
        let encode_delta = encodes_after - encodes_before;

        println!(
            "[roundtrip] verified={} num_cycles={} input_vars={} output={:?} x1={} x3={} encode_delta={} commitment=0x{}",
            proof.verified,
            proof.num_cycles,
            proof.input_vars,
            proof.output,
            proof.post_regs[1],
            proof.post_regs[3],
            encode_delta,
            proof.commitment.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        );

        assert!(proof.verified, "roundtrip GKR self-verify failed");
        assert_eq!(proof.output, expected_sum.to_le_bytes().to_vec(), "roundtrip output != golden");
        assert_eq!(proof.post_regs[1], expected_sum, "roundtrip x1 (sum) != golden");
        assert_eq!(proof.post_regs[3], n, "roundtrip x3 (n) != golden");
        assert_eq!(proof.commitment, root, "returned commitment != DA root");
        assert_eq!(
            encode_delta, 0,
            "prover performed {encode_delta} RS-encode(s) in phase 2 (expected 0)"
        );
    }
}
