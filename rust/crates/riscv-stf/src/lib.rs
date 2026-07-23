//! RV32IM CPU-verifier proven over Expander GKR with rsema1d as the input PCS.
//!
//! Library surface for embedding the prover in a C-ABI shared library
//! (`libaccprover`) and, through it, the ev-reth `accProof` RPC. The two binaries
//! (`riscv_pipeline`, `riscv_pipeline_full`) keep their own module trees and are
//! unaffected. The proven circuit (`circuit_full`) is unchanged.

pub mod circuit_elf;
pub mod circuit_full;
pub mod circuit_gp;
pub mod prove_elf;
pub mod circuit_stf;
// Input-committed STF executor (the correct accidental-computer shape): commits
// ONLY the block data, computes the transition as internal wires.
pub mod circuit_ac;
pub mod stf_ac;
// R4: data-parallel batch keccak over the committed DA block data.
pub mod batch_keccak;
// R2: deterministic (hint-free) 256-bit modular arithmetic over GF2 (secp256k1).
pub mod u256;
// R5: secp256k1 + ECDSA ecrecover (native reference; circuit layer builds on u256).
pub mod secp256k1;
// R6: EIP-1559 tx RLP (native reference; circuit parser builds on this layout).
pub mod rlp;
// R7: Ethereum world-state MPT root (native reference; circuit MPT builds on this).
pub mod mpt;
// R7: value-transfer state transition + post-state root (native reference).
pub mod stf_transfer;
// R7: in-circuit Ethereum MPT primitives (minimal-RLP, node hashing).
pub mod mpt_circuit;
// R7c: in-circuit MPT inclusion + update gadget (parent root + witness -> post root).
pub mod mpt_inclusion;
// R7 assembly: rsema1d-committed transfer STF proof (committed DA data -> post_state_root).
pub mod stf_prove;
// R8: full EVM interpreter (native semantics reference; circuit step-fn builds on this).
pub mod evm;
// R8: in-circuit EVM step function (GF2), matching the native interpreter.
pub mod evm_circuit;
pub mod evm_gkr;
pub mod block_stf;
// R7d: REAL dense multi-account MPT state-root transition (chained account updates
// against a consistent post-trie) + ev-reth RealBlockData parsing.
pub mod block_real;
pub mod emulator;
// RV32I "accidental computer" (CORRECT GKR regime): committed input = program +
// pre-state; the per-cycle trace is intermediate. Mirrors evm_circuit/block_stf.
pub mod rv32_circuit;
pub mod rv32_prove;
#[cfg(test)]
mod rv32_circuit_tests;
pub mod evm_asm;
// The RV32 proof's native EVM oracle IS the exact evm-core source that evm-xcheck
// proves byte-identical to revm 26.0.1 (parent workspace). #[path] points the
// module straight at that file — drift-free, no copy.
#[path = "../../evm-core/src/lib.rs"]
pub mod evm_core;
pub mod evm_prove;
pub mod evm_rv32;
pub mod gf128;
pub mod loader;
pub mod prove;
pub mod stf;

pub use evm_prove::{prove_evm, EvmProof};
pub use prove::{prove_execution, AccProof};
pub use stf::{
    apply_native, prove_block_stf, BlockInput, BlockStfProof, StfOutcome, TxData, MAXTX_TXS,
};
