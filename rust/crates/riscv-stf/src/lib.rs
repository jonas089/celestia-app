//! RV32IM "accidental computer" prover: prove committed rv32i execution over
//! Expander GKR with rsema1d as the sole (reused, DA-side) input polynomial
//! commitment. The per-cycle execution trace is intermediate (never committed);
//! only the input layer (program + pre-state) is committed, and its commitment
//! IS the on-DA rsema1d commitment. The proof binds an in-circuit state root so
//! each block attests `pre_root -> post_root` like a rollup.

pub mod batch_keccak;
pub mod emulator;
pub mod rv32_circuit;
pub mod rv32_prove;
pub mod u256;

#[cfg(test)]
mod rv32_circuit_tests;
