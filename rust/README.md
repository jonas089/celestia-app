# The Accidental Computer — reviewer's map

This tree welds a **Celestia data-availability (rsema1d) commitment** to a **GKR
proof of RV32I execution**: the DA commitment over a block's data is reused as the
*sole* polynomial commitment for the proof, so the prover **re-encodes nothing**.
That is the "accidental computer": the DA layer's tensor encoding already commits
the block, and GKR opens against exactly that commitment.

This document is a map for reviewing soundness and completeness. Read the four
crates in this order; each is small and single-purpose.

## Crate map (`rust/crates/`)

| Crate | Responsibility | Key files |
|-------|----------------|-----------|
| **`riscv-stf`** | The RV32I VM as an in-circuit state-transition function + the GKR proving weld | `rv32_circuit.rs`, `emulator.rs`, `rv32_prove.rs`, `u256.rs`, `batch_keccak.rs` |
| **`rsema1d-pcs`** | Adapts the rsema1d DA commitment to Expander's `ExpanderPCS<GF2ExtConfig>` (commit / open / verify), and the "install a DA-side commitment" hand-off | `lib.rs` |
| **`rsema1d-sys`** | Thin FFI to the Go `pkg/rsema1d` encoder (the DA side) | `lib.rs`, `build.rs` |
| **`rv32-rollup`** | The sovereign rollup node: seals blocks, runs the fibre DA upload, opens the proof, serves the explorer RPC | `main.rs`, `programs/txproc.rs` |

Third-party, unmodified, vendored only for a pinned build: `rust/vendor/expander`
(the Polyhedra Expander GKR prover). It contains **none** of our code — the
integration lives entirely in `rsema1d-pcs`.

## 1. Where the VM is implemented — `riscv-stf/src/rv32_circuit.rs`

`step()` is one RV32I instruction as a circuit over GF(2). Every instruction in the
base integer ISA is decoded and executed here; the same file is the single source
of truth for "how each opcode is constrained."

- **Decode** (fetch → opcode/rd/funct3/rs1/rs2/funct7 + the five immediate forms)
  mirrors the RV32I encoding tables one-to-one.
- **Execute**: the ALU (`add/sub/sll/slt/sltu/xor/srl/sra/or/and` and their `*i`
  immediate forms), `lui/auipc`, loads/stores (all widths, sign/zero extension),
  branches (`beq/bne/blt/bge/bltu/bgeu`), and `jal/jalr`. Each result is computed
  and selected by its opcode/funct3 one-hot — a full-ISA multiplexer, so every
  cycle is sound regardless of which instruction ran.
- **Arithmetic gadgets** are in `u256.rs` (ripple/`carry-save` adders, `lt`, `eq`,
  `select`) — the primitives the ECC frontend lowers to GF(2) gates.
- `run()` threads the machine state (`regs`, `pc`, bounded word-memory) across
  `n_steps` unrolled `step`s; `x0` is hardwired to 0; halted programs idle.

**Native reference**: `emulator.rs` is a plain Rust RV32I interpreter used as the
golden model. Every proof asserts the in-circuit post-state equals the emulator's,
so correctness is differential (`rv32_circuit_tests.rs`, one test per opcode group).

**State root**: `batch_keccak.rs` computes keccak-256 in-circuit over the rollup
state region; `rv32_prove.rs` binds `pre_root → post_root` so a block is a genuine
verifiable state transition.

## 2. The rsema1d encoding pipeline — `rsema1d-pcs/src/lib.rs` (+ `pkg/rsema1d`, Go)

This is the "accidental" part. The commitment is a Reed–Solomon / tensor encoding
of the block's committed input, produced **once on the DA side** and reused:

- `pkg/rsema1d` (Go) encodes the input square and produces the Merkle/RLC
  commitment — this is what Celestia stores. `rsema1d-sys` is the FFI to it.
- `rsema1d-pcs` implements Expander's `ExpanderPCS`: `commit` builds the same
  square, `open` produces sampled-row openings, `verify` checks them. The DA-side
  commitment is installed via `install_da_commitment_from_serialized` and the
  prover opens against it.
- Soundness hook: `rsema1d_sys::encode_call_count()` is asserted to increase by
  **exactly zero** across `prove`+`verify` (`rv32_prove.rs`) — a machine-check
  that the prover reused the DA commitment and never re-encoded.

## 3. GKR proof generation — `riscv-stf/src/rv32_prove.rs`

The weld, in two phases so the DA encode can happen out of process:

1. **`rv32_prepare`** — run the emulator (golden), assign the in-circuit witness
   (`build_block`), compile the layered circuit (cached process-wide; it is
   input-independent), solve the witness, export the Expander circuit, and
   serialize the input layer for the DA encoder. No encode, no prove.
2. **`rv32_prove_prepared`** — install the DA-side commitment, run Expander GKR
   `prove`/`verify`, and assert the zero-re-encode property and that the DA root
   is embedded in the proof.

`prove_rv32_block` chains both with a stand-in single encode (tests/standalone).
`rv32_prepare_batch` / `rv32_prove_prepared_batch` do the same for up to **8
sequentially-chained sub-blocks packed into the 8 GF2x8 SIMD lanes** — one proof,
~8× the transactions, same prover cost. A process-global reentrant lock serializes
the (global) DA-commitment critical section so proving is thread-safe.

## 4. The node — `rv32-rollup/src/main.rs`

Seals a block per interval: carries the persistent balance state, expands a
transfer submission into 8 chained lanes, runs the fibre upload
(`fibre_upload` → encode+settle `MsgPayForFibre` on Celestia), opens the batched
GKR proof against the settled commitment, and serves per-block records to the
explorer (`test/accidental-computer/dashboard`). `programs/txproc.rs` is the single
deployed contract (a signed balance-transfer STF), shown in the explorer with its
source and compiled rv32 bytecode.

## Running it

`make start` builds the celestia-app + rollup images and launches the full stack
(validator + fibre server + rollup + submitter + explorer on :3000). `make stop`
tears it down. See the top-level `Makefile`.
