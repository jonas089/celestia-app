# rv32i rollup — real-time block-proof dashboard

A small React + Vite dashboard showing a **single rollup** (namespace
`rv32-rollup`) and its **live, per-block proof feed**. Each block executes
committed rv32i over persistent VM state; the program, input and pre-state are
posted to Celestia DA, and the block is GKR-proven by reusing the **on-DA
rsema1d commitment** as the sole polynomial commitment (the prover re-encodes
nothing) while binding `pre_root → post_root` in-circuit. Everything displayed
comes from real accProof backend records — nothing is mocked or hardcoded.

## What it shows

- A header with the rollup name, namespace + DA namespace, live summary stats,
  and a connection pill.
- A **per-block card feed**, newest first, that polls `/api/blockproofs` every
  ~2s and updates live. Each card shows the block number, cycles executed, and
  an animated status badge (`queued → proving → proved` / `failed`).
- Expanding a card reveals the verification checklist (Expander GKR accepted the
  rv32i execution; the GKR input commitment is the reused rsema1d/DA encoding),
  the commitment and state root, and the compiled rv32i program disassembled
  into words.
- Per-block **Verify** and **Get Proof** actions, backed by
  `/api/blockproof?ns=..&height=..`.

## Run it

```bash
pnpm install
pnpm dev
# open http://localhost:3000
```

The Go API server (`test/accidental-computer/api`) must be running on `:8088`,
and the rv32-rollup's accProof RPC on `:8545`:

```bash
cd ../api && go run .
```

Or launch the whole UI (API + dashboard) via the repo Makefile: `make ui-up`
(stop with `make ui-down`). `make start` brings up the full stack and the UI.

## Configuration

| Env var         | Default                 | Purpose                    |
| --------------- | ----------------------- | -------------------------- |
| `VITE_API_BASE` | `http://localhost:8088` | Base URL of the Go API.    |

The Go API also honors `ROLLUP_NS` (default `rv32-rollup`) to select the rollup
namespace it surfaces, and `ACCPROOF_RPC` (default `http://localhost:8545`) for
the rollup's accProof RPC.

## Build

```bash
pnpm build      # outputs to dist/
pnpm preview    # serve the production build locally
```

## Notes

- The only network host contacted is `VITE_API_BASE`.
- Theme-aware: follows the viewer's light/dark preference.
- Dependencies are intentionally minimal: React + Vite, plain CSS, native
  `fetch`.
