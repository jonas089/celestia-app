# ev-reth rollup — real-time block-proof dashboard

A small React + Vite dashboard showing a **single rollup** (namespace
`ev-reth-rollup`) and its **live, per-block proof feed**. Every block's real EVM
state transition is proven as a span of RV32IM chunks over the guest ELF
(`prove_elf`), reusing the **rsema1d DA encoding** as the GKR input commitment
(byte-identical to Go/DA). Everything displayed comes from real accProof backend
records — nothing is mocked or hardcoded.

## What it shows

- A header with the rollup name, namespace + DA namespace, and live summary
  stats (blocks / proved / verified / golden-composed) plus a connection pill.
- A **per-block card feed**, newest first, that polls `/api/blockproofs` every
  ~2s and updates live. Each card shows:
  - height, EVM block number, and an animated status badge that transitions
    `queued → proving (spinner) → proved (green)` / `failed (red)`;
  - the **tx composition** — total txs and the breakdown of normal transfers,
    contract deployments, and contract write-calls (deployments are flagged);
  - proof metrics — state root, the rsema1d commitment with a **`== DA`**
    indicator (go-match), `#chunks (verified / go-match / chain-valid)`, the
    composed flag, and proof size / prove time.
- Per-block **Verify** and **Get Proof** buttons (on every block). Verify shows
  a checklist (Expander-verified + commitment `== DA` + chain-valid + composed)
  from a per-block `/api/blockproof?ns=..&height=..` call; Get Proof shows the
  commitment, state root, chunk count, and proof size, and downloads the
  verified proof metadata as JSON.

## Run it

```bash
pnpm install
pnpm dev
# open http://localhost:3000
```

The Go API server (`test/accidental-computer/api`) must be running on `:8088`,
and the accProof backend (ev-reth node or the `accproof-serve` harness) on
`:8545`:

```bash
cd ../api && go run .
```

Or launch the whole UI (API + dashboard) via the repo Makefile: `make ui-up`
(stop with `make ui-down`).

## Configuration

| Env var         | Default                 | Purpose                    |
| --------------- | ----------------------- | -------------------------- |
| `VITE_API_BASE` | `http://localhost:8088` | Base URL of the Go API.    |

The Go API also honors `ROLLUP_NS` (default `ev-reth-rollup`) to select the
single rollup namespace it surfaces, and `ACCPROOF_RPC` (default
`http://localhost:8545`) for the accProof backend. The backend prunes block
records older than `ACCPROOF_PRUNE_SECS` (default `3600`, ~1 hour).

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
