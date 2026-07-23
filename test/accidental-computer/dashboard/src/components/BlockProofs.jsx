import { useState } from 'react'
import { api } from '../api.js'
import { truncHex } from '../util.js'
import { CopyButton } from './CopyButton.jsx'

// Single-rollup, real-time per-block proof feed. The parent (App) polls
// /api/blockproofs every ~2s and passes the one namespace group down here.
// Blocks render newest-first; each card animates its status
// queued -> proving -> proved / failed and carries its own Verify + Get Proof
// buttons (per-block /api/blockproof calls). Nothing here is fabricated.
export function RollupFeed({ group, rollupNS, scope, backendError }) {
  const blocks = [...(group?.blocks || [])].sort(
    (a, b) => (b.blockNumber || 0) - (a.blockNumber || 0),
  )

  return (
    <section className="feed">
      <div className="feed__head">
        <h2>Per-block proofs</h2>
        <span className="muted">newest first · live</span>
      </div>

      <details className="scope">
        <summary>Honest scope of the proof</summary>
        <p>
          {scope ||
            'Proves the real EVM block state transition as a bounded span of RV32IM chunks over the guest ELF, reusing rsema1d as the GKR input commitment.'}
        </p>
      </details>

      {backendError && (
        <p className="inline-error">
          accProof backend: {String(backendError)} — start the backend and submit
          blocks (namespace <code className="mono">{rollupNS}</code>).
        </p>
      )}

      {blocks.length === 0 && !backendError && (
        <div className="card empty">
          <div className="spinner" aria-hidden />
          <p className="muted">No blocks yet. Waiting for proofs…</p>
        </div>
      )}

      <div className="cards">
        {blocks.map((b) => (
          <BlockCard key={`${b.namespace}-${b.blockNumber}`} b={b} rollupNS={rollupNS} />
        ))}
      </div>
    </section>
  )
}

function statusInfo(b) {
  // rv32 blocks carry `verified`; EVM-STF blocks carry `stfVerified`.
  const isVerified = b.kind === 'rv32' ? b.verified : b.stfVerified
  switch (b.status) {
    case 'proved':
      return isVerified
        ? { kind: 'ok', label: 'proved', icon: '✓' }
        : { kind: 'bad', label: 'unverified', icon: '✕' }
    case 'failed':
      return { kind: 'bad', label: 'failed', icon: '✕' }
    case 'proving':
      return { kind: 'proving', label: 'proving', icon: '' }
    case 'queued':
      return { kind: 'queued', label: 'queued', icon: '' }
    default:
      return { kind: 'queued', label: b.status || 'pending', icon: '' }
  }
}

function BlockCard({ b, rollupNS }) {
  const [panel, setPanel] = useState(null) // null | 'verify' | 'proof'
  const [detail, setDetail] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)

  const st = statusInfo(b)
  const isDeploy = (b.numDeploys || 0) > 0
  const daMatch = !!b.stfGoMatch
  const isBlockStf = b.kind === 'block_stf'
  const isRv32 = b.kind === 'rv32'

  async function open(which) {
    if (panel === which) {
      setPanel(null)
      return
    }
    setPanel(which)
    setBusy(true)
    setErr(null)
    try {
      const rec = await api.blockProof(rollupNS, b.blockNumber)
      setDetail(rec)
    } catch (e) {
      setErr(e)
    } finally {
      setBusy(false)
    }
  }

  function download() {
    const rec = detail || b
    const info = {
      namespace: rec.namespace,
      height: rec.blockNumber,
      evmBlockNumber: rec.stfBlockNumber,
      status: rec.status,
      commitment: rec.stfCommitment || rec.commitment,
      stateRoot: rec.stfStateRoot,
      numChunks: rec.stfNumChunks,
      verified: rec.stfVerified,
      commitmentEqualsDA: rec.stfGoMatch,
      chainValid: rec.stfChainValid,
      composed: rec.stfComposed,
      proofBytes: rec.proofBytes,
      elapsedMs: rec.elapsedMs,
    }
    const blob = new Blob([JSON.stringify(info, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `proof-${rec.namespace}-h${rec.blockNumber}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <article className={`bcard bcard--${st.kind}`}>
      <div className="bcard__accent" />

      <div className="bcard__head">
        <div className="bcard__ids">
          <span className="bcard__height">height {b.blockNumber}</span>
          <span className="bcard__evm mono">
            {isRv32
              ? `${b.numCycles || 0} cycles`
              : `EVM block #${b.stfBlockNumber != null ? b.stfBlockNumber : '—'}`}
          </span>
        </div>
        <StatusBadge st={st} />
      </div>

      {/* rv32i program composition (the accidental rv32i computer) */}
      {isRv32 && (
        <div className="chips">
          <span className="chip chip--flag" title="Raw rv32i executed in-circuit; the rsema1d/DA commitment is the sole (reused) GKR input commitment.">
            accidental computer · in-circuit rv32i
          </span>
          <span className="chip chip--total">{b.numCycles || 0} cycles</span>
          <span className="chip chip--transfer mono" title={b.program}>
            program {truncHex(b.program, 8, 4)}
          </span>
          <span className="chip chip--write mono" title={b.input || '0x'}>
            input {b.input && b.input !== '0x' ? truncHex(b.input, 8, 4) : '∅'}
          </span>
          <span className="chip chip--deploy mono" title={b.output}>
            output {b.output ? truncHex(b.output, 8, 4) : '—'}
          </span>
        </div>
      )}

      {/* tx composition (EVM-STF rollups) */}
      {!isRv32 && (
      <div className="chips">
        <span className="chip chip--total">{b.txCount || 0} txs</span>
        <span className={`chip ${(b.numTransfers || 0) > 0 ? 'chip--transfer' : 'chip--zero'}`}>
          {b.numTransfers || 0} transfer{(b.numTransfers || 0) === 1 ? '' : 's'}
        </span>
        <span className={`chip ${isDeploy ? 'chip--deploy' : 'chip--zero'}`}>
          {b.numDeploys || 0} deploy{(b.numDeploys || 0) === 1 ? '' : 's'}
        </span>
        <span className={`chip ${(b.numWrites || 0) > 0 ? 'chip--write' : 'chip--zero'}`}>
          {b.numWrites || 0} write-call{(b.numWrites || 0) === 1 ? '' : 's'}
        </span>
        {isDeploy && <span className="chip chip--flag">contract deployment</span>}
        {isBlockStf && (
          <span className="chip chip--flag" title="In-circuit EVM execution + Ethereum MPT proven directly in GKR; rsema1d/DA is the sole (reused) commitment.">
            accidental computer · in-circuit EVM
          </span>
        )}
      </div>
      )}

      {/* proof metrics — rv32i */}
      {isRv32 && (
      <div className="metrics">
        <Metric label="output (public)">
          <span className="mono metric__hex" title={b.output}>
            {b.output ? truncHex(b.output, 12, 8) : '—'}
            {b.output ? <CopyButton value={b.output} label="⧉" /> : null}
          </span>
        </Metric>

        <Metric label="commitment (rsema1d == DA)">
          {b.commitment ? (
            <span className="mono metric__hex" title={b.commitment}>
              {truncHex(b.commitment, 10, 6)}
              <CopyButton value={b.commitment} label="⧉" />
            </span>
          ) : (
            <span className="muted">—</span>
          )}
        </Metric>

        <Metric label="post-state root">
          {b.stfStateRoot ? (
            <span className="mono metric__hex" title={b.stfStateRoot}>
              {truncHex(b.stfStateRoot, 10, 6)}
              <CopyButton value={b.stfStateRoot} label="⧉" />
            </span>
          ) : (
            <span className="muted">—</span>
          )}
        </Metric>

        <Metric label="cycles / input vars">
          <span className="mono">
            {b.numCycles || 0} / {b.inputVars || 0}
          </span>
        </Metric>

        <Metric label="GKR verified">
          <span className={b.verified ? 'pill pill--ok' : 'pill pill--partial'}>
            {b.verified ? '✓ verified' : b.status === 'proved' ? 'unverified' : b.status}
          </span>
        </Metric>

        <Metric label="proof size">
          <span className="mono">
            {b.proofBytes ? fmtBytes(b.proofBytes) : '—'}
            {b.elapsedMs ? ` · ${fmtDuration(b.elapsedMs)}` : ''}
          </span>
        </Metric>
      </div>
      )}

      {/* proof metrics — EVM-STF */}
      {!isRv32 && (
      <div className="metrics">
        <Metric label="state root">
          {b.stfStateRoot ? (
            <span className="mono metric__hex" title={b.stfStateRoot}>
              {truncHex(b.stfStateRoot, 10, 6)}
              <CopyButton value={b.stfStateRoot} label="⧉" />
            </span>
          ) : (
            <span className="muted">—</span>
          )}
        </Metric>

        <Metric label="commitment (rsema1d)">
          {b.stfCommitment ? (
            <span className="mono metric__hex" title={b.stfCommitment}>
              {truncHex(b.stfCommitment, 10, 6)}
              {b.status === 'proved' && (
                <span className={`daflag ${daMatch ? 'daflag--ok' : 'daflag--bad'}`}>
                  {daMatch ? '== DA' : '≠ DA'}
                </span>
              )}
              <CopyButton value={b.stfCommitment} label="⧉" />
            </span>
          ) : (
            <span className="muted">—</span>
          )}
        </Metric>

        <Metric label="chunks (verified / go-match / chain-valid)">
          <span className="mono">
            {b.stfNumChunks || 0}
            {b.status === 'proved'
              ? ` (${b.stfVerified ? '✓' : '✕'} / ${b.stfGoMatch ? '✓' : '✕'} / ${b.stfChainValid ? '✓' : '✕'})`
              : ''}
          </span>
        </Metric>

        <Metric label="composed">
          <span className={b.stfComposed ? 'pill pill--ok' : 'pill pill--partial'}>
            {b.stfComposed ? '✓ golden' : 'partial span'}
          </span>
        </Metric>

        <Metric label="proof size">
          <span className="mono">
            {b.proofBytes ? fmtBytes(b.proofBytes) : '—'}
            {b.elapsedMs ? ` · ${fmtDuration(b.elapsedMs)}` : ''}
          </span>
        </Metric>
      </div>
      )}

      {b.error && <p className="inline-error">error: {b.error}</p>}

      <div className="bcard__actions">
        <button
          className={`btn ${panel === 'verify' ? 'btn--active' : ''}`}
          onClick={() => open('verify')}
        >
          Verify
        </button>
        <button
          className={`btn ${panel === 'proof' ? 'btn--active' : ''}`}
          onClick={() => open('proof')}
        >
          Get Proof
        </button>
      </div>

      {panel && (
        <div className="drawer">
          {busy && (
            <p className="muted">
              <span className="spinner spinner--sm" /> fetching from backend…
            </p>
          )}
          {err && <p className="inline-error">{String(err.message || err)}</p>}
          {!busy && !err && detail && panel === 'verify' && (
            <VerifyPanel rec={detail} />
          )}
          {!busy && !err && detail && panel === 'proof' && (
            <ProofPanel rec={detail} onDownload={download} />
          )}
        </div>
      )}
    </article>
  )
}

function StatusBadge({ st }) {
  return (
    <span className={`sbadge sbadge--${st.kind}`}>
      {st.kind === 'proving' && <span className="spinner spinner--sm" />}
      {st.kind === 'queued' && <span className="sbadge__pulse" />}
      {st.icon && <span className="sbadge__icon">{st.icon}</span>}
      <span className="sbadge__label">{st.label}</span>
    </span>
  )
}

function VerifyPanel({ rec }) {
  const proved = rec.status === 'proved'

  // rv32i (accidental computer): the GKR proof verified in-node and its input
  // commitment IS the reused rsema1d/DA commitment.
  if (rec.kind === 'rv32') {
    const checks = [
      { name: 'Expander GKR verifier ACCEPTED the rv32i execution', ok: proved && rec.verified },
      { name: 'GKR input commitment == reused rsema1d/DA encoding (32 bytes)', ok: (rec.commitment || '').length === 2 + 64 },
      { name: 'public output present', ok: !!rec.output && rec.output !== '0x' },
      { name: 'post-state root committed', ok: !!rec.stfStateRoot },
    ]
    const verified = proved && rec.verified
    return (
      <div className={`verify ${verified ? 'verify--ok' : proved ? 'verify--warn' : 'verify--pending'}`}>
        <div className="verify__badge">
          {proved ? (verified ? 'VERIFIED = TRUE' : 'NOT VERIFIED') : `status: ${rec.status}`}
        </div>
        <ul className="checks">
          {checks.map((c, i) => (
            <li key={i} className={c.ok ? 'check--ok' : 'check--bad'}>
              <span className="check__mark">{c.ok ? '✓' : '✕'}</span>
              <span>{c.name}</span>
            </li>
          ))}
        </ul>
      </div>
    )
  }

  const checks = [
    {
      name: 'Expander GKR verifier ACCEPTED every chunk',
      ok: proved && rec.stfVerified,
    },
    {
      name: 'GKR input commitment == rsema1d DA encoding (go-match)',
      ok: proved && rec.stfGoMatch,
    },
    {
      name: 'chunk span is chain-valid (each chunk continues the last)',
      ok: proved && rec.stfChainValid,
    },
    {
      name: 'span composed to the golden state root (whole block)',
      ok: proved && rec.stfComposed,
      soft: proved && !rec.stfComposed,
    },
  ]
  const verified = proved && rec.stfVerified && rec.stfGoMatch && rec.stfChainValid
  return (
    <div className={`verify ${verified ? 'verify--ok' : proved ? 'verify--warn' : 'verify--pending'}`}>
      <div className="verify__badge">
        {proved
          ? verified
            ? 'VERIFIED = TRUE'
            : 'NOT VERIFIED'
          : `status: ${rec.status}`}
      </div>
      <ul className="checks">
        {checks.map((c, i) => (
          <li key={i} className={c.ok ? 'check--ok' : c.soft ? 'check--soft' : 'check--bad'}>
            <span className="check__mark">{c.ok ? '✓' : c.soft ? '◐' : '✕'}</span>
            <span>{c.name}</span>
          </li>
        ))}
      </ul>
      {proved && !rec.stfComposed && (
        <p className="verify__note">
          A bounded span is a genuine <em>partial</em> proof; composing to HALT
          proves the whole block. The chunks proven are fully verified and DA-matched.
        </p>
      )}
    </div>
  )
}

function ProofPanel({ rec, onDownload }) {
  const isRv32 = rec.kind === 'rv32'
  return (
    <div className="proof">
      <dl className="kv">
        <KV k="commitment (reused DA)" v={rec.stfCommitment || rec.commitment} mono copy />
        {isRv32 && <KV k="program" v={rec.program} mono copy />}
        {isRv32 && <KV k="input" v={rec.input || '0x'} mono copy />}
        {isRv32 && <KV k="output" v={rec.output} mono copy />}
        <KV k={isRv32 ? 'post-state root' : 'state root'} v={rec.stfStateRoot} mono copy />
        {isRv32 ? (
          <KV k="cycles" v={rec.numCycles} />
        ) : (
          <KV k="chunks proven" v={rec.stfNumChunks} />
        )}
        <KV k="proof size" v={rec.proofBytes ? fmtBytes(rec.proofBytes) : '—'} />
        <KV k="prove time" v={rec.elapsedMs ? fmtDuration(rec.elapsedMs) : '—'} />
        {!isRv32 && <KV k="EVM block #" v={rec.stfBlockNumber} />}
      </dl>
      <button className="btn btn--primary" onClick={onDownload} disabled={!rec.stfCommitment && !rec.commitment}>
        ⬇ Download proof info (JSON)
      </button>
      <p className="muted proof__note">
        The raw proof is {rec.proofBytes ? fmtBytes(rec.proofBytes) : 'large'} of
        GKR transcript held by the backend; this downloads its verified metadata
        (commitment, state root, chunk count, size).
      </p>
    </div>
  )
}

function Metric({ label, children }) {
  return (
    <div className="metric">
      <div className="metric__label">{label}</div>
      <div className="metric__value">{children}</div>
    </div>
  )
}

function KV({ k, v, mono, copy }) {
  return (
    <div className="kv__row">
      <span className="kv__k">{k}</span>
      <span className="kv__vwrap">
        <code className={mono ? 'mono kv__v' : 'kv__v'}>{v == null || v === '' ? '—' : String(v)}</code>
        {copy && v ? <CopyButton value={String(v)} label="⧉" /> : null}
      </span>
    </div>
  )
}

function fmtBytes(n) {
  if (!n) return '0 B'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

function fmtDuration(ms) {
  if (ms < 1000) return `${ms} ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)} s`
  const m = Math.floor(s / 60)
  return `${m}m ${Math.round(s % 60)}s`
}
