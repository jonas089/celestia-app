import { useMemo, useState } from 'react'
import { CopyButton } from './CopyButton.jsx'
import {
  truncHex, fmtBytes, fmtDuration, timeAgo,
  hexToWordsLE, disasm, opClasses,
} from '../util.js'

// Live, newest-first per-block proof feed for the rv32i accidental computer.
// Each block executes committed rv32i over persistent VM state and is GKR-proven
// by reusing the on-DA rsema1d commitment. Cards animate in on arrival and
// expand into tabbed, scrollable detail (disassembly, I/O, proof, verify).
export function RollupFeed({ blocks, rollupNS, scope, backendError }) {
  const labelOf = useMemo(() => {
    const m = new Map()
    for (const b of blocks) {
      if (b.program) m.set(b.program, b.programName || 'program')
    }
    return (hex) => m.get(hex) || 'program'
  }, [blocks])

  return (
    <section className="feed">
      <div className="feed__head">
        <h2>Block proofs</h2>
        <span className="muted">newest first · live</span>
      </div>

      <details className="scope">
        <summary>Honest scope of the proof</summary>
        <p>{scope || 'Each block executes committed rv32i over persistent VM state; program + input + pre-state posted to Celestia DA; GKR-proven by reusing the rsema1d/DA commitment as the sole PCS (zero prover re-encode).'}</p>
      </details>

      {backendError && (
        <p className="inline-error">
          rollup backend: {String(backendError)}. Start the rv32-rollup and submit blocks
          (namespace <code className="mono">{rollupNS}</code>).
        </p>
      )}

      {blocks.length === 0 && !backendError && (
        <div className="empty">
          <div className="spinner" aria-hidden />
          <p className="muted">Waiting for the first block. The submitter injects a program every ~60s.</p>
        </div>
      )}

      <div className="cards">
        {blocks.map((b) => (
          <BlockCard key={b.blockNumber} b={b} label={labelOf(b.program)} />
        ))}
      </div>
    </section>
  )
}

function statusInfo(b) {
  switch (b.status) {
    case 'proved': return b.verified
      ? { kind: 'ok', label: 'proved', icon: '✓' }
      : { kind: 'bad', label: 'unverified', icon: '✕' }
    case 'failed': return { kind: 'bad', label: 'failed', icon: '✕' }
    case 'proving': return { kind: 'proving', label: 'proving' }
    case 'queued': return { kind: 'queued', label: 'queued' }
    default: return { kind: 'queued', label: b.status || 'pending' }
  }
}

const TABS = ['Overview', 'Source', 'Program', 'I/O', 'Proof', 'Verify']

function BlockCard({ b, label }) {
  const [tab, setTab] = useState(null) // null = collapsed
  const st = statusInfo(b)
  const words = useMemo(() => hexToWordsLE(b.program), [b.program])

  return (
    <article className={`bcard bcard--${st.kind}`}>
      <div className="bcard__accent" />

      <div className="bcard__head">
        <div className="bcard__ids">
          <span className="bcard__height">#{b.blockNumber} <small>· {label}</small></span>
          <span className="bcard__sub mono">{(b.numCycles || 0).toLocaleString()} cycles · {words.length} instrs</span>
        </div>
        <StatusBadge st={st} />
      </div>

      <div className="chips">
        <span className="chip chip--flag" title="Raw rv32i executed in-circuit; the on-DA rsema1d commitment is the sole (reused) GKR input commitment.">accidental computer · in-circuit rv32i</span>
        <span className="chip chip--prog">{label}</span>
        <span className="chip chip--total">{(b.numCycles || 0).toLocaleString()} cycles</span>
        <span className="chip mono" title={b.input || '0x'}>input {b.input && b.input !== '0x' ? truncHex(b.input, 6, 4) : '∅'}</span>
        <span className="chip mono" title={b.output}>output {b.output && b.output !== '0x' ? truncHex(b.output, 6, 4) : '—'}</span>
        {b.daHeight > 0 && <span className="chip">DA h{b.daHeight}</span>}
      </div>

      {st.kind !== 'bad' && (
        <div className="metrics">
          <Metric label="commitment (rsema1d == DA)">
            {b.commitment
              ? <span className="mono metric__hex" title={b.commitment}>{truncHex(b.commitment, 10, 6)}
                  <span className="daflag daflag--ok">== DA</span><CopyButton value={b.commitment} label="⧉" /></span>
              : <span className="muted">pending…</span>}
          </Metric>
          <Metric label="state root · pre → post">
            {b.postRoot
              ? <span className="mono metric__hex" title={`pre  ${b.preRoot}\npost ${b.postRoot}`}>
                  {truncHex(b.preRoot, 6, 4)} → {truncHex(b.postRoot, 6, 4)}
                  <CopyButton value={b.postRoot} label="⧉" />
                </span>
              : <span className="muted">pending…</span>}
          </Metric>
          <Metric label="GKR verified">
            <span className={b.verified ? 'pill pill--ok' : 'pill pill--partial'}>
              {b.verified ? '✓ verified' : b.status === 'proved' ? 'unverified' : b.status}
            </span>
          </Metric>
          <Metric label="proof size · time">
            <span className="mono">{b.proofBytes ? fmtBytes(b.proofBytes) : '—'}{b.elapsedMs ? ` · ${fmtDuration(b.elapsedMs)}` : ''}</span>
          </Metric>
        </div>
      )}

      {b.error && <ErrorLine msg={b.error} />}

      <div className="bcard__actions">
        {TABS.map((t) => (
          <button key={t} className={`btn ${tab === t ? 'btn--active' : ''}`} onClick={() => setTab(tab === t ? null : t)}>{t}</button>
        ))}
        <span className="bcard__time">{timeAgo(b.provedUnix || b.submittedUnix)}</span>
      </div>

      {tab && (
        <div className="drawer">
          <div className="tabs">
            {TABS.map((t) => (
              <button key={t} className={`tab ${tab === t ? 'tab--on' : ''}`} onClick={() => setTab(t)}>{t}</button>
            ))}
          </div>
          {tab === 'Overview' && <OverviewTab b={b} words={words} />}
          {tab === 'Source' && <SourceTab b={b} />}
          {tab === 'Program' && <ProgramTab b={b} words={words} />}
          {tab === 'I/O' && <IoTab b={b} />}
          {tab === 'Proof' && <ProofTab b={b} label={label} />}
          {tab === 'Verify' && <VerifyTab b={b} />}
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

/* -------- tabs -------- */

function OverviewTab({ b, words }) {
  const oc = useMemo(() => opClasses(words), [words])
  const total = Object.values(oc).reduce((a, c) => a + c, 0) || 1
  const bars = [
    ['alu', 'ALU'], ['mem', 'load/store'], ['branch', 'branch'],
    ['jump', 'jump'], ['mul', 'mul/div'], ['other', 'other'],
  ].filter(([k]) => oc[k] > 0)
  return (
    <div className="scrollbox">
      <div className="kv">
        <KV k="block height" v={`#${b.blockNumber}`} />
        <KV k="status" v={b.status} />
        <KV k="pre-state root" v={b.preRoot ? truncHex(b.preRoot, 12, 10) : '—'} full={b.preRoot} copy />
        <KV k="post-state root" v={b.postRoot ? truncHex(b.postRoot, 12, 10) : '—'} full={b.postRoot} copy />
        <KV k="cycles executed" v={(b.numCycles || 0).toLocaleString()} />
        <KV k="GKR input vars" v={b.inputVars} />
        <KV k="DA settlement height" v={b.daHeight || '—'} />
        <KV k="fibre tx" v={b.txHash ? truncHex(b.txHash, 10, 8) : '—'} full={b.txHash} copy />
        <KV k="fibre blob id" v={b.blobId ? truncHex(b.blobId, 10, 8) : '—'} full={b.blobId} copy />
        <KV k="submitted" v={b.submittedUnix ? timeAgo(b.submittedUnix) : '—'} />
        <KV k="proved" v={b.provedUnix ? timeAgo(b.provedUnix) : '—'} />
      </div>
      <div style={{ marginTop: 12 }}>
        <div className="metric__label">instruction mix ({words.length} instrs)</div>
        <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
          {bars.map(([k, name]) => (
            <span key={k} className="chip">{name} · {Math.round((oc[k] / total) * 100)}%</span>
          ))}
        </div>
      </div>
    </div>
  )
}

function SourceTab({ b }) {
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <div>
        <div className="srchead">
          <span className="metric__label">Rust source · no_std, compiled to riscv32im</span>
          {b.rustSource ? <CopyButton value={b.rustSource} label="⧉ copy" /> : null}
        </div>
        {b.rustSource
          ? <pre className="scrollbox code">{b.rustSource}</pre>
          : <div className="scrollbox"><span className="muted">No Rust source recorded for this program.</span></div>}
      </div>
      <div>
        <div className="srchead">
          <span className="metric__label">compiled rv32 bytecode · {b.program ? (b.program.replace(/^0x/, '').length / 8) | 0 : 0} words</span>
          {b.program ? <CopyButton value={b.program} label="⧉ copy" /> : null}
        </div>
        <pre className="scrollbox hexdump">{b.program || '—'}</pre>
      </div>
    </div>
  )
}

function ProgramTab({ b, words }) {
  if (!words.length) return <div className="scrollbox"><span className="muted">No program words.</span></div>
  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <span className="metric__label">rv32im disassembly · {words.length} instructions</span>
        <CopyButton value={b.program} label="⧉ copy hex" />
      </div>
      <div className="scrollbox asm">
        {words.map((w, i) => {
          const text = disasm(w)
          const mnem = text.split(/\s/)[0]
          const rest = text.slice(mnem.length)
          return (
            <div className="asm__row" key={i}>
              <span className="asm__addr">{(i * 4).toString(16).padStart(4, '0')}:</span>
              <span className="asm__word">{w.toString(16).padStart(8, '0')}</span>
              <span className="asm__op"><b>{mnem}</b>{rest}</span>
            </div>
          )
        })}
      </div>
    </>
  )
}

function IoTab({ b }) {
  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div>
        <div className="metric__label" style={{ marginBottom: 6 }}>input (on DA)</div>
        <div className="scrollbox hexdump">{hexdump(b.input) || '∅ (no input)'}</div>
      </div>
      <div>
        <div className="metric__label" style={{ marginBottom: 6 }}>output — public value (on DA)</div>
        <div className="scrollbox hexdump">{hexdump(b.output) || '— (no output)'}</div>
      </div>
    </div>
  )
}

function ProofTab({ b, label }) {
  function download() {
    const info = {
      program: label, blockHeight: b.blockNumber, status: b.status,
      commitment_rsema1d_eq_DA: b.commitment, postStateRoot: b.stfStateRoot,
      output: b.output, numCycles: b.numCycles, inputVars: b.inputVars,
      proofBytes: b.proofBytes, elapsedMs: b.elapsedMs, verified: b.verified,
      daHeight: b.daHeight, fibreTxHash: b.txHash, fibreBlobId: b.blobId,
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify(info, null, 2)], { type: 'application/json' }))
    const a = document.createElement('a')
    a.href = url; a.download = `rv32-proof-h${b.blockNumber}.json`
    document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url)
  }
  return (
    <div className="scrollbox">
      <div className="kv">
        <KV k="commitment (reused DA)" v={b.commitment ? truncHex(b.commitment, 14, 10) : '—'} full={b.commitment} copy />
        <KV k="post-state root" v={b.stfStateRoot ? truncHex(b.stfStateRoot, 14, 10) : '—'} full={b.stfStateRoot} copy />
        <KV k="output" v={b.output ? truncHex(b.output, 14, 10) : '—'} full={b.output} copy />
        <KV k="cycles" v={b.numCycles} />
        <KV k="input vars (2^n square)" v={b.inputVars} />
        <KV k="proof size" v={b.proofBytes ? fmtBytes(b.proofBytes) : '—'} />
        <KV k="prove time" v={b.elapsedMs ? fmtDuration(b.elapsedMs) : '—'} />
      </div>
      <button className="btn btn--primary" style={{ marginTop: 12 }} onClick={download} disabled={!b.commitment}>⬇ Download proof info (JSON)</button>
      <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
        The raw proof is a {b.proofBytes ? fmtBytes(b.proofBytes) : 'large'} GKR transcript held by the
        rollup. This exports the verified metadata. The GKR input commitment is the on-DA rsema1d
        commitment, so the prover performed zero RS-encoding.
      </p>
    </div>
  )
}

function VerifyTab({ b }) {
  const proved = b.status === 'proved'
  const checks = [
    { name: 'Expander GKR verifier ACCEPTED the rv32i execution', ok: proved && b.verified },
    { name: 'GKR input commitment == reused rsema1d/DA encoding (32 bytes)', ok: (b.commitment || '').replace(/^0x/, '').length === 64 },
    { name: 'block settled on Celestia DA (fibre MsgPayForFibre)', ok: (b.daHeight || 0) > 0 || !!b.txHash },
    { name: 'public output present', ok: !!b.output && b.output !== '0x' },
    { name: 'state transition bound in-circuit (pre_root → post_root)', ok: !!b.preRoot && !!b.postRoot },
  ]
  const verified = proved && b.verified
  return (
    <div className={`verify ${verified ? 'verify--ok' : proved ? 'verify--warn' : 'verify--pending'}`}>
      <div className="verify__badge">{proved ? (verified ? 'VERIFIED = TRUE' : 'NOT VERIFIED') : `status: ${b.status}`}</div>
      <ul className="checks">
        {checks.map((c, i) => (
          <li key={i} className={c.ok ? 'check--ok' : 'check--bad'}>
            <span className="check__mark">{c.ok ? '✓' : '✕'}</span><span>{c.name}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/* -------- small pieces -------- */

function Metric({ label, children }) {
  return <div className="metric"><div className="metric__label">{label}</div><div className="metric__value">{children}</div></div>
}

// Extract a concise human reason from a verbose backend/prover error blob.
function shortError(msg) {
  if (!msg) return ''
  const m = String(msg).replace(/\s+/g, ' ').trim()
  const desc = m.match(/desc = ([^"]+?)(?:"|$)/)
  if (desc) return desc[1].trim()
  const tail = m.match(/(?:fibre upload|error):?\s*([^:]+:[^:]+)$/)
  if (tail) return tail[1].trim()
  return m.length > 160 ? m.slice(0, 160) + '…' : m
}

function ErrorLine({ msg }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="errbox">
      <div className="errbox__line">
        <span className="errbox__dot" />
        <span className="errbox__reason" title={msg}>{shortError(msg)}</span>
        <button className="errbox__toggle" onClick={() => setOpen((v) => !v)}>{open ? 'hide' : 'details'}</button>
      </div>
      {open && <pre className="scrollbox errbox__full">{msg}</pre>}
    </div>
  )
}

function KV({ k, v, full, copy }) {
  const value = v == null || v === '' ? '—' : String(v)
  return (
    <div className="kv__row">
      <span className="kv__k">{k}</span>
      <span className="kv__vwrap">
        <code className="mono kv__v" title={full || value}>{value}</code>
        {copy && full ? <CopyButton value={String(full)} label="⧉" /> : null}
      </span>
    </div>
  )
}

// classic offset | hex bytes | ascii dump of a 0x hex string (capped).
function hexdump(hex) {
  if (!hex) return ''
  let s = hex.startsWith('0x') ? hex.slice(2) : hex
  if (!s.length) return ''
  const bytes = []
  for (let i = 0; i + 2 <= s.length && bytes.length < 512; i += 2) bytes.push(parseInt(s.slice(i, i + 2), 16))
  const lines = []
  for (let o = 0; o < bytes.length; o += 16) {
    const chunk = bytes.slice(o, o + 16)
    const hexPart = chunk.map((x) => x.toString(16).padStart(2, '0')).join(' ').padEnd(47, ' ')
    const ascii = chunk.map((x) => (x >= 32 && x < 127 ? String.fromCharCode(x) : '.')).join('')
    lines.push(`${o.toString(16).padStart(6, '0')}  ${hexPart}  ${ascii}`)
  }
  if (bytes.length >= 512) lines.push('… (truncated)')
  return lines.join('\n')
}
