import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError, API_BASE } from './api.js'
import { RollupFeed } from './components/BlockProofs.jsx'
import { CopyButton } from './components/CopyButton.jsx'
import { AuroraBg } from './components/AuroraBg.jsx'
import { fmtDuration, useNow } from './util.js'

const POLL_MS = 2000

export default function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [conn, setConn] = useState('connecting')
  const alive = useRef(true)
  const now = useNow(250)

  const load = useCallback(async () => {
    try {
      const d = await api.blockProofs()
      if (!alive.current) return
      setData(d); setError(null); setConn('live')
    } catch (err) {
      if (!alive.current) return
      setError(err); setConn('stale')
    }
  }, [])

  useEffect(() => {
    alive.current = true
    load()
    const t = setInterval(load, POLL_MS)
    return () => { alive.current = false; clearInterval(t) }
  }, [load])

  const group = data?.namespaces?.[0] || null
  const rollupName = data?.rollupName || 'rv32i rollup'
  const rollupNS = data?.rollupNS || group?.namespace || 'rv32-rollup'
  const blocks = useMemo(
    () => [...(group?.blocks || [])].sort((a, b) => (b.blockNumber || 0) - (a.blockNumber || 0)),
    [group],
  )

  return (
    <div className="page">
      <AuroraBg blocks={blocks} />
      <TopBar conn={conn} rollupName={rollupName} rollupNS={rollupNS} daNamespace={group?.daNamespace} />

      {error && !data && <FatalError error={error} />}

      {data && (
        <>
          <Hero blocks={blocks} now={now} group={group} />
          <ChainState blocks={blocks} />
          <RollupFeed blocks={blocks} now={now} rollupNS={rollupNS} scope={data.scope} backendError={data.error} />
        </>
      )}

      <Footer />
    </div>
  )
}

/* ------------------------------ top bar ------------------------------ */

function TopBar({ conn, rollupName, rollupNS, daNamespace }) {
  const label = conn === 'live' ? 'live' : conn === 'stale' ? 'reconnecting' : 'connecting'
  return (
    <header className="top">
      <div className="top__title">
        <h1>{rollupName}</h1>
        <span className="top__sub">accidental computer · rv32i on Celestia DA</span>
      </div>
      <div className={`live live--${conn}`} title={`polling every ${POLL_MS / 1000}s`}>
        <span className="live__dot" />{label}
      </div>
    </header>
  )
}

/* ------------------ hero: block countdown + live totals ------------------ */

// The rollup seals a block on a fixed interval (RV32_BLOCK_SECS = 60s + the
// submitter's 60s cadence), so the countdown is a fixed 60s that RESTARTS each
// time a block lands (the anchor is the latest block, so elapsed resets to 0).
const BLOCK_SECS = 60

function Hero({ blocks, now, group }) {
  const nowSec = now / 1000
  const proved = blocks.filter((b) => b.status === 'proved')
  const proving = blocks.find((b) => b.status === 'proving')
  const latest = blocks.find((b) => b.provedUnix) // most recent landed block
  const cadence = BLOCK_SECS
  const anchor = latest?.provedUnix || 0
  const elapsed = anchor ? Math.max(0, nowSec - anchor) : 0
  const remain = Math.max(0, cadence - elapsed)
  // proving block clock: elapsed since it was submitted (started).
  const provingSecs = proving ? Math.max(0, Math.floor(nowSec - (proving.submittedUnix || nowSec))) : 0
  // Show the fixed 60s countdown for the whole window; only once it's overdue
  // (a block is late relative to the interval) switch to proving / landing.
  const mode = !anchor || elapsed < cadence ? 'wait' : (proving ? 'proving' : 'due')
  const progress = mode === 'wait' ? Math.min(1, elapsed / cadence) : 1

  const totalTx = blocks.reduce((s, b) => s + (b.numTx || 0), 0)
  const tps = proved.reduce((s, b) => s + (b.elapsedMs > 0 ? (b.numTx || 0) * 1000 / b.elapsedMs : 0), 0)
  const avgTps = proved.length ? tps / proved.length : 0
  const avgMs = proved.length ? Math.round(proved.reduce((s, b) => s + (b.elapsedMs || 0), 0) / proved.length) : 0

  return (
    <section className="hero">
      <BlockClock mode={mode} progress={progress}
        provingNum={proving?.blockNumber} provingSecs={provingSecs}
        remainSecs={Math.ceil(remain)} nextNum={(latest?.blockNumber || 0) + 1} />
      <div className="totals">
        <Total value={(group?.provedCount ?? proved.length).toLocaleString()} label="blocks proved" />
        <Total value={totalTx.toLocaleString()} label="transactions" />
        <Total value={avgTps > 0 ? avgTps.toFixed(2) : '—'} label="avg TPS" />
        <Total value={avgMs ? fmtDuration(avgMs) : '—'} label="avg prove time" />
      </div>
    </section>
  )
}

// Large countdown ring. wait = filling toward the next block; proving = a teal
// arc sweeps while a proof runs; due = a block is landing (past the estimate).
function BlockClock({ mode, progress, provingNum, provingSecs, remainSecs, nextNum }) {
  const R = 92, C = 2 * Math.PI * R
  const p = mode === 'proving' ? 0.3 : progress
  return (
    <div className={`clock clock--${mode}`}>
      <svg viewBox="0 0 220 220" className="clock__svg">
        <circle cx="110" cy="110" r={R} className="clock__track" />
        <circle cx="110" cy="110" r={R} className="clock__ring"
          style={{ strokeDasharray: C, strokeDashoffset: C * (1 - p) }} />
      </svg>
      <div className="clock__face">
        {mode === 'proving' ? (
          <>
            <div className="clock__eyebrow">proving block #{provingNum}</div>
            <div className="clock__big">{provingSecs}<span className="clock__unit">s</span></div>
            <div className="clock__small">generating GKR proof</div>
          </>
        ) : mode === 'due' ? (
          <>
            <div className="clock__eyebrow">next block #{nextNum}</div>
            <div className="clock__big clock__big--soon">landing</div>
            <div className="clock__small">any moment</div>
          </>
        ) : (
          <>
            <div className="clock__eyebrow">next block #{nextNum}</div>
            <div className="clock__big">~{remainSecs}<span className="clock__unit">s</span></div>
            <div className="clock__small">estimated</div>
          </>
        )}
      </div>
    </div>
  )
}

function Total({ value, label }) {
  const prev = useRef(value)
  const [bump, setBump] = useState(false)
  useEffect(() => {
    if (prev.current !== value) { prev.current = value; setBump(true); const t = setTimeout(() => setBump(false), 600); return () => clearTimeout(t) }
  }, [value])
  return (
    <div className="total">
      <div className={`total__v${bump ? ' bump' : ''}`}>{value}</div>
      <div className="total__l">{label}</div>
    </div>
  )
}

/* ---------------------------- chain state ---------------------------- */

function ChainState({ blocks }) {
  const [open, setOpen] = useState(false)
  const proved = blocks.filter((b) => b.status === 'proved')
  const last = proved[0]
  const cum = blocks.reduce((s, b) => s + (b.numCycles || 0), 0)
  const daH = Math.max(0, ...blocks.map((b) => b.daHeight || 0))
  const contract = blocks.find((b) => b.programName)?.programName || 'transactions'

  return (
    <section className="state">
      <button className="state__row" onClick={() => setOpen((v) => !v)}>
        <span className="state__k">current state root</span>
        <span className="state__v mono">{last?.postRoot ? shortMid(last.postRoot) : '—'}</span>
        <span className={`chev ${open ? 'chev--open' : ''}`} aria-hidden>›</span>
      </button>
      {open && (
        <div className="state__more">
          <Row k="state root" full={last?.postRoot} />
          <Row k="latest block" v={blocks[0] ? `#${blocks[0].blockNumber} · ${blocks[0].status}` : '—'} />
          <Row k="deployed contract" v={contract} />
          <Row k="cumulative cycles" v={cum.toLocaleString()} />
          <Row k="last DA settlement height" v={daH || '—'} />
          <Row k="last fibre blob" full={last?.blobId} />
        </div>
      )}
    </section>
  )
}

function Row({ k, v, full }) {
  return (
    <div className="srow">
      <span className="srow__k">{k}</span>
      {full
        ? <span className="srow__v mono full-hash">{full}<CopyButton value={full} label="copy" /></span>
        : <span className="srow__v">{v ?? '—'}</span>}
    </div>
  )
}

/* ------------------------------ helpers ------------------------------ */

// Only place we shorten a hash: the collapsed one-line preview. Everything
// expandable shows the full value.
function shortMid(h) {
  if (!h) return '—'
  const s = h.startsWith('0x') ? h.slice(2) : h
  return s.length > 20 ? `0x${s.slice(0, 8)}…${s.slice(-8)}` : h
}

function FatalError({ error }) {
  const msg = error instanceof ApiError ? error.message : String(error?.message || error)
  return (
    <section className="fatal">
      <h2>Can't reach the explorer API</h2>
      <p className="mono">{msg}</p>
      <p className="muted">Start the stack, then submit blocks.</p>
      <pre className="hint"><code>make start   # chain + rollup + API ({API_BASE}) + this UI</code></pre>
    </section>
  )
}

function Footer() {
  return <footer className="foot">API <code className="mono">{API_BASE}</code> · live every {POLL_MS / 1000}s</footer>
}
