import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError, API_BASE } from './api.js'
import { RollupFeed } from './components/BlockProofs.jsx'
import { CopyButton } from './components/CopyButton.jsx'
import {
  truncHex, fmtDuration, timeAgo, makeProgramLabeler,
} from './util.js'

const POLL_MS = 2000

export default function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [conn, setConn] = useState('connecting')
  const alive = useRef(true)

  const load = useCallback(async () => {
    try {
      const d = await api.blockProofs()
      if (!alive.current) return
      setData(d)
      setError(null)
      setConn('live')
      setLastUpdated(Date.now())
    } catch (err) {
      if (!alive.current) return
      setError(err)
      setConn('stale')
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
      <div className="bg" aria-hidden />
      <Header conn={conn} lastUpdated={lastUpdated} rollupName={rollupName}
        rollupNS={rollupNS} daNamespace={group?.daNamespace} group={group} blocks={blocks} />

      {error && !data && <FatalError error={error} />}

      {data && (
        <>
          <div className="grid2">
            <VmStatePanel blocks={blocks} />
            <ProgramsPanel blocks={blocks} />
          </div>
          <RollupFeed blocks={blocks} rollupNS={rollupNS} scope={data.scope} backendError={data.error} />
        </>
      )}

      <Footer />
    </div>
  )
}

/* ---------------- header ---------------- */

function Header({ conn, lastUpdated, rollupName, rollupNS, daNamespace, group, blocks }) {
  const proved = blocks.filter((b) => b.status === 'proved')
  const totalCycles = blocks.reduce((s, b) => s + (b.numCycles || 0), 0)
  const avgMs = proved.length
    ? Math.round(proved.reduce((s, b) => s + (b.elapsedMs || 0), 0) / proved.length)
    : null
  const stats = [
    { label: 'blocks', value: group?.blockCount ?? blocks.length },
    { label: 'proved', value: group?.provedCount ?? proved.length },
    { label: 'GKR verified', value: group?.verifiedCount ?? blocks.filter((b) => b.verified).length },
    { label: 'vm cycles', value: totalCycles.toLocaleString() },
    { label: 'avg prove', value: avgMs != null ? fmtDuration(avgMs) : '—' },
  ]
  return (
    <header className="hdr">
      <div className="hdr__top">
        <div className="hdr__brand">
          <span className="hdr__badge">rsema1d · accidental computer</span>
          <h1 className="hdr__title">{rollupName}</h1>
          <div className="hdr__ns mono" title={daNamespace || rollupNS}>
            namespace <strong>{rollupNS}</strong>
            {daNamespace ? <span className="hdr__da"> · DA {truncHex(daNamespace, 8, 6)}</span> : null}
          </div>
        </div>
        <ConnPill conn={conn} lastUpdated={lastUpdated} />
      </div>

      <p className="hdr__lede">
        Each block runs a committed rv32i program over persistent VM state. The program,
        input, and pre-state are posted to Celestia DA, and the block is GKR-proven by reusing
        that on-DA rsema1d commitment as its only polynomial commitment. The prover re-encodes
        nothing; the execution trace stays intermediate.
      </p>

      <div className="statrow">
        {stats.map((s) => <Stat key={s.label} {...s} />)}
      </div>
    </header>
  )
}

function Stat({ label, value }) {
  const prev = useRef(value)
  const [flip, setFlip] = useState(false)
  useEffect(() => {
    if (prev.current !== value) {
      prev.current = value
      setFlip(true)
      const t = setTimeout(() => setFlip(false), 500)
      return () => clearTimeout(t)
    }
  }, [value])
  return (
    <div className="stat">
      <div className={`stat__value${flip ? ' flip' : ''}`}>{value}</div>
      <div className="stat__label">{label}</div>
    </div>
  )
}

function ConnPill({ conn, lastUpdated }) {
  const label = conn === 'live' ? 'live' : conn === 'stale' ? 'reconnecting' : 'connecting'
  return (
    <div className={`conn conn--${conn}`} title={`polling every ${POLL_MS / 1000}s`}>
      <span className="conn__dot" />
      <span className="conn__label">{label}</span>
      {lastUpdated && <span className="conn__time mono">{new Date(lastUpdated).toLocaleTimeString()}</span>}
    </div>
  )
}

/* ---------------- persistent VM state ---------------- */

function VmStatePanel({ blocks }) {
  const latest = blocks[0]
  const proved = blocks.filter((b) => b.status === 'proved')
  const lastProved = proved[0]
  const totalCycles = blocks.reduce((s, b) => s + (b.numCycles || 0), 0)
  const daHeights = blocks.map((b) => b.daHeight || 0).filter(Boolean)
  return (
    <div className="panel">
      <div className="panel__head"><span className="panel__dot" /><h3>Persistent VM state</h3></div>
      <div className="vmrow">
        <span className="vmrow__k">current post-state root</span>
        <span className="vmrow__v">
          {lastProved?.stfStateRoot
            ? <span className="mono">{truncHex(lastProved.stfStateRoot, 10, 8)}<CopyButton value={lastProved.stfStateRoot} label="⧉" /></span>
            : <span className="muted">—</span>}
        </span>
      </div>
      <div className="vmrow">
        <span className="vmrow__k">latest block</span>
        <span className="vmrow__v">{latest ? `#${latest.blockNumber} · ${latest.status}` : '—'}</span>
      </div>
      <div className="vmrow">
        <span className="vmrow__k">cumulative cycles executed</span>
        <span className="vmrow__v">{totalCycles.toLocaleString()}</span>
      </div>
      <div className="vmrow">
        <span className="vmrow__k">last DA settlement height</span>
        <span className="vmrow__v mono">{daHeights.length ? Math.max(...daHeights) : '—'}</span>
      </div>
      <div className="vmrow">
        <span className="vmrow__k">last fibre blob</span>
        <span className="vmrow__v">
          {lastProved?.blobId
            ? <span className="mono">{truncHex(lastProved.blobId, 8, 6)}<CopyButton value={lastProved.blobId} label="⧉" /></span>
            : <span className="muted">—</span>}
        </span>
      </div>
    </div>
  )
}

/* ---------------- deployed programs ---------------- */

function ProgramsPanel({ blocks }) {
  const programs = useMemo(() => {
    const label = makeProgramLabeler()
    const map = new Map()
    // iterate oldest→newest so labels are assigned in first-seen order
    for (const b of [...blocks].reverse()) {
      const hex = b.program
      if (!hex) continue
      if (!map.has(hex)) {
        const { name, idx } = label(hex)
        map.set(hex, { hex, name: b.programName || name, idx, runs: 0, lastBlock: 0, lastOutput: '' })
      }
      const e = map.get(hex)
      e.runs++
      if ((b.blockNumber || 0) >= e.lastBlock) { e.lastBlock = b.blockNumber || 0; e.lastOutput = b.output }
    }
    return [...map.values()].sort((a, b) => a.idx - b.idx)
  }, [blocks])

  return (
    <div className="panel">
      <div className="panel__head"><span className="panel__dot" /><h3>Deployed programs</h3></div>
      {programs.length === 0 && <p className="muted" style={{ fontSize: 12.5, margin: '6px 0' }}>No programs executed yet.</p>}
      {programs.map((p) => (
        <div className="prog" key={p.hex}>
          <div className="prog__tag">{p.idx + 1}</div>
          <div className="prog__body">
            <div className="prog__name">{p.name}</div>
            <div className="prog__meta mono">{truncHex(p.hex, 10, 6)} · {(p.hex.replace(/^0x/, '').length / 8) | 0} instrs</div>
          </div>
          <div className="prog__runs">{p.runs} run{p.runs === 1 ? '' : 's'} · #{p.lastBlock}</div>
        </div>
      ))}
    </div>
  )
}

/* ---------------- fallbacks ---------------- */

function FatalError({ error }) {
  const msg = error instanceof ApiError ? error.message : String(error?.message || error)
  return (
    <section className="panel state state--error">
      <div className="state__icon">!</div>
      <h2>Cannot reach the explorer API</h2>
      <p className="mono">{msg}</p>
      <p className="muted">Start the Go API (it must listen on the base URL below) and the rv32-rollup, then submit blocks.</p>
      <pre className="hint"><code>make start   # brings up the chain, rollup, API ({API_BASE}) and this UI</code></pre>
    </section>
  )
}

function Footer() {
  return (
    <footer className="footer">
      API <code className="mono">{API_BASE}</code> · polling every {POLL_MS / 1000}s
    </footer>
  )
}
