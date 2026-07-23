import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError, API_BASE } from './api.js'
import { RollupFeed } from './components/BlockProofs.jsx'

const POLL_MS = 2000

export default function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)
  // "connecting" until the first successful poll; then "live" / "stale".
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
    return () => {
      alive.current = false
      clearInterval(t)
    }
  }, [load])

  const group = data?.namespaces?.[0] || null
  const rollupName = data?.rollupName || 'rv32i rollup'
  const rollupNS = data?.rollupNS || group?.namespace || 'rv32-rollup'

  return (
    <div className="page">
      <Header
        conn={conn}
        lastUpdated={lastUpdated}
        rollupName={rollupName}
        rollupNS={rollupNS}
        daNamespace={group?.daNamespace}
        group={group}
      />

      {error && !data && <FatalError error={error} />}

      {data && (
        <RollupFeed
          group={group}
          rollupNS={rollupNS}
          rollupName={rollupName}
          scope={data.scope}
          backendError={data.error}
        />
      )}

      <Footer />
    </div>
  )
}

function Header({ conn, lastUpdated, rollupName, rollupNS, daNamespace, group }) {
  const stats = [
    { label: 'blocks', value: group?.blockCount ?? 0 },
    { label: 'proved', value: group?.provedCount ?? 0 },
    { label: 'verified', value: group?.verifiedCount ?? 0 },
  ]
  return (
    <header className="hdr">
      <div className="hdr__top">
        <div className="hdr__brand">
          <span className="hdr__badge">rsema1d · rv32i STF</span>
          <h1 className="hdr__title">{rollupName}</h1>
          <div className="hdr__ns mono" title={daNamespace || rollupNS}>
            namespace <strong>{rollupNS}</strong>
            {daNamespace ? <span className="hdr__da"> · DA {daNamespace}</span> : null}
          </div>
        </div>
        <ConnPill conn={conn} lastUpdated={lastUpdated} />
      </div>

      <p className="hdr__lede">
        Every block executes a committed <strong>rv32i program</strong> over persistent
        VM state; program + input + output + trace + state are posted to Celestia DA,
        and the block is <strong>GKR-proven by reusing the rsema1d DA encoding</strong> as
        the sole polynomial commitment — byte-identical to Go/DA, zero prover re-encode
        (the accidental rv32i computer). Proofs are self-verified before they land here;
        nothing is mocked, and proving is serialized (one block at a time).
      </p>

      <div className="statrow">
        {stats.map((s) => (
          <div className="stat" key={s.label}>
            <div className="stat__value">{s.value}</div>
            <div className="stat__label">{s.label}</div>
          </div>
        ))}
      </div>
    </header>
  )
}

function ConnPill({ conn, lastUpdated }) {
  const label =
    conn === 'live' ? 'live' : conn === 'stale' ? 'reconnecting' : 'connecting'
  return (
    <div className={`conn conn--${conn}`} title={`polling every ${POLL_MS / 1000}s`}>
      <span className="conn__dot" />
      <span className="conn__label">{label}</span>
      {lastUpdated && (
        <span className="conn__time mono">
          {new Date(lastUpdated).toLocaleTimeString()}
        </span>
      )}
    </div>
  )
}

function FatalError({ error }) {
  const msg =
    error instanceof ApiError ? error.message : String(error?.message || error)
  return (
    <section className="card state state--error">
      <div className="state__icon">!</div>
      <h2>Cannot reach the dashboard API</h2>
      <p className="mono state__msg">{msg}</p>
      <p className="muted">
        Start the Go API (it must listen on the base URL below) and the accProof
        backend, then submit blocks.
      </p>
      <pre className="hint">
        <code>
          cd test/accidental-computer/api{'\n'}go run . # serves {API_BASE}
        </code>
      </pre>
    </section>
  )
}

function Footer() {
  return (
    <footer className="footer">
      <span className="muted">
        API base <code className="mono">{API_BASE}</code> · override with{' '}
        <code className="mono">VITE_API_BASE</code> · polls every {POLL_MS / 1000}s
      </span>
    </footer>
  )
}
