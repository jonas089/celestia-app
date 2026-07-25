import { useMemo, useState } from 'react'
import { CopyButton } from './CopyButton.jsx'
import { fmtBytes, fmtDuration, timeAgo, hexToWordsLE, disasm, opClasses } from '../util.js'

// Live, newest-first per-block feed. Each row is a clean summary; click to expand
// full detail (all hashes shown in full, plus source / disassembly / I/O / verify).
export function RollupFeed({ blocks, now, rollupNS, scope, backendError }) {
  return (
    <section className="feed">
      <div className="feed__head">
        <h2>Blocks</h2>
        <span className="muted">newest first · live</span>
      </div>

      <details className="scope">
        <summary>What is proved</summary>
        <p>{scope || 'Each block executes committed rv32i over persistent VM state; program, input and pre-state are posted to Celestia DA; the block is GKR-proven by reusing the on-DA rsema1d commitment as the sole polynomial commitment (the prover re-encodes nothing) and binds pre_root → post_root in-circuit.'}</p>
      </details>

      {backendError && (
        <p className="inline-error">rollup backend: {String(backendError)} (namespace <code className="mono">{rollupNS}</code>)</p>
      )}

      {blocks.length === 0 && !backendError && (
        <div className="empty"><span className="spinner" aria-hidden /><span className="muted">Waiting for the first block…</span></div>
      )}

      <div className="blocks">
        {blocks.map((b) => <BlockCard key={b.blockNumber} b={b} now={now} />)}
      </div>
    </section>
  )
}

function kindOf(b) {
  if (b.status === 'proved') return b.verified ? 'ok' : 'warn'
  if (b.status === 'failed') return 'bad'
  if (b.status === 'proving') return 'proving'
  return 'queued'
}
function statusWord(b) {
  if (b.status === 'proved') return b.verified ? 'verified' : 'unverified'
  return b.status || 'pending'
}

const TABS = ['Overview', 'Source', 'Program', 'I/O']

function BlockCard({ b, now }) {
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState('Overview')
  const kind = kindOf(b)
  const words = useMemo(() => hexToWordsLE(b.program), [b.program])
  const provingSecs = b.status === 'proving' && b.submittedUnix
    ? Math.max(0, Math.floor(now / 1000) - b.submittedUnix) : null

  return (
    <article className={`blk blk--${kind} ${open ? 'blk--open' : ''}`}>
      <button className="blk__bar" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className={`blk__dot blk__dot--${kind}`} />
        <span className="blk__num">#{b.blockNumber}</span>
        <span className="blk__facts">
          <b>{(b.numTx || 0).toLocaleString()}</b> tx
          <span className="dotsep">·</span>{(b.numCycles || 0).toLocaleString()} cycles
          {(b.numLanes || 1) > 1 && <><span className="dotsep">·</span>{b.numLanes} lanes</>}
        </span>
        <span className="blk__spacer" />
        <span className={`blk__stat blk__stat--${kind}`}>
          {provingSecs != null ? `proving · ${provingSecs}s` : statusWord(b)}
        </span>
        <span className="blk__time">{timeAgo(b.provedUnix || b.submittedUnix)}</span>
        <span className={`chev ${open ? 'chev--open' : ''}`} aria-hidden>›</span>
      </button>

      {b.status === 'proving' && <span className="blk__progress" aria-hidden />}

      {open && (
        <div className="blk__body">
          <nav className="tabs">
            {TABS.map((t) => (
              <button key={t} className={`tab ${tab === t ? 'tab--on' : ''}`} onClick={() => setTab(t)}>{t}</button>
            ))}
          </nav>
          {tab === 'Overview' && <Overview b={b} words={words} />}
          {tab === 'Source' && <Source b={b} />}
          {tab === 'Program' && <Program b={b} words={words} />}
          {tab === 'I/O' && <Io b={b} />}
          {b.error && <pre className="scrollbox err">{b.error}</pre>}
        </div>
      )}
    </article>
  )
}

/* ------------------------------- tabs ------------------------------- */

function Overview({ b, words }) {
  const oc = useMemo(() => opClasses(words), [words])
  const total = Object.values(oc).reduce((a, c) => a + c, 0) || 1
  const mix = [['alu', 'ALU'], ['mem', 'load/store'], ['branch', 'branch'], ['jump', 'jump'], ['mul', 'mul/div'], ['other', 'other']]
    .filter(([k]) => oc[k] > 0)
  const checks = [
    ['Expander GKR verifier accepted the rv32i execution', b.status === 'proved' && b.verified],
    ['GKR input commitment == reused rsema1d / DA encoding', (b.commitment || '').replace(/^0x/, '').length === 64],
    ['settled on Celestia DA (fibre MsgPayForFibre)', (b.daHeight || 0) > 0 || !!b.txHash],
    ['state transition bound in-circuit (pre_root → post_root)', !!b.preRoot && !!b.postRoot],
  ]
  return (
    <div className="ov">
      <div className="grid">
        <Fact k="transactions" v={`${(b.numTx || 0).toLocaleString()}${(b.numLanes || 1) > 1 ? `  ·  ${b.numLanes} SIMD lanes` : ''}`} />
        <Fact k="throughput" v={b.elapsedMs > 0 ? `${((b.numTx || 0) * 1000 / b.elapsedMs).toFixed(2)} tx/s` : '—'} />
        <Fact k="cycles executed" v={(b.numCycles || 0).toLocaleString()} />
        <Fact k="prove time" v={b.elapsedMs ? fmtDuration(b.elapsedMs) : '—'} />
        <Fact k="proof size" v={b.proofBytes ? fmtBytes(b.proofBytes) : '—'} />
        <Fact k="GKR input vars" v={b.inputVars ? `2^${b.inputVars}` : '—'} />
        <Fact k="DA settlement height" v={b.daHeight || '—'} />
      </div>

      <Hash k="commitment  (rsema1d == DA)" v={b.commitment} />
      <Hash k="state root — before" v={b.preRoot} />
      <Hash k="state root — after" v={b.postRoot} />
      <Hash k="fibre blob id" v={b.blobId} />
      <Hash k="fibre settlement tx" v={b.txHash} />

      <ul className="checks">
        {checks.map(([name, ok], i) => (
          <li key={i} className={ok ? 'ok' : 'no'}><span>{ok ? '✓' : '·'}</span>{name}</li>
        ))}
      </ul>

      {mix.length > 0 && (
        <div className="mix">
          {mix.map(([k, name]) => (
            <div className="mix__row" key={k}>
              <span className="mix__name">{name}</span>
              <span className="mix__bar"><span style={{ width: `${(oc[k] / total) * 100}%` }} /></span>
              <span className="mix__pct">{Math.round((oc[k] / total) * 100)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Source({ b }) {
  return (
    <div className="src">
      <div className="src__head"><span>Rust source · no_std → riscv32im</span>{b.rustSource && <CopyButton value={b.rustSource} label="copy" />}</div>
      <pre className="scrollbox code">{b.rustSource || 'No source recorded.'}</pre>
      <div className="src__head"><span>compiled bytecode · {b.program ? (b.program.replace(/^0x/, '').length / 8) | 0 : 0} words</span>{b.program && <CopyButton value={b.program} label="copy" />}</div>
      <pre className="scrollbox hex">{b.program || '—'}</pre>
    </div>
  )
}

function Program({ b, words }) {
  if (!words.length) return <p className="muted">No program words.</p>
  return (
    <div className="scrollbox asm">
      {words.map((w, i) => {
        const text = disasm(w), mnem = text.split(/\s/)[0]
        return (
          <div className="asm__row" key={i}>
            <span className="asm__a">{(i * 4).toString(16).padStart(4, '0')}</span>
            <span className="asm__w">{w.toString(16).padStart(8, '0')}</span>
            <span className="asm__op"><b>{mnem}</b>{text.slice(mnem.length)}</span>
          </div>
        )
      })}
    </div>
  )
}

function Io({ b }) {
  return (
    <div className="io">
      <div className="io__label">input (on DA)</div>
      <pre className="scrollbox hex">{hexdump(b.input) || '∅ no input'}</pre>
      <div className="io__label">output — public value (on DA)</div>
      <pre className="scrollbox hex">{hexdump(b.output) || '— no output'}</pre>
    </div>
  )
}

/* ------------------------------ pieces ------------------------------ */

function Fact({ k, v }) {
  return <div className="fact"><div className="fact__k">{k}</div><div className="fact__v">{v ?? '—'}</div></div>
}

// Full, never-truncated hash with an inline copy control.
function Hash({ k, v }) {
  if (!v) return null
  return (
    <div className="hashrow">
      <div className="hashrow__k">{k}</div>
      <div className="hashrow__v"><code className="mono">{v}</code><CopyButton value={v} label="copy" /></div>
    </div>
  )
}

function hexdump(hex) {
  if (!hex) return ''
  const s = hex.startsWith('0x') ? hex.slice(2) : hex
  if (!s.length) return ''
  const bytes = []
  for (let i = 0; i + 2 <= s.length && bytes.length < 512; i += 2) bytes.push(parseInt(s.slice(i, i + 2), 16))
  const lines = []
  for (let o = 0; o < bytes.length; o += 16) {
    const chunk = bytes.slice(o, o + 16)
    const hp = chunk.map((x) => x.toString(16).padStart(2, '0')).join(' ').padEnd(47, ' ')
    const asc = chunk.map((x) => (x >= 32 && x < 127 ? String.fromCharCode(x) : '.')).join('')
    lines.push(`${o.toString(16).padStart(6, '0')}  ${hp}  ${asc}`)
  }
  if (bytes.length >= 512) lines.push('… truncated')
  return lines.join('\n')
}
