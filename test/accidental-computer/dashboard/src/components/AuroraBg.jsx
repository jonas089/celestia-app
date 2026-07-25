import { useEffect, useRef } from 'react'

// Ambient, GPU-cheap "activity monitor" backdrop: a few large, soft colour
// fields drift on smooth sine paths (an aurora / lava-lamp feel), and every time
// a new block lands a quiet ring ripples out from centre. Low-contrast so it
// never fights the foreground; respects prefers-reduced-motion.
export function AuroraBg({ blocks }) {
  const ref = useRef(null)
  const ripples = useRef([])
  const lastBlock = useRef(0)

  // Ripple on each new block.
  useEffect(() => {
    const n = blocks?.[0]?.blockNumber || 0
    if (n && n !== lastBlock.current) {
      if (lastBlock.current !== 0) ripples.current.push({ t0: performance.now() })
      lastBlock.current = n
    }
  }, [blocks])

  useEffect(() => {
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    const canvas = ref.current
    const ctx = canvas.getContext('2d')
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    let w = 0, h = 0, raf = 0

    const orbs = [
      { color: 'rgba(122,162,255,0.22)', ax: 0.28, ay: 0.20, sx: 0.045, sy: 0.037, ph: 0.0, r: 0.55 },
      { color: 'rgba(99,230,190,0.16)', ax: 0.30, ay: 0.24, sx: 0.033, sy: 0.051, ph: 1.7, r: 0.5 },
      { color: 'rgba(177,151,252,0.16)', ax: 0.24, ay: 0.28, sx: 0.057, sy: 0.029, ph: 3.1, r: 0.5 },
      { color: 'rgba(80,120,220,0.14)', ax: 0.34, ay: 0.16, sx: 0.026, sy: 0.044, ph: 4.6, r: 0.6 },
    ]

    function resize() {
      w = canvas.clientWidth; h = canvas.clientHeight
      canvas.width = Math.max(1, w * dpr); canvas.height = Math.max(1, h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener('resize', resize)

    const start = performance.now()
    function frame(now) {
      const t = (now - start) / 1000
      ctx.clearRect(0, 0, w, h)
      const base = Math.min(w, h)

      ctx.globalCompositeOperation = 'lighter'
      for (const o of orbs) {
        const x = w * (0.5 + o.ax * Math.sin(t * o.sx * 6.28 + o.ph))
        const y = h * (0.5 + o.ay * Math.cos(t * o.sy * 6.28 + o.ph * 1.3))
        const rad = base * o.r
        const g = ctx.createRadialGradient(x, y, 0, x, y, rad)
        g.addColorStop(0, o.color)
        g.addColorStop(1, 'transparent')
        ctx.fillStyle = g
        ctx.fillRect(0, 0, w, h)
      }

      // block-arrival ripples
      const cx = w / 2, cy = h * 0.42
      ripples.current = ripples.current.filter((rp) => {
        const age = (now - rp.t0) / 1000
        if (age > 2.4) return false
        const rr = age * base * 0.5
        const alpha = Math.max(0, 0.16 * (1 - age / 2.4))
        ctx.globalCompositeOperation = 'lighter'
        ctx.beginPath()
        ctx.arc(cx, cy, rr, 0, 6.2832)
        ctx.strokeStyle = `rgba(122,162,255,${alpha})`
        ctx.lineWidth = 2
        ctx.stroke()
        return true
      })

      ctx.globalCompositeOperation = 'source-over'
      raf = requestAnimationFrame(frame)
    }

    if (reduce) {
      // static single paint
      frame(start)
    } else {
      raf = requestAnimationFrame(frame)
    }
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [])

  return <canvas className="aurora" ref={ref} aria-hidden />
}
