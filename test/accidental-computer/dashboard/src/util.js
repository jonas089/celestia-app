// Small presentation helpers. No formatting of *values* — only display trimming.

export function truncHex(hex, head = 10, tail = 6) {
  if (!hex) return ''
  const s = String(hex)
  if (s.length <= head + tail + 1) return s
  return `${s.slice(0, head)}…${s.slice(-tail)}`
}

export function shortEq(a, b) {
  return a && b && a.toLowerCase() === b.toLowerCase()
}

// Deterministic palette so each rollup band gets a stable, distinct colour.
const BAND_COLORS = [
  '#6ea8fe', // blue
  '#63e6be', // teal
  '#ffd43b', // amber
  '#ff8787', // red
  '#b197fc', // violet
  '#69db7c', // green
  '#ffa94d', // orange
  '#4dd4ff', // cyan
]

export function bandColor(i) {
  return BAND_COLORS[i % BAND_COLORS.length]
}
