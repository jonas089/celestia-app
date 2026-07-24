// Presentation + light client-side decoding helpers. Nothing here fabricates
// data — the rv32 disassembler simply decodes the on-DA program words so the
// explorer can show what the VM actually executed.

export function truncHex(hex, head = 10, tail = 6) {
  if (!hex) return ''
  const s = String(hex)
  if (s.length <= head + tail + 1) return s
  return `${s.slice(0, head)}…${s.slice(-tail)}`
}

export function shortEq(a, b) {
  return a && b && a.toLowerCase() === b.toLowerCase()
}

export function fmtBytes(n) {
  if (!n) return '0 B'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

export function fmtDuration(ms) {
  if (ms == null) return '—'
  if (ms < 1000) return `${ms} ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)} s`
  const m = Math.floor(s / 60)
  return `${m}m ${Math.round(s % 60)}s`
}

export function timeAgo(unixSec) {
  if (!unixSec) return ''
  const d = Math.max(0, Math.floor(Date.now() / 1000) - unixSec)
  if (d < 2) return 'just now'
  if (d < 60) return `${d}s ago`
  if (d < 3600) return `${Math.floor(d / 60)}m ago`
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`
  return `${Math.floor(d / 86400)}d ago`
}

// Deterministic palette so each distinct program gets a stable, distinct colour.
const BAND_COLORS = [
  '#6ea8fe', '#63e6be', '#ffd43b', '#ff8787',
  '#b197fc', '#69db7c', '#ffa94d', '#4dd4ff',
]
export function bandColor(i) {
  return BAND_COLORS[i % BAND_COLORS.length]
}

// ---- RV32IM disassembler (compact) --------------------------------------
// Decodes the on-DA program (hex of little-endian u32 words) into readable
// mnemonics. Covers the RV32I base + M extension the rollup's emulator runs.

const ABI = [
  'zero', 'ra', 'sp', 'gp', 'tp', 't0', 't1', 't2',
  's0', 's1', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5',
  'a6', 'a7', 's2', 's3', 's4', 's5', 's6', 's7',
  's8', 's9', 's10', 's11', 't3', 't4', 't5', 't6',
]
const r = (n) => ABI[n & 31]
const signed = (v, bits) => (v & (1 << (bits - 1)) ? v - (1 << bits) : v)

export function hexToWordsLE(hex) {
  if (!hex) return []
  let s = hex.startsWith('0x') ? hex.slice(2) : hex
  if (s.length % 2) s = '0' + s
  const bytes = []
  for (let i = 0; i < s.length; i += 2) bytes.push(parseInt(s.slice(i, i + 2), 16))
  const words = []
  for (let i = 0; i + 4 <= bytes.length; i += 4) {
    words.push((bytes[i] | (bytes[i + 1] << 8) | (bytes[i + 2] << 16) | (bytes[i + 3] << 24)) >>> 0)
  }
  return words
}

export function disasm(word) {
  const w = word >>> 0
  const op = w & 0x7f
  const rd = (w >>> 7) & 31
  const f3 = (w >>> 12) & 7
  const rs1 = (w >>> 15) & 31
  const rs2 = (w >>> 20) & 31
  const f7 = (w >>> 25) & 127
  const iImm = signed(w >>> 20, 12)
  switch (op) {
    case 0x37: return `lui   ${r(rd)}, 0x${((w >>> 12) >>> 0).toString(16)}`
    case 0x17: return `auipc ${r(rd)}, 0x${((w >>> 12) >>> 0).toString(16)}`
    case 0x6f: {
      const imm = signed(
        (((w >>> 31) & 1) << 20) | (((w >>> 12) & 0xff) << 12) |
        (((w >>> 20) & 1) << 11) | (((w >>> 21) & 0x3ff) << 1), 21)
      return `jal   ${r(rd)}, ${imm}`
    }
    case 0x67: return `jalr  ${r(rd)}, ${iImm}(${r(rs1)})`
    case 0x63: {
      const imm = signed(
        (((w >>> 31) & 1) << 12) | (((w >>> 7) & 1) << 11) |
        (((w >>> 25) & 0x3f) << 5) | (((w >>> 8) & 0xf) << 1), 13)
      const m = { 0: 'beq', 1: 'bne', 4: 'blt', 5: 'bge', 6: 'bltu', 7: 'bgeu' }[f3] || 'b?'
      return `${m.padEnd(5)} ${r(rs1)}, ${r(rs2)}, ${imm}`
    }
    case 0x03: {
      const m = { 0: 'lb', 1: 'lh', 2: 'lw', 4: 'lbu', 5: 'lhu' }[f3] || 'l?'
      return `${m.padEnd(5)} ${r(rd)}, ${iImm}(${r(rs1)})`
    }
    case 0x23: {
      const imm = signed(((f7 << 5) | rd), 12)
      const m = { 0: 'sb', 1: 'sh', 2: 'sw' }[f3] || 's?'
      return `${m.padEnd(5)} ${r(rs2)}, ${imm}(${r(rs1)})`
    }
    case 0x13: {
      const m = { 0: 'addi', 2: 'slti', 3: 'sltiu', 4: 'xori', 6: 'ori', 7: 'andi' }[f3]
      if (m) return `${m.padEnd(5)} ${r(rd)}, ${r(rs1)}, ${iImm}`
      if (f3 === 1) return `slli  ${r(rd)}, ${r(rs1)}, ${rs2}`
      if (f3 === 5) return `${f7 ? 'srai' : 'srli'} ${r(rd)}, ${r(rs1)}, ${rs2}`
      return 'op-imm?'
    }
    case 0x33: {
      if (f7 === 1) {
        const m = ['mul', 'mulh', 'mulhsu', 'mulhu', 'div', 'divu', 'rem', 'remu'][f3]
        return `${m.padEnd(5)} ${r(rd)}, ${r(rs1)}, ${r(rs2)}`
      }
      let m
      if (f3 === 0) m = f7 ? 'sub' : 'add'
      else if (f3 === 5) m = f7 ? 'sra' : 'srl'
      else m = { 1: 'sll', 2: 'slt', 3: 'sltu', 4: 'xor', 6: 'or', 7: 'and' }[f3] || 'op?'
      return `${m.padEnd(5)} ${r(rd)}, ${r(rs1)}, ${r(rs2)}`
    }
    case 0x73: return f3 === 0 ? (w & 0x100000 ? 'ebreak' : 'ecall') : `csr f3=${f3}`
    case 0x0f: return 'fence'
    default: return `.word 0x${w.toString(16).padStart(8, '0')}`
  }
}

// Opcode-class histogram for a program's words — a quick "shape" of the code.
export function opClasses(words) {
  const bucket = { alu: 0, mem: 0, branch: 0, jump: 0, mul: 0, other: 0 }
  for (const w of words) {
    const op = w & 0x7f
    const f7 = (w >>> 25) & 127
    if (op === 0x33 && f7 === 1) bucket.mul++
    else if (op === 0x33 || op === 0x13 || op === 0x37 || op === 0x17) bucket.alu++
    else if (op === 0x03 || op === 0x23) bucket.mem++
    else if (op === 0x63) bucket.branch++
    else if (op === 0x6f || op === 0x67) bucket.jump++
    else bucket.other++
  }
  return bucket
}
