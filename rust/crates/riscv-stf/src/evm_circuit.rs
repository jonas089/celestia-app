//! R8 (circuit) — in-circuit EVM step-function over GF2, matching the native
//! `evm` interpreter (which is byte-faithful to ev-reth's ev-revm). Executes
//! committed bytecode by threading (stack, sp, pc, storage, halted) across a
//! bounded number of unrolled steps; every wire but the committed input layer is
//! pinned by sumcheck. Deterministic — no advice.
//!
//! Efficiency: bounds are tight (small stack/storage/steps sized to the test
//! program), and each step reads one opcode + one immediate byte via a mux over
//! the (small) committed code, computes the implemented opcodes' effects, and
//! selects by a one-hot. Grows opcode-by-opcode; each rung diff-tested vs native.

use crate::u256::{add, divmod, eq, is_zero, lt, mul_full, mul_low, select, sub, BITS};
use expander_compiler::frontend::*;

pub type Word = Vec<Variable>; // 256-bit LE

/// A bounded EVM machine state threaded through the step function.
pub struct Evm {
    pub stack: Vec<Word>, // [SD] words
    pub sp: Vec<Variable>, // stack depth as a small LE value (SP_BITS)
    pub pc: Vec<Variable>, // program counter, small LE value (PC_BITS)
    pub skeys: Vec<Word>,  // storage keys [SS]
    pub svals: Vec<Word>,  // storage values [SS]
    pub mem: Vec<Vec<Variable>>, // [MEM_BYTES] bytes (each 8 bits)
    pub ret: Vec<Vec<Variable>>, // [MEM_BYTES] captured RETURN data bytes
    pub ret_len: Vec<Variable>,  // RETURN length (PC_BITS value)
    pub halted: Variable,
}

pub struct Cfg {
    pub sd: usize,       // stack depth bound
    pub ss: usize,       // storage slots bound
    pub code_len: usize, // committed code length
    pub mem_bytes: usize, // memory byte bound
    pub sp_bits: usize,
    pub pc_bits: usize,
}

/// Per-step context that is NOT threaded state: committed environment inputs
/// (calldata/caller/callvalue/address) plus compile-time enables for the heavy
/// opcode groups (DIV family, ADDMOD/MULMOD, EXP, KECCAK256). This lives OUTSIDE
/// `Evm`/`Cfg` deliberately: `Evm`/`Cfg` are constructed by other modules
/// (`block_stf`, `evm_gkr`) via field literals and `run()` keeps its original
/// signature, so extending them would break those callers. Threading env +
/// enables here keeps the light opcode set cheap (heavy gadgets are only wired
/// when the corresponding enable is set) without touching any other file.
#[derive(Clone)]
pub struct StepOpts {
    pub calldata: Vec<Vec<Variable>>, // committed calldata bytes (each 8 bits)
    pub caller: Word,
    pub callvalue: Word,
    pub address: Word,
    pub en_div: bool,    // DIV MOD SDIV SMOD  (one 256-bit long-division / step)
    pub en_mulmod: bool, // ADDMOD MULMOD      (one 512-bit long-division + mul / step)
    pub en_exp: bool,    // EXP                (256 square-and-multiply / step)
    pub en_keccak: bool, // KECCAK256          (one keccak_f / step)
}
impl StepOpts {
    /// Empty env, all heavy opcodes disabled (used by the legacy `run`).
    pub fn empty<C: Config>(api: &mut impl RootAPI<C>) -> Self {
        StepOpts {
            calldata: vec![],
            caller: word_zero(api),
            callvalue: word_zero(api),
            address: word_zero(api),
            en_div: false,
            en_mulmod: false,
            en_exp: false,
            en_keccak: false,
        }
    }
}

fn word_zero<C: Config>(api: &mut impl RootAPI<C>) -> Word {
    vec![api.constant(0); BITS]
}
/// A 256-bit constant word (LE bits) for a small u64 value.
fn const_word<C: Config>(api: &mut impl RootAPI<C>, v: u64) -> Word {
    (0..BITS).map(|b| if b < 64 { api.constant(((v >> b) & 1) as u32) } else { api.constant(0) }).collect()
}
/// Signed 256-bit less-than: a < b interpreting both as two's-complement.
/// If signs differ, the negative one is smaller; else compare unsigned.
fn signed_lt<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Variable {
    let sign_a = a[BITS - 1];
    let sign_b = b[BITS - 1];
    let ult = lt(api, a, b);
    let diff_sign = api.add(sign_a, sign_b); // XOR in GF2
    // diff_sign ? sign_a : ult
    let t1 = api.mul(diff_sign, sign_a);
    let nd = api.sub(1, diff_sign);
    let t2 = api.mul(nd, ult);
    api.add(t1, t2)
}
fn small_const<C: Config>(api: &mut impl RootAPI<C>, v: u32, n: usize) -> Vec<Variable> {
    (0..n).map(|b| api.constant((v >> b) & 1)).collect()
}
fn small_eq<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], v: u32) -> Variable {
    let mut acc = api.constant(1);
    for (b, &ab) in a.iter().enumerate() {
        let want = (v >> b) & 1;
        let t = if want == 1 { ab } else { api.sub(1, ab) };
        acc = api.mul(acc, t);
    }
    acc
}
fn small_add_const<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], c: u32) -> Vec<Variable> {
    // a + c over a.len() bits (c small).
    let n = a.len();
    let cb: Vec<Variable> = (0..n).map(|b| api.constant((c >> b) & 1)).collect();
    let mut out = Vec::with_capacity(n);
    let mut carry = api.constant(0);
    for i in 0..n {
        let ab = api.add(a[i], cb[i]);
        let s = api.add(ab, carry);
        let and = api.mul(a[i], cb[i]);
        let cc = api.mul(carry, ab);
        carry = api.add(and, cc);
        out.push(s);
    }
    out
}
fn small_sub_const<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], c: u32) -> Vec<Variable> {
    let n = a.len();
    let nb: Vec<Variable> = (0..n).map(|b| { let cb = api.constant((c >> b) & 1); api.sub(1, cb) }).collect();
    let mut out = Vec::with_capacity(n);
    let mut carry = api.constant(1);
    for i in 0..n {
        let ab = api.add(a[i], nb[i]);
        let s = api.add(ab, carry);
        let and = api.mul(a[i], nb[i]);
        let cc = api.mul(carry, ab);
        carry = api.add(and, cc);
        out.push(s);
    }
    out
}

/// Read code byte at a dynamic index `idx` (LE bits) via a mux over `code` bytes
/// (each 8 committed Variables). Returns 8 bits.
fn read_code<C: Config>(api: &mut impl RootAPI<C>, code: &[Vec<Variable>], idx: &[Variable]) -> Vec<Variable> {
    let mut out = vec![api.constant(0); 8];
    for (j, byte) in code.iter().enumerate() {
        let sel = small_eq(api, idx, j as u32);
        for b in 0..8 {
            let t = api.mul(sel, byte[b]);
            out[b] = api.add(out[b], t);
        }
    }
    out
}

/// Read the stack word at a dynamic index `idx` (small LE value) via mux.
fn read_stack_at<C: Config>(api: &mut impl RootAPI<C>, stack: &[Word], idx: &[Variable]) -> Word {
    let mut out = word_zero(api);
    for (i, w) in stack.iter().enumerate() {
        let sel = small_eq(api, idx, i as u32);
        for b in 0..BITS { let t = api.mul(sel, w[b]); out[b] = api.add(out[b], t); }
    }
    out
}
/// Read stack word at depth `sp - 1 - k` (top is k=0).
fn read_stack<C: Config>(api: &mut impl RootAPI<C>, st: &Evm, k: u32) -> Word {
    let idx = small_sub_const(api, &st.sp, 1 + k);
    read_stack_at(api, &st.stack, &idx)
}
/// a - b for small LE values. `b` may be shorter than `a` (missing high bits
/// treated as 0); result has `a.len()` bits.
fn small_sub<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let n = a.len();
    let zero = api.constant(0);
    let nb: Vec<Variable> = (0..n).map(|i| { let bi = *b.get(i).unwrap_or(&zero); api.sub(1, bi) }).collect();
    let mut out = Vec::with_capacity(n);
    let mut carry = api.constant(1);
    for i in 0..n {
        let ab = api.add(a[i], nb[i]); let s = api.add(ab, carry);
        let and = api.mul(a[i], nb[i]); let cc = api.mul(carry, ab); carry = api.add(and, cc);
        out.push(s);
    }
    out
}

/// Write `val` into stack position `pos` (LE value); returns the new stack array
/// (position pos replaced, others kept).
fn write_stack<C: Config>(api: &mut impl RootAPI<C>, stack: &[Word], pos: &[Variable], val: &Word) -> Vec<Word> {
    (0..stack.len())
        .map(|i| {
            let sel = small_eq(api, pos, i as u32);
            select(api, sel, val, &stack[i])
        })
        .collect()
}

fn shl8(w: &[Variable], zero: Variable) -> Word {
    (0..BITS).map(|b| if b >= 8 { w[b - 8] } else { zero }).collect()
}
fn set_low_byte(w: &[Variable], byte: &[Variable]) -> Word {
    (0..BITS).map(|b| if b < 8 { byte[b] } else { w[b] }).collect()
}
fn word_le_byte(w: &[Variable], k: usize) -> Vec<Variable> {
    w[k * 8..k * 8 + 8].to_vec()
}
/// Logical shift-right by a constant k (LE bit vector).
fn shr_const(w: &[Variable], k: usize, zero: Variable) -> Word {
    (0..BITS).map(|b| if b + k < BITS { w[b + k] } else { zero }).collect()
}
/// Barrel shift-left by a dynamic amount given by the low 8 bits `s` (LE). If the
/// full shift word has any bit >= 8 set (`ge256` == 1), the result is forced 0.
fn barrel_shl<C: Config>(api: &mut impl RootAPI<C>, val: &Word, s: &[Variable], ge256: Variable) -> Word {
    let mut acc = val.clone();
    for k in 0..8 {
        let shifted = crate::u256::shl_const(api, &acc, 1usize << k);
        acc = select(api, s[k], &shifted, &acc);
    }
    // zero out if shift >= 256
    let keep = api.sub(1, ge256);
    (0..BITS).map(|b| api.mul(keep, acc[b])).collect()
}
/// Barrel logical shift-right by a dynamic amount (low 8 bits `s`), forced 0 if `ge256`.
fn barrel_shr<C: Config>(api: &mut impl RootAPI<C>, val: &Word, s: &[Variable], ge256: Variable, zero: Variable) -> Word {
    let mut acc = val.clone();
    for k in 0..8 {
        let shifted = shr_const(&acc, 1usize << k, zero);
        acc = select(api, s[k], &shifted, &acc);
    }
    let keep = api.sub(1, ge256);
    (0..BITS).map(|b| api.mul(keep, acc[b])).collect()
}
/// Read memory byte at dynamic index `idx` (small LE value) via mux over mem.
fn read_mem_byte<C: Config>(api: &mut impl RootAPI<C>, mem: &[Vec<Variable>], idx: &[Variable]) -> Vec<Variable> {
    let mut out = vec![api.constant(0); 8];
    for (p, byte) in mem.iter().enumerate() {
        let sel = small_eq(api, idx, p as u32);
        for b in 0..8 { let t = api.mul(sel, byte[b]); out[b] = api.add(out[b], t); }
    }
    out
}

/// Read calldata byte at dynamic index `idx` (small LE value) via mux over the
/// committed calldata bytes. Returns 0 for out-of-range indices (empty calldata
/// => always 0), matching native `ctx.calldata` OOB semantics.
fn read_calldata_byte<C: Config>(api: &mut impl RootAPI<C>, cd: &[Vec<Variable>], idx: &[Variable]) -> Vec<Variable> {
    let mut out = vec![api.constant(0); 8];
    for (p, byte) in cd.iter().enumerate() {
        let sel = small_eq(api, idx, p as u32);
        for b in 0..8 { let t = api.mul(sel, byte[b]); out[b] = api.add(out[b], t); }
    }
    out
}

/// 256-bit two's-complement negation: (0 - x) mod 2^256.
fn neg_word<C: Config>(api: &mut impl RootAPI<C>, x: &Word) -> Word {
    let z = word_zero(api);
    let (d, _b) = sub(api, &z, x);
    d
}

/// One EVM step over committed `code`, threading `st` and reading committed
/// environment / heavy-opcode enables from `opts`. Implements the standard
/// opcode set diff-tested against `crate::evm` (see module tests for the exact
/// list). `CALL*`/`CREATE*` remain the one architectural gap (see the comment at
/// the end of this function).
pub fn step<C: Config>(api: &mut impl RootAPI<C>, st: Evm, code: &[Vec<Variable>], cfg: &Cfg, opts: &StepOpts) -> Evm {
    let zero = api.constant(0);
    let op = read_code(api, code, &st.pc);
    // opcode selectors
    let is_stop = small_eq(api, &op, 0x00);
    let is_add = small_eq(api, &op, 0x01);
    let is_mul = small_eq(api, &op, 0x02);
    let is_sub = small_eq(api, &op, 0x03);
    let is_lt = small_eq(api, &op, 0x10);
    let is_gt = small_eq(api, &op, 0x11);
    let is_eq = small_eq(api, &op, 0x14);
    let is_iszero = small_eq(api, &op, 0x15);
    let is_and = small_eq(api, &op, 0x16);
    let is_or = small_eq(api, &op, 0x17);
    let is_xor = small_eq(api, &op, 0x18);
    let is_pop = small_eq(api, &op, 0x50);
    let is_mload = small_eq(api, &op, 0x51);
    let is_mstore = small_eq(api, &op, 0x52);
    let is_sload = small_eq(api, &op, 0x54);
    let is_sstore = small_eq(api, &op, 0x55);
    let is_jump = small_eq(api, &op, 0x56);
    let is_jumpi = small_eq(api, &op, 0x57);
    let is_jumpdest = small_eq(api, &op, 0x5b);
    let is_return = small_eq(api, &op, 0xf3);
    // tranche-1 extra opcodes
    let is_not = small_eq(api, &op, 0x19);
    let is_byte = small_eq(api, &op, 0x1a);
    let is_shl = small_eq(api, &op, 0x1b);
    let is_shr = small_eq(api, &op, 0x1c);
    let is_sar = small_eq(api, &op, 0x1d);
    let is_signext = small_eq(api, &op, 0x0b);
    let is_slt = small_eq(api, &op, 0x12);
    let is_sgt = small_eq(api, &op, 0x13);
    let is_codecopy = small_eq(api, &op, 0x39);
    // tranche-2 arithmetic (heavy, gated by opts)
    let is_div = small_eq(api, &op, 0x04);
    let is_sdiv = small_eq(api, &op, 0x05);
    let is_mod = small_eq(api, &op, 0x06);
    let is_smod = small_eq(api, &op, 0x07);
    let is_addmod = small_eq(api, &op, 0x08);
    let is_mulmod = small_eq(api, &op, 0x09);
    let is_exp = small_eq(api, &op, 0x0a);
    // memory / pc / misc
    let is_mstore8 = small_eq(api, &op, 0x53);
    let is_pc = small_eq(api, &op, 0x58);
    let is_keccak = small_eq(api, &op, 0x20);
    let is_revert = small_eq(api, &op, 0xfd);
    // environment / context
    let is_address = small_eq(api, &op, 0x30);
    let is_caller = small_eq(api, &op, 0x33);
    let is_callvalue = small_eq(api, &op, 0x34);
    let is_calldataload = small_eq(api, &op, 0x35);
    let is_calldatasize = small_eq(api, &op, 0x36);
    let is_calldatacopy = small_eq(api, &op, 0x37);
    let is_codesize = small_eq(api, &op, 0x38);
    // LOG0..LOG4 (0xa0..0xa4): stack effect only (pop 2 + ntopics); each variant
    // handled directly in the sp-update chain via `log_sel[t]`.
    let log_sel: Vec<Variable> = (0..5).map(|t| small_eq(api, &op, 0xa0 + t)).collect();
    // DUP1-16 = 0x80..0x8f (bits4-7 = 1000); SWAP1-16 = 0x90..0x9f (bits4-7 = 1001).
    let is_dup = { let b6 = api.sub(1, op[6]); let b5 = api.sub(1, op[5]); let b4 = api.sub(1, op[4]); let t = api.mul(op[7], b6); let t = api.mul(t, b5); api.mul(t, b4) };
    let is_swap = { let b6 = api.sub(1, op[6]); let b5 = api.sub(1, op[5]); let t = api.mul(op[7], b6); let t = api.mul(t, b5); api.mul(t, op[4]) };
    // n = (op & 0x0f) + 1 for DUP/SWAP (6-bit value).
    let dn: Vec<Variable> = { let mut v = op[0..4].to_vec(); v.push(zero); v.push(zero); v };
    let n_val = small_add_const(api, &dn, 1);
    // PUSHn: op in [0x60,0x7f] => op[7]==0, op[6]==1, op[5]==1.
    let is_push = { let n7 = api.sub(1, op[7]); let t = api.mul(n7, op[6]); api.mul(t, op[5]) };
    // n-1 = low 5 bits of op; n in 1..=32.
    let nm1: Vec<Variable> = op[0..5].to_vec();

    let active = api.sub(1, st.halted);

    let top = read_stack(api, &st, 0);
    let second = read_stack(api, &st, 1);

    // --- PUSHn: assemble the n-byte big-endian immediate at code[pc+1 .. pc+1+n] ---
    let mut push_word = word_zero(api);
    for i in 0..32 {
        let idx = small_add_const(api, &st.pc, (i + 1) as u32);
        let imm_byte = read_code(api, code, &idx);
        // use byte i iff i < n  <=>  i <= nm1  <=>  NOT (nm1 < i)
        let nm1_lt_i = { // nm1 < i (constant i)
            let ib = small_const(api, i as u32, 5);
            let (_d, borrow) = { // nm1 - i borrow => nm1 < i
                let nb: Vec<Variable> = ib.iter().map(|&x| api.sub(1, x)).collect();
                let mut carry = api.constant(1); let mut out = vec![];
                for k in 0..5 { let ab = api.add(nm1[k], nb[k]); let s = api.add(ab, carry); let a2 = api.mul(nm1[k], nb[k]); let c2 = api.mul(carry, ab); carry = api.add(a2, c2); out.push(s); }
                (out, api.sub(1, carry))
            };
            borrow
        };
        let use_i = api.sub(1, nm1_lt_i); // i < n
        let shifted = shl8(&push_word, zero);
        let ored = set_low_byte(&shifted, &imm_byte);
        push_word = select(api, use_i, &ored, &push_word);
    }

    // --- binary ops (all pop2, push1 at sp-2): ADD/MUL/SUB/LT/GT/EQ/AND/OR/XOR ---
    let sum = { let s = add(api, &top, &second); s[0..BITS].to_vec() };
    let mul_v = { let p = mul_full(api, &top, &second); p[0..BITS].to_vec() };
    let sub_v = { let (d, _b) = sub(api, &top, &second); d }; // top - second mod 2^256
    let lt_b = lt(api, &top, &second);
    let gt_b = lt(api, &second, &top);
    let eq_b = eq(api, &top, &second);
    let iszero_b = is_zero(api, &top);
    let and_v: Vec<Variable> = (0..BITS).map(|b| api.mul(top[b], second[b])).collect();
    let or_v: Vec<Variable> = (0..BITS).map(|b| { let x = api.add(top[b], second[b]); let y = api.mul(top[b], second[b]); api.add(x, y) }).collect();
    let xor_v: Vec<Variable> = (0..BITS).map(|b| api.add(top[b], second[b])).collect();
    // NOT (unary): complement each bit of top.
    let not_v: Vec<Variable> = (0..BITS).map(|b| api.sub(1, top[b])).collect();
    // Shift amount for SHL/SHR/SAR: low 8 bits of `top`, and ge256 iff any higher bit set.
    let s8: Vec<Variable> = top[0..8].to_vec();
    let top_hi = top[8..BITS].to_vec();
    let hi_zero = is_zero(api, &top_hi);
    let ge256 = api.sub(1, hi_zero);
    // SHL/SHR operate on `second` (the value), shifted by `top`.
    let shl_v = barrel_shl(api, &second, &s8, ge256);
    let shr_v = barrel_shr(api, &second, &s8, ge256, zero);
    // SAR: arithmetic right shift = logical shr with high `shift` bits set to sign.
    let sar_v = {
        let sign = second[BITS - 1];
        let ones = vec![api.constant(1); BITS];
        let lshr_ones = barrel_shr(api, &ones, &s8, ge256, zero); // low (256-s) bits set
        // high_mask = NOT(lshr_ones): high s bits (and all bits when ge256)
        let high_mask: Vec<Variable> = (0..BITS).map(|b| api.sub(1, lshr_ones[b])).collect();
        (0..BITS).map(|b| { let sm = api.mul(sign, high_mask[b]); api.add(shr_v[b], sm) }).collect::<Vec<_>>()
    };
    // BYTE(i=top, x=second): byte (31-i) of x if i<32 else 0.
    let byte_v = {
        let mut out = vec![api.constant(0); BITS];
        for i in 0..32usize {
            let ci = const_word(api, i as u64);
            let sel = eq(api, &top, &ci);
            // LE byte (31 - i) of second -> low 8 bits of result
            let src = (31 - i) * 8;
            for b in 0..8 { let t = api.mul(sel, second[src + b]); out[b] = api.add(out[b], t); }
        }
        out
    };
    // SIGNEXTEND(i=top, x=second): extend x from byte i's high bit.
    let signext_v = {
        // default = x (covers i >= 31 and i >= 32)
        let mut out = second.clone();
        for i in 0..31usize {
            let ci = const_word(api, i as u64);
            let sel = eq(api, &top, &ci);
            let bit = (i + 1) * 8 - 1;
            let sign = second[bit];
            // ext[b] = b <= bit ? second[b] : sign
            let ext: Vec<Variable> = (0..BITS).map(|b| if b <= bit { second[b] } else { sign }).collect();
            out = select(api, sel, &ext, &out);
        }
        out
    };
    // SLT/SGT (signed): compare via sign bits + unsigned magnitude.
    let slt_bit = signed_lt(api, &top, &second);
    let sgt_bit = signed_lt(api, &second, &top);
    let mut slt_v = vec![api.constant(0); BITS]; slt_v[0] = slt_bit;
    let mut sgt_v = vec![api.constant(0); BITS]; sgt_v[0] = sgt_bit;
    let mut lt_v = vec![api.constant(0); BITS]; lt_v[0] = lt_b;
    let mut gt_v = vec![api.constant(0); BITS]; gt_v[0] = gt_b;
    let mut eq_v = vec![api.constant(0); BITS]; eq_v[0] = eq_b;
    let mut iszero_v = vec![api.constant(0); BITS]; iszero_v[0] = iszero_b;

    // --- DIV/MOD/SDIV/SMOD (pop2 push1): one shared 256-bit long division ---
    // DIV/MOD are unsigned (dividend=top, divisor=second). SDIV/SMOD operate on
    // magnitudes then re-apply signs. A single `divmod` per step handles both by
    // muxing its operands, so the gate cost is one long-division regardless.
    let (mut div_v, mut mod_v, mut sdiv_v, mut smod_v) = (word_zero(api), word_zero(api), word_zero(api), word_zero(api));
    if opts.en_div {
        let sa = top[BITS - 1];
        let sb = second[BITS - 1];
        let neg_top = neg_word(api, &top);
        let neg_sec = neg_word(api, &second);
        let abs_a = select(api, sa, &neg_top, &top);
        let abs_b = select(api, sb, &neg_sec, &second);
        let is_signed_div = { let s = api.add(is_sdiv, is_smod); s };
        let dividend = select(api, is_signed_div, &abs_a, &top);
        let divisor = select(api, is_signed_div, &abs_b, &second);
        let (q, r) = divmod(api, &dividend, &divisor);
        div_v = q.clone();
        mod_v = r.clone();
        // SDIV sign = sa XOR sb; SMOD sign = sa (remainder follows dividend).
        let qsign = api.add(sa, sb);
        let neg_q = neg_word(api, &q);
        let neg_r = neg_word(api, &r);
        sdiv_v = select(api, qsign, &neg_q, &q);
        smod_v = select(api, sa, &neg_r, &r);
    }

    // --- EXP (pop2 push1): base=top, exp=second, base^exp mod 2^256 ---
    // Square-and-multiply over all 256 exponent bits (exp is a witness, so every
    // bit must be handled for correctness). Two 256-bit truncating muls per bit.
    let mut exp_v = word_zero(api);
    if opts.en_exp {
        let mut result = const_word(api, 1);
        let mut sq = top.clone();
        for i in 0..BITS {
            let bit = second[i];
            let prod = mul_low(api, &result, &sq, BITS);
            result = select(api, bit, &prod, &result);
            if i + 1 < BITS {
                sq = mul_low(api, &sq, &sq, BITS);
            }
        }
        exp_v = result;
    }

    // combine binary results by opcode
    let mut bin = sum.clone();
    bin = select(api, is_mul, &mul_v, &bin);
    bin = select(api, is_sub, &sub_v, &bin);
    bin = select(api, is_lt, &lt_v, &bin);
    bin = select(api, is_gt, &gt_v, &bin);
    bin = select(api, is_eq, &eq_v, &bin);
    bin = select(api, is_and, &and_v, &bin);
    bin = select(api, is_or, &or_v, &bin);
    bin = select(api, is_xor, &xor_v, &bin);
    bin = select(api, is_shl, &shl_v, &bin);
    bin = select(api, is_shr, &shr_v, &bin);
    bin = select(api, is_sar, &sar_v, &bin);
    bin = select(api, is_byte, &byte_v, &bin);
    bin = select(api, is_signext, &signext_v, &bin);
    bin = select(api, is_slt, &slt_v, &bin);
    bin = select(api, is_sgt, &sgt_v, &bin);
    bin = select(api, is_div, &div_v, &bin);
    bin = select(api, is_mod, &mod_v, &bin);
    bin = select(api, is_sdiv, &sdiv_v, &bin);
    bin = select(api, is_smod, &smod_v, &bin);
    bin = select(api, is_exp, &exp_v, &bin);
    let is_bin = {
        let mut s = is_add;
        for x in [is_mul, is_sub, is_lt, is_gt, is_eq, is_and, is_or, is_xor,
                  is_shl, is_shr, is_sar, is_byte, is_signext, is_slt, is_sgt,
                  is_div, is_mod, is_sdiv, is_smod, is_exp] { s = api.add(s, x); }
        s // selectors mutually exclusive => sum == OR
    };

    // --- ADDMOD/MULMOD (pop3 push1): (a+b)%n and (a*b)%n, n==0 => 0 ---
    // One shared long division over a 512-bit dividend (the padded sum for
    // ADDMOD, the full product for MULMOD), muxed by opcode.
    let n3 = read_stack(api, &st, 2); // third stack item = modulus n
    let mut ter_v = word_zero(api); // (result at sp-3)
    if opts.en_mulmod {
        let sum257 = add(api, &top, &second); // 257 bits
        let sum512: Vec<Variable> = (0..512).map(|i| if i < sum257.len() { sum257[i] } else { api.constant(0) }).collect();
        let prod = mul_full(api, &top, &second); // <= 512 bits
        let prod512: Vec<Variable> = (0..512).map(|i| if i < prod.len() { prod[i] } else { api.constant(0) }).collect();
        let dividend = select(api, is_mulmod, &prod512, &sum512);
        let (_q, r) = divmod(api, &dividend, &n3); // r is 256 bits (n3.len()==BITS)
        ter_v = r;
    }
    let is_ter = { let s = api.add(is_addmod, is_mulmod); s };

    // SLOAD: storage[top]
    let mut sload_val = word_zero(api);
    for i in 0..cfg.ss {
        let m = eq(api, &st.skeys[i], &top);
        for b in 0..BITS { let t = api.mul(m, st.svals[i][b]); sload_val[b] = api.add(sload_val[b], t); }
    }

    // MLOAD: word whose LE byte j = mem[off + 31 - j], off = low bits of top.
    let off_idx: Vec<Variable> = top[0..cfg.sp_bits.max(6)].to_vec();
    let mut mload_val = word_zero(api);
    for j in 0..32 {
        let idx = small_add_const(api, &off_idx, (31 - j) as u32);
        let byte = read_mem_byte(api, &st.mem, &idx);
        for b in 0..8 { mload_val[j * 8 + b] = byte[b]; }
    }

    // --- environment / context push values ---
    let cds_word = const_word(api, opts.calldata.len() as u64); // CALLDATASIZE
    let cs_word = const_word(api, cfg.code_len as u64);         // CODESIZE
    // PC pushes the index of the PC opcode itself (= current st.pc, pre-advance).
    let pc_word: Word = (0..BITS).map(|b| if b < st.pc.len() { st.pc[b] } else { zero }).collect();
    let mut env_push = word_zero(api);
    env_push = select(api, is_address, &opts.address, &env_push);
    env_push = select(api, is_caller, &opts.caller, &env_push);
    env_push = select(api, is_callvalue, &opts.callvalue, &env_push);
    env_push = select(api, is_calldatasize, &cds_word, &env_push);
    env_push = select(api, is_codesize, &cs_word, &env_push);
    env_push = select(api, is_pc, &pc_word, &env_push);
    let is_env_push = { let mut s = is_address; for x in [is_caller, is_callvalue, is_calldatasize, is_codesize, is_pc] { s = api.add(s, x); } s };

    // CALLDATALOAD: 32 bytes big-endian at off=top; LE byte j = calldata[off+31-j].
    let mut cdl_val = word_zero(api);
    for j in 0..32 {
        let idx = small_add_const(api, &off_idx, (31 - j) as u32);
        let byte = read_calldata_byte(api, &opts.calldata, &idx);
        for b in 0..8 { cdl_val[j * 8 + b] = byte[b]; }
    }

    // KECCAK256: hash mem[off..off+len] (off=top, len=second, len<=135). LE byte k
    // of the pushed word = digest byte (31-k) since native does from_bytes_be(h).
    let mut keccak_v = word_zero(api);
    if opts.en_keccak {
        let mut msg = Vec::with_capacity(135 * 8);
        for i in 0..135 {
            let idx = small_add_const(api, &off_idx, i as u32);
            let byte = read_mem_byte(api, &st.mem, &idx);
            for b in 0..8 { msg.push(byte[b]); }
        }
        let len_bits: Vec<Variable> = second[0..8].to_vec();
        let digest = crate::batch_keccak::keccak256_varlen(api, &msg, &len_bits);
        for k in 0..32 { for b in 0..8 { keccak_v[k * 8 + b] = digest[(31 - k) * 8 + b]; } }
    }

    let sp_m1 = small_sub_const(api, &st.sp, 1);
    let sp_m2 = small_sub_const(api, &st.sp, 2);
    let sp_m3 = small_sub_const(api, &st.sp, 3);

    // DUP: copy stack[sp - n] ; SWAP: swap top (sp-1) with stack[sp-1-n].
    let sp6: Vec<Variable> = { let mut v = st.sp.clone(); while v.len() < 6 { v.push(zero); } v };
    let dup_idx = small_sub(api, &sp6, &n_val);
    let dup_val = read_stack_at(api, &st.stack, &dup_idx);
    let sp_m1_6 = small_sub_const(api, &sp6, 1);
    let swap_b_idx = small_sub(api, &sp_m1_6, &n_val);
    let val_b = read_stack_at(api, &st.stack, &swap_b_idx);
    let a_idx = sp_m1.clone(); // top position = sp-1

    let stack_push = write_stack(api, &st.stack, &st.sp, &push_word);
    let stack_bin = write_stack(api, &st.stack, &sp_m2, &bin);
    let stack_sload = write_stack(api, &st.stack, &sp_m1, &sload_val);
    let stack_mload = write_stack(api, &st.stack, &sp_m1, &mload_val);
    let stack_iszero = write_stack(api, &st.stack, &sp_m1, &iszero_v);
    let stack_not = write_stack(api, &st.stack, &sp_m1, &not_v);
    let stack_dup = write_stack(api, &st.stack, &st.sp, &dup_val);
    let stack_env = write_stack(api, &st.stack, &st.sp, &env_push);       // env push (+1)
    let stack_cdl = write_stack(api, &st.stack, &sp_m1, &cdl_val);        // CALLDATALOAD (net 0)
    let stack_keccak = write_stack(api, &st.stack, &sp_m2, &keccak_v);    // KECCAK (pop2 push1)
    let stack_ter = write_stack(api, &st.stack, &sp_m3, &ter_v);          // ADDMOD/MULMOD (pop3 push1)
    let mut stack_swap = st.stack.clone();
    for i in 0..cfg.sd {
        let sel_a = small_eq(api, &a_idx, i as u32);
        let sel_b = small_eq(api, &swap_b_idx, i as u32);
        let mut w = st.stack[i].clone();
        w = select(api, sel_b, &top, &w);   // position b gets old top
        w = select(api, sel_a, &val_b, &w); // top position gets old b
        stack_swap[i] = w;
    }

    let mut new_stack = st.stack.clone();
    for i in 0..cfg.sd {
        let mut w = st.stack[i].clone();
        w = select(api, is_push, &stack_push[i], &w);
        w = select(api, is_bin, &stack_bin[i], &w);
        w = select(api, is_sload, &stack_sload[i], &w);
        w = select(api, is_mload, &stack_mload[i], &w);
        w = select(api, is_iszero, &stack_iszero[i], &w);
        w = select(api, is_not, &stack_not[i], &w);
        w = select(api, is_dup, &stack_dup[i], &w);
        w = select(api, is_swap, &stack_swap[i], &w);
        w = select(api, is_env_push, &stack_env[i], &w);
        w = select(api, is_calldataload, &stack_cdl[i], &w);
        w = select(api, is_keccak, &stack_keccak[i], &w);
        w = select(api, is_ter, &stack_ter[i], &w);
        w = select(api, active, &w, &st.stack[i]);
        new_stack[i] = w;
    }

    // --- sp update ---
    let sp_p1 = small_add_const(api, &st.sp, 1);
    let mut new_sp = st.sp.clone();
    new_sp = select_small(api, is_push, &sp_p1, &new_sp);   // +1
    new_sp = select_small(api, is_bin, &sp_m1, &new_sp);    // -1 (pop2 push1)
    new_sp = select_small(api, is_pop, &sp_m1, &new_sp);    // -1
    new_sp = select_small(api, is_sstore, &sp_m2, &new_sp); // -2
    new_sp = select_small(api, is_mstore, &sp_m2, &new_sp); // -2
    new_sp = select_small(api, is_dup, &sp_p1, &new_sp);    // +1
    new_sp = select_small(api, is_jump, &sp_m1, &new_sp);   // -1
    new_sp = select_small(api, is_jumpi, &sp_m2, &new_sp);  // -2
    new_sp = select_small(api, is_codecopy, &sp_m3, &new_sp); // -3 (dst,src,len)
    new_sp = select_small(api, is_env_push, &sp_p1, &new_sp);   // +1 (ADDRESS/CALLER/CALLVALUE/CALLDATASIZE/CODESIZE/PC)
    new_sp = select_small(api, is_keccak, &sp_m1, &new_sp);    // -1 (pop2 push1)
    new_sp = select_small(api, is_ter, &sp_m2, &new_sp);       // -2 (ADDMOD/MULMOD pop3 push1)
    new_sp = select_small(api, is_mstore8, &sp_m2, &new_sp);   // -2 (off,val)
    new_sp = select_small(api, is_calldatacopy, &sp_m3, &new_sp); // -3 (dst,src,len)
    // LOG0..LOG4: pop (off, len) + ntopics topics = -(2 + ntopics).
    for t in 0..5 {
        let sp_dec = small_sub_const(api, &st.sp, 2 + t as u32);
        new_sp = select_small(api, log_sel[t], &sp_dec, &new_sp);
    }
    new_sp = select_small(api, active, &new_sp, &st.sp);
    // MLOAD/CALLDATALOAD net sp change is 0 (pop 1, push 1). SWAP: 0.

    // --- pc update: push +1+n (=2+nm1); others +1; stop/return stay ---
    let pc_p1 = small_add_const(api, &st.pc, 1);
    let mut new_pc = pc_p1.clone();
    // push advance = pc + 2 + nm1
    let pc_push = { let base = small_add_const(api, &st.pc, 2); // + nm1 (5-bit) via ripple
        let n = base.len(); let mut out = vec![]; let mut carry = zero;
        for k in 0..n { let nb = if k < 5 { nm1[k] } else { zero }; let ab = api.add(base[k], nb); let s = api.add(ab, carry); let a2 = api.mul(base[k], nb); let c2 = api.mul(carry, ab); carry = api.add(a2, c2); out.push(s); }
        out };
    // JUMP: pc = dest (low pc bits of top). JUMPI: cond ? dest : pc+1.
    let dest: Vec<Variable> = st.pc.iter().enumerate().map(|(i, _)| top[i]).collect();
    let jcond = { let z = is_zero(api, &second); api.sub(1, z) }; // second != 0
    let jumpi_pc = select_small(api, jcond, &dest, &pc_p1);
    new_pc = select_small(api, is_push, &pc_push, &new_pc);
    new_pc = select_small(api, is_jump, &dest, &new_pc);
    new_pc = select_small(api, is_jumpi, &jumpi_pc, &new_pc);
    new_pc = select_small(api, is_stop, &st.pc, &new_pc);
    new_pc = select_small(api, is_return, &st.pc, &new_pc);
    new_pc = select_small(api, is_revert, &st.pc, &new_pc);
    new_pc = select_small(api, active, &new_pc, &st.pc);

    // --- MSTORE: mem[off..off+32] = second (big-endian), off = low bits of top ---
    // --- CODECOPY: mem[dst+i] = code[src+i] for i<len; dst=top, src=second, len=third ---
    let do_mstore = api.mul(is_mstore, active);
    let do_mstore8 = api.mul(is_mstore8, active);
    let do_codecopy = api.mul(is_codecopy, active);
    let do_calldatacopy = api.mul(is_calldatacopy, active);
    let cc_third = read_stack(api, &st, 2); // len
    let cc_src_idx: Vec<Variable> = second[0..off_idx.len()].to_vec(); // src base (low bits of second)
    let mut new_mem = st.mem.clone();
    for p in 0..cfg.mem_bytes {
        let mut byte = st.mem[p].clone();
        // MSTORE
        for k in 0..32 {
            let offk = small_add_const(api, &off_idx, k as u32);
            let hit0 = small_eq(api, &offk, p as u32);
            let hit = api.mul(hit0, do_mstore);
            let vb = word_le_byte(&second, 31 - k); // BE byte k = LE byte 31-k
            byte = (0..8).map(|b| { let d = api.add(vb[b], byte[b]); let t = api.mul(hit, d); api.add(byte[b], t) }).collect();
        }
        // MSTORE8: mem[off] = second & 0xff (off = top low bits = off_idx).
        {
            let hit0 = small_eq(api, &off_idx, p as u32);
            let hit = api.mul(hit0, do_mstore8);
            byte = (0..8).map(|b| { let d = api.add(second[b], byte[b]); let t = api.mul(hit, d); api.add(byte[b], t) }).collect();
        }
        // CODECOPY / CALLDATACOPY: byte p from code[src+i] / calldata[src+i] when dst+i==p and i<len.
        for i in 0..cfg.mem_bytes {
            let dsti = small_add_const(api, &off_idx, i as u32); // dst + i
            let at_p = small_eq(api, &dsti, p as u32);           // dst+i == p
            let ci = const_word(api, i as u64);
            let i_lt_len = lt(api, &ci, &cc_third);              // i < len
            let at_p_lt = api.mul(at_p, i_lt_len);
            let srci = small_add_const(api, &cc_src_idx, i as u32); // src + i
            // CODECOPY source
            let hit_cc = api.mul(at_p_lt, do_codecopy);
            let cb = read_code(api, code, &srci);
            byte = (0..8).map(|b| { let d = api.add(cb[b], byte[b]); let t = api.mul(hit_cc, d); api.add(byte[b], t) }).collect();
            // CALLDATACOPY source
            let hit_cd = api.mul(at_p_lt, do_calldatacopy);
            let db = read_calldata_byte(api, &opts.calldata, &srci);
            byte = (0..8).map(|b| { let d = api.add(db[b], byte[b]); let t = api.mul(hit_cd, d); api.add(byte[b], t) }).collect();
        }
        new_mem[p] = byte;
    }

    // --- SSTORE: storage[top] = second ---
    let do_sstore = api.mul(is_sstore, active);
    let mut new_svals = st.svals.clone();
    for i in 0..cfg.ss {
        let m0 = eq(api, &st.skeys[i], &top);
        let m = api.mul(m0, do_sstore);
        new_svals[i] = select(api, m, &second, &st.svals[i]);
    }

    // --- RETURN / REVERT: capture ret = mem[off..], ret_len = second; halt ---
    // REVERT (0xfd) captures the same return data and halts; the single-frame
    // circuit does not model the storage/world rollback a real REVERT triggers
    // inside a sub-call (there are no sub-calls here). The returned bytes are the
    // byte-for-byte diff target vs native `return_data`.
    let is_ret_or_rev = api.add(is_return, is_revert);
    let do_return = api.mul(is_ret_or_rev, active);
    let mut new_ret = st.ret.clone();
    for p in 0..cfg.mem_bytes {
        let idx = small_add_const(api, &off_idx, p as u32);
        let mb = read_mem_byte(api, &st.mem, &idx);
        // ret[p] = do_return ? mem[off+p] : old ret[p]
        new_ret[p] = (0..8).map(|b| { let d = api.add(mb[b], st.ret[p][b]); let t = api.mul(do_return, d); api.add(st.ret[p][b], t) }).collect();
    }
    let ret_len_new: Vec<Variable> = { let ln: Vec<Variable> = second[0..st.ret_len.len()].to_vec(); select_small(api, do_return, &ln, &st.ret_len) };

    // --- halted: STOP or RETURN or REVERT ---
    let stop_or_ret = { let s = api.add(is_stop, is_ret_or_rev); let a = api.mul(is_stop, is_ret_or_rev); api.sub(s, a) };
    let set_halt = api.mul(stop_or_ret, active);
    let new_halted = { let s = api.add(st.halted, set_halt); let a = api.mul(st.halted, set_halt); api.sub(s, a) };

    // ================= CALL / CREATE — architectural gap =================
    // CALL(0xf1)/CALLCODE(0xf2)/DELEGATECALL(0xf4)/STATICCALL(0xfa)/CREATE(0xf0)/
    // CREATE2(0xf5) require SUB-FRAME execution: a fresh (stack, mem, pc, storage)
    // machine running the callee's committed code to completion, whose result
    // (success flag, return data, and — for successful CALL/CREATE — world/storage
    // mutations) is folded back into this frame. `step` here is a single flat
    // select over one machine's wires; it has no room to nest a whole second
    // interpreter mid-step.
    //
    // A bounded design that WOULD work: maintain an explicit frame stack of depth
    // D (D fixed at compile time). Each frame carries its own bounded (stack, mem,
    // storage, pc, code-window, calldata, return buffer). The global `run` loop
    // executes a fixed budget of micro-steps; a CALL/CREATE opcode PUSHES a new
    // frame (initialised from committed callee code + gas split) and switches the
    // "active frame" pointer; STOP/RETURN/REVERT/end-of-code POPS the active frame,
    // writing its return data into the parent's out-region and pushing the
    // success flag. Every micro-step is then a select over "which frame is active"
    // in addition to the opcode one-hot, so cost multiplies by D and by the total
    // micro-step budget (sum of all frames' steps). This is a well-defined but
    // large extension; it is deliberately NOT stubbed/faked here. CALL/CREATE are
    // the sole opcode family without an in-circuit implementation.

    Evm { stack: new_stack, sp: new_sp, pc: new_pc, skeys: st.skeys, svals: new_svals, mem: new_mem, ret: new_ret, ret_len: ret_len_new, halted: new_halted }
}

fn select_small<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..a.len()).map(|i| { let d = api.add(a[i], b[i]); let t = api.mul(sel, d); api.add(b[i], t) }).collect()
}

/// Run N steps of the EVM over committed code, from an initial state, with EMPTY
/// environment context and the heavy opcode groups DISABLED (the light opcode
/// set only). Signature preserved for existing callers (`block_stf`, `evm_gkr`).
pub fn run<C: Config>(api: &mut impl RootAPI<C>, mut st: Evm, code: &[Vec<Variable>], cfg: &Cfg, n_steps: usize) -> Evm {
    let opts = StepOpts::empty(api);
    for _ in 0..n_steps {
        st = step(api, st, code, cfg, &opts);
    }
    st
}

/// Run N steps with explicit environment context / heavy-opcode enables.
pub fn run_opts<C: Config>(api: &mut impl RootAPI<C>, mut st: Evm, code: &[Vec<Variable>], cfg: &Cfg, opts: &StepOpts, n_steps: usize) -> Evm {
    for _ in 0..n_steps {
        st = step(api, st, code, cfg, opts);
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::u256::bigint_to_bits;
    use num_bigint::BigInt;

    const SD: usize = 4;
    const SS: usize = 2;
    const CODE_LEN: usize = 10;
    const MEM_BYTES: usize = 8;
    const N: usize = 8;
    const SP_BITS: usize = 4;
    const PC_BITS: usize = 6;

    fn fresh_state<B: RootAPI<GF2Config>>(api: &mut B, skeys: Vec<Word>, svals: Vec<Word>, sd: usize, ss: usize, mem_bytes: usize) -> Evm {
        let zw = vec![api.constant(0); BITS];
        Evm {
            stack: vec![zw.clone(); sd],
            sp: vec![api.constant(0); SP_BITS],
            pc: vec![api.constant(0); PC_BITS],
            skeys,
            svals,
            mem: vec![vec![api.constant(0); 8]; mem_bytes],
            ret: vec![vec![api.constant(0); 8]; mem_bytes],
            ret_len: vec![api.constant(0); PC_BITS],
            halted: api.constant(0),
        }
    }

    // increment: 60 00 54 60 01 01 60 00 55 00
    declare_circuit!(IncCircuit {
        code: [[Variable; 8]; CODE_LEN],
        skey0: [Variable; BITS],
        sval0: [Variable; BITS],
        out_val0: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for IncCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let cfg = Cfg { sd: SD, ss: SS, code_len: CODE_LEN, mem_bytes: MEM_BYTES, sp_bits: SP_BITS, pc_bits: PC_BITS };
            let code: Vec<Vec<Variable>> = (0..CODE_LEN).map(|i| self.code[i].to_vec()).collect();
            let zeroword = vec![api.constant(0); BITS];
            let st = fresh_state(api, vec![self.skey0.to_vec(), zeroword.clone()], vec![self.sval0.to_vec(), zeroword.clone()], SD, SS, MEM_BYTES);
            let fin = run(api, st, &code, &cfg, N);
            for b in 0..BITS { api.assert_is_equal(fin.svals[0][b], self.out_val0[b]); }
        }
    }

    // deploy init code: PUSH10 <runtime> PUSH1 0 MSTORE PUSH1 0x0a PUSH1 0x16 RETURN
    // returns the 10 runtime bytes (602a60005560016000f3). Exercises PUSHn/MSTORE/RETURN.
    const DCODE: usize = 24;
    const DMEM: usize = 32;
    const DN: usize = 8;
    declare_circuit!(DeployCircuit {
        code: [[Variable; 8]; DCODE],
        ret_out: [[PublicVariable; 8]; 10], // expected returned runtime bytes
    });
    impl Define<GF2Config> for DeployCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let cfg = Cfg { sd: 8, ss: 1, code_len: DCODE, mem_bytes: DMEM, sp_bits: SP_BITS, pc_bits: PC_BITS };
            let code: Vec<Vec<Variable>> = (0..DCODE).map(|i| self.code[i].to_vec()).collect();
            let zw = vec![api.constant(0); BITS];
            let st = Evm {
                stack: vec![zw.clone(); 8],
                sp: vec![api.constant(0); SP_BITS],
                pc: vec![api.constant(0); PC_BITS],
                skeys: vec![zw.clone()],
                svals: vec![zw.clone()],
                mem: vec![vec![api.constant(0); 8]; DMEM],
                ret: vec![vec![api.constant(0); 8]; DMEM],
                ret_len: vec![api.constant(0); PC_BITS],
                halted: api.constant(0),
            };
            let fin = run(api, st, &code, &cfg, DN);
            for i in 0..10 { for b in 0..8 { api.assert_is_equal(fin.ret[i][b], self.ret_out[i][b]); } }
        }
    }

    // arith golden: 600360040160020260005260206000f3
    //  PUSH1 3, PUSH1 4, ADD, PUSH1 2, MUL, PUSH1 0, MSTORE, PUSH1 0x20, PUSH1 0, RETURN -> 14
    const ACODE: usize = 16;
    declare_circuit!(ArithCircuit {
        code: [[Variable; 8]; ACODE],
        ret_last: [PublicVariable; 8], // last return byte (should be 0x0e)
    });
    impl Define<GF2Config> for ArithCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let cfg = Cfg { sd: 8, ss: 1, code_len: ACODE, mem_bytes: 32, sp_bits: SP_BITS, pc_bits: PC_BITS };
            let code: Vec<Vec<Variable>> = (0..ACODE).map(|i| self.code[i].to_vec()).collect();
            let zw = vec![api.constant(0); BITS];
            let st = Evm {
                stack: vec![zw.clone(); 8], sp: vec![api.constant(0); SP_BITS], pc: vec![api.constant(0); PC_BITS],
                skeys: vec![zw.clone()], svals: vec![zw.clone()],
                mem: vec![vec![api.constant(0); 8]; 32], ret: vec![vec![api.constant(0); 8]; 32],
                ret_len: vec![api.constant(0); PC_BITS], halted: api.constant(0),
            };
            let fin = run(api, st, &code, &cfg, 12);
            // return value 14 big-endian in 32 bytes -> ret[31] = 0x0e
            for b in 0..8 { api.assert_is_equal(fin.ret[31][b], self.ret_last[b]); }
        }
    }

    #[test]
    fn incircuit_evm_arith_matches_native() {
        let code_hex = "600360040160020260005260206000f3";
        let code: Vec<u8> = (0..code_hex.len() / 2).map(|i| u8::from_str_radix(&code_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        let ctx = crate::evm::CallCtx { code: code.clone(), ..Default::default() };
        let mut storage = std::collections::BTreeMap::new();
        let r = crate::evm::execute(&ctx, &mut storage, 1_000_000);
        assert_eq!(*r.return_data.last().unwrap(), 0x0e);
        let CompileResult { witness_solver, layered_circuit } = compile(&ArithCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = ArithCircuit::<GF2>::default();
        for i in 0..ACODE { let byte = if i < code.len() { code[i] } else { 0 }; for b in 0..8 { asg.code[i][b] = (((byte >> b) & 1) as u32).into(); } }
        for b in 0..8 { asg.ret_last[b] = (((0x0eu8 >> b) & 1) as u32).into(); }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit arith (ADD+MUL) != native (14)");
    }

    #[test]
    fn incircuit_evm_deploy_returns_runtime() {
        let code_hex = "69602a60005560016000f3600052600a6016f3";
        let code: Vec<u8> = (0..code_hex.len() / 2).map(|i| u8::from_str_radix(&code_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        // native golden
        let ctx = crate::evm::CallCtx { code: code.clone(), ..Default::default() };
        let mut storage = std::collections::BTreeMap::new();
        let r = crate::evm::execute(&ctx, &mut storage, 1_000_000);
        assert!(r.success);
        let runtime: Vec<u8> = (0..10).map(|i| u8::from_str_radix(&"602a60005560016000f3"[2 * i..2 * i + 2], 16).unwrap()).collect();
        assert_eq!(r.return_data, runtime, "native deploy return mismatch");

        let CompileResult { witness_solver, layered_circuit } =
            compile(&DeployCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = DeployCircuit::<GF2>::default();
        for i in 0..DCODE { let byte = if i < code.len() { code[i] } else { 0 }; for b in 0..8 { asg.code[i][b] = (((byte >> b) & 1) as u32).into(); } }
        for i in 0..10 { for b in 0..8 { asg.ret_out[i][b] = (((runtime[i] >> b) & 1) as u32).into(); } }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit deploy return != native runtime");
    }

    // dup/swap: PUSH1 5, PUSH1 7, SWAP1, POP, DUP1, ADD, MSTORE, RETURN -> 14
    #[test]
    fn incircuit_evm_dup_swap_matches_native() {
        let code_hex = "6005600790508001600052602060 00f3".replace(' ', "");
        let code: Vec<u8> = (0..code_hex.len() / 2).map(|i| u8::from_str_radix(&code_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        let ctx = crate::evm::CallCtx { code: code.clone(), ..Default::default() };
        let mut storage = std::collections::BTreeMap::new();
        let r = crate::evm::execute(&ctx, &mut storage, 1_000_000);
        assert_eq!(*r.return_data.last().unwrap(), 14, "native dup/swap != 14");
        let CompileResult { witness_solver, layered_circuit } = compile(&ArithCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = ArithCircuit::<GF2>::default();
        for i in 0..ACODE { let byte = if i < code.len() { code[i] } else { 0 }; for b in 0..8 { asg.code[i][b] = (((byte >> b) & 1) as u32).into(); } }
        for b in 0..8 { asg.ret_last[b] = (((14u8 >> b) & 1) as u32).into(); }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit dup/swap != native (14)");
    }

    // jump: PUSH1 4, JUMP, STOP, JUMPDEST, PUSH1 0x2a, PUSH1 0, SSTORE, STOP
    //  the JUMP must skip the STOP@3 to reach JUMPDEST@4; else storage stays 0.
    #[test]
    fn incircuit_evm_jump_matches_native() {
        let code_hex = "600456005b602a60005500";
        let code: Vec<u8> = (0..code_hex.len() / 2).map(|i| u8::from_str_radix(&code_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        let ctx = crate::evm::CallCtx { code: code.clone(), ..Default::default() };
        let mut storage = std::collections::BTreeMap::new();
        let r = crate::evm::execute(&ctx, &mut storage, 1_000_000);
        assert!(r.success);
        assert_eq!(storage.get(&BigInt::from(0u32)), Some(&BigInt::from(0x2au32)), "native jump result");
        let CompileResult { witness_solver, layered_circuit } = compile(&IncCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = IncCircuit::<GF2>::default();
        for i in 0..CODE_LEN { let byte = if i < code.len() { code[i] } else { 0 }; for b in 0..8 { asg.code[i][b] = (((byte >> b) & 1) as u32).into(); } }
        for (i, bit) in bigint_to_bits(&BigInt::from(0u32), BITS).into_iter().enumerate() { asg.skey0[i] = (bit as u32).into(); }
        for (i, bit) in bigint_to_bits(&BigInt::from(0u32), BITS).into_iter().enumerate() { asg.sval0[i] = (bit as u32).into(); }
        for (i, bit) in bigint_to_bits(&BigInt::from(0x2au32), BITS).into_iter().enumerate() { asg.out_val0[i] = (bit as u32).into(); }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit jump != native (storage[0]=0x2a)");
    }

    /// Run `code_hex` in-circuit via ArithCircuit (checks the last RETURN byte
    /// equals native's), asserting the tranche-1 opcode matches the native EVM.
    fn check_ret_last(code_hex: &str) {
        let code: Vec<u8> = (0..code_hex.len() / 2).map(|i| u8::from_str_radix(&code_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        assert!(code.len() <= ACODE, "program too long for ArithCircuit");
        let ctx = crate::evm::CallCtx { code: code.clone(), ..Default::default() };
        let mut storage = std::collections::BTreeMap::new();
        let r = crate::evm::execute(&ctx, &mut storage, 1_000_000);
        assert!(r.success, "native exec failed for {code_hex}");
        let want = *r.return_data.last().expect("native returned no data");
        let CompileResult { witness_solver, layered_circuit } = compile(&ArithCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = ArithCircuit::<GF2>::default();
        for i in 0..ACODE { let byte = if i < code.len() { code[i] } else { 0 }; for b in 0..8 { asg.code[i][b] = (((byte >> b) & 1) as u32).into(); } }
        for b in 0..8 { asg.ret_last[b] = (((want >> b) & 1) as u32).into(); }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit != native (last ret byte {want:#04x}) for {code_hex}");
    }

    // Real spamoor deploy init-code (uses CODECOPY 0x39): copies the 10-byte
    // runtime at offset 0x0c to mem[0] and RETURNs it. Verifies CODECOPY in-circuit
    // returns the runtime byte-for-byte vs native.
    #[test]
    fn incircuit_codecopy_deploy() {
        let code_hex = "600a600c600039600a6000f3602a60005560016000f3";
        let code: Vec<u8> = (0..code_hex.len() / 2).map(|i| u8::from_str_radix(&code_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        let ctx = crate::evm::CallCtx { code: code.clone(), ..Default::default() };
        let mut storage = std::collections::BTreeMap::new();
        let r = crate::evm::execute(&ctx, &mut storage, 1_000_000);
        assert!(r.success);
        let runtime: Vec<u8> = (0..10).map(|i| u8::from_str_radix(&"602a60005560016000f3"[2 * i..2 * i + 2], 16).unwrap()).collect();
        assert_eq!(r.return_data, runtime, "native codecopy-deploy return mismatch");
        let CompileResult { witness_solver, layered_circuit } = compile(&DeployCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = DeployCircuit::<GF2>::default();
        for i in 0..DCODE { let byte = if i < code.len() { code[i] } else { 0 }; for b in 0..8 { asg.code[i][b] = (((byte >> b) & 1) as u32).into(); } }
        for i in 0..10 { for b in 0..8 { asg.ret_out[i][b] = (((runtime[i] >> b) & 1) as u32).into(); } }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit CODECOPY deploy != native runtime");
    }

    // NOT 0x0f -> ...f0 ; last byte 0xf0
    #[test]
    fn incircuit_not() { check_ret_last("600f1960005260206000f3"); }
    // SHL: value=1, shift=4 -> 0x10
    #[test]
    fn incircuit_shl() { check_ret_last("600160041b60005260206000f3"); }
    // SHR: value=0xff00, shift=8 -> 0xff
    #[test]
    fn incircuit_shr() { check_ret_last("61ff0060081c60005260206000f3"); }
    // BYTE: byte 31 of 0x00ff -> 0xff (i=top=31, x=second=0x00ff)
    #[test]
    fn incircuit_byte() { check_ret_last("6100ff601f1a60005260206000f3"); }
    // SLT: top=3 <s second=5 -> 1. PUSH1 5; PUSH1 3; SLT
    #[test]
    fn incircuit_slt() { check_ret_last("600560031260005260206000f3"); }
    // SGT: top=5 >s second=3 -> 1. PUSH1 3; PUSH1 5; SGT
    #[test]
    fn incircuit_sgt() { check_ret_last("600360051360005260206000f3"); }
    // SAR: -8 >>s 1 = -4 -> last byte 0xfc. PUSH1 8; PUSH1 0; SUB(=-8); PUSH1 1; SAR
    #[test]
    fn incircuit_sar() { check_ret_last("600860000360011d60005260206000f3"); }
    // SIGNEXTEND(0, 0xff) = all-ff -> last byte 0xff. PUSH1 0xff; PUSH1 0; SIGNEXTEND
    #[test]
    fn incircuit_signext() { check_ret_last("60ff60000b60005260206000f3"); }

    #[test]
    fn incircuit_evm_increment_matches_native() {
        // native golden
        let code_bytes = [0x60u8, 0x00, 0x54, 0x60, 0x01, 0x01, 0x60, 0x00, 0x55, 0x00];
        let ctx = crate::evm::CallCtx { code: code_bytes.to_vec(), ..Default::default() };
        let mut storage = std::collections::BTreeMap::new();
        storage.insert(BigInt::from(0u32), BigInt::from(5u32));
        let r = crate::evm::execute(&ctx, &mut storage, 1_000_000);
        assert!(r.success);
        let post = storage.get(&BigInt::from(0u32)).cloned().unwrap();
        assert_eq!(post, BigInt::from(6u32));

        let CompileResult { witness_solver, layered_circuit } =
            compile(&IncCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = IncCircuit::<GF2>::default();
        for i in 0..CODE_LEN { for b in 0..8 { asg.code[i][b] = (((code_bytes[i] >> b) & 1) as u32).into(); } }
        for (i, bit) in bigint_to_bits(&BigInt::from(0u32), BITS).into_iter().enumerate() { asg.skey0[i] = (bit as u32).into(); }
        for (i, bit) in bigint_to_bits(&BigInt::from(5u32), BITS).into_iter().enumerate() { asg.sval0[i] = (bit as u32).into(); }
        for (i, bit) in bigint_to_bits(&post, BITS).into_iter().enumerate() { asg.out_val0[i] = (bit as u32).into(); }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit EVM increment != native (storage[0] should be 6)");
    }

    // ========================================================================
    //  Tranche-2 opcode differential tests.
    //
    //  Golden values ALWAYS come from `crate::evm::execute` (the native EVM that
    //  is itself byte-verified against ev-reth's revm). Two shapes are used:
    //   * full-program tests (light opcodes) run the whole program in-circuit
    //     via `run`/`run_opts` and compare the 32-byte RETURN word / storage;
    //   * single-step isolation tests (the DIV/MOD/EXP family) pre-load the
    //     operand stack and execute exactly ONE step, comparing the resulting
    //     stack-top to native's result for the equivalent PUSH..OP..RETURN
    //     program. Isolation is REQUIRED because the 256-bit long-division /
    //     square-and-multiply gadgets are very deep (~30k+ layers each); a
    //     multi-step unroll of them does not compile in reasonable memory. The
    //     golden is still native `execute`, so the test remains a true diff.
    // ========================================================================

    /// 32-byte big-endian encoding of a non-negative `v < 2^256`.
    fn u256_be(v: &BigInt) -> Vec<u8> {
        let (_s, b) = v.to_bytes_be();
        let mut out = vec![0u8; 32];
        let n = b.len().min(32);
        out[32 - n..].copy_from_slice(&b[b.len() - n..]);
        out
    }
    /// Native golden for a stack op: build `PUSH32 op0; ..; PUSH32 opk; OP;
    /// MSTORE@0; RETURN 32`, execute it, return the 32-byte result as a BigInt.
    /// `operands[0]` is the deepest stack item, `operands[last]` the top.
    fn heavy_golden(op: u8, operands: &[BigInt]) -> BigInt {
        let mut prog = Vec::new();
        for v in operands { prog.push(0x7f); prog.extend_from_slice(&u256_be(v)); }
        prog.push(op);
        prog.extend_from_slice(&[0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xf3]);
        let ctx = crate::evm::CallCtx { code: prog, ..Default::default() };
        let mut storage = std::collections::BTreeMap::new();
        let r = crate::evm::execute(&ctx, &mut storage, 100_000_000);
        assert!(r.success, "native golden exec failed for op {op:#04x}");
        BigInt::from_bytes_be(num_bigint::Sign::Plus, &r.return_data)
    }

    // ---- single-step DIV/MOD/SDIV/SMOD (en_div) ---------------------------
    const HSP: usize = 8;
    const HPC: usize = 8;
    declare_circuit!(DivStepCircuit {
        op: [Variable; 8],
        s0: [Variable; BITS], // stack[0] = divisor (second)
        s1: [Variable; BITS], // stack[1] = dividend (top)
        out: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for DivStepCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let cfg = Cfg { sd: 4, ss: 1, code_len: 1, mem_bytes: 1, sp_bits: HSP, pc_bits: HPC };
            let mut opts = StepOpts::empty(api);
            opts.en_div = true;
            let zw = vec![api.constant(0); BITS];
            let st = Evm {
                stack: vec![self.s0.to_vec(), self.s1.to_vec(), zw.clone(), zw.clone()],
                sp: small_const(api, 2, HSP),
                pc: vec![api.constant(0); HPC],
                skeys: vec![zw.clone()], svals: vec![zw.clone()],
                mem: vec![vec![api.constant(0); 8]; 1], ret: vec![vec![api.constant(0); 8]; 1],
                ret_len: vec![api.constant(0); HPC], halted: api.constant(0),
            };
            let code = vec![self.op.to_vec()];
            let fin = run_opts(api, st, &code, &cfg, &opts, 1);
            for b in 0..BITS { api.assert_is_equal(fin.stack[0][b], self.out[b]); }
        }
    }

    #[test]
    fn incircuit_div_mod_sdiv_smod() {
        let CompileResult { witness_solver, layered_circuit } =
            compile(&DivStepCircuit::default(), CompileOptions::default()).unwrap();
        let two256 = BigInt::from(1u8) << 256;
        let neg = |v: i64| -> BigInt { if v < 0 { &two256 - BigInt::from(-v) } else { BigInt::from(v) } };
        // (opcode, second=divisor, top=dividend)
        let cases: Vec<(u8, BigInt, BigInt)> = vec![
            (0x04, BigInt::from(3u32), BigInt::from(20u32)),  // DIV 20/3 = 6
            (0x06, BigInt::from(5u32), BigInt::from(17u32)),  // MOD 17%5 = 2
            (0x05, BigInt::from(2u32), neg(-6)),              // SDIV -6/2 = -3
            (0x07, BigInt::from(3u32), neg(-7)),              // SMOD -7%3 = -1
        ];
        for (op, second, top) in cases {
            let golden = heavy_golden(op, &[second.clone(), top.clone()]);
            let mut asg = DivStepCircuit::<GF2>::default();
            for b in 0..8 { asg.op[b] = (((op >> b) & 1) as u32).into(); }
            for (i, bit) in bigint_to_bits(&second, BITS).into_iter().enumerate() { asg.s0[i] = (bit as u32).into(); }
            for (i, bit) in bigint_to_bits(&top, BITS).into_iter().enumerate() { asg.s1[i] = (bit as u32).into(); }
            for (i, bit) in bigint_to_bits(&golden, BITS).into_iter().enumerate() { asg.out[i] = (bit as u32).into(); }
            let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
            assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit != native for op {op:#04x} ({top}/{second})");
        }
    }

    // ---- single-step ADDMOD/MULMOD (en_mulmod) ----------------------------
    declare_circuit!(TerStepCircuit {
        op: [Variable; 8],
        s0: [Variable; BITS], // stack[0] = n (modulus, deepest)
        s1: [Variable; BITS], // stack[1] = b (second)
        s2: [Variable; BITS], // stack[2] = a (top)
        out: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for TerStepCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let cfg = Cfg { sd: 4, ss: 1, code_len: 1, mem_bytes: 1, sp_bits: HSP, pc_bits: HPC };
            let mut opts = StepOpts::empty(api);
            opts.en_mulmod = true;
            let zw = vec![api.constant(0); BITS];
            let st = Evm {
                stack: vec![self.s0.to_vec(), self.s1.to_vec(), self.s2.to_vec(), zw.clone()],
                sp: small_const(api, 3, HSP),
                pc: vec![api.constant(0); HPC],
                skeys: vec![zw.clone()], svals: vec![zw.clone()],
                mem: vec![vec![api.constant(0); 8]; 1], ret: vec![vec![api.constant(0); 8]; 1],
                ret_len: vec![api.constant(0); HPC], halted: api.constant(0),
            };
            let code = vec![self.op.to_vec()];
            let fin = run_opts(api, st, &code, &cfg, &opts, 1);
            for b in 0..BITS { api.assert_is_equal(fin.stack[0][b], self.out[b]); }
        }
    }

    #[test]
    fn incircuit_addmod_mulmod() {
        let CompileResult { witness_solver, layered_circuit } =
            compile(&TerStepCircuit::default(), CompileOptions::default()).unwrap();
        // (opcode, n, b, a)  result at stack[0]
        let cases: Vec<(u8, BigInt, BigInt, BigInt)> = vec![
            (0x08, BigInt::from(16u32), BigInt::from(255u32), BigInt::from(255u32)), // (255+255)%16 = 14
            (0x09, BigInt::from(16u32), BigInt::from(255u32), BigInt::from(255u32)), // (255*255)%16 = 1
        ];
        for (op, n, b, a) in cases {
            let golden = heavy_golden(op, &[n.clone(), b.clone(), a.clone()]);
            let mut asg = TerStepCircuit::<GF2>::default();
            for bb in 0..8 { asg.op[bb] = (((op >> bb) & 1) as u32).into(); }
            for (i, bit) in bigint_to_bits(&n, BITS).into_iter().enumerate() { asg.s0[i] = (bit as u32).into(); }
            for (i, bit) in bigint_to_bits(&b, BITS).into_iter().enumerate() { asg.s1[i] = (bit as u32).into(); }
            for (i, bit) in bigint_to_bits(&a, BITS).into_iter().enumerate() { asg.s2[i] = (bit as u32).into(); }
            for (i, bit) in bigint_to_bits(&golden, BITS).into_iter().enumerate() { asg.out[i] = (bit as u32).into(); }
            let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
            assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit != native for op {op:#04x}");
        }
    }

    // ---- single-step EXP (en_exp) -----------------------------------------
    declare_circuit!(ExpStepCircuit {
        s0: [Variable; BITS], // stack[0] = exp (second)
        s1: [Variable; BITS], // stack[1] = base (top)
        out: [PublicVariable; BITS],
    });
    impl Define<GF2Config> for ExpStepCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let cfg = Cfg { sd: 4, ss: 1, code_len: 1, mem_bytes: 1, sp_bits: HSP, pc_bits: HPC };
            let mut opts = StepOpts::empty(api);
            opts.en_exp = true;
            let zw = vec![api.constant(0); BITS];
            let st = Evm {
                stack: vec![self.s0.to_vec(), self.s1.to_vec(), zw.clone(), zw.clone()],
                sp: small_const(api, 2, HSP),
                pc: vec![api.constant(0); HPC],
                skeys: vec![zw.clone()], svals: vec![zw.clone()],
                mem: vec![vec![api.constant(0); 8]; 1], ret: vec![vec![api.constant(0); 8]; 1],
                ret_len: vec![api.constant(0); HPC], halted: api.constant(0),
            };
            let code = vec![(0..8).map(|b| api.constant(((0x0au32) >> b) & 1)).collect::<Vec<_>>()]; // EXP
            let fin = run_opts(api, st, &code, &cfg, &opts, 1);
            for b in 0..BITS { api.assert_is_equal(fin.stack[0][b], self.out[b]); }
        }
    }

    #[test]
    fn incircuit_exp() {
        let CompileResult { witness_solver, layered_circuit } =
            compile(&ExpStepCircuit::default(), CompileOptions::default()).unwrap();
        // base=3, exp=5 => 243
        let (base, exp) = (BigInt::from(3u32), BigInt::from(5u32));
        let golden = heavy_golden(0x0a, &[exp.clone(), base.clone()]);
        assert_eq!(golden, BigInt::from(243u32), "native EXP 3**5 must be 243");
        let mut asg = ExpStepCircuit::<GF2>::default();
        for (i, bit) in bigint_to_bits(&exp, BITS).into_iter().enumerate() { asg.s0[i] = (bit as u32).into(); }
        for (i, bit) in bigint_to_bits(&base, BITS).into_iter().enumerate() { asg.s1[i] = (bit as u32).into(); }
        for (i, bit) in bigint_to_bits(&golden, BITS).into_iter().enumerate() { asg.out[i] = (bit as u32).into(); }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit EXP != native (3**5=243)");
    }

    // ---- light opcodes via full programs: MSTORE8, PC, LOG0-4, REVERT -----
    // These programs stage a single result byte with MSTORE8 and RETURN/REVERT
    // exactly 1 byte, so `GMEM` (hence the O(mem^2) copy loop cost) stays tiny.
    const GCODE: usize = 32;
    const GMEM: usize = 4;
    const GSTEPS: usize = 14;
    const GSD: usize = 8;
    declare_circuit!(GenCircuit {
        code: [[Variable; 8]; GCODE],
        ret0: [PublicVariable; 8], // single returned byte
    });
    impl Define<GF2Config> for GenCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let cfg = Cfg { sd: GSD, ss: 1, code_len: GCODE, mem_bytes: GMEM, sp_bits: 8, pc_bits: 8 };
            let code: Vec<Vec<Variable>> = (0..GCODE).map(|i| self.code[i].to_vec()).collect();
            let zw = vec![api.constant(0); BITS];
            let st = Evm {
                stack: vec![zw.clone(); GSD], sp: vec![api.constant(0); 8], pc: vec![api.constant(0); 8],
                skeys: vec![zw.clone()], svals: vec![zw.clone()],
                mem: vec![vec![api.constant(0); 8]; GMEM], ret: vec![vec![api.constant(0); 8]; GMEM],
                ret_len: vec![api.constant(0); 8], halted: api.constant(0),
            };
            let fin = run(api, st, &code, &cfg, GSTEPS);
            for b in 0..8 { api.assert_is_equal(fin.ret[0][b], self.ret0[b]); }
        }
    }

    #[test]
    fn incircuit_light_mstore8_pc_log_revert() {
        let CompileResult { witness_solver, layered_circuit } =
            compile(&GenCircuit::default(), CompileOptions::default()).unwrap();
        // Each program stores byte 0x2a (or the PC value) at mem[0] via MSTORE8
        // then RETURN/REVERT 1 byte. (code_hex, expect_success).
        let progs: Vec<(&str, bool)> = vec![
            ("602a60005360016000f3", true),                       // MSTORE8 mem[0]=0x2a; RETURN 1 => 0x2a
            ("60005058600053600160 00f3", true),                  // PUSH0;POP;PC(@3);MSTORE8;RETURN1 => 3
            ("60006000a0602a60005360016000f3", true),             // LOG0
            ("600060006000a1602a60005360016000f3", true),         // LOG1
            ("6000600060006000a2602a60005360016000f3", true),     // LOG2
            ("60006000600060006000a3602a60005360016000f3", true), // LOG3
            ("600060006000600060006000a4602a60005360016000f3", true), // LOG4
            ("602a600053600160 00fd", false),                     // REVERT 1 byte (0x2a)
        ];
        for (code_hex, expect_success) in progs {
            let code_hex = code_hex.replace(' ', "");
            let mut code: Vec<u8> = (0..code_hex.len() / 2).map(|i| u8::from_str_radix(&code_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
            assert!(code.len() <= GCODE, "program {code_hex} too long");
            code.resize(GCODE, 0); // pad to committed code length (trailing STOP, unreached)
            let ctx = crate::evm::CallCtx { code: code.clone(), ..Default::default() };
            let mut storage = std::collections::BTreeMap::new();
            let r = crate::evm::execute(&ctx, &mut storage, 1_000_000);
            assert_eq!(r.success, expect_success, "native success mismatch for {code_hex}");
            assert_eq!(r.return_data.len(), 1, "program {code_hex} must return 1 byte");
            let mut asg = GenCircuit::<GF2>::default();
            for i in 0..GCODE { for b in 0..8 { asg.code[i][b] = (((code[i] >> b) & 1) as u32).into(); } }
            for b in 0..8 { asg.ret0[b] = (((r.return_data[0] >> b) & 1) as u32).into(); }
            let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
            assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit != native for {code_hex}");
        }
    }

    // ---- environment opcodes via full programs ----------------------------
    const ECODE: usize = 24;
    const EMEM: usize = 32;
    const ESTEPS: usize = 8;
    const CDLEN: usize = 32;
    declare_circuit!(EnvCircuit {
        code: [[Variable; 8]; ECODE],
        cd: [[Variable; 8]; CDLEN],
        caller: [Variable; BITS],
        callvalue: [Variable; BITS],
        address: [Variable; BITS],
        ret: [[PublicVariable; 8]; 32],
    });
    impl Define<GF2Config> for EnvCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let cfg = Cfg { sd: 8, ss: 1, code_len: ECODE, mem_bytes: EMEM, sp_bits: 8, pc_bits: 8 };
            let code: Vec<Vec<Variable>> = (0..ECODE).map(|i| self.code[i].to_vec()).collect();
            let opts = StepOpts {
                calldata: (0..CDLEN).map(|i| self.cd[i].to_vec()).collect(),
                caller: self.caller.to_vec(),
                callvalue: self.callvalue.to_vec(),
                address: self.address.to_vec(),
                en_div: false, en_mulmod: false, en_exp: false, en_keccak: false,
            };
            let zw = vec![api.constant(0); BITS];
            let st = Evm {
                stack: vec![zw.clone(); 8], sp: vec![api.constant(0); 8], pc: vec![api.constant(0); 8],
                skeys: vec![zw.clone(); 2], svals: vec![zw.clone(); 2],
                mem: vec![vec![api.constant(0); 8]; EMEM], ret: vec![vec![api.constant(0); 8]; EMEM],
                ret_len: vec![api.constant(0); 8], halted: api.constant(0),
            };
            let fin = run_opts(api, st, &code, &cfg, &opts, ESTEPS);
            for i in 0..32 { for b in 0..8 { api.assert_is_equal(fin.ret[i][b], self.ret[i][b]); } }
        }
    }

    #[test]
    fn incircuit_env_opcodes() {
        let CompileResult { witness_solver, layered_circuit } =
            compile(&EnvCircuit::default(), CompileOptions::default()).unwrap();
        let caller = [0x11u8; 20];
        let address = [0x22u8; 20];
        let value = BigInt::from(0x1234u32);
        let calldata: Vec<u8> = (0..CDLEN).map(|i| i as u8).collect();
        let caller_w = BigInt::from_bytes_be(num_bigint::Sign::Plus, &caller);
        let address_w = BigInt::from_bytes_be(num_bigint::Sign::Plus, &address);
        // CALLDATALOAD@0; CALLDATACOPY 32; CALLDATASIZE; CODESIZE; CALLER; CALLVALUE; ADDRESS.
        let progs: Vec<&str> = vec![
            "60003560005260206000f3",   // CALLDATALOAD 0
            "6020600060003760206000f3", // CALLDATACOPY(dst0,src0,len32); RETURN 32
            "3660005260206000f3",       // CALLDATASIZE => 32
            "3860005260206000f3",       // CODESIZE => ECODE
            "3360005260206000f3",       // CALLER
            "3460005260206000f3",       // CALLVALUE
            "3060005260206000f3",       // ADDRESS
        ];
        for code_hex in progs {
            let mut code: Vec<u8> = (0..code_hex.len() / 2).map(|i| u8::from_str_radix(&code_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
            assert!(code.len() <= ECODE);
            code.resize(ECODE, 0); // pad so native CODESIZE == ECODE (committed code length)
            let ctx = crate::evm::CallCtx {
                code: code.clone(), calldata: calldata.clone(), caller, address, value: value.clone(), ..Default::default()
            };
            let mut storage = std::collections::BTreeMap::new();
            let r = crate::evm::execute(&ctx, &mut storage, 1_000_000);
            assert!(r.success, "native env exec failed for {code_hex}");
            assert_eq!(r.return_data.len(), 32);
            let mut asg = EnvCircuit::<GF2>::default();
            for i in 0..ECODE { for b in 0..8 { asg.code[i][b] = (((code[i] >> b) & 1) as u32).into(); } }
            for i in 0..CDLEN { for b in 0..8 { asg.cd[i][b] = (((calldata[i] >> b) & 1) as u32).into(); } }
            for (i, bit) in bigint_to_bits(&caller_w, BITS).into_iter().enumerate() { asg.caller[i] = (bit as u32).into(); }
            for (i, bit) in bigint_to_bits(&value, BITS).into_iter().enumerate() { asg.callvalue[i] = (bit as u32).into(); }
            for (i, bit) in bigint_to_bits(&address_w, BITS).into_iter().enumerate() { asg.address[i] = (bit as u32).into(); }
            for i in 0..32 { for b in 0..8 { asg.ret[i][b] = (((r.return_data[i] >> b) & 1) as u32).into(); } }
            let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
            assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit env != native for {code_hex}");
        }
    }

    // ---- KECCAK256 via full program (en_keccak) ---------------------------
    const KCODE: usize = 20;
    const KMEM: usize = 34;
    const KSTEPS: usize = 12;
    declare_circuit!(KeccakCircuit {
        code: [[Variable; 8]; KCODE],
        ret: [[PublicVariable; 8]; 32],
    });
    impl Define<GF2Config> for KeccakCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let cfg = Cfg { sd: 8, ss: 1, code_len: KCODE, mem_bytes: KMEM, sp_bits: 8, pc_bits: 8 };
            let mut opts = StepOpts::empty(api);
            opts.en_keccak = true;
            let code: Vec<Vec<Variable>> = (0..KCODE).map(|i| self.code[i].to_vec()).collect();
            let zw = vec![api.constant(0); BITS];
            let st = Evm {
                stack: vec![zw.clone(); 8], sp: vec![api.constant(0); 8], pc: vec![api.constant(0); 8],
                skeys: vec![zw.clone()], svals: vec![zw.clone()],
                mem: vec![vec![api.constant(0); 8]; KMEM], ret: vec![vec![api.constant(0); 8]; KMEM],
                ret_len: vec![api.constant(0); 8], halted: api.constant(0),
            };
            let fin = run_opts(api, st, &code, &cfg, &opts, KSTEPS);
            for i in 0..32 { for b in 0..8 { api.assert_is_equal(fin.ret[i][b], self.ret[i][b]); } }
        }
    }

    #[test]
    fn incircuit_keccak256() {
        // MSTORE8 mem[0]=0x2a; KECCAK256(off=0,len=1); MSTORE digest@0; RETURN 32.
        let code_hex = "602a600053600160002060005260206000f3";
        let mut code: Vec<u8> = (0..code_hex.len() / 2).map(|i| u8::from_str_radix(&code_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        assert!(code.len() <= KCODE);
        code.resize(KCODE, 0);
        let ctx = crate::evm::CallCtx { code: code.clone(), ..Default::default() };
        let mut storage = std::collections::BTreeMap::new();
        let r = crate::evm::execute(&ctx, &mut storage, 1_000_000);
        assert!(r.success);
        assert_eq!(r.return_data.len(), 32, "keccak program must return 32 bytes");
        let CompileResult { witness_solver, layered_circuit } =
            compile(&KeccakCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = KeccakCircuit::<GF2>::default();
        for i in 0..KCODE { for b in 0..8 { asg.code[i][b] = (((code[i] >> b) & 1) as u32).into(); } }
        for i in 0..32 { for b in 0..8 { asg.ret[i][b] = (((r.return_data[i] >> b) & 1) as u32).into(); } }
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit KECCAK256 != native");
    }
}
