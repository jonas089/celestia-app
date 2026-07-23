//! RV32IM CPU-verifier with register/memory consistency by OFFLINE MEMORY
//! CHECKING (a grand-product argument over GF(2^128)) instead of O(cycles x state)
//! in-circuit state threading.
//!
//! Everything is still bit-level over GF2 (api.add == XOR, api.mul == AND). The
//! per-cycle TRANSITION constraints are unchanged in spirit (ROM fetch by a
//! threaded pc, decode, immediates, ALU by bit-decomposition, next-pc / control
//! flow). What changes is CONSISTENCY: we no longer reconstruct the register file
//! and data memory each cycle. Instead every register/memory access is modelled
//! as a (addr, value, timestamp) tuple and we prove a multiset equality
//!
//!     multiset(init writes  U  all write-backs)  ==  multiset(all reads  U  final reads)
//!
//! via  PROD read-fingerprints == PROD write-fingerprints, where each tuple is
//! fingerprinted into one GF(2^128) element
//!
//!     h  = addr + value*beta + ts*beta^2                  (RLC of the tuple)
//!     fp = alpha + h                                      (product-offset form)
//!
//! with TWO Fiat-Shamir challenges alpha, beta in GF(2^128), derived from the
//! committed trace and injected as PUBLIC INPUTS (see main_gp.rs for the
//! binding). The offset `alpha` (a) makes every factor non-zero with
//! overwhelming probability -- without it the tuple (addr,val,ts)=(0,0,0), which
//! occurs for register x0's init write and its first read, would fingerprint to 0
//! and zero out BOTH product sides, making the check vacuous -- and (b) gives a
//! degree-1 (in alpha) factor, so PROD_R == PROD_W over random alpha iff the RLC
//! multisets {h_i} match (Schwartz-Zippel), iff the tuple multisets match. The
//! grand product is a layered tree of the in-circuit GF(2^128) multiplies.
//!
//! Construction (Blum et al. / Spice / Ceno "no-multiplicity" offline memory
//! checking): every access (read OR write) emits a read tuple of the CURRENT cell
//! (addr, old_value, prev_ts) and a write-back tuple (addr, new_value, now_ts).
//! Reads write back the same value; writes write back the new value. `now_ts` is
//! a unique per-slot compile-time constant; `prev_ts` is a committed hint that is
//! range-checked `prev_ts < now_ts` (the monotonicity check that makes reading a
//! *future* write impossible). Init writes seed each cell at ts=0; final reads
//! drain each cell at its last-write ts. Inactive slots emit an identical
//! read==write tuple, which contributes an equal factor to both product sides and
//! therefore cancels.

use crate::emulator::StepRecord;
use crate::gf128;
use expander_compiler::frontend::*;

pub const XLEN: usize = 32;
pub const NREG: usize = 32;
// Timestamp bit width. Every access emits <= SLOTS events/cycle, so the largest
// now_ts is 4*cycles+1; 32 bits covers up to ~1B cycles (far beyond any realistic
// trace) so long executions never overflow the monotonic-timestamp range check.
pub const TS: usize = 32;
pub const SLOTS: usize = 4; // rs1-read, rs2-read, mem, rd-write
pub const NOUT: usize = 3;

// Config set by the driver before compile(). NCYC / NMEM drive the Vec field
// lengths of the template instance, so the compiled circuit scales with cycles.
pub static mut NCYC: usize = 0;
pub static mut NMEM: usize = 0;
pub static mut PROG_LEN: usize = 0;
pub static mut PROGRAM: Vec<u32> = Vec::new();
pub static mut MEM_ADDRS: Vec<u32> = Vec::new();
pub static mut MEM_INIT: Vec<u32> = Vec::new();
/// Which entries of the `naddr = NREG + NMEM` final-value array feed the NOUT
/// public outputs. Sentinel [0;NOUT] (all x0, which is always 0) => use the
/// legacy default [2, 3, NREG+NMEM-1] so `main_gp` is unaffected.
pub static mut OUT_IDX: [usize; NOUT] = [0; NOUT];
/// Legacy fixed base kept for the standalone bins that read `ckt::BASE` as a
/// const. The circuit's ROM/pc use the configurable [`PROG_BASE`] below (default
/// 0), so legacy callers are unaffected.
pub const BASE: u32 = 0;
/// Configurable program base address for the ROM-fetch gate and the initial pc.
/// Defaults to 0 (== BASE); `rv32_prove` sets it so arbitrary-based programs
/// verify without touching the legacy bins.
pub static mut PROG_BASE: u32 = 0;
/// Per-register initial values seeded at ts=0. Empty => all zero (the default,
/// exactly the legacy behaviour). `rv32_prove` sets this to prove a block that
/// begins from a persistent pre-register state (`pre_regs`).
pub static mut REG_INIT: Vec<u32> = Vec::new();

fn ncyc() -> usize { unsafe { NCYC } }
fn nmem() -> usize { unsafe { NMEM } }
fn prog() -> &'static [u32] { unsafe { PROGRAM.as_slice() } }
fn mem_addrs() -> &'static [u32] { unsafe { MEM_ADDRS.as_slice() } }
fn mem_init() -> &'static [u32] { unsafe { MEM_INIT.as_slice() } }
fn prog_base() -> u32 { unsafe { PROG_BASE } }
fn reg_init(r: usize) -> u32 { unsafe { REG_INIT.get(r).copied().unwrap_or(0) } }

declare_circuit!(RiscvCircuitGp {
    insn: [[Variable; XLEN]],       // NCYC
    rs1_val: [[Variable; XLEN]],    // NCYC
    rs2_val: [[Variable; XLEN]],    // NCYC
    rd_val: [[Variable; XLEN]],     // NCYC
    tprev: [[Variable; TS]],        // NCYC*SLOTS  (slot-major: c*SLOTS + slot)
    vold_c: [[Variable; XLEN]],     // NCYC  (old memory WORD at addr&!3, load & store)
    vold_d: [[Variable; XLEN]],     // NCYC  (old register value for rd write)
    div_q: [[Variable; XLEN]],      // NCYC  (DIV/DIVU quotient hint; constrained)
    div_r: [[Variable; XLEN]],      // NCYC  (REM/REMU remainder hint; constrained)
    fin_val: [[Variable; XLEN]],    // NREG + NMEM  (final value per address)
    fin_ts: [[Variable; TS]],       // NREG + NMEM  (final last-write ts per address)
    alpha: [PublicVariable; 128],   // Fiat-Shamir product-offset challenge (public)
    beta: [PublicVariable; 128],    // Fiat-Shamir RLC challenge (public)
    out: [[PublicVariable; XLEN]],  // NOUT
});

// -------------------------------- assignment --------------------------------

/// All prover hints for one trace, produced by replaying the memory model.
pub struct Hints {
    pub tprev: Vec<[u32; SLOTS]>, // per cycle
    pub vold_c: Vec<u32>,         // per cycle (old memory WORD, load & store)
    pub vold_d: Vec<u32>,         // per cycle (rd old value)
    pub div_q: Vec<u32>,          // per cycle (DIV/DIVU quotient)
    pub div_r: Vec<u32>,          // per cycle (REM/REMU remainder)
    pub fin_val: Vec<u32>,        // per address (NREG + NMEM)
    pub fin_ts: Vec<u32>,         // per address
}

fn set_word<const N: usize>(dst: &mut [GF2], word: u32) {
    for i in 0..N {
        dst[i] = ((word >> i) & 1).into();
    }
}

pub fn template() -> RiscvCircuitGp<Variable> {
    let n = ncyc();
    let naddr = NREG + nmem();
    RiscvCircuitGp {
        insn: vec![[Variable::default(); XLEN]; n],
        rs1_val: vec![[Variable::default(); XLEN]; n],
        rs2_val: vec![[Variable::default(); XLEN]; n],
        rd_val: vec![[Variable::default(); XLEN]; n],
        tprev: vec![[Variable::default(); TS]; n * SLOTS],
        vold_c: vec![[Variable::default(); XLEN]; n],
        vold_d: vec![[Variable::default(); XLEN]; n],
        div_q: vec![[Variable::default(); XLEN]; n],
        div_r: vec![[Variable::default(); XLEN]; n],
        fin_val: vec![[Variable::default(); XLEN]; naddr],
        fin_ts: vec![[Variable::default(); TS]; naddr],
        alpha: [Variable::default(); 128],
        beta: [Variable::default(); 128],
        out: vec![[Variable::default(); XLEN]; NOUT],
    }
}

pub fn build_assignment(
    trace: &[StepRecord],
    hints: &Hints,
    alpha: u128,
    beta: u128,
    out_vals: &[u32],
) -> RiscvCircuitGp<GF2> {
    let n = ncyc();
    let naddr = NREG + nmem();
    let mut a = RiscvCircuitGp::<GF2> {
        insn: vec![[GF2::default(); XLEN]; n],
        rs1_val: vec![[GF2::default(); XLEN]; n],
        rs2_val: vec![[GF2::default(); XLEN]; n],
        rd_val: vec![[GF2::default(); XLEN]; n],
        tprev: vec![[GF2::default(); TS]; n * SLOTS],
        vold_c: vec![[GF2::default(); XLEN]; n],
        vold_d: vec![[GF2::default(); XLEN]; n],
        div_q: vec![[GF2::default(); XLEN]; n],
        div_r: vec![[GF2::default(); XLEN]; n],
        fin_val: vec![[GF2::default(); XLEN]; naddr],
        fin_ts: vec![[GF2::default(); TS]; naddr],
        alpha: [GF2::default(); 128],
        beta: [GF2::default(); 128],
        out: vec![[GF2::default(); XLEN]; NOUT],
    };
    for c in 0..n {
        set_word::<XLEN>(&mut a.insn[c], trace[c].insn);
        set_word::<XLEN>(&mut a.rs1_val[c], trace[c].rs1_val);
        set_word::<XLEN>(&mut a.rs2_val[c], trace[c].rs2_val);
        set_word::<XLEN>(&mut a.rd_val[c], trace[c].rd_val);
        set_word::<XLEN>(&mut a.vold_c[c], hints.vold_c[c]);
        set_word::<XLEN>(&mut a.vold_d[c], hints.vold_d[c]);
        set_word::<XLEN>(&mut a.div_q[c], hints.div_q[c]);
        set_word::<XLEN>(&mut a.div_r[c], hints.div_r[c]);
        for s in 0..SLOTS {
            set_word::<TS>(&mut a.tprev[c * SLOTS + s], hints.tprev[c][s]);
        }
    }
    for i in 0..naddr {
        set_word::<XLEN>(&mut a.fin_val[i], hints.fin_val[i]);
        set_word::<TS>(&mut a.fin_ts[i], hints.fin_ts[i]);
    }
    for i in 0..128 {
        a.alpha[i] = (((alpha >> i) & 1) as u32).into();
        a.beta[i] = (((beta >> i) & 1) as u32).into();
    }
    for k in 0..NOUT {
        set_word::<XLEN>(&mut a.out[k], out_vals[k]);
    }
    a
}

/// Tamper helper: corrupt a committed loaded value (a memory read result).
pub fn tamper_rd_val(a: &mut RiscvCircuitGp<GF2>, cycle: usize, word: u32) {
    set_word::<XLEN>(&mut a.rd_val[cycle], word);
}

fn flip(x: GF2) -> GF2 { if x.is_zero() { 1u32.into() } else { 0u32.into() } }

/// Diagnostic tamper helpers (fields other than `insn` are private to this module
/// because declare_circuit! only marks the first field `pub`).
pub fn flip_fin_val(a: &mut RiscvCircuitGp<GF2>, addr_idx: usize, bit: usize) {
    a.fin_val[addr_idx][bit] = flip(a.fin_val[addr_idx][bit]);
}
pub fn flip_insn(a: &mut RiscvCircuitGp<GF2>, cycle: usize, bit: usize) {
    a.insn[cycle][bit] = flip(a.insn[cycle][bit]);
}
pub fn flip_tprev(a: &mut RiscvCircuitGp<GF2>, slot_index: usize, bit: usize) {
    a.tprev[slot_index][bit] = flip(a.tprev[slot_index][bit]);
}

// ------------------------------- bit gadgets -------------------------------

pub fn not1<C: Config>(api: &mut impl RootAPI<C>, x: Variable) -> Variable { api.sub(1, x) }
pub fn or1<C: Config>(api: &mut impl RootAPI<C>, a: Variable, b: Variable) -> Variable {
    let x = api.add(a, b);
    let y = api.mul(a, b);
    api.add(x, y)
}
pub fn mux1<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: Variable, b: Variable) -> Variable {
    let d = api.add(a, b);
    let t = api.mul(sel, d);
    api.add(b, t)
}
pub fn mux32<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| mux1(api, sel, a[i], b[i])).collect()
}
pub fn xor32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| api.add(a[i], b[i])).collect()
}
pub fn and32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| api.mul(a[i], b[i])).collect()
}
pub fn or32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..XLEN).map(|i| { let x = api.add(a[i], b[i]); let y = api.mul(a[i], b[i]); api.add(x, y) }).collect()
}
pub fn add32_cin<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable], cin: Variable) -> Vec<Variable> {
    let mut sum = vec![api.constant(0); XLEN];
    let mut carry = cin;
    for i in 0..XLEN {
        let ab = api.add(a[i], b[i]);
        sum[i] = api.add(ab, carry);
        let and_ab = api.mul(a[i], b[i]);
        let c_and = api.mul(carry, ab);
        carry = api.add(and_ab, c_and);
    }
    sum
}
pub fn add32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    add32_cin(api, a, b, zero)
}
pub fn sub32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let nb: Vec<Variable> = b.iter().map(|&x| not1(api, x)).collect();
    let one = api.constant(1);
    add32_cin(api, a, &nb, one)
}
pub fn sll32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], shamt: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut cur = a.to_vec();
    for (k, &sbit) in shamt.iter().enumerate().take(5) {
        let sh = 1usize << k;
        let shifted: Vec<Variable> = (0..XLEN).map(|i| if i >= sh { cur[i - sh] } else { zero }).collect();
        cur = (0..XLEN).map(|i| mux1(api, sbit, shifted[i], cur[i])).collect();
    }
    cur
}
pub fn srl32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], shamt: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut cur = a.to_vec();
    for (k, &sbit) in shamt.iter().enumerate().take(5) {
        let sh = 1usize << k;
        let shifted: Vec<Variable> = (0..XLEN).map(|i| if i + sh < XLEN { cur[i + sh] } else { zero }).collect();
        cur = (0..XLEN).map(|i| mux1(api, sbit, shifted[i], cur[i])).collect();
    }
    cur
}
pub fn eq_const<C: Config>(api: &mut impl RootAPI<C>, bits: &[Variable], value: u32) -> Variable {
    let mut acc = api.constant(1);
    for (i, &b) in bits.iter().enumerate() {
        let want = (value >> i) & 1;
        let term = if want == 1 { b } else { not1(api, b) };
        acc = api.mul(acc, term);
    }
    acc
}
pub fn eq32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Variable {
    let mut acc = api.constant(1);
    for i in 0..XLEN {
        let x = api.add(a[i], b[i]);
        let eqb = not1(api, x);
        acc = api.mul(acc, eqb);
    }
    acc
}
pub fn const_word<C: Config>(api: &mut impl RootAPI<C>, word: u32) -> Vec<Variable> {
    (0..XLEN).map(|i| api.constant((word >> i) & 1)).collect()
}

/// Borrow-out of the TS-bit subtraction (a_const - b_bits). 1 iff a_const < b.
pub fn borrow_lt<C: Config>(api: &mut impl RootAPI<C>, a_const: u32, b: &[Variable]) -> Variable {
    let mut borrow = api.constant(0);
    for i in 0..TS {
        let ai = api.constant((a_const >> i) & 1);
        let not_ai = not1(api, ai);
        let t1 = api.mul(not_ai, b[i]); // ~a & b
        let axb = api.add(ai, b[i]);
        let not_axb = not1(api, axb);
        let t2 = api.mul(not_axb, borrow); // ~(a^b) & borrow_in
        borrow = or1(api, t1, t2);
    }
    borrow
}

/// Generic 2-way mux over equal-length bit slices: returns `a` if sel==1 else `b`.
pub fn mux_vec<C: Config>(api: &mut impl RootAPI<C>, sel: Variable, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    (0..a.len()).map(|i| mux1(api, sel, a[i], b[i])).collect()
}

/// Unsigned less-than over full XLEN: borrow-out of (a - b). ==1 iff a <u b.
pub fn ltu32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Variable {
    let mut borrow = api.constant(0);
    for i in 0..XLEN {
        let not_ai = not1(api, a[i]);
        let t1 = api.mul(not_ai, b[i]); // ~a & b
        let axb = api.add(a[i], b[i]);
        let not_axb = not1(api, axb);
        let t2 = api.mul(not_axb, borrow); // ~(a^b) & borrow_in
        borrow = or1(api, t1, t2);
    }
    borrow
}

/// 1 iff the XLEN word is all zero.
pub fn is_zero32<C: Config>(api: &mut impl RootAPI<C>, x: &[Variable]) -> Variable {
    let mut acc = api.constant(1);
    for i in 0..XLEN {
        let nz = not1(api, x[i]);
        acc = api.mul(acc, nz);
    }
    acc
}

/// Two's-complement negation of an XLEN word: (~x) + 1.
pub fn neg32<C: Config>(api: &mut impl RootAPI<C>, x: &[Variable]) -> Vec<Variable> {
    let nx: Vec<Variable> = x.iter().map(|&b| not1(api, b)).collect();
    let mut one = vec![api.constant(0); XLEN];
    one[0] = api.constant(1);
    add32(api, &nx, &one)
}

/// Absolute value of a signed XLEN word: negate iff the sign bit is set.
pub fn abs32<C: Config>(api: &mut impl RootAPI<C>, x: &[Variable]) -> Vec<Variable> {
    let n = neg32(api, x);
    mux_vec(api, x[XLEN - 1], &n, x)
}

/// Arithmetic shift right by shamt[0..5]: like srl32 but the vacated high bits are
/// filled with the ORIGINAL sign bit (a[31]) rather than zero.
pub fn sra32<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], shamt: &[Variable]) -> Vec<Variable> {
    let sign = a[XLEN - 1];
    let mut cur = a.to_vec();
    for (k, &sbit) in shamt.iter().enumerate().take(5) {
        let sh = 1usize << k;
        let shifted: Vec<Variable> =
            (0..XLEN).map(|i| if i + sh < XLEN { cur[i + sh] } else { sign }).collect();
        cur = (0..XLEN).map(|i| mux1(api, sbit, shifted[i], cur[i])).collect();
    }
    cur
}

/// Ripple-carry add of two equal-length bit slices, result truncated to len bits.
pub fn add_bits<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable]) -> Vec<Variable> {
    let n = a.len();
    let mut sum = vec![api.constant(0); n];
    let mut carry = api.constant(0);
    for i in 0..n {
        let ab = api.add(a[i], b[i]);
        sum[i] = api.add(ab, carry);
        let and_ab = api.mul(a[i], b[i]);
        let c_and = api.mul(carry, ab);
        carry = api.add(and_ab, c_and);
    }
    sum
}

/// Low `n` bits of the product of two n-bit slices (schoolbook shift-add).
pub fn mul_low_bits<C: Config>(api: &mut impl RootAPI<C>, a: &[Variable], b: &[Variable], n: usize) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut acc = vec![zero; n];
    for i in 0..n {
        let partial: Vec<Variable> =
            (0..n).map(|k| if k >= i { api.mul(a[i], b[k - i]) } else { zero }).collect();
        acc = add_bits(api, &acc, &partial);
    }
    acc
}

/// Zero-extend an XLEN word to 64 bits.
pub fn zext64<C: Config>(api: &mut impl RootAPI<C>, x: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut v = vec![zero; 64];
    v[..XLEN].copy_from_slice(&x[..XLEN]);
    v
}
/// Sign-extend an XLEN word to 64 bits.
pub fn sext64<C: Config>(_api: &mut impl RootAPI<C>, x: &[Variable]) -> Vec<Variable> {
    let mut v = vec![x[XLEN - 1]; 64];
    v[..XLEN].copy_from_slice(&x[..XLEN]);
    v
}

/// Zero-extend an arbitrary short bit slice to XLEN.
pub fn zext_to32<C: Config>(api: &mut impl RootAPI<C>, bits: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut v = vec![zero; XLEN];
    v[..bits.len()].copy_from_slice(bits);
    v
}
/// Sign-extend an arbitrary short bit slice to XLEN.
pub fn sext_to32<C: Config>(_api: &mut impl RootAPI<C>, bits: &[Variable]) -> Vec<Variable> {
    let s = bits[bits.len() - 1];
    let mut v = vec![s; XLEN];
    v[..bits.len()].copy_from_slice(bits);
    v
}

// --------------------------- GF(2^128) fingerprint --------------------------

/// Embed a k-bit little-endian source into a 128-bit GF element (high bits zero).
pub fn embed<C: Config>(api: &mut impl RootAPI<C>, src: &[Variable]) -> Vec<Variable> {
    let zero = api.constant(0);
    let mut v = vec![zero; 128];
    for (i, &s) in src.iter().enumerate().take(128) {
        v[i] = s;
    }
    v
}

/// factor = alpha + (addr + value*beta + ts*beta^2)  (all in GF(2^128)).
/// The RLC uses `beta`/`beta^2`; the offset `alpha` guarantees the factor is
/// non-zero w.h.p. (crucially for the (0,0,0) tuple) and makes it degree-1 in
/// alpha. `value` is 32-bit and `ts` is TS-bit, so the multiplies are bounded.
pub fn factor<C: Config>(
    api: &mut impl RootAPI<C>,
    addr: &[Variable],
    val: &[Variable],
    ts: &[Variable],
    alpha: &[Variable],
    beta: &[Variable],
    beta2: &[Variable],
) -> Vec<Variable> {
    let addr_e = embed(api, addr);
    let val_e = embed(api, val);
    let ts_e = embed(api, ts);
    let vg = gf128::mul_bounded(api, &val_e, beta, XLEN);
    let tg = gf128::mul_bounded(api, &ts_e, beta2, TS);
    let s1 = gf128::add(api, &addr_e, &vg);
    let h = gf128::add(api, &s1, &tg);
    gf128::add(api, alpha, &h)
}

/// Balanced product tree of GF(2^128) elements.
pub fn product_tree<C: Config>(api: &mut impl RootAPI<C>, mut layer: Vec<Vec<Variable>>) -> Vec<Variable> {
    if layer.is_empty() {
        return gf128::const_gf(api, 1);
    }
    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len() / 2 + 1);
        let mut i = 0;
        while i < layer.len() {
            if i + 1 < layer.len() {
                next.push(gf128::mul(api, &layer[i], &layer[i + 1]));
                i += 2;
            } else {
                next.push(layer[i].clone());
                i += 1;
            }
        }
        layer = next;
    }
    layer.pop().unwrap()
}

// -------------------------------- the circuit -------------------------------

impl Define<GF2Config> for RiscvCircuitGp<Variable> {
    fn define<Builder: RootAPI<GF2Config>>(&self, api: &mut Builder) {
        let zero = api.constant(0);
        let n = ncyc();
        let nm = nmem();
        let naddr = NREG + nm;
        let prog_len = prog().len();

        // Fiat-Shamir challenges: alpha (product offset), beta (RLC) and beta^2.
        let alpha: Vec<Variable> = self.alpha.to_vec();
        let beta: Vec<Variable> = self.beta.to_vec();
        let beta2 = gf128::mul(api, &beta, &beta);

        let mut write_fps: Vec<Vec<Variable>> = Vec::new();
        let mut read_fps: Vec<Vec<Variable>> = Vec::new();

        // ---- init writes: seed each cell at ts = 0 (all compile-time constants).
        for r in 0..NREG {
            let addr = const_word(api, r as u32);
            let val = const_word(api, reg_init(r));
            let ts = (0..TS).map(|_| zero).collect::<Vec<_>>();
            write_fps.push(factor(api, &addr, &val, &ts, &alpha, &beta, &beta2));
        }
        for k in 0..nm {
            let addr = const_word(api, mem_addrs()[k]);
            let val = const_word(api, mem_init()[k]);
            let ts = (0..TS).map(|_| zero).collect::<Vec<_>>();
            write_fps.push(factor(api, &addr, &val, &ts, &alpha, &beta, &beta2));
        }

        // ---- threaded program counter (small: threading pc is O(cycles), fine).
        let mut pc = const_word(api, prog_base());

        for c in 0..n {
            let insn = self.insn[c].to_vec();
            let rs1v = self.rs1_val[c].to_vec();
            let rs2v = self.rs2_val[c].to_vec();
            let rdv = self.rd_val[c].to_vec();

            // (0) FETCH: committed insn must equal ROM[pc]; pc must hit exactly one slot.
            let mut insn_rom = vec![zero; XLEN];
            let mut hit = api.constant(0);
            for j in 0..prog_len {
                let addr_j = const_word(api, prog_base() + 4 * j as u32);
                let m = eq32(api, &pc, &addr_j);
                hit = api.add(hit, m);
                let word = prog()[j];
                for b in 0..XLEN {
                    let wbit = api.constant((word >> b) & 1);
                    let t = api.mul(m, wbit);
                    insn_rom[b] = api.add(insn_rom[b], t);
                }
            }
            api.assert_is_equal(hit, 1);
            for b in 0..XLEN {
                api.assert_is_equal(insn[b], insn_rom[b]);
            }

            // (1) Decode.
            let opcode = &insn[0..7];
            let rd_idx = &insn[7..12];
            let funct3 = &insn[12..15];
            let rs1_idx = &insn[15..20];
            let rs2_idx = &insn[20..25];
            let funct7 = &insn[25..32];

            let imm_i = {
                let mut v = vec![zero; XLEN];
                for i in 0..12 { v[i] = insn[20 + i]; }
                for i in 12..XLEN { v[i] = insn[31]; }
                v
            };
            let imm_s = {
                let mut v = vec![zero; XLEN];
                for i in 0..5 { v[i] = insn[7 + i]; }
                for i in 5..12 { v[i] = insn[20 + i]; }
                for i in 12..XLEN { v[i] = insn[31]; }
                v
            };
            let imm_b = {
                let mut v = vec![zero; XLEN];
                v[0] = zero;
                for i in 1..5 { v[i] = insn[7 + i]; }
                for i in 5..11 { v[i] = insn[20 + i]; }
                v[11] = insn[7];
                for i in 12..XLEN { v[i] = insn[31]; }
                v
            };
            let imm_j = {
                let mut v = vec![zero; XLEN];
                v[0] = zero;
                for i in 1..11 { v[i] = insn[20 + i]; }
                v[11] = insn[20];
                for i in 12..20 { v[i] = insn[i]; }
                for i in 20..XLEN { v[i] = insn[31]; }
                v
            };
            // U-type immediate: insn[31:12] placed in the high 20 bits, low 12 zero.
            let imm_u = {
                let mut v = vec![zero; XLEN];
                for i in 12..XLEN { v[i] = insn[i]; }
                v
            };

            // (2) Class + sub-op selectors.
            let is_op = eq_const(api, opcode, 0b0110011);
            let is_opimm = eq_const(api, opcode, 0b0010011);
            let is_load = eq_const(api, opcode, 0b0000011);
            let is_store = eq_const(api, opcode, 0b0100011);
            let is_branch = eq_const(api, opcode, 0b1100011);
            let is_jal = eq_const(api, opcode, 0b1101111);
            let is_jalr = eq_const(api, opcode, 0b1100111);
            let is_lui = eq_const(api, opcode, 0b0110111);
            let is_auipc = eq_const(api, opcode, 0b0010111);
            let is_alu = api.add(is_op, is_opimm);
            let f3_0 = eq_const(api, funct3, 0x0);
            let f3_1 = eq_const(api, funct3, 0x1);
            let f3_2 = eq_const(api, funct3, 0x2);
            let f3_3 = eq_const(api, funct3, 0x3);
            let f3_4 = eq_const(api, funct3, 0x4);
            let f3_5 = eq_const(api, funct3, 0x5);
            let f3_6 = eq_const(api, funct3, 0x6);
            let f3_7 = eq_const(api, funct3, 0x7);
            let f7_zero = eq_const(api, funct7, 0x00);
            let f7_sub = eq_const(api, funct7, 0x20);
            let f7_m = eq_const(api, funct7, 0x01); // RV32M funct7

            let operand_b = mux32(api, is_opimm, &imm_i, &rs2v);
            let shamt: Vec<Variable> = operand_b[0..5].to_vec();

            // (3) ALU results. operand_b already selects imm_i (OPIMM) vs rs2v (OP).
            let add_res = add32(api, &rs1v, &operand_b);
            let sub_res = sub32(api, &rs1v, &rs2v);
            let xor_res = xor32(api, &rs1v, &operand_b);
            let or_res = or32(api, &rs1v, &operand_b);
            let and_res = and32(api, &rs1v, &operand_b);
            let sll_res = sll32(api, &rs1v, &shamt);
            let srl_res = srl32(api, &rs1v, &shamt);
            let sra_res = sra32(api, &rs1v, &shamt);

            // Compares: SLT(I) signed, SLTU(I) unsigned. Result is a boolean in bit 0.
            let ltu_ob = ltu32(api, &rs1v, &operand_b);
            // signed lt = ltu XOR sign(a) XOR sign(b)  (b = operand_b, so SLTI uses imm).
            let lt_s = {
                let t = api.add(ltu_ob, rs1v[XLEN - 1]);
                api.add(t, operand_b[XLEN - 1])
            };
            let slt_res = { let mut v = vec![zero; XLEN]; v[0] = lt_s; v };
            let sltu_res = { let mut v = vec![zero; XLEN]; v[0] = ltu_ob; v };

            // ---- RV32M multiply: 64-bit products, take low/high 32. ----
            let a_u = zext64(api, &rs1v);
            let b_u = zext64(api, &rs2v);
            let a_s = sext64(api, &rs1v);
            let b_s = sext64(api, &rs2v);
            let prod_uu = mul_low_bits(api, &a_u, &b_u, 64);
            let prod_ss = mul_low_bits(api, &a_s, &b_s, 64);
            let prod_su = mul_low_bits(api, &a_s, &b_u, 64);
            let mul_res = prod_uu[0..XLEN].to_vec();           // MUL: low 32 (sign-agnostic)
            let mulhu_res = prod_uu[XLEN..2 * XLEN].to_vec();  // MULHU: high 32 (u*u)
            let mulh_res = prod_ss[XLEN..2 * XLEN].to_vec();   // MULH: high 32 (s*s)
            let mulhsu_res = prod_su[XLEN..2 * XLEN].to_vec(); // MULHSU: high 32 (s*u)

            // ---- RV32M divide: results from committed q/r hints, with the special
            // cases (div-by-zero / signed overflow) forced; hints constrained below.
            let div_q = self.div_q[c].to_vec();
            let div_r = self.div_r[c].to_vec();
            let b_zero = is_zero32(api, &rs2v);
            let a_is_min = eq_const(api, &rs1v, 0x8000_0000);
            let b_is_neg1 = eq_const(api, &rs2v, 0xFFFF_FFFF);
            let ovf = api.mul(a_is_min, b_is_neg1);
            let allones = const_word(api, 0xFFFF_FFFF);
            let intmin = const_word(api, 0x8000_0000);
            let zeros = const_word(api, 0);
            // unsigned: DIVU = b==0 ? all-ones : q ; REMU = b==0 ? a : r
            let divu_res = mux_vec(api, b_zero, &allones, &div_q);
            let remu_res = mux_vec(api, b_zero, &rs1v, &div_r);
            // signed: DIV = b==0 ? -1 : (ovf ? INT_MIN : q) ; REM = b==0 ? a : (ovf ? 0 : r)
            let div_norm = mux_vec(api, ovf, &intmin, &div_q);
            let div_res = mux_vec(api, b_zero, &allones, &div_norm);
            let rem_norm = mux_vec(api, ovf, &zeros, &div_r);
            let rem_res = mux_vec(api, b_zero, &rs1v, &rem_norm);

            // ---- ALU op selectors (mutually exclusive). is_op gates funct7 so RV32M
            // (funct7==0x01) and SUB/SRA (funct7==0x20) don't collide with the base
            // ops (funct7==0x00). OPIMM has no funct7 except the SR shifts.
            let a2 = |api: &mut Builder, x, y| api.mul(x, y);
            let a3 = |api: &mut Builder, x, y, z| { let t = api.mul(x, y); api.mul(t, z) };
            let sel_add = { let t1 = a3(api, is_op, f3_0, f7_zero); let t2 = a2(api, is_opimm, f3_0); api.add(t1, t2) };
            let sel_sub = a3(api, is_op, f3_0, f7_sub);
            let sel_sll = { let t1 = a3(api, is_op, f3_1, f7_zero); let t2 = a2(api, is_opimm, f3_1); api.add(t1, t2) };
            let sel_slt = { let t1 = a3(api, is_op, f3_2, f7_zero); let t2 = a2(api, is_opimm, f3_2); api.add(t1, t2) };
            let sel_sltu = { let t1 = a3(api, is_op, f3_3, f7_zero); let t2 = a2(api, is_opimm, f3_3); api.add(t1, t2) };
            let sel_xor = { let t1 = a3(api, is_op, f3_4, f7_zero); let t2 = a2(api, is_opimm, f3_4); api.add(t1, t2) };
            let sel_srl = { let t1 = a3(api, is_op, f3_5, f7_zero); let t2 = a3(api, is_opimm, f3_5, f7_zero); api.add(t1, t2) };
            let sel_sra = { let t1 = a3(api, is_op, f3_5, f7_sub); let t2 = a3(api, is_opimm, f3_5, f7_sub); api.add(t1, t2) };
            let sel_or = { let t1 = a3(api, is_op, f3_6, f7_zero); let t2 = a2(api, is_opimm, f3_6); api.add(t1, t2) };
            let sel_and = { let t1 = a3(api, is_op, f3_7, f7_zero); let t2 = a2(api, is_opimm, f3_7); api.add(t1, t2) };
            let sel_mul = a3(api, is_op, f3_0, f7_m);
            let sel_mulh = a3(api, is_op, f3_1, f7_m);
            let sel_mulhsu = a3(api, is_op, f3_2, f7_m);
            let sel_mulhu = a3(api, is_op, f3_3, f7_m);
            let sel_div = a3(api, is_op, f3_4, f7_m);
            let sel_divu = a3(api, is_op, f3_5, f7_m);
            let sel_rem = a3(api, is_op, f3_6, f7_m);
            let sel_remu = a3(api, is_op, f3_7, f7_m);

            let sels = [
                sel_add, sel_sub, sel_xor, sel_or, sel_and, sel_sll, sel_srl, sel_sra,
                sel_slt, sel_sltu, sel_mul, sel_mulh, sel_mulhsu, sel_mulhu,
                sel_div, sel_divu, sel_rem, sel_remu,
            ];
            let ress: [&Vec<Variable>; 18] = [
                &add_res, &sub_res, &xor_res, &or_res, &and_res, &sll_res, &srl_res, &sra_res,
                &slt_res, &sltu_res, &mul_res, &mulh_res, &mulhsu_res, &mulhu_res,
                &div_res, &divu_res, &rem_res, &remu_res,
            ];
            let mut alu_rd = vec![zero; XLEN];
            for (sel, res) in sels.iter().zip(ress.iter()) {
                for b in 0..XLEN {
                    let t = api.mul(*sel, res[b]);
                    alu_rd[b] = api.add(alu_rd[b], t);
                }
            }

            // ---- Division correctness: constrain the q/r hints via the Euclid
            // identity dividend = divisor*quotient + remainder (64-bit, exact) plus
            // range/sign checks. Special cases (b==0, signed INT_MIN/-1) bypass this.
            let one = api.constant(1);
            let not_bzero = not1(api, b_zero);
            let not_ovf = not1(api, ovf);
            // unsigned group active (and divisor nonzero).
            let act_u = { let g = api.add(sel_divu, sel_remu); api.mul(g, not_bzero) };
            {
                let dq = zext64(api, &div_q);
                let bq = mul_low_bits(api, &b_u, &dq, 64);
                let dr = zext64(api, &div_r);
                let sum = add_bits(api, &bq, &dr);
                let target = zext64(api, &rs1v);
                for b in 0..64 {
                    let d = api.add(sum[b], target[b]);
                    let g = api.mul(act_u, d);
                    api.assert_is_equal(g, 0);
                }
                // remainder < divisor (unsigned).
                let lt = ltu32(api, &div_r, &rs2v);
                let nlt = api.sub(one, lt);
                let g = api.mul(act_u, nlt);
                api.assert_is_equal(g, 0);
            }
            // signed group active (divisor nonzero and not the INT_MIN/-1 overflow).
            let act_s = {
                let g = api.add(sel_div, sel_rem);
                let g = api.mul(g, not_bzero);
                api.mul(g, not_ovf)
            };
            {
                let dq = sext64(api, &div_q);
                let bq = mul_low_bits(api, &b_s, &dq, 64);
                let dr = sext64(api, &div_r);
                let sum = add_bits(api, &bq, &dr);
                let target = sext64(api, &rs1v);
                for b in 0..64 {
                    let d = api.add(sum[b], target[b]);
                    let g = api.mul(act_s, d);
                    api.assert_is_equal(g, 0);
                }
                // |remainder| < |divisor|.
                let ar = abs32(api, &div_r);
                let ab = abs32(api, &rs2v);
                let lt = ltu32(api, &ar, &ab);
                let nlt = api.sub(one, lt);
                let g = api.mul(act_s, nlt);
                api.assert_is_equal(g, 0);
                // sign(remainder) == sign(dividend) OR remainder == 0 (truncated div).
                let r_zero = is_zero32(api, &div_r);
                let sign_diff = api.add(div_r[XLEN - 1], rs1v[XLEN - 1]);
                let same_sign = not1(api, sign_diff);
                let sign_ok = or1(api, same_sign, r_zero);
                let nok = not1(api, sign_ok);
                let g = api.mul(act_s, nok);
                api.assert_is_equal(g, 0);
            }

            // (4) link value (pc+4), U-type / AUIPC derived values.
            let four = const_word(api, 4);
            let pc_plus4 = add32(api, &pc, &four);
            let auipc_res = add32(api, &pc, &imm_u);

            // (5) memory address; the grand-product tuple keys on the WORD address
            // (addr & !3). Sub-word LOAD/STORE extract/insert against the committed
            // old memory word (self.vold_c[c]), which the grand product pins to the
            // current cell value.
            let load_addr = add32(api, &rs1v, &imm_i);
            let store_addr = add32(api, &rs1v, &imm_s);
            let mem_addr = mux32(api, is_store, &store_addr, &load_addr);
            let off0 = mem_addr[0];
            let off1 = mem_addr[1];
            let word_addr = { let mut v = mem_addr.clone(); v[0] = zero; v[1] = zero; v };
            let old_word = self.vold_c[c].to_vec();

            // byte / halfword extract from old_word by the 2-bit byte offset.
            let b0 = old_word[0..8].to_vec();
            let b1 = old_word[8..16].to_vec();
            let b2 = old_word[16..24].to_vec();
            let b3 = old_word[24..32].to_vec();
            let lo_b = mux_vec(api, off0, &b1, &b0);
            let hi_b = mux_vec(api, off0, &b3, &b2);
            let sel_byte = mux_vec(api, off1, &hi_b, &lo_b); // 8 bits
            let h0 = old_word[0..16].to_vec();
            let h1 = old_word[16..32].to_vec();
            let sel_half = mux_vec(api, off1, &h1, &h0); // 16 bits
            let lb_res = sext_to32(api, &sel_byte);
            let lbu_res = zext_to32(api, &sel_byte);
            let lh_res = sext_to32(api, &sel_half);
            let lhu_res = zext_to32(api, &sel_half);
            let lw_res = old_word.clone();
            // load result mux by funct3: LB=0, LH=1, LW=2, LBU=4, LHU=5.
            let load_res = {
                let mut v = vec![zero; XLEN];
                let parts: [(Variable, &Vec<Variable>); 5] =
                    [(f3_0, &lb_res), (f3_1, &lh_res), (f3_2, &lw_res), (f3_4, &lbu_res), (f3_5, &lhu_res)];
                for (sel, r) in parts.iter() {
                    for b in 0..XLEN { let t = api.mul(*sel, r[b]); v[b] = api.add(v[b], t); }
                }
                v
            };

            // store insert into old_word: SB(f3=0) byte, SH(f3=1) half, SW(f3=2) word.
            let noff0 = not1(api, off0);
            let noff1 = not1(api, off1);
            let lane0 = api.mul(noff1, noff0);
            let lane1 = api.mul(noff1, off0);
            let lane2 = api.mul(off1, noff0);
            let lane3 = api.mul(off1, off0);
            let lanes = [lane0, lane1, lane2, lane3];
            let sb_word = {
                let mut v = old_word.clone();
                for j in 0..4 {
                    for k in 0..8 {
                        v[j * 8 + k] = mux1(api, lanes[j], rs2v[k], old_word[j * 8 + k]);
                    }
                }
                v
            };
            let sh_word = {
                // off1==0 -> low half = rs2[0..16]; off1==1 -> high half = rs2[0..16].
                let mut v = old_word.clone();
                for k in 0..16 {
                    v[k] = mux1(api, off1, old_word[k], rs2v[k]);
                    v[16 + k] = mux1(api, off1, rs2v[k], old_word[16 + k]);
                }
                v
            };
            let new_word = {
                let mut v = vec![zero; XLEN];
                let parts: [(Variable, &Vec<Variable>); 3] =
                    [(f3_0, &sb_word), (f3_1, &sh_word), (f3_2, &rs2v)];
                for (sel, r) in parts.iter() {
                    for b in 0..XLEN { let t = api.mul(*sel, r[b]); v[b] = api.add(v[b], t); }
                }
                v
            };

            // (6) computed rd for every register-writing class, validated against the
            // committed rd_val. LOAD's rd = sub-word extract of the memory-checked old
            // word; JAL/JALR link = pc+4; LUI = imm_u; AUIPC = pc+imm_u.
            let link = api.add(is_jal, is_jalr);
            let mut computed_rd = alu_rd.clone();
            for b in 0..XLEN {
                let t_link = api.mul(link, pc_plus4[b]);
                let t_lui = api.mul(is_lui, imm_u[b]);
                let t_auipc = api.mul(is_auipc, auipc_res[b]);
                let t_load = api.mul(is_load, load_res[b]);
                computed_rd[b] = api.add(computed_rd[b], t_link);
                computed_rd[b] = api.add(computed_rd[b], t_lui);
                computed_rd[b] = api.add(computed_rd[b], t_auipc);
                computed_rd[b] = api.add(computed_rd[b], t_load);
            }
            let check_rd = {
                let t = api.add(is_alu, is_jal);
                let t = api.add(t, is_jalr);
                let t = api.add(t, is_lui);
                let t = api.add(t, is_auipc);
                api.add(t, is_load)
            };
            for b in 0..XLEN {
                let diff = api.add(computed_rd[b], rdv[b]);
                let g = api.mul(check_rd, diff);
                api.assert_is_equal(g, 0);
            }

            // (7) write-enable / rd nonzero for the register-write slot. Every class
            // that writes a register participates so its write-back is recorded.
            let write_enable = {
                let t = api.add(is_alu, is_load);
                let t = api.add(t, is_jal);
                let t = api.add(t, is_jalr);
                let t = api.add(t, is_lui);
                api.add(t, is_auipc)
            };
            let rd_is_zero = eq_const(api, rd_idx, 0);
            let rd_nonzero = not1(api, rd_is_zero);
            let active_d = api.mul(write_enable, rd_nonzero);
            let active_c = api.add(is_load, is_store);

            // Register-index address words (5-bit zero-extended).
            let rs1_addr = { let mut v = vec![zero; XLEN]; for i in 0..5 { v[i] = rs1_idx[i]; } v };
            let rs2_addr = { let mut v = vec![zero; XLEN]; for i in 0..5 { v[i] = rs2_idx[i]; } v };
            let rd_addr = { let mut v = vec![zero; XLEN]; for i in 0..5 { v[i] = rd_idx[i]; } v };

            let vold_d = self.vold_d[c].to_vec();

            // Emit the 4 slots. Each slot: read tuple + write-back tuple.
            // (addr, read_val, write_val, active) computed below; now_ts constant.
            let null_val = vec![zero; XLEN];
            for slot in 0..SLOTS {
                let now_ts_c = (c * SLOTS + slot + 1) as u32; // unique, >= 1
                let tprev = self.tprev[c * SLOTS + slot].to_vec();

                let (addr, read_val, write_val, active): (Vec<Variable>, Vec<Variable>, Vec<Variable>, Variable) =
                    match slot {
                        0 => (rs1_addr.clone(), rs1v.clone(), rs1v.clone(), api.constant(1)),
                        1 => (rs2_addr.clone(), rs2v.clone(), rs2v.clone(), api.constant(1)),
                        2 => {
                            // mem: both load & store READ the old word; a store WRITES the
                            // inserted new word, a load rewrites the same old word.
                            let rval = old_word.clone();
                            let wval = mux32(api, is_store, &new_word, &old_word);
                            (word_addr.clone(), rval, wval, active_c)
                        }
                        _ => (rd_addr.clone(), vold_d.clone(), rdv.clone(), active_d),
                    };

                // now_ts as bits.
                let now_ts_bits: Vec<Variable> = (0..TS).map(|i| api.constant((now_ts_c >> i) & 1)).collect();

                // Range check prev_ts < now_ts when active (monotonicity).
                let borrow = borrow_lt(api, now_ts_c.wrapping_sub(1), &tprev); // 1 iff (now-1) < tprev
                let g = api.mul(active, borrow);
                api.assert_is_equal(g, 0);

                // Active: real read (addr, read_val, tprev) + writeback (addr, write_val, now_ts).
                // Inactive: identical read==write null tuple (cancels across sides).
                let r_addr = mux32(api, active, &addr, &null_val);
                let r_val = mux32(api, active, &read_val, &null_val);
                let r_ts: Vec<Variable> = (0..TS).map(|i| mux1(api, active, tprev[i], now_ts_bits[i])).collect();
                let w_addr = mux32(api, active, &addr, &null_val);
                let w_val = mux32(api, active, &write_val, &null_val);
                let w_ts = now_ts_bits.clone(); // both null and active writeback use now_ts

                read_fps.push(factor(api, &r_addr, &r_val, &r_ts, &alpha, &beta, &beta2));
                write_fps.push(factor(api, &w_addr, &w_val, &w_ts, &alpha, &beta, &beta2));
            }

            // (8) next pc + control flow; thread pc. All six branch conditions,
            // JAL and JALR are handled.
            let eq = eq32(api, &rs1v, &rs2v);
            let neq = not1(api, eq);
            let ltu_rr = ltu32(api, &rs1v, &rs2v);
            let lts_rr = { let t = api.add(ltu_rr, rs1v[XLEN - 1]); api.add(t, rs2v[XLEN - 1]) };
            let nlts = not1(api, lts_rr);
            let nltu = not1(api, ltu_rr);
            // taken condition selected by funct3: BEQ0 BNE1 BLT4 BGE5 BLTU6 BGEU7.
            let taken_cond = {
                let mut acc = api.mul(f3_0, eq);
                let t = api.mul(f3_1, neq); acc = api.add(acc, t);
                let t = api.mul(f3_4, lts_rr); acc = api.add(acc, t);
                let t = api.mul(f3_5, nlts); acc = api.add(acc, t);
                let t = api.mul(f3_6, ltu_rr); acc = api.add(acc, t);
                let t = api.mul(f3_7, nltu); acc = api.add(acc, t);
                acc
            };
            let branch_taken = api.mul(is_branch, taken_cond);
            let branch_target = add32(api, &pc, &imm_b);
            let jal_target = add32(api, &pc, &imm_j);
            // JALR target = (rs1 + imm_i) with bit 0 cleared.
            let jalr_target = { let mut v = add32(api, &rs1v, &imm_i); v[0] = zero; v };
            let mut np = pc_plus4.clone();
            np = mux32(api, branch_taken, &branch_target, &np);
            np = mux32(api, is_jal, &jal_target, &np);
            np = mux32(api, is_jalr, &jalr_target, &np);
            pc = np;
        }

        // ---- final reads: drain each cell (addr const, final value+ts hints).
        for r in 0..NREG {
            let addr = const_word(api, r as u32);
            let val = self.fin_val[r].to_vec();
            let ts = self.fin_ts[r].to_vec();
            read_fps.push(factor(api, &addr, &val, &ts, &alpha, &beta, &beta2));
        }
        for k in 0..nm {
            let addr = const_word(api, mem_addrs()[k]);
            let val = self.fin_val[NREG + k].to_vec();
            let ts = self.fin_ts[NREG + k].to_vec();
            read_fps.push(factor(api, &addr, &val, &ts, &alpha, &beta, &beta2));
        }

        // ---- grand product: assert PROD(read) == PROD(write).
        let pr = product_tree(api, read_fps);
        let pw = product_tree(api, write_fps);
        for b in 0..128 {
            api.assert_is_equal(pr[b], pw[b]);
        }

        // ---- public outputs: expose OUT_IDX[k]'s final value (validated by the
        // grand product). Sentinel [0,0,0] -> legacy [2, 3, NREG+NMEM-1].
        let out_idx = {
            let cfg = unsafe { OUT_IDX };
            if cfg == [0usize; NOUT] {
                [2, 3, NREG + (nm - 1)]
            } else {
                cfg
            }
        };
        for k in 0..NOUT {
            for b in 0..XLEN {
                api.assert_is_equal(self.fin_val[out_idx[k]][b], self.out[k][b]);
            }
        }

        let _ = naddr;
    }
}
