//! Genuine minimal EVM interpreter — a real opcode dispatch loop over EVM
//! bytecode, operating on true 256-bit words (8 x u32 little-endian limbs),
//! with a stack, byte-addressable memory, a persistent storage map, and gas
//! accounting. Correct EVM semantics for the supported opcode subset.
//!
//! Zero dependencies on purpose: this crate is the native EVM reference shared
//! by the RISC-V proof (its output/storage is what the proven RV32 execution
//! must reproduce) AND by `evm-xcheck`, which asserts it agrees byte-for-byte
//! with revm's ACTUAL execution on the same input (drift check).
//!
//! Supported opcodes: STOP, ADD, MUL, SUB, LT, GT, EQ, ISZERO, AND, OR, XOR,
//! NOT, BYTE, SHL, SHR, POP, MLOAD, MSTORE, MSTORE8, SLOAD, SSTORE, JUMP,
//! JUMPI, PC, MSIZE, JUMPDEST, PUSH1..PUSH32, DUP1..DUP16, SWAP1..SWAP16,
//! CALLDATALOAD, CALLDATASIZE, CALLDATACOPY, CODESIZE, RETURN, REVERT.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// U256: 256-bit unsigned word as 8 little-endian u32 limbs (limb[0] = LSB).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, Hash)]
pub struct U256(pub [u32; 8]);

impl U256 {
    pub const ZERO: U256 = U256([0; 8]);
    pub const ONE: U256 = U256([1, 0, 0, 0, 0, 0, 0, 0]);

    pub fn from_u64(x: u64) -> U256 {
        U256([x as u32, (x >> 32) as u32, 0, 0, 0, 0, 0, 0])
    }

    /// Interpret `bytes` (big-endian, up to 32 bytes, as EVM PUSH does) as a U256.
    pub fn from_be_bytes(bytes: &[u8]) -> U256 {
        let mut full = [0u8; 32];
        let n = bytes.len().min(32);
        full[32 - n..].copy_from_slice(&bytes[bytes.len() - n..]);
        let mut limbs = [0u32; 8];
        for i in 0..8 {
            // limb i occupies big-endian bytes [28-4i .. 32-4i)
            let off = 28 - 4 * i;
            limbs[i] = u32::from_be_bytes([full[off], full[off + 1], full[off + 2], full[off + 3]]);
        }
        U256(limbs)
    }

    pub fn to_be_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..8 {
            let off = 28 - 4 * i;
            out[off..off + 4].copy_from_slice(&self.0[i].to_be_bytes());
        }
        out
    }

    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&l| l == 0)
    }

    pub fn low_u64(&self) -> u64 {
        (self.0[0] as u64) | ((self.0[1] as u64) << 32)
    }

    /// True if the value exceeds usize range that we allow for mem offsets, etc.
    pub fn fits_usize(&self) -> bool {
        self.0[2..].iter().all(|&l| l == 0) && self.0[1] == 0
    }

    pub fn wrapping_add(&self, o: &U256) -> U256 {
        let mut r = [0u32; 8];
        let mut carry = 0u64;
        for i in 0..8 {
            let s = self.0[i] as u64 + o.0[i] as u64 + carry;
            r[i] = s as u32;
            carry = s >> 32;
        }
        U256(r)
    }

    pub fn wrapping_sub(&self, o: &U256) -> U256 {
        let mut r = [0u32; 8];
        let mut borrow = 0i64;
        for i in 0..8 {
            let d = self.0[i] as i64 - o.0[i] as i64 - borrow;
            if d < 0 {
                r[i] = (d + (1i64 << 32)) as u32;
                borrow = 1;
            } else {
                r[i] = d as u32;
                borrow = 0;
            }
        }
        U256(r)
    }

    /// Full 256-bit multiply, low 256 bits (EVM MUL wraps mod 2^256).
    pub fn wrapping_mul(&self, o: &U256) -> U256 {
        let mut r = [0u64; 8];
        for i in 0..8 {
            let mut carry = 0u64;
            for j in 0..(8 - i) {
                let idx = i + j;
                let cur = r[idx] + (self.0[i] as u64) * (o.0[j] as u64) + carry;
                r[idx] = cur & 0xffff_ffff;
                carry = cur >> 32;
            }
        }
        let mut out = [0u32; 8];
        for i in 0..8 {
            out[i] = r[i] as u32;
        }
        U256(out)
    }

    pub fn bitand(&self, o: &U256) -> U256 {
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = self.0[i] & o.0[i];
        }
        U256(r)
    }
    pub fn bitor(&self, o: &U256) -> U256 {
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = self.0[i] | o.0[i];
        }
        U256(r)
    }
    pub fn bitxor(&self, o: &U256) -> U256 {
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = self.0[i] ^ o.0[i];
        }
        U256(r)
    }
    pub fn bitnot(&self) -> U256 {
        let mut r = [0u32; 8];
        for i in 0..8 {
            r[i] = !self.0[i];
        }
        U256(r)
    }

    /// Unsigned less-than.
    pub fn lt(&self, o: &U256) -> bool {
        for i in (0..8).rev() {
            if self.0[i] != o.0[i] {
                return self.0[i] < o.0[i];
            }
        }
        false
    }

    /// Left shift by `sh` bits (EVM SHL: shift amount is the top-of-stack arg).
    pub fn shl(&self, sh: u32) -> U256 {
        if sh >= 256 {
            return U256::ZERO;
        }
        let word = (sh / 32) as usize;
        let bit = sh % 32;
        let mut r = [0u32; 8];
        for i in (0..8).rev() {
            if i < word {
                continue;
            }
            let src = i - word;
            let mut v = self.0[src] as u64;
            if bit != 0 {
                v <<= bit;
                if src >= 1 {
                    v |= (self.0[src - 1] as u64) >> (32 - bit);
                }
            }
            r[i] = v as u32;
        }
        U256(r)
    }

    /// Logical right shift by `sh` bits (EVM SHR).
    pub fn shr(&self, sh: u32) -> U256 {
        if sh >= 256 {
            return U256::ZERO;
        }
        let word = (sh / 32) as usize;
        let bit = sh % 32;
        let mut r = [0u32; 8];
        for i in 0..8 {
            let src = i + word;
            if src >= 8 {
                continue;
            }
            let mut v = self.0[src] as u64;
            if bit != 0 {
                v >>= bit;
                if src + 1 < 8 {
                    v |= (self.0[src + 1] as u64) << (32 - bit);
                }
            }
            r[i] = v as u32;
        }
        U256(r)
    }

    /// EVM BYTE: byte at big-endian index `i` (0 = most significant), else 0.
    pub fn byte(&self, i: &U256) -> U256 {
        if !i.fits_usize() {
            return U256::ZERO;
        }
        let idx = i.low_u64() as usize;
        if idx >= 32 {
            return U256::ZERO;
        }
        U256::from_u64(self.to_be_bytes()[idx] as u64)
    }
}

// ---------------------------------------------------------------------------
// Opcodes
// ---------------------------------------------------------------------------

pub mod op {
    pub const STOP: u8 = 0x00;
    pub const ADD: u8 = 0x01;
    pub const MUL: u8 = 0x02;
    pub const SUB: u8 = 0x03;
    pub const LT: u8 = 0x10;
    pub const GT: u8 = 0x11;
    pub const EQ: u8 = 0x14;
    pub const ISZERO: u8 = 0x15;
    pub const AND: u8 = 0x16;
    pub const OR: u8 = 0x17;
    pub const XOR: u8 = 0x18;
    pub const NOT: u8 = 0x19;
    pub const BYTE: u8 = 0x1a;
    pub const SHL: u8 = 0x1b;
    pub const SHR: u8 = 0x1c;
    pub const POP: u8 = 0x50;
    pub const MLOAD: u8 = 0x51;
    pub const MSTORE: u8 = 0x52;
    pub const MSTORE8: u8 = 0x53;
    pub const SLOAD: u8 = 0x54;
    pub const SSTORE: u8 = 0x55;
    pub const JUMP: u8 = 0x56;
    pub const JUMPI: u8 = 0x57;
    pub const PC: u8 = 0x58;
    pub const MSIZE: u8 = 0x59;
    pub const JUMPDEST: u8 = 0x5b;
    pub const PUSH0: u8 = 0x5f;
    pub const PUSH1: u8 = 0x60;
    pub const PUSH8: u8 = 0x67;
    pub const PUSH32: u8 = 0x7f;
    pub const DUP1: u8 = 0x80;
    pub const DUP16: u8 = 0x8f;
    pub const SWAP1: u8 = 0x90;
    pub const SWAP16: u8 = 0x9f;
    pub const CALLDATALOAD: u8 = 0x35;
    pub const CALLDATASIZE: u8 = 0x36;
    pub const CALLDATACOPY: u8 = 0x37;
    pub const CODESIZE: u8 = 0x38;
    pub const RETURN: u8 = 0xf3;
    pub const REVERT: u8 = 0xfd;
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Halt {
    Stop,
    Return,
    Revert,
}

#[derive(Clone, Debug)]
pub struct ExecResult {
    /// true iff the call halted via STOP or RETURN (not REVERT and no error).
    pub success: bool,
    pub halt: Result<Halt, String>,
    pub return_data: Vec<u8>,
    /// Final persistent storage (only slots that were written or pre-set).
    pub storage: BTreeMap<U256, U256>,
    pub gas_used: u64,
    /// Number of EVM opcodes actually executed (dispatch iterations).
    pub steps: u64,
}

impl ExecResult {
    pub fn storage_digest(&self) -> [u8; 32] {
        // Deterministic fold of the sorted (key,val) storage pairs. Not keccak;
        // an order-independent digest for cross-checking the storage set.
        let mut acc = [0u8; 32];
        for (k, v) in &self.storage {
            if v.is_zero() {
                continue; // zero slots are absent in the EVM state
            }
            let kb = k.to_be_bytes();
            let vb = v.to_be_bytes();
            for i in 0..32 {
                acc[i] ^= kb[i].rotate_left((i as u32) & 7) ^ vb[i].wrapping_add(0x9b);
            }
        }
        acc
    }
}

pub struct Interpreter<'a> {
    code: &'a [u8],
    calldata: &'a [u8],
    pc: usize,
    stack: Vec<U256>,
    mem: Vec<u8>,
    storage: BTreeMap<U256, U256>,
    gas: u64,
    gas_limit: u64,
    steps: u64,
    jumpdests: Vec<bool>,
    /// RETURN/REVERT output data.
    take_return: Vec<u8>,
}

const STACK_LIMIT: usize = 1024;

impl<'a> Interpreter<'a> {
    fn valid_jumpdests(code: &[u8]) -> Vec<bool> {
        let mut v = vec![false; code.len()];
        let mut i = 0;
        while i < code.len() {
            let o = code[i];
            if o == op::JUMPDEST {
                v[i] = true;
                i += 1;
            } else if (op::PUSH1..=op::PUSH32).contains(&o) {
                i += 1 + (o - op::PUSH1 + 1) as usize;
            } else {
                i += 1;
            }
        }
        v
    }

    fn push(&mut self, v: U256) -> Result<(), String> {
        if self.stack.len() >= STACK_LIMIT {
            return Err("stack overflow".into());
        }
        self.stack.push(v);
        Ok(())
    }
    fn pop(&mut self) -> Result<U256, String> {
        self.stack.pop().ok_or_else(|| "stack underflow".into())
    }

    fn use_gas(&mut self, g: u64) -> Result<(), String> {
        self.gas = self.gas.checked_sub(g).ok_or_else(|| "out of gas".to_string())?;
        Ok(())
    }

    fn mem_expand(&mut self, off: usize, len: usize) -> Result<(), String> {
        if len == 0 {
            return Ok(());
        }
        let end = off.checked_add(len).ok_or("memory offset overflow")?;
        let need = (end + 31) / 32 * 32; // round up to word
        if need > self.mem.len() {
            // memory expansion gas: 3*words + words^2/512 (quadratic), charged on delta
            let old_words = (self.mem.len() / 32) as u64;
            let new_words = (need / 32) as u64;
            let cost = |w: u64| 3 * w + w * w / 512;
            let delta = cost(new_words).saturating_sub(cost(old_words));
            self.use_gas(delta)?;
            self.mem.resize(need, 0);
        }
        Ok(())
    }

    fn mem_store(&mut self, off: usize, bytes: &[u8]) -> Result<(), String> {
        self.mem_expand(off, bytes.len())?;
        self.mem[off..off + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
    fn mem_load32(&mut self, off: usize) -> Result<U256, String> {
        self.mem_expand(off, 32)?;
        Ok(U256::from_be_bytes(&self.mem[off..off + 32]))
    }
}

/// Static-ish gas costs (Berlin-ish base tiers). SLOAD/SSTORE use a simplified
/// warm/cold-agnostic cost; this is honest gas *accounting*, not a bit-exact
/// match of revm's EIP-2929/2200 metering (the cross-check asserts output +
/// storage, not gas).
fn base_gas(opcode: u8) -> u64 {
    use op::*;
    match opcode {
        STOP | RETURN | REVERT => 0,
        JUMPDEST => 1,
        ADD | SUB | LT | GT | EQ | ISZERO | AND | OR | XOR | NOT | BYTE | SHL | SHR | POP
        | PC | MSIZE | PUSH0 | CALLDATASIZE | CODESIZE => 2,
        MUL => 5,
        PUSH1..=PUSH32 | DUP1..=DUP16 | SWAP1..=SWAP16 | CALLDATALOAD => 3,
        MLOAD | MSTORE | MSTORE8 => 3,
        JUMP => 8,
        JUMPI => 10,
        CALLDATACOPY => 3,
        SLOAD => 100,
        SSTORE => 100,
        _ => 0,
    }
}

/// Execute `code` with `calldata` against `pre_storage`, returning the result.
pub fn execute(
    code: &[u8],
    calldata: &[u8],
    pre_storage: &BTreeMap<U256, U256>,
    gas_limit: u64,
) -> ExecResult {
    let mut it = Interpreter {
        code,
        calldata,
        pc: 0,
        stack: Vec::with_capacity(64),
        mem: Vec::new(),
        storage: pre_storage.clone(),
        gas: gas_limit,
        gas_limit,
        steps: 0,
        jumpdests: Interpreter::valid_jumpdests(code),
        take_return: Vec::new(),
    };
    let halt = run(&mut it);
    let success = matches!(halt, Ok(Halt::Stop) | Ok(Halt::Return));
    // Determine return data.
    let return_data = it.take_return.clone();
    ExecResult {
        success,
        halt,
        return_data,
        storage: it.storage,
        gas_used: it.gas_limit - it.gas,
        steps: it.steps,
    }
}

fn run(it: &mut Interpreter) -> Result<Halt, String> {
    use op::*;
    loop {
        if it.pc >= it.code.len() {
            return Ok(Halt::Stop); // implicit STOP at end of code
        }
        let opcode = it.code[it.pc];
        it.steps += 1;
        it.use_gas(base_gas(opcode))?;

        match opcode {
            STOP => return Ok(Halt::Stop),
            ADD => {
                let a = it.pop()?;
                let b = it.pop()?;
                it.push(a.wrapping_add(&b))?;
            }
            MUL => {
                let a = it.pop()?;
                let b = it.pop()?;
                it.push(a.wrapping_mul(&b))?;
            }
            SUB => {
                let a = it.pop()?;
                let b = it.pop()?;
                it.push(a.wrapping_sub(&b))?;
            }
            LT => {
                let a = it.pop()?;
                let b = it.pop()?;
                it.push(if a.lt(&b) { U256::ONE } else { U256::ZERO })?;
            }
            GT => {
                let a = it.pop()?;
                let b = it.pop()?;
                it.push(if b.lt(&a) { U256::ONE } else { U256::ZERO })?;
            }
            EQ => {
                let a = it.pop()?;
                let b = it.pop()?;
                it.push(if a == b { U256::ONE } else { U256::ZERO })?;
            }
            ISZERO => {
                let a = it.pop()?;
                it.push(if a.is_zero() { U256::ONE } else { U256::ZERO })?;
            }
            AND => {
                let a = it.pop()?;
                let b = it.pop()?;
                it.push(a.bitand(&b))?;
            }
            OR => {
                let a = it.pop()?;
                let b = it.pop()?;
                it.push(a.bitor(&b))?;
            }
            XOR => {
                let a = it.pop()?;
                let b = it.pop()?;
                it.push(a.bitxor(&b))?;
            }
            NOT => {
                let a = it.pop()?;
                it.push(a.bitnot())?;
            }
            BYTE => {
                let i = it.pop()?;
                let x = it.pop()?;
                it.push(x.byte(&i))?;
            }
            SHL => {
                let sh = it.pop()?;
                let v = it.pop()?;
                let s = if sh.fits_usize() { sh.low_u64() as u32 } else { 256 };
                it.push(v.shl(s))?;
            }
            SHR => {
                let sh = it.pop()?;
                let v = it.pop()?;
                let s = if sh.fits_usize() { sh.low_u64() as u32 } else { 256 };
                it.push(v.shr(s))?;
            }
            POP => {
                it.pop()?;
            }
            MLOAD => {
                let off = it.pop()?;
                if !off.fits_usize() {
                    return Err("MLOAD offset too large".into());
                }
                let v = it.mem_load32(off.low_u64() as usize)?;
                it.push(v)?;
            }
            MSTORE => {
                let off = it.pop()?;
                let val = it.pop()?;
                if !off.fits_usize() {
                    return Err("MSTORE offset too large".into());
                }
                it.mem_store(off.low_u64() as usize, &val.to_be_bytes())?;
            }
            MSTORE8 => {
                let off = it.pop()?;
                let val = it.pop()?;
                if !off.fits_usize() {
                    return Err("MSTORE8 offset too large".into());
                }
                let b = [val.to_be_bytes()[31]];
                it.mem_store(off.low_u64() as usize, &b)?;
            }
            SLOAD => {
                let k = it.pop()?;
                let v = it.storage.get(&k).copied().unwrap_or(U256::ZERO);
                it.push(v)?;
            }
            SSTORE => {
                let k = it.pop()?;
                let v = it.pop()?;
                it.storage.insert(k, v);
            }
            JUMP => {
                let dst = it.pop()?;
                if !dst.fits_usize() {
                    return Err("JUMP dest too large".into());
                }
                let d = dst.low_u64() as usize;
                if d >= it.jumpdests.len() || !it.jumpdests[d] {
                    return Err(format!("invalid JUMP dest {d}"));
                }
                it.pc = d;
                continue;
            }
            JUMPI => {
                let dst = it.pop()?;
                let cond = it.pop()?;
                if !cond.is_zero() {
                    if !dst.fits_usize() {
                        return Err("JUMPI dest too large".into());
                    }
                    let d = dst.low_u64() as usize;
                    if d >= it.jumpdests.len() || !it.jumpdests[d] {
                        return Err(format!("invalid JUMPI dest {d}"));
                    }
                    it.pc = d;
                    continue;
                }
            }
            PC => {
                let pc = it.pc as u64;
                it.push(U256::from_u64(pc))?;
            }
            MSIZE => {
                it.push(U256::from_u64(it.mem.len() as u64))?;
            }
            JUMPDEST => {}
            PUSH0 => {
                it.push(U256::ZERO)?;
            }
            PUSH1..=PUSH32 => {
                let n = (opcode - PUSH1 + 1) as usize;
                let start = it.pc + 1;
                let end = (start + n).min(it.code.len());
                let bytes = &it.code[start..end];
                // If the code is truncated, EVM pads with zero bytes on the right.
                let mut buf = vec![0u8; n];
                buf[..bytes.len()].copy_from_slice(bytes);
                it.push(U256::from_be_bytes(&buf))?;
                it.pc += 1 + n;
                continue;
            }
            DUP1..=DUP16 => {
                let d = (opcode - DUP1 + 1) as usize;
                if it.stack.len() < d {
                    return Err("DUP underflow".into());
                }
                let v = it.stack[it.stack.len() - d];
                it.push(v)?;
            }
            SWAP1..=SWAP16 => {
                let d = (opcode - SWAP1 + 1) as usize;
                let n = it.stack.len();
                if n < d + 1 {
                    return Err("SWAP underflow".into());
                }
                it.stack.swap(n - 1, n - 1 - d);
            }
            CALLDATALOAD => {
                let off = it.pop()?;
                let mut buf = [0u8; 32];
                if off.fits_usize() {
                    let o = off.low_u64() as usize;
                    for i in 0..32 {
                        if o + i < it.calldata.len() {
                            buf[i] = it.calldata[o + i];
                        }
                    }
                }
                it.push(U256::from_be_bytes(&buf))?;
            }
            CALLDATASIZE => {
                it.push(U256::from_u64(it.calldata.len() as u64))?;
            }
            CALLDATACOPY => {
                let dst = it.pop()?;
                let src = it.pop()?;
                let len = it.pop()?;
                if !dst.fits_usize() || !len.fits_usize() {
                    return Err("CALLDATACOPY args too large".into());
                }
                let (d, l) = (dst.low_u64() as usize, len.low_u64() as usize);
                let s = if src.fits_usize() { src.low_u64() as usize } else { usize::MAX };
                let mut buf = vec![0u8; l];
                for i in 0..l {
                    if s != usize::MAX && s.wrapping_add(i) < it.calldata.len() {
                        buf[i] = it.calldata[s + i];
                    }
                }
                it.mem_store(d, &buf)?;
            }
            CODESIZE => {
                it.push(U256::from_u64(it.code.len() as u64))?;
            }
            RETURN => {
                let off = it.pop()?;
                let len = it.pop()?;
                if !off.fits_usize() || !len.fits_usize() {
                    return Err("RETURN args too large".into());
                }
                let (o, l) = (off.low_u64() as usize, len.low_u64() as usize);
                it.mem_expand(o, l)?;
                it.take_return = it.mem[o..o + l].to_vec();
                return Ok(Halt::Return);
            }
            REVERT => {
                let off = it.pop()?;
                let len = it.pop()?;
                if !off.fits_usize() || !len.fits_usize() {
                    return Err("REVERT args too large".into());
                }
                let (o, l) = (off.low_u64() as usize, len.low_u64() as usize);
                it.mem_expand(o, l)?;
                it.take_return = it.mem[o..o + l].to_vec();
                return Ok(Halt::Revert);
            }
            _ => return Err(format!("unsupported opcode {opcode:#04x} at pc {}", it.pc)),
        }
        it.pc += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u256_roundtrip_and_ops() {
        let a = U256::from_u64(0xdead_beef);
        assert_eq!(U256::from_be_bytes(&a.to_be_bytes()), a);
        let b = U256::from_u64(0x1111);
        assert_eq!(a.wrapping_add(&b), U256::from_u64(0xdead_beef + 0x1111));
        assert_eq!(a.wrapping_sub(&b), U256::from_u64(0xdead_beef - 0x1111));
        assert_eq!(
            U256::from_u64(0xffff_ffff).wrapping_mul(&U256::from_u64(0x2)),
            U256::from_u64(0x1_ffff_fffe)
        );
        assert!(b.lt(&a));
        assert_eq!(U256::from_u64(1).shl(8), U256::from_u64(256));
        assert_eq!(U256::from_u64(256).shr(8), U256::from_u64(1));
    }

    #[test]
    fn trivial_add_return() {
        // PUSH1 3; PUSH1 5; ADD; PUSH1 0; MSTORE; PUSH1 32; PUSH1 0; RETURN
        use op::*;
        let code = vec![
            PUSH1, 3, PUSH1, 5, ADD, PUSH1, 0, MSTORE, PUSH1, 32, PUSH1, 0, RETURN,
        ];
        let r = execute(&code, &[], &BTreeMap::new(), 100000);
        assert!(r.success);
        assert_eq!(U256::from_be_bytes(&r.return_data), U256::from_u64(8));
    }

    #[test]
    fn sstore_sload_roundtrip() {
        use op::*;
        // store 0x2a at slot 1, then load slot 1 and return it
        let code = vec![
            PUSH1, 0x2a, PUSH1, 1, SSTORE, // sstore(slot1, 42)
            PUSH1, 1, SLOAD, PUSH1, 0, MSTORE, // mstore(0, sload(1))
            PUSH1, 32, PUSH1, 0, RETURN,
        ];
        let r = execute(&code, &[], &BTreeMap::new(), 100000);
        assert!(r.success);
        assert_eq!(U256::from_be_bytes(&r.return_data), U256::from_u64(42));
        assert_eq!(r.storage.get(&U256::from_u64(1)), Some(&U256::from_u64(42)));
    }
}
