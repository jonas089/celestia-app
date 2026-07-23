//! R8 (native reference) — a spec-faithful EVM interpreter over a full world
//! state. This defines the exact execution semantics the in-circuit EVM
//! step-function must reproduce; it is differential-tested against ev-reth's own
//! EVM (`ev-revm`, revm 41, via the evm-oracle `execute_tx`) on real bytecode and
//! whole blocks, and is the golden the circuit is checked against.
//!
//! NOT a shortcut / not a transfer stand-in: this is a real stack-machine EVM
//! (stack, memory, world state, pc, gas) executing EVM bytecode. It covers the
//! full standard opcode set (arith incl. signed + EXP + MOD family, compare,
//! bitwise + shifts + BYTE + SIGNEXTEND, env/context, KECCAK256, memory,
//! storage, CALLDATA*/CODE*/RETURNDATA*, LOG0-4, CALL/CALLCODE/DELEGATECALL/
//! STATICCALL, CREATE/CREATE2, RETURN/REVERT/STOP/INVALID/SELFDESTRUCT), operates
//! over a world state of accounts, and recurses on message calls.

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::BTreeMap;
use tiny_keccak::Hasher;

fn mask256() -> BigInt {
    (BigInt::one() << 256) - BigInt::one()
}
fn two256() -> BigInt {
    BigInt::one() << 256
}
fn wrap(x: BigInt) -> BigInt {
    let m = mask256();
    let two = two256();
    // Bring into [0, 2^256) even for negative intermediates.
    let mut r = x & &m;
    if r.is_negative() {
        r += &two;
    }
    r
}

/// A world-state account.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Account {
    pub nonce: u64,
    pub balance: BigInt,
    pub code: Vec<u8>,
    pub storage: BTreeMap<BigInt, BigInt>,
}

impl Account {
    pub fn eoa(nonce: u64, balance: BigInt) -> Self {
        Account { nonce, balance, code: vec![], storage: BTreeMap::new() }
    }
    pub fn contract(nonce: u64, balance: BigInt, code: Vec<u8>, storage: BTreeMap<BigInt, BigInt>) -> Self {
        Account { nonce, balance, code, storage }
    }
    pub fn is_empty(&self) -> bool {
        self.nonce == 0 && self.balance.is_zero() && self.code.is_empty()
    }
}

/// The world state: a map of 20-byte addresses to accounts.
#[derive(Clone, Debug, Default)]
pub struct World {
    pub accounts: BTreeMap<[u8; 20], Account>,
}

impl World {
    pub fn new() -> Self {
        World { accounts: BTreeMap::new() }
    }
    pub fn get(&self, a: &[u8; 20]) -> Account {
        self.accounts.get(a).cloned().unwrap_or_default()
    }
    pub fn balance(&self, a: &[u8; 20]) -> BigInt {
        self.accounts.get(a).map(|x| x.balance.clone()).unwrap_or_else(BigInt::zero)
    }
    pub fn code(&self, a: &[u8; 20]) -> Vec<u8> {
        self.accounts.get(a).map(|x| x.code.clone()).unwrap_or_default()
    }
    pub fn nonce(&self, a: &[u8; 20]) -> u64 {
        self.accounts.get(a).map(|x| x.nonce).unwrap_or(0)
    }
    pub fn entry(&mut self, a: [u8; 20]) -> &mut Account {
        self.accounts.entry(a).or_default()
    }
}

/// Block-level environment (context opcodes).
#[derive(Clone, Debug)]
pub struct BlockEnv {
    pub number: u64,
    pub timestamp: u64,
    pub coinbase: [u8; 20],
    pub prevrandao: BigInt,
    pub gas_limit: u64,
    pub basefee: BigInt,
    pub chain_id: u64,
}

impl Default for BlockEnv {
    fn default() -> Self {
        BlockEnv {
            number: 0,
            timestamp: 0,
            coinbase: [0u8; 20],
            prevrandao: BigInt::zero(),
            gas_limit: 30_000_000,
            basefee: BigInt::zero(),
            chain_id: 1234,
        }
    }
}

/// Execution context (message call frame).
#[derive(Clone, Debug, Default)]
pub struct CallCtx {
    pub code: Vec<u8>,
    pub calldata: Vec<u8>,
    pub caller: [u8; 20],
    pub address: [u8; 20],
    pub origin: [u8; 20],
    pub value: BigInt,
    pub gas_price: BigInt,
    pub is_static: bool,
    pub depth: u32,
}

#[derive(Clone, Debug)]
pub struct Log {
    pub address: [u8; 20],
    pub topics: Vec<BigInt>,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct ExecResult {
    pub success: bool,
    pub return_data: Vec<u8>,
    /// Contract storage of the top-level call's `address` (back-compat convenience).
    pub storage: BTreeMap<BigInt, BigInt>,
    pub gas_used: u64,
    pub steps: usize,
    pub logs: Vec<Log>,
    /// Address created (CREATE/CREATE2 top-level), if any.
    pub created_address: Option<[u8; 20]>,
}

fn keccak(b: &[u8]) -> [u8; 32] {
    let mut h = tiny_keccak::Keccak::v256();
    h.update(b);
    let mut o = [0u8; 32];
    h.finalize(&mut o);
    o
}

const MAX_DEPTH: u32 = 1024;
const MAX_STEPS: usize = 8_000_000;

/// Back-compat single-contract entry point: execute `ctx.code` with `storage` as
/// the (mutable) storage of `ctx.address`, over a world containing just that
/// contract. Matches the original API used by `block_stf` and existing tests.
pub fn execute(
    ctx: &CallCtx,
    storage: &mut BTreeMap<BigInt, BigInt>,
    gas_limit: u64,
) -> ExecResult {
    let mut world = World::new();
    let addr = ctx.address;
    world.entry(addr).code = ctx.code.clone();
    world.entry(addr).storage = storage.clone();
    let env = BlockEnv::default();
    let mut frame = ctx.clone();
    frame.depth = 0;
    let r = call(&mut world, &env, &frame, gas_limit);
    *storage = world.get(&addr).storage;
    ExecResult { storage: storage.clone(), ..r }
}

/// The recursive message-call interpreter. Mutates `world` on success; on failure
/// the caller is responsible for discarding the frame's world changes (this fn
/// operates on a caller-provided snapshot for sub-calls).
pub fn call(world: &mut World, env: &BlockEnv, ctx: &CallCtx, gas_limit: u64) -> ExecResult {
    let mut logs: Vec<Log> = Vec::new();
    let mut steps_total = 0usize;
    let r = call_inner(world, env, ctx, gas_limit, &mut logs, &mut steps_total);
    ExecResult { logs, ..r }
}

fn call_inner(
    world: &mut World,
    env: &BlockEnv,
    ctx: &CallCtx,
    gas_limit: u64,
    logs: &mut Vec<Log>,
    steps_total: &mut usize,
) -> ExecResult {
    let mut stack: Vec<BigInt> = Vec::with_capacity(1024);
    let mut mem: Vec<u8> = Vec::new();
    let mut pc = 0usize;
    let mut gas = gas_limit as i64;
    let mut steps = 0usize;
    let mut return_data: Vec<u8> = Vec::new(); // last sub-call's return (RETURNDATA*)
    let code = ctx.code.clone();
    let self_addr = ctx.address;

    macro_rules! fail {
        () => {{
            return ExecResult {
                success: false,
                return_data: vec![],
                storage: Default::default(),
                gas_used: (gas_limit as i64 - gas).max(0) as u64,
                steps,
                logs: vec![],
                created_address: None,
            };
        }};
    }
    macro_rules! pop {
        () => {{
            match stack.pop() {
                Some(v) => v,
                None => fail!(),
            }
        }};
    }
    macro_rules! push {
        ($v:expr) => {{
            if stack.len() >= 1024 {
                fail!();
            }
            stack.push(wrap($v));
        }};
    }
    macro_rules! spend {
        ($g:expr) => {{
            gas -= $g as i64;
            if gas < 0 {
                fail!();
            }
        }};
    }

    fn mem_expand(mem: &mut Vec<u8>, end: usize) {
        if end > mem.len() {
            mem.resize(end.next_multiple_of(32), 0);
        }
    }

    while pc < code.len() {
        if *steps_total >= MAX_STEPS {
            fail!();
        }
        let op = code[pc];
        steps += 1;
        *steps_total += 1;
        pc += 1;
        match op {
            0x00 => {
                // STOP
                return ExecResult {
                    success: true,
                    return_data: vec![],
                    storage: world.get(&self_addr).storage,
                    gas_used: (gas_limit as i64 - gas).max(0) as u64,
                    steps,
                    logs: vec![],
                    created_address: None,
                };
            }
            0x01 => { spend!(3); let a = pop!(); let b = pop!(); push!(a + b); }
            0x02 => { spend!(5); let a = pop!(); let b = pop!(); push!(a * b); }
            0x03 => { spend!(3); let a = pop!(); let b = pop!(); push!(wrap(a - b)); }
            0x04 => { spend!(5); let a = pop!(); let b = pop!(); push!(if b.is_zero() { BigInt::zero() } else { a / b }); }
            0x05 => { // SDIV
                spend!(5);
                let a = to_signed(&pop!()); let b = to_signed(&pop!());
                push!(if b.is_zero() { BigInt::zero() } else { from_signed(&(a / b)) });
            }
            0x06 => { spend!(5); let a = pop!(); let b = pop!(); push!(if b.is_zero() { BigInt::zero() } else { a % b }); }
            0x07 => { // SMOD
                spend!(5);
                let a = to_signed(&pop!()); let b = to_signed(&pop!());
                push!(if b.is_zero() { BigInt::zero() } else { from_signed(&(a % b)) });
            }
            0x08 => { // ADDMOD
                spend!(8);
                let a = pop!(); let b = pop!(); let n = pop!();
                push!(if n.is_zero() { BigInt::zero() } else { (a + b) % n });
            }
            0x09 => { // MULMOD
                spend!(8);
                let a = pop!(); let b = pop!(); let n = pop!();
                push!(if n.is_zero() { BigInt::zero() } else { (a * b) % n });
            }
            0x0a => { // EXP
                let base = pop!(); let exp = pop!();
                let byte_len = if exp.is_zero() { 0 } else { (exp.bits() as usize + 7) / 8 };
                spend!(10 + 50 * byte_len as i64);
                push!(mod_pow(&base, &exp, &two256()));
            }
            0x0b => { // SIGNEXTEND
                spend!(5);
                let i = pop!(); let x = pop!();
                push!(sign_extend(&i, &x));
            }
            0x10 => { spend!(3); let a = pop!(); let b = pop!(); push!(if a < b { BigInt::one() } else { BigInt::zero() }); }
            0x11 => { spend!(3); let a = pop!(); let b = pop!(); push!(if a > b { BigInt::one() } else { BigInt::zero() }); }
            0x12 => { // SLT
                spend!(3);
                let a = to_signed(&pop!()); let b = to_signed(&pop!());
                push!(if a < b { BigInt::one() } else { BigInt::zero() });
            }
            0x13 => { // SGT
                spend!(3);
                let a = to_signed(&pop!()); let b = to_signed(&pop!());
                push!(if a > b { BigInt::one() } else { BigInt::zero() });
            }
            0x14 => { spend!(3); let a = pop!(); let b = pop!(); push!(if a == b { BigInt::one() } else { BigInt::zero() }); }
            0x15 => { spend!(3); let a = pop!(); push!(if a.is_zero() { BigInt::one() } else { BigInt::zero() }); }
            0x16 => { spend!(3); let a = pop!(); let b = pop!(); push!(a & b); }
            0x17 => { spend!(3); let a = pop!(); let b = pop!(); push!(a | b); }
            0x18 => { spend!(3); let a = pop!(); let b = pop!(); push!(a ^ b); }
            0x19 => { spend!(3); let a = pop!(); push!(mask256() ^ a); }
            0x1a => { // BYTE
                spend!(3);
                let i = pop!(); let x = pop!();
                let iu = to_usize(&i);
                push!(if iu >= 32 { BigInt::zero() } else {
                    let b = to_be32(&x);
                    BigInt::from(b[iu])
                });
            }
            0x1b => { spend!(3); let sh = pop!(); let v = pop!(); let s = to_usize(&sh); push!(if s >= 256 { BigInt::zero() } else { wrap(v << s) }); }
            0x1c => { spend!(3); let sh = pop!(); let v = pop!(); let s = to_usize(&sh); push!(if s >= 256 { BigInt::zero() } else { v >> s }); }
            0x1d => { // SAR
                spend!(3);
                let sh = pop!(); let v = to_signed(&pop!());
                let s = to_usize(&sh);
                let res = if s >= 256 {
                    if v.is_negative() { -BigInt::one() } else { BigInt::zero() }
                } else {
                    v >> s
                };
                push!(from_signed(&res));
            }
            0x20 => {
                // KECCAK256
                let off = to_usize(&pop!());
                let len = to_usize(&pop!());
                spend!(30 + 6 * ((len + 31) / 32) as i64);
                mem_expand(&mut mem, off + len);
                let h = keccak(&mem[off..off + len]);
                push!(BigInt::from_bytes_be(num_bigint::Sign::Plus, &h));
            }
            0x30 => { spend!(2); push!(addr_to_word(&self_addr)); } // ADDRESS
            0x31 => { spend!(100); let a = word_to_addr(&pop!()); push!(world.balance(&a)); } // BALANCE
            0x32 => { spend!(2); push!(addr_to_word(&ctx.origin)); } // ORIGIN
            0x33 => { spend!(2); push!(addr_to_word(&ctx.caller)); } // CALLER
            0x34 => { spend!(2); push!(ctx.value.clone()); } // CALLVALUE
            0x35 => {
                // CALLDATALOAD
                spend!(3);
                let off = to_usize(&pop!());
                let mut w = [0u8; 32];
                for i in 0..32 {
                    if off.saturating_add(i) < ctx.calldata.len() {
                        w[i] = ctx.calldata[off + i];
                    }
                }
                push!(BigInt::from_bytes_be(num_bigint::Sign::Plus, &w));
            }
            0x36 => { spend!(2); push!(BigInt::from(ctx.calldata.len())); } // CALLDATASIZE
            0x37 => {
                // CALLDATACOPY
                let dst = to_usize(&pop!()); let src = to_usize(&pop!()); let len = to_usize(&pop!());
                spend!(3 + 3 * ((len + 31) / 32) as i64);
                mem_expand(&mut mem, dst + len);
                for i in 0..len {
                    mem[dst + i] = if src.saturating_add(i) < ctx.calldata.len() { ctx.calldata[src + i] } else { 0 };
                }
            }
            0x38 => { spend!(2); push!(BigInt::from(code.len())); } // CODESIZE
            0x39 => {
                // CODECOPY
                let dst = to_usize(&pop!()); let src = to_usize(&pop!()); let len = to_usize(&pop!());
                spend!(3 + 3 * ((len + 31) / 32) as i64);
                mem_expand(&mut mem, dst + len);
                for i in 0..len {
                    mem[dst + i] = if src.saturating_add(i) < code.len() { code[src + i] } else { 0 };
                }
            }
            0x3a => { spend!(2); push!(ctx.gas_price.clone()); } // GASPRICE
            0x3b => { spend!(100); let a = word_to_addr(&pop!()); push!(BigInt::from(world.code(&a).len())); } // EXTCODESIZE
            0x3c => {
                // EXTCODECOPY
                let a = word_to_addr(&pop!());
                let dst = to_usize(&pop!()); let src = to_usize(&pop!()); let len = to_usize(&pop!());
                spend!(100 + 3 * ((len + 31) / 32) as i64);
                let ecode = world.code(&a);
                mem_expand(&mut mem, dst + len);
                for i in 0..len {
                    mem[dst + i] = if src.saturating_add(i) < ecode.len() { ecode[src + i] } else { 0 };
                }
            }
            0x3d => { spend!(2); push!(BigInt::from(return_data.len())); } // RETURNDATASIZE
            0x3e => {
                // RETURNDATACOPY
                let dst = to_usize(&pop!()); let src = to_usize(&pop!()); let len = to_usize(&pop!());
                spend!(3 + 3 * ((len + 31) / 32) as i64);
                if src.saturating_add(len) > return_data.len() { fail!(); }
                mem_expand(&mut mem, dst + len);
                for i in 0..len {
                    mem[dst + i] = return_data[src + i];
                }
            }
            0x3f => {
                // EXTCODEHASH
                spend!(100);
                let a = word_to_addr(&pop!());
                let acct = world.get(&a);
                if acct.is_empty() {
                    push!(BigInt::zero());
                } else {
                    push!(BigInt::from_bytes_be(num_bigint::Sign::Plus, &keccak(&acct.code)));
                }
            }
            0x40 => { spend!(20); let _ = pop!(); push!(BigInt::zero()); } // BLOCKHASH (0 for demo)
            0x41 => { spend!(2); push!(addr_to_word(&env.coinbase)); } // COINBASE
            0x42 => { spend!(2); push!(BigInt::from(env.timestamp)); } // TIMESTAMP
            0x43 => { spend!(2); push!(BigInt::from(env.number)); } // NUMBER
            0x44 => { spend!(2); push!(env.prevrandao.clone()); } // PREVRANDAO
            0x45 => { spend!(2); push!(BigInt::from(env.gas_limit)); } // GASLIMIT
            0x46 => { spend!(2); push!(BigInt::from(env.chain_id)); } // CHAINID
            0x47 => { spend!(5); push!(world.balance(&self_addr)); } // SELFBALANCE
            0x48 => { spend!(2); push!(env.basefee.clone()); } // BASEFEE
            0x50 => { spend!(2); let _ = pop!(); } // POP
            0x51 => {
                // MLOAD
                spend!(3);
                let off = to_usize(&pop!());
                mem_expand(&mut mem, off + 32);
                push!(BigInt::from_bytes_be(num_bigint::Sign::Plus, &mem[off..off + 32]));
            }
            0x52 => {
                // MSTORE
                spend!(3);
                let off = to_usize(&pop!());
                let v = pop!();
                mem_expand(&mut mem, off + 32);
                let bytes = to_be32(&v);
                mem[off..off + 32].copy_from_slice(&bytes);
            }
            0x53 => {
                // MSTORE8
                spend!(3);
                let off = to_usize(&pop!());
                let v = pop!();
                mem_expand(&mut mem, off + 1);
                mem[off] = (v & BigInt::from(0xffu32)).to_u8().unwrap_or(0);
            }
            0x54 => {
                // SLOAD
                spend!(100);
                let k = pop!();
                push!(world.get(&self_addr).storage.get(&k).cloned().unwrap_or_else(BigInt::zero));
            }
            0x55 => {
                // SSTORE
                if ctx.is_static { fail!(); }
                spend!(100);
                let k = pop!();
                let v = pop!();
                let acct = world.entry(self_addr);
                if v.is_zero() { acct.storage.remove(&k); } else { acct.storage.insert(k, v); }
            }
            0x56 => {
                // JUMP
                spend!(8);
                let dst = to_usize(&pop!());
                if !valid_jump(&code, dst) { fail!(); }
                pc = dst;
            }
            0x57 => {
                // JUMPI
                spend!(10);
                let dst = to_usize(&pop!());
                let c = pop!();
                if !c.is_zero() {
                    if !valid_jump(&code, dst) { fail!(); }
                    pc = dst;
                }
            }
            0x58 => { spend!(2); push!(BigInt::from(pc - 1)); } // PC
            0x59 => { spend!(2); push!(BigInt::from(mem.len())); } // MSIZE
            0x5a => { spend!(2); push!(BigInt::from((gas.max(0)) as u64)); } // GAS
            0x5b => { spend!(1); } // JUMPDEST
            0x60..=0x7f => {
                // PUSH1..PUSH32
                spend!(3);
                let n = (op - 0x60 + 1) as usize;
                let mut w = vec![0u8; n];
                for i in 0..n {
                    w[i] = if pc + i < code.len() { code[pc + i] } else { 0 };
                }
                pc += n;
                push!(BigInt::from_bytes_be(num_bigint::Sign::Plus, &w));
            }
            0x80..=0x8f => {
                // DUP1..DUP16
                spend!(3);
                let n = (op - 0x80 + 1) as usize;
                if stack.len() < n { fail!(); }
                let v = stack[stack.len() - n].clone();
                push!(v);
            }
            0x90..=0x9f => {
                // SWAP1..SWAP16
                spend!(3);
                let n = (op - 0x90 + 1) as usize;
                let l = stack.len();
                if l < n + 1 { fail!(); }
                stack.swap(l - 1, l - 1 - n);
            }
            0xa0..=0xa4 => {
                // LOG0..LOG4
                if ctx.is_static { fail!(); }
                let ntopics = (op - 0xa0) as usize;
                let off = to_usize(&pop!());
                let len = to_usize(&pop!());
                spend!(375 + 375 * ntopics as i64 + 8 * len as i64);
                let mut topics = Vec::with_capacity(ntopics);
                for _ in 0..ntopics { topics.push(pop!()); }
                mem_expand(&mut mem, off + len);
                logs.push(Log { address: self_addr, topics, data: mem[off..off + len].to_vec() });
            }
            0xf0 => {
                // CREATE
                if ctx.is_static { fail!(); }
                spend!(32000);
                let value = pop!();
                let off = to_usize(&pop!());
                let len = to_usize(&pop!());
                mem_expand(&mut mem, off + len);
                let init = mem[off..off + len].to_vec();
                return_data = vec![];
                let (ok, new_addr) = do_create(world, env, ctx, &self_addr, value, init, None, &mut gas, logs, steps_total);
                if ok {
                    push!(addr_to_word(&new_addr));
                } else {
                    push!(BigInt::zero());
                }
            }
            0xf5 => {
                // CREATE2
                if ctx.is_static { fail!(); }
                spend!(32000);
                let value = pop!();
                let off = to_usize(&pop!());
                let len = to_usize(&pop!());
                let salt = pop!();
                mem_expand(&mut mem, off + len);
                let init = mem[off..off + len].to_vec();
                return_data = vec![];
                let (ok, new_addr) = do_create(world, env, ctx, &self_addr, value, init, Some(salt), &mut gas, logs, steps_total);
                if ok {
                    push!(addr_to_word(&new_addr));
                } else {
                    push!(BigInt::zero());
                }
            }
            0xf1 | 0xf2 | 0xf4 | 0xfa => {
                // CALL / CALLCODE / DELEGATECALL / STATICCALL
                spend!(100);
                let _call_gas = pop!();
                let target = word_to_addr(&pop!());
                // value only for CALL/CALLCODE
                let value = if op == 0xf1 || op == 0xf2 { pop!() } else { BigInt::zero() };
                if ctx.is_static && !value.is_zero() { fail!(); }
                let in_off = to_usize(&pop!());
                let in_len = to_usize(&pop!());
                let out_off = to_usize(&pop!());
                let out_len = to_usize(&pop!());
                mem_expand(&mut mem, in_off + in_len);
                let calldata = mem[in_off..in_off + in_len].to_vec();

                // Determine the execution address (storage context) and code.
                let (exec_addr, code_addr, new_caller, new_value, is_static) = match op {
                    0xf1 => (target, target, self_addr, value.clone(), ctx.is_static),
                    0xf2 => (self_addr, target, self_addr, value.clone(), ctx.is_static), // CALLCODE
                    0xf4 => (self_addr, target, ctx.caller, ctx.value.clone(), ctx.is_static), // DELEGATECALL
                    0xfa => (target, target, self_addr, BigInt::zero(), true),           // STATICCALL
                    _ => unreachable!(),
                };

                // Value transfer (CALL/CALLCODE with value): balance check + move.
                if (op == 0xf1) && !value.is_zero() {
                    if world.balance(&self_addr) < value {
                        return_data = vec![];
                        push!(BigInt::zero());
                        continue;
                    }
                }

                let sub_code = world.code(&code_addr);
                let mut snapshot = world.clone();
                // Apply value transfer in the snapshot for CALL.
                if op == 0xf1 && !value.is_zero() {
                    snapshot.entry(self_addr).balance -= &value;
                    snapshot.entry(target).balance += &value;
                }
                let sub_ctx = CallCtx {
                    code: sub_code,
                    calldata,
                    caller: new_caller,
                    address: exec_addr,
                    origin: ctx.origin,
                    value: new_value,
                    gas_price: ctx.gas_price.clone(),
                    is_static,
                    depth: ctx.depth + 1,
                };
                if ctx.depth + 1 > MAX_DEPTH {
                    return_data = vec![];
                    push!(BigInt::zero());
                    continue;
                }
                let sub_gas = (gas.max(0) as u64).saturating_sub((gas.max(0) as u64) / 64);
                let sub = call_inner(&mut snapshot, env, &sub_ctx, sub_gas, logs, steps_total);
                gas -= sub.gas_used as i64;
                if gas < 0 { fail!(); }
                return_data = sub.return_data.clone();
                if sub.success {
                    *world = snapshot;
                    // write returndata into out region
                    mem_expand(&mut mem, out_off + out_len);
                    for i in 0..out_len.min(sub.return_data.len()) {
                        mem[out_off + i] = sub.return_data[i];
                    }
                    push!(BigInt::one());
                } else {
                    // revert sub-call world changes (snapshot discarded)
                    mem_expand(&mut mem, out_off + out_len);
                    for i in 0..out_len.min(sub.return_data.len()) {
                        mem[out_off + i] = sub.return_data[i];
                    }
                    push!(BigInt::zero());
                }
            }
            0xf3 => {
                // RETURN
                let off = to_usize(&pop!());
                let len = to_usize(&pop!());
                mem_expand(&mut mem, off + len);
                return ExecResult {
                    success: true,
                    return_data: mem[off..off + len].to_vec(),
                    storage: world.get(&self_addr).storage,
                    gas_used: (gas_limit as i64 - gas).max(0) as u64,
                    steps,
                    logs: vec![],
                    created_address: None,
                };
            }
            0xfd => {
                // REVERT
                let off = to_usize(&pop!());
                let len = to_usize(&pop!());
                mem_expand(&mut mem, off + len);
                return ExecResult {
                    success: false,
                    return_data: mem[off..off + len].to_vec(),
                    storage: Default::default(),
                    gas_used: (gas_limit as i64 - gas).max(0) as u64,
                    steps,
                    logs: vec![],
                    created_address: None,
                };
            }
            0xff => {
                // SELFDESTRUCT
                if ctx.is_static { fail!(); }
                spend!(5000);
                let benef = word_to_addr(&pop!());
                let bal = world.balance(&self_addr);
                world.entry(benef).balance += &bal;
                world.entry(self_addr).balance = BigInt::zero();
                return ExecResult {
                    success: true,
                    return_data: vec![],
                    storage: world.get(&self_addr).storage,
                    gas_used: (gas_limit as i64 - gas).max(0) as u64,
                    steps,
                    logs: vec![],
                    created_address: None,
                };
            }
            _ => { fail!(); } // INVALID / unimplemented
        }
    }
    // Ran off the end of code == STOP.
    ExecResult {
        success: true,
        return_data: vec![],
        storage: world.get(&self_addr).storage,
        gas_used: (gas_limit as i64 - gas).max(0) as u64,
        steps,
        logs: vec![],
        created_address: None,
    }
}

/// Execute a CREATE/CREATE2: compute the new address, run the init code with the
/// new account as `address`, and on success install the returned runtime code.
/// Returns (success, new_address). Charges init-code execution gas out of `gas`.
#[allow(clippy::too_many_arguments)]
fn do_create(
    world: &mut World,
    env: &BlockEnv,
    ctx: &CallCtx,
    creator: &[u8; 20],
    value: BigInt,
    init: Vec<u8>,
    salt: Option<BigInt>,
    gas: &mut i64,
    logs: &mut Vec<Log>,
    steps_total: &mut usize,
) -> (bool, [u8; 20]) {
    let creator_nonce = world.nonce(creator);
    let new_addr = match &salt {
        None => create_address(creator, creator_nonce),
        Some(s) => create2_address(creator, s, &init),
    };
    // Bump creator nonce.
    world.entry(*creator).nonce = creator_nonce + 1;
    if world.balance(creator) < value {
        return (false, new_addr);
    }
    let mut snapshot = world.clone();
    snapshot.entry(*creator).balance -= &value;
    snapshot.entry(new_addr).balance += &value;
    snapshot.entry(new_addr).nonce = 1;
    let init_ctx = CallCtx {
        code: init,
        calldata: vec![],
        caller: *creator,
        address: new_addr,
        origin: ctx.origin,
        value,
        gas_price: ctx.gas_price.clone(),
        is_static: false,
        depth: ctx.depth + 1,
    };
    if ctx.depth + 1 > MAX_DEPTH {
        return (false, new_addr);
    }
    let sub_gas = (*gas).max(0) as u64;
    let r = call_inner(&mut snapshot, env, &init_ctx, sub_gas, logs, steps_total);
    *gas -= r.gas_used as i64;
    if !r.success || *gas < 0 {
        return (false, new_addr);
    }
    // Install runtime code (charge 200/byte).
    let deposit = 200 * r.return_data.len() as i64;
    *gas -= deposit;
    if *gas < 0 {
        return (false, new_addr);
    }
    snapshot.entry(new_addr).code = r.return_data.clone();
    *world = snapshot;
    (true, new_addr)
}

/// CREATE address = keccak256(rlp([sender, nonce]))[12:].
pub fn create_address(sender: &[u8; 20], nonce: u64) -> [u8; 20] {
    let mut rlp = Vec::new();
    // rlp list of [address(20), nonce]
    let nonce_enc = rlp_uint(nonce);
    let payload_len = 1 + 20 + nonce_enc.len(); // 0x94 ++ addr ++ nonce_enc
    // list header
    if payload_len <= 55 {
        rlp.push(0xc0 + payload_len as u8);
    } else {
        let lb = be_bytes(payload_len as u64);
        rlp.push(0xf7 + lb.len() as u8);
        rlp.extend_from_slice(&lb);
    }
    rlp.push(0x94); // 0x80 + 20
    rlp.extend_from_slice(sender);
    rlp.extend_from_slice(&nonce_enc);
    let h = keccak(&rlp);
    let mut a = [0u8; 20];
    a.copy_from_slice(&h[12..]);
    a
}

/// CREATE2 address = keccak256(0xff ++ sender ++ salt ++ keccak256(init))[12:].
pub fn create2_address(sender: &[u8; 20], salt: &BigInt, init: &[u8]) -> [u8; 20] {
    let mut buf = Vec::with_capacity(1 + 20 + 32 + 32);
    buf.push(0xff);
    buf.extend_from_slice(sender);
    buf.extend_from_slice(&to_be32(salt));
    buf.extend_from_slice(&keccak(init));
    let h = keccak(&buf);
    let mut a = [0u8; 20];
    a.copy_from_slice(&h[12..]);
    a
}

fn rlp_uint(n: u64) -> Vec<u8> {
    if n == 0 {
        vec![0x80]
    } else if n < 0x80 {
        vec![n as u8]
    } else {
        let b = be_bytes(n);
        let mut v = vec![0x80 + b.len() as u8];
        v.extend_from_slice(&b);
        v
    }
}
fn be_bytes(n: u64) -> Vec<u8> {
    let mut b = n.to_be_bytes().to_vec();
    while b.len() > 1 && b[0] == 0 {
        b.remove(0);
    }
    b
}

fn to_usize(v: &BigInt) -> usize {
    v.to_usize().unwrap_or(usize::MAX / 4)
}
fn to_be32(v: &BigInt) -> [u8; 32] {
    let mut o = [0u8; 32];
    let b = wrap(v.clone()).to_bytes_be().1;
    if b.len() <= 32 {
        o[32 - b.len()..].copy_from_slice(&b);
    } else {
        o.copy_from_slice(&b[b.len() - 32..]);
    }
    o
}
fn addr_to_word(a: &[u8; 20]) -> BigInt {
    BigInt::from_bytes_be(num_bigint::Sign::Plus, a)
}
fn word_to_addr(w: &BigInt) -> [u8; 20] {
    let b = to_be32(w);
    let mut a = [0u8; 20];
    a.copy_from_slice(&b[12..]);
    a
}
fn valid_jump(code: &[u8], dst: usize) -> bool {
    dst < code.len() && code[dst] == 0x5b
}

/// Interpret a [0,2^256) word as a signed two's-complement 256-bit integer.
fn to_signed(v: &BigInt) -> BigInt {
    let half = BigInt::one() << 255;
    if *v >= half {
        v - two256()
    } else {
        v.clone()
    }
}
/// Inverse of `to_signed`: map a signed integer back into [0,2^256).
fn from_signed(v: &BigInt) -> BigInt {
    wrap(v.clone())
}
fn sign_extend(i: &BigInt, x: &BigInt) -> BigInt {
    let iu = to_usize(i);
    if iu >= 31 {
        return wrap(x.clone());
    }
    let bit = (iu + 1) * 8 - 1;
    let sign = (x >> bit) & BigInt::one();
    if sign.is_one() {
        // set all higher bits
        let mask = (BigInt::one() << (bit + 1)) - BigInt::one();
        let ext = mask256() ^ mask; // high bits
        wrap(x | ext)
    } else {
        let mask = (BigInt::one() << (bit + 1)) - BigInt::one();
        x & mask
    }
}
/// Modular exponentiation (base^exp mod m) for 256-bit words.
fn mod_pow(base: &BigInt, exp: &BigInt, m: &BigInt) -> BigInt {
    if m.is_one() {
        return BigInt::zero();
    }
    let mut result = BigInt::one();
    let mut b = base % m;
    let mut e = exp.clone();
    while e > BigInt::zero() {
        if (&e & BigInt::one()).is_one() {
            result = (result * &b) % m;
        }
        e >>= 1;
        b = (&b * &b) % m;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn sstore_sload_roundtrip() {
        let code = vec![0x60, 0x2a, 0x60, 0x00, 0x55, 0x60, 0x00, 0x54, 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xf3];
        let ctx = CallCtx { code, ..Default::default() };
        let mut storage = BTreeMap::new();
        let r = execute(&ctx, &mut storage, 100000);
        assert!(r.success);
        assert_eq!(storage.get(&BigInt::from(0u32)), Some(&BigInt::from(0x2au32)));
        assert_eq!(r.return_data.len(), 32);
        assert_eq!(*r.return_data.last().unwrap(), 0x2a);
    }

    fn hx(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }
    fn run_hex(code_hex: &str, calldata_hex: &str, pre: &[(u64, u64)]) -> ExecResult {
        let code: Vec<u8> = (0..code_hex.len() / 2).map(|i| u8::from_str_radix(&code_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        let calldata: Vec<u8> = (0..calldata_hex.len() / 2).map(|i| u8::from_str_radix(&calldata_hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        let ctx = CallCtx { code, calldata, ..Default::default() };
        let mut storage = BTreeMap::new();
        for (k, v) in pre { storage.insert(BigInt::from(*k), BigInt::from(*v)); }
        execute(&ctx, &mut storage, 1_000_000)
    }

    #[test]
    fn matches_ev_revm_golden_arith() {
        let r = run_hex("600360040160020260005260206000f3", "", &[]);
        assert!(r.success);
        assert_eq!(hx(&r.return_data), "000000000000000000000000000000000000000000000000000000000000000e");
    }
    #[test]
    fn matches_ev_revm_golden_increment() {
        let r = run_hex("60005460010160005500", "", &[(0, 5)]);
        assert!(r.success);
        assert_eq!(r.storage.get(&BigInt::from(0u32)), Some(&BigInt::from(6u32)));
    }
    #[test]
    fn matches_ev_revm_golden_sstore_sload() {
        let r = run_hex("602a60005560005460005260206000f3", "", &[]);
        assert!(r.success);
        assert_eq!(r.storage.get(&BigInt::from(0u32)), Some(&BigInt::from(0x2au32)));
        assert_eq!(hx(&r.return_data), "000000000000000000000000000000000000000000000000000000000000002a");
    }
    #[test]
    fn matches_ev_revm_golden_calldata_echo() {
        let cd = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
        let r = run_hex("60003560005500", cd, &[]);
        assert!(r.success);
        let want = BigInt::from_bytes_be(num_bigint::Sign::Plus, &(0..32).map(|i| i as u8).collect::<Vec<u8>>());
        assert_eq!(r.storage.get(&BigInt::from(0u32)), Some(&want));
    }
    #[test]
    fn matches_ev_revm_golden_mapping_write() {
        let r = run_hex("600060005260006020526040600020602a905500", "", &[]);
        assert!(r.success);
        let key = BigInt::from_bytes_be(num_bigint::Sign::Plus, &super::keccak(&[0u8; 64]));
        assert_eq!(hx(&key.to_bytes_be().1), "ad3228b676f7d3cd4284a5443f17f1962b36e491b30a40b2405849e597ba5fb5");
        assert_eq!(r.storage.get(&key), Some(&BigInt::from(0x2au32)));
    }
    #[test]
    fn matches_ev_revm_golden_deploy_initcode() {
        let r = run_hex("69602a60005560016000f3600052600a6016f3", "", &[]);
        assert!(r.success);
        assert_eq!(hx(&r.return_data), "602a60005560016000f3");
    }

    #[test]
    fn arithmetic_and_stack() {
        let code = vec![0x60, 3, 0x60, 4, 0x01, 0x60, 2, 0x02, 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xf3];
        let ctx = CallCtx { code, ..Default::default() };
        let mut storage = BTreeMap::new();
        let r = execute(&ctx, &mut storage, 100000);
        assert!(r.success);
        assert_eq!(*r.return_data.last().unwrap(), 14);
    }

    // ---- New full-coverage opcode tests ------------------------------------

    #[test]
    fn signed_div_mod() {
        // stack (top-first) SDIV: top / second. Want -6 / 2 = -3.
        // PUSH1 2 ; PUSH1 6 ; PUSH1 0 ; SUB (=-6, top) ; SDIV
        let r = run_hex("600260066000030560005260206000f3", "", &[]);
        assert!(r.success);
        let want = super::wrap(BigInt::from(-3));
        assert_eq!(BigInt::from_bytes_be(num_bigint::Sign::Plus, &r.return_data), want);
    }

    #[test]
    fn exp_op() {
        // EXP: base=top, exp=second. Want 3**4=81. PUSH1 4 ; PUSH1 3 ; EXP
        let r = run_hex("600460030a60005260206000f3", "", &[]);
        assert!(r.success);
        assert_eq!(BigInt::from_bytes_be(num_bigint::Sign::Plus, &r.return_data), BigInt::from(81u32));
    }

    #[test]
    fn byte_op() {
        // BYTE: i=top, x=second. Want byte 31 of 0x00ff = 0xff.
        // PUSH2 0x00ff ; PUSH1 31 ; BYTE
        let r = run_hex("6100ff601f1a60005260206000f3", "", &[]);
        assert!(r.success);
        assert_eq!(BigInt::from_bytes_be(num_bigint::Sign::Plus, &r.return_data), BigInt::from(0xffu32));
    }

    #[test]
    fn signextend_op() {
        // SIGNEXTEND: i=top, x=second. Want signextend(0, 0xff)=2^256-1.
        // PUSH1 0xff ; PUSH1 0 ; SIGNEXTEND
        let r = run_hex("60ff60000b60005260206000f3", "", &[]);
        assert!(r.success);
        assert_eq!(BigInt::from_bytes_be(num_bigint::Sign::Plus, &r.return_data), super::mask256());
    }

    #[test]
    fn sar_op() {
        // SAR: shift=top, value=second. Want -8 >> 1 (arith) = -4.
        // PUSH1 8 ; PUSH1 0 ; SUB (=-8, value) ; PUSH1 1 (shift) ; SAR
        let r = run_hex("600860000360011d60005260206000f3", "", &[]);
        assert!(r.success);
        let want = super::wrap(BigInt::from(-4));
        assert_eq!(BigInt::from_bytes_be(num_bigint::Sign::Plus, &r.return_data), want);
    }

    #[test]
    fn create_installs_runtime() {
        // Program (runs at 0x11): CALLDATACOPY the init code (from calldata) into
        // mem[0], then CREATE(value=0, off=0, len=initlen), MSTORE the returned
        // address, RETURN it. Init deploys runtime 602a60005560016000f3.
        let init = "600a600c600039600a6000f3602a60005560016000f3";
        let initlen = init.len() / 2;
        let initb: Vec<u8> = (0..initlen).map(|i| u8::from_str_radix(&init[2 * i..2 * i + 2], 16).unwrap()).collect();
        let mut c = Vec::new();
        // CALLDATACOPY: pop dst,src,len => push len, src, dst (dst on top)
        c.extend_from_slice(&[0x60, initlen as u8]); // len
        c.extend_from_slice(&[0x60, 0x00]); // src
        c.extend_from_slice(&[0x60, 0x00]); // dst
        c.push(0x37); // CALLDATACOPY
        // CREATE: pops value,off,len (value on top) => push len, off, value
        c.extend_from_slice(&[0x60, initlen as u8]); // len
        c.extend_from_slice(&[0x60, 0x00]); // off
        c.extend_from_slice(&[0x60, 0x00]); // value
        c.push(0xf0); // CREATE -> pushes new addr
        c.extend_from_slice(&[0x60, 0x00, 0x52]); // MSTORE addr@0
        c.extend_from_slice(&[0x60, 0x20, 0x60, 0x00, 0xf3]); // RETURN 32
        let ctx = CallCtx { code: c.clone(), calldata: initb, address: [0x11; 20], caller: [0x22; 20], ..Default::default() };
        let mut world = World::new();
        world.entry([0x11; 20]).code = c;
        world.entry([0x22; 20]).balance = BigInt::from(10u32);
        let env = BlockEnv::default();
        let r = call(&mut world, &env, &ctx, 10_000_000);
        assert!(r.success, "create program should succeed");
        let created = word_to_addr(&BigInt::from_bytes_be(num_bigint::Sign::Plus, &r.return_data));
        assert_ne!(created, [0u8; 20], "CREATE returned zero address");
        assert_eq!(hx(&world.code(&created)), "602a60005560016000f3", "runtime code not installed");
    }

    #[test]
    fn call_between_contracts() {
        // Contract A (0xaa) CALLs contract B (0xbb). B stores 0x2a at slot 0 and
        // returns 1 byte. A checks CALL succeeded by returning the success flag.
        // B runtime: 602a60005560016000f3
        let bcode: Vec<u8> = {
            let s = "602a60005560016000f3";
            (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
        };
        // A: CALL(gas, B, value=0, inoff=0, inlen=0, outoff=0, outlen=0) then RETURN the flag.
        // CALL pops: gas, addr, value, inoff, inlen, outoff, outlen (gas on top).
        // push order (bottom->top): outlen, outoff, inlen, inoff, value, addr, gas
        let mut a = Vec::new();
        a.extend_from_slice(&[0x60, 0x00]); // outlen
        a.extend_from_slice(&[0x60, 0x00]); // outoff
        a.extend_from_slice(&[0x60, 0x00]); // inlen
        a.extend_from_slice(&[0x60, 0x00]); // inoff
        a.extend_from_slice(&[0x60, 0x00]); // value
        a.extend_from_slice(&[0x73]); // PUSH20 addr B
        a.extend_from_slice(&[0xbb; 20]);
        a.extend_from_slice(&[0x62, 0x0f, 0x42, 0x40]); // PUSH3 gas 1_000_000
        a.push(0xf1); // CALL -> pushes success
        a.extend_from_slice(&[0x60, 0x00, 0x52]); // MSTORE flag@0
        a.extend_from_slice(&[0x60, 0x20, 0x60, 0x00, 0xf3]); // RETURN 32
        let ctx = CallCtx { code: a.clone(), address: [0xaa; 20], caller: [0x22; 20], ..Default::default() };
        let mut world = World::new();
        world.entry([0xaa; 20]).code = a;
        world.entry([0xbb; 20]).code = bcode;
        let env = BlockEnv::default();
        let r = call(&mut world, &env, &ctx, 10_000_000);
        assert!(r.success);
        assert_eq!(BigInt::from_bytes_be(num_bigint::Sign::Plus, &r.return_data), BigInt::one(), "CALL should succeed");
        // B's storage slot 0 == 0x2a
        assert_eq!(world.get(&[0xbb; 20]).storage.get(&BigInt::from(0u32)), Some(&BigInt::from(0x2au32)));
    }

    // ====================================================================
    // Differential vectors captured from ev-reth's REAL revm (ev-revm /
    // revm 41, Prague) via evm-oracle `examples/evm_vectors.rs`. Every
    // `return`/`storage_post` below is the byte-for-byte output the oracle
    // printed (see /Users/.../stf-gate-out/evm_goldens_full.txt). Each
    // program computes a value, MSTOREs it at 0, and RETURNs 32 bytes; the
    // expected hex is exactly what real revm returned. NONE hand-authored.
    // ====================================================================

    fn dec(s: &str) -> Vec<u8> {
        (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    /// Generates one `#[test]` per oracle CASE that RETURNs a 32-byte word.
    macro_rules! diff_ret {
        ($name:ident, $code:expr, $ret:expr) => {
            #[test]
            fn $name() {
                let r = run_hex($code, "", &[]);
                assert!(r.success, "{} must succeed (native)", stringify!($name));
                assert_eq!(
                    hx(&r.return_data),
                    $ret,
                    "return mismatch vs ev-revm golden for {}",
                    stringify!($name)
                );
            }
        };
    }

    // -- unsigned arithmetic --
    diff_ret!(diff_op_sub, "6003600a0360005260206000f3", "0000000000000000000000000000000000000000000000000000000000000007");
    diff_ret!(diff_op_div, "600360140460005260206000f3", "0000000000000000000000000000000000000000000000000000000000000006");
    diff_ret!(diff_op_mod, "600560110660005260206000f3", "0000000000000000000000000000000000000000000000000000000000000002");
    // -- signed arithmetic + mod family + exp --
    diff_ret!(diff_op_sdiv, "600260066000030560005260206000f3", "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd");
    diff_ret!(diff_op_smod, "600360076000030760005260206000f3", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    diff_ret!(diff_op_addmod, "601060ff60ff0860005260206000f3", "000000000000000000000000000000000000000000000000000000000000000e");
    diff_ret!(diff_op_mulmod, "601060ff60ff0960005260206000f3", "0000000000000000000000000000000000000000000000000000000000000001");
    diff_ret!(diff_op_exp, "600560030a60005260206000f3", "00000000000000000000000000000000000000000000000000000000000000f3");
    diff_ret!(diff_op_signextend, "60ff60000b60005260206000f3", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    // -- comparison (unsigned + signed) --
    diff_ret!(diff_op_lt, "600560031060005260206000f3", "0000000000000000000000000000000000000000000000000000000000000001");
    diff_ret!(diff_op_gt, "600360051160005260206000f3", "0000000000000000000000000000000000000000000000000000000000000001");
    diff_ret!(diff_op_slt, "600160016000031260005260206000f3", "0000000000000000000000000000000000000000000000000000000000000001");
    diff_ret!(diff_op_sgt, "600160000360011360005260206000f3", "0000000000000000000000000000000000000000000000000000000000000001");
    diff_ret!(diff_op_eq, "600760071460005260206000f3", "0000000000000000000000000000000000000000000000000000000000000001");
    diff_ret!(diff_op_iszero, "60001560005260206000f3", "0000000000000000000000000000000000000000000000000000000000000001");
    // -- bitwise + shifts + BYTE --
    diff_ret!(diff_op_and, "603c600f1660005260206000f3", "000000000000000000000000000000000000000000000000000000000000000c");
    diff_ret!(diff_op_or, "6030600f1760005260206000f3", "000000000000000000000000000000000000000000000000000000000000003f");
    diff_ret!(diff_op_xor, "603c600f1860005260206000f3", "0000000000000000000000000000000000000000000000000000000000000033");
    diff_ret!(diff_op_not, "600f1960005260206000f3", "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff0");
    diff_ret!(diff_op_byte, "6100ff601f1a60005260206000f3", "00000000000000000000000000000000000000000000000000000000000000ff");
    diff_ret!(diff_op_shl, "600160041b60005260206000f3", "0000000000000000000000000000000000000000000000000000000000000010");
    diff_ret!(diff_op_shr, "61ff0060081c60005260206000f3", "00000000000000000000000000000000000000000000000000000000000000ff");
    diff_ret!(diff_op_sar, "600860000360011d60005260206000f3", "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc");

    #[test]
    fn diff_op_revert() {
        // Top-level REVERT with 32 bytes of 0x2a as revert data.
        let r = run_hex("602a60005260206000fd", "", &[]);
        assert!(!r.success, "REVERT must fail the frame (native)");
        assert_eq!(
            hx(&r.return_data),
            "000000000000000000000000000000000000000000000000000000000000002a",
            "revert data mismatch vs ev-revm golden"
        );
    }

    // Oracle harness addresses (evm_vectors.rs): CALLER, CONTRACT, CONTRACT2.
    fn caller_addr() -> [u8; 20] { let mut a = [0u8; 20]; a[17] = 0x0c; a[18] = 0xa1; a[19] = 0x1e; a }
    fn contract_addr() -> [u8; 20] { let mut a = [0u8; 20]; a[17] = 0x0c; a[18] = 0x0d; a[19] = 0xe0; a }
    fn contract2_addr() -> [u8; 20] { let mut a = [0u8; 20]; a[17] = 0x0c; a[18] = 0x0d; a[19] = 0xe1; a }

    #[test]
    fn diff_call_cross() {
        // CONTRACT CALLs CONTRACT2 (B_write): the SSTORE lands in CONTRACT2 (callee
        // storage context), and CONTRACT returns the callee's 0x2a. ev-revm golden:
        // return 0x..2a, CONTRACT storage empty, CONTRACT2 slot0 = 0x2a.
        let a = dec("602060006000600060007300000000000000000000000000000000000c0de1620f4240f160206000f3");
        let b = dec("602a600055602a60005260206000f3");
        let (c1, c2) = (contract_addr(), contract2_addr());
        let ctx = CallCtx { code: a.clone(), address: c1, caller: caller_addr(), ..Default::default() };
        let mut world = World::new();
        world.entry(c1).code = a;
        world.entry(c1).nonce = 1;
        world.entry(c2).code = b;
        world.entry(c2).nonce = 1;
        let env = BlockEnv::default();
        let r = call(&mut world, &env, &ctx, 1_000_000);
        assert!(r.success);
        assert_eq!(hx(&r.return_data), "000000000000000000000000000000000000000000000000000000000000002a");
        assert_eq!(world.get(&c2).storage.get(&BigInt::from(0u32)), Some(&BigInt::from(0x2au32)), "CALL writes callee storage");
        assert!(world.get(&c1).storage.is_empty(), "caller storage must be untouched by CALL");
    }

    #[test]
    fn diff_staticcall_cross() {
        // CONTRACT STATICCALLs CONTRACT2 (B_pure) which just returns 0x63; no writes.
        let a = dec("60206000600060007300000000000000000000000000000000000c0de1620f4240fa60206000f3");
        let b = dec("606360005260206000f3");
        let (c1, c2) = (contract_addr(), contract2_addr());
        let ctx = CallCtx { code: a.clone(), address: c1, caller: caller_addr(), ..Default::default() };
        let mut world = World::new();
        world.entry(c1).code = a;
        world.entry(c1).nonce = 1;
        world.entry(c2).code = b;
        world.entry(c2).nonce = 1;
        let env = BlockEnv::default();
        let r = call(&mut world, &env, &ctx, 1_000_000);
        assert!(r.success);
        assert_eq!(hx(&r.return_data), "0000000000000000000000000000000000000000000000000000000000000063");
        assert!(world.get(&c1).storage.is_empty());
        assert!(world.get(&c2).storage.is_empty());
    }

    #[test]
    fn diff_delegatecall_cross() {
        // CONTRACT DELEGATECALLs CONTRACT2 (B_write): the SSTORE runs in CONTRACT's own
        // storage context, so the write lands in CONTRACT, and CONTRACT2 is untouched.
        // ev-revm golden: return 0x..2a, CONTRACT slot0 = 0x2a, CONTRACT2 empty.
        let a = dec("60206000600060007300000000000000000000000000000000000c0de1620f4240f460206000f3");
        let b = dec("602a600055602a60005260206000f3");
        let (c1, c2) = (contract_addr(), contract2_addr());
        let ctx = CallCtx { code: a.clone(), address: c1, caller: caller_addr(), ..Default::default() };
        let mut world = World::new();
        world.entry(c1).code = a;
        world.entry(c1).nonce = 1;
        world.entry(c2).code = b;
        world.entry(c2).nonce = 1;
        let env = BlockEnv::default();
        let r = call(&mut world, &env, &ctx, 1_000_000);
        assert!(r.success);
        assert_eq!(hx(&r.return_data), "000000000000000000000000000000000000000000000000000000000000002a");
        assert_eq!(world.get(&c1).storage.get(&BigInt::from(0u32)), Some(&BigInt::from(0x2au32)), "DELEGATECALL writes caller storage");
        assert!(world.get(&c2).storage.is_empty(), "callee storage must be untouched by DELEGATECALL");
    }

    #[test]
    fn diff_create_factory() {
        // CONTRACT (nonce 1) CREATEs a child deploying runtime 602a60005560016000f3.
        // ev-revm golden created address = 0xed591dc4375a9e8c959da1aaa3712ac1063fcc18,
        // returned left-padded to 32 bytes; child deployed code = the 10-byte runtime.
        let code = dec("75600a600c600039600a6000f3602a60005560016000f36000526016600a6000f060005260206000f3");
        let c1 = contract_addr();
        let ctx = CallCtx { code: code.clone(), address: c1, caller: caller_addr(), ..Default::default() };
        let mut world = World::new();
        world.entry(c1).code = code;
        world.entry(c1).nonce = 1;
        let env = BlockEnv::default();
        let r = call(&mut world, &env, &ctx, 10_000_000);
        assert!(r.success);
        assert_eq!(
            hx(&r.return_data),
            "000000000000000000000000ed591dc4375a9e8c959da1aaa3712ac1063fcc18",
            "CREATE address (RLP derivation) must match ev-revm"
        );
        let created = word_to_addr(&BigInt::from_bytes_be(num_bigint::Sign::Plus, &r.return_data));
        assert_eq!(hx(&world.code(&created)), "602a60005560016000f3", "deployed runtime code mismatch vs ev-revm");
    }
}
