//! A genuine EVM interpreter expressed as an RV32I program (base subset only, so
//! the GKR CPU circuit needs zero new instructions). Real opcode dispatch over
//! EVM bytecode; 256-bit words as 8x u32 little-endian limbs held on an in-RAM
//! stack; byte-addressable EVM memory and a linear-scan storage map; gas is
//! tracked by the native reference (`evm-core`) — this RV32 program proves the
//! stack/memory/storage state transition, cross-checked byte-for-byte.
//!
//! Memory layout (byte addresses; code/calldata/EVM-memory are word-per-byte so
//! only word LW/SW are used):
//!   x20 CODE_BASE  0x0001_0000   code[i]      @ CODE  + i*4
//!   x21 CALL_BASE  0x0002_0000   calldata[i]  @ CALL  + i*4
//!   x22 MEM_BASE   0x0003_0000   evm_mem[i]   @ MEM   + i*4
//!   x23 STACK_BASE 0x0004_0000   slot d limb i@ STACK + d*32 + i*4
//!   x24 STOR_BASE  0x0005_0000   entry e: key[8] @ +e*64, val[8] @ +e*64+32
//!   x25 RES_BASE   0x0006_0000   return bytes  @ RES + i*4
//! Scalars: x1 pc, x5 sp(depth), x26 storage_count, x27 calldata_size,
//!   x28 code_size, x29 return_len, x30 halt_code (1 STOP,2 RETURN,3 REVERT,4 ERR).

use crate::emulator::Cpu;
use crate::evm_asm::Asm;
use crate::evm_core::U256;
use std::collections::BTreeMap;

pub const CODE_BASE: u32 = 0x0001_0000;
pub const CALL_BASE: u32 = 0x0002_0000;
pub const MEM_BASE: u32 = 0x0003_0000;
pub const STACK_BASE: u32 = 0x0004_0000;
pub const STOR_BASE: u32 = 0x0005_0000;
pub const RES_BASE: u32 = 0x0006_0000;
pub const RESULT_ADDR: u32 = 0x0000_7000; // combined result-fold word cell
pub const PROG_BASE: u32 = 0; // instruction ROM base

// register aliases
const PC: u32 = 1;
const SP: u32 = 5;
const RCODE: u32 = 20;
const RCALL: u32 = 21;
const RMEM: u32 = 22;
const RSTACK: u32 = 23;
const RSTOR: u32 = 24;
const RRES: u32 = 25;
const RSTORN: u32 = 26;
const RCALLN: u32 = 27;
const RCODEN: u32 = 28;
const RRETLEN: u32 = 29;
const RHALT: u32 = 30;

// temps (must avoid the reserved ones above; x2/x3 reserved for public output)
const T0: u32 = 6;
const T1: u32 = 7;
const T2: u32 = 8;
const T3: u32 = 9;
const T4: u32 = 10;
const T5: u32 = 11;
const T6: u32 = 12;
const T7: u32 = 13;
const T8: u32 = 14;
const T9: u32 = 15;
const TA: u32 = 16;
const TB: u32 = 17;
const TC: u32 = 18;
const TD: u32 = 19;
const TE: u32 = 31;
const TF: u32 = 4;

// ------------------------------ emit helpers -------------------------------

/// dst = base + idx*4  (word address).
fn word_addr(a: &mut Asm, dst: u32, base: u32, idx: u32, tmp: u32) {
    a.slli(tmp, idx, 2);
    a.add(dst, base, tmp);
}

/// dst = STACK_BASE + depth*32  (base address of stack slot `depth`).
fn slot_addr(a: &mut Asm, dst: u32, depth: u32, tmp: u32) {
    a.slli(tmp, depth, 5);
    a.add(dst, RSTACK, tmp);
}

/// carry-out of s=x+y using three scratch regs (o result, plus t1,t2).
fn carry_out3(a: &mut Asm, o: u32, x: u32, y: u32, s: u32, t1: u32, t2: u32) {
    a.and(o, x, y); // o = x & y
    a.or(t1, x, y); // t1 = x | y
    a.xori(t2, s, -1); // t2 = ~s
    a.and(t1, t1, t2); // t1 = (x|y) & ~s
    a.or(o, o, t1); // o = (x&y) | ((x|y)&~s)
    a.srli(o, o, 31); // top bit
}

/// full-adder for one 32-bit limb: out = x + y + cin (mod 2^32); cout in `cout`.
/// scratch: t1,t2,t3. (out may alias x or y.)
#[allow(clippy::too_many_arguments)]
fn adc32(a: &mut Asm, out: u32, cout: u32, x: u32, y: u32, cin: u32, t1: u32, t2: u32, t3: u32) {
    // s1 = x + y ; c1 = carry(x,y,s1)
    a.add(t1, x, y); // s1
    carry_out3(a, cout, x, y, t1, t2, t3); // c1 -> cout
    // s = s1 + cin ; c2 = carry(s1,cin,s)
    a.add(out, t1, cin); // s
    carry_out3(a, t3, t1, cin, out, t2, t1); // c2 -> t3 (t1 free now)
    a.or(cout, cout, t3); // cout = c1|c2
}

/// full-subtractor for one limb: out = x - y - bin (mod 2^32); borrow-out in
/// `bout`. Implemented via adc32(x, ~y, 1-bin): borrow = 1 - carry.
/// scratch: t1..t4.
#[allow(clippy::too_many_arguments)]
fn sbb32(a: &mut Asm, out: u32, bout: u32, x: u32, y: u32, bin: u32, t1: u32, t2: u32, t3: u32, t4: u32) {
    a.xori(t4, y, -1); // ~y
    a.xori(bin, bin, 1); // notbin = 1 - bin   (NOTE: clobbers caller's bin!)
    adc32(a, out, bout, x, t4, bin, t1, t2, t3); // out = x+~y+notbin ; carry
    a.xori(bin, bin, 1); // restore bin to original
    a.xori(bout, bout, 1); // borrow = 1 - carry
}

// ------------------------------ interpreter --------------------------------

/// Assemble the interpreter program (label-resolved RV32I words at PROG_BASE).
pub fn assemble_interpreter() -> Vec<u32> {
    let mut a = Asm::new();

    // ---- startup: set base registers, pc=0, sp=0 ----
    a.li(RCODE, T0, CODE_BASE);
    a.li(RCALL, T0, CALL_BASE);
    a.li(RMEM, T0, MEM_BASE);
    a.li(RSTACK, T0, STACK_BASE);
    a.li(RSTOR, T0, STOR_BASE);
    a.li(RRES, T0, RES_BASE);
    a.addi(PC, 0, 0);
    a.addi(SP, 0, 0);
    a.addi(RHALT, 0, 0);
    a.addi(RRETLEN, 0, 0);
    // x26 storage_count, x27 calldata_size, x28 code_size are seeded from memory
    // header words the driver writes at fixed absolute addresses 0x8000/0x8004/0x8008.
    a.li(T0, T1, 0x8000);
    a.lw(RSTORN, T0, 0);
    a.lw(RCALLN, T0, 4);
    a.lw(RCODEN, T0, 8);

    // ================= dispatch =================
    a.label("dispatch");
    // if pc >= code_size -> implicit STOP. pc and code_size are small (< 2^31),
    // so pc < codeN  <=>  (pc - codeN) has its sign bit set. cheap: sub + srli.
    a.beq(PC, RCODEN, "op_stop"); // pc == codeN -> stop
    a.sub(T4, PC, RCODEN); // pc - codeN
    a.srli(T5, T4, 31); // sign bit: 1 iff pc < codeN
    a.beq(T5, 0, "op_stop"); // pc > codeN -> stop
    // op = code[pc]
    word_addr(&mut a, T0, RCODE, PC, T1);
    a.lw(T0, T0, 0); // T0 = opcode (0..255)

    // ---- range dispatch: PUSH (0x60..0x7f): (op & 0xE0)==0x60 ----
    a.andi(T1, T0, 0xE0);
    a.li(T2, T3, 0x60);
    a.beq(T1, T2, "op_push");
    // DUP (0x80..0x8f): (op & 0xF0)==0x80
    a.andi(T1, T0, 0xF0);
    a.li(T2, T3, 0x80);
    a.beq(T1, T2, "op_dup");
    // SWAP (0x90..0x9f): (op & 0xF0)==0x90
    a.li(T2, T3, 0x90);
    a.beq(T1, T2, "op_swap");

    // ---- exact-match opcodes ----
    for (opc, lbl) in [
        (0x00, "op_stop"),
        (0x01, "op_add"),
        (0x03, "op_sub"),
        (0x10, "op_lt"),
        (0x11, "op_gt"),
        (0x14, "op_eq"),
        (0x15, "op_iszero"),
        (0x16, "op_and"),
        (0x17, "op_or"),
        (0x18, "op_xor"),
        (0x19, "op_not"),
        (0x50, "op_pop"),
        (0x51, "op_mload"),
        (0x52, "op_mstore"),
        (0x54, "op_sload"),
        (0x55, "op_sstore"),
        (0x56, "op_jump"),
        (0x57, "op_jumpi"),
        (0x5b, "op_jumpdest"),
        (0x35, "op_calldataload"),
        (0xf3, "op_return"),
        (0xfd, "op_revert"),
    ] {
        a.li(T1, T2, opc);
        a.beq(T0, T1, lbl);
    }
    // unknown -> error halt
    a.addi(RHALT, 0, 4);
    a.j("halt");

    // ---------------- handlers ----------------

    // STOP
    a.label("op_stop");
    a.addi(RHALT, 0, 1);
    a.j("halt");

    // JUMPDEST: pc += 1
    a.label("op_jumpdest");
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // POP: sp -= 1
    a.label("op_pop");
    a.addi(SP, SP, -1);
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // --- binary bitwise AND/OR/XOR: r[sp-2] = a op b ; sp-=1 ---
    // a = slot sp-1, b = slot sp-2, write b slot.
    for (opc, name) in [("and", "op_and"), ("or", "op_or"), ("xor", "op_xor")] {
        a.label(name);
        a.addi(T8, SP, -1);
        slot_addr(&mut a, TA, T8, T0); // aslot
        a.addi(T9, SP, -2);
        slot_addr(&mut a, TB, T9, T0); // bslot
        for i in 0..8 {
            a.lw(T0, TA, i * 4);
            a.lw(T1, TB, i * 4);
            match opc {
                "and" => a.and(T2, T0, T1),
                "or" => a.or(T2, T0, T1),
                _ => a.xor(T2, T0, T1),
            }
            a.sw(T2, TB, i * 4);
        }
        a.addi(SP, SP, -1);
        a.addi(PC, PC, 1);
        a.j("dispatch");
    }

    // NOT: r[sp-1] = ~a  (in place)
    a.label("op_not");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0);
    for i in 0..8 {
        a.lw(T0, TA, i * 4);
        a.xori(T0, T0, -1);
        a.sw(T0, TA, i * 4);
    }
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // ADD: r = a + b (256-bit) ; result to b slot ; sp-=1
    a.label("op_add");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0); // aslot
    a.addi(T9, SP, -2);
    slot_addr(&mut a, TB, T9, T0); // bslot
    a.addi(TC, 0, 0); // carry=0
    for i in 0..8 {
        a.lw(T0, TA, i * 4);
        a.lw(T1, TB, i * 4);
        // out = a+b+carry ; newcarry
        adc32(&mut a, T2, TD, T0, T1, TC, T3, T4, T5);
        a.sw(T2, TB, i * 4);
        a.add(TC, TD, 0); // carry = cout
    }
    a.addi(SP, SP, -1);
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // SUB: r = a - b (256-bit) ; result to b slot ; sp-=1
    a.label("op_sub");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0);
    a.addi(T9, SP, -2);
    slot_addr(&mut a, TB, T9, T0);
    a.addi(TC, 0, 0); // borrow=0
    for i in 0..8 {
        a.lw(T0, TA, i * 4);
        a.lw(T1, TB, i * 4);
        sbb32(&mut a, T2, TD, T0, T1, TC, T3, T4, T5, T6);
        a.sw(T2, TB, i * 4);
        a.add(TC, TD, 0); // borrow = bout
    }
    a.addi(SP, SP, -1);
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // LT: (a < b) unsigned 256-bit -> 1/0 ; result to b slot ; sp-=1
    // borrow-out of (a - b) == (a<b).
    emit_cmp(&mut a, "op_lt", false);
    // GT: (a > b) == (b < a)
    emit_cmp(&mut a, "op_gt", true);

    // EQ: (a==b)?1:0
    a.label("op_eq");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0);
    a.addi(T9, SP, -2);
    slot_addr(&mut a, TB, T9, T0);
    a.addi(TC, 0, 0); // acc
    for i in 0..8 {
        a.lw(T0, TA, i * 4);
        a.lw(T1, TB, i * 4);
        a.xor(T0, T0, T1);
        a.or(TC, TC, T0);
    }
    // result = (acc==0)?1:0
    emit_store_bool_iszero(&mut a, TB, TC);
    a.addi(SP, SP, -1);
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // ISZERO: (a==0)?1:0 in place
    a.label("op_iszero");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0);
    a.addi(TC, 0, 0);
    for i in 0..8 {
        a.lw(T0, TA, i * 4);
        a.or(TC, TC, T0);
    }
    emit_store_bool_iszero(&mut a, TA, TC);
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // PUSH: n = op-0x5f ; read n big-endian bytes from code[pc+1..] into slot sp
    a.label("op_push");
    a.addi(T9, T0, -0x5f); // n
    // clear 8 limbs of slot sp
    slot_addr(&mut a, TB, SP, T0); // slot base
    a.addi(T1, 0, 0);
    for i in 0..8 {
        a.sw(T1, TB, i * 4);
    }
    // k = 0
    a.addi(TA, 0, 0); // k
    a.label("push_loop");
    a.beq(TA, T9, "push_done"); // k==n?
    // byte = code[pc+1+k]
    a.addi(T0, PC, 1);
    a.add(T0, T0, TA); // pc+1+k
    word_addr(&mut a, T1, RCODE, T0, T2);
    a.lw(T1, T1, 0); // byte
    // p = n-1-k
    a.sub(T2, T9, TA);
    a.addi(T2, T2, -1); // p
    // limbidx = p>>2 ; limbaddr = slot + limbidx*4
    a.srli(T3, T2, 2);
    a.slli(T3, T3, 2);
    a.add(T3, TB, T3); // limb addr
    // sh = (p&3)<<3 ; val = byte<<sh
    a.andi(T4, T2, 3);
    a.slli(T4, T4, 3);
    a.sll(T1, T1, T4);
    // cur |= val
    a.lw(T5, T3, 0);
    a.or(T5, T5, T1);
    a.sw(T5, T3, 0);
    a.addi(TA, TA, 1); // k++
    a.j("push_loop");
    a.label("push_done");
    a.addi(SP, SP, 1);
    // pc += 1 + n
    a.addi(PC, PC, 1);
    a.add(PC, PC, T9);
    a.j("dispatch");

    // DUP-d: d = op-0x7f ; copy slot (sp-d) to slot sp ; sp+=1
    a.label("op_dup");
    a.addi(T9, T0, -0x7f); // d
    a.sub(T8, SP, T9); // src depth = sp-d
    slot_addr(&mut a, TA, T8, T0); // src
    slot_addr(&mut a, TB, SP, T0); // dst
    for i in 0..8 {
        a.lw(T0, TA, i * 4);
        a.sw(T0, TB, i * 4);
    }
    a.addi(SP, SP, 1);
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // SWAP-d: d = op-0x8f ; swap slot sp-1 and slot sp-1-d
    a.label("op_swap");
    a.addi(T9, T0, -0x8f); // d
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0); // top
    a.sub(T8, T8, T9);
    slot_addr(&mut a, TB, T8, T0); // other
    for i in 0..8 {
        a.lw(T0, TA, i * 4);
        a.lw(T1, TB, i * 4);
        a.sw(T1, TA, i * 4);
        a.sw(T0, TB, i * 4);
    }
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // MSTORE: off=pop(low limb), val=pop ; write val big-endian to EVM mem[off..+32]
    a.label("op_mstore");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0); // off slot
    a.lw(TC, TA, 0); // off (low 32 bits)
    a.addi(T9, SP, -2);
    slot_addr(&mut a, TB, T9, T0); // val slot
    // for be byte position p in 0..32: limb i=(31-p)/4, local=(31-p)&3 (0=LSB of limb)
    for p in 0..32u32 {
        let overall = 31 - p; // LSB byte index of value
        let limb = overall / 4;
        let local = overall % 4;
        a.lw(T0, TB, (limb * 4) as i32);
        if local != 0 {
            a.srli(T0, T0, local * 8);
        }
        a.andi(T0, T0, 0xFF); // byte value
        // dst = MEM + (off+p)*4
        a.addi(T1, TC, p as i32);
        word_addr(&mut a, T2, RMEM, T1, T3);
        a.sw(T0, T2, 0);
    }
    a.addi(SP, SP, -2);
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // MLOAD: off=pop ; read 32 big-endian bytes from EVM mem -> push U256
    a.label("op_mload");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0);
    a.lw(TC, TA, 0); // off
    slot_addr(&mut a, TB, T8, T0); // reuse slot sp-1 as destination (pop then push => same slot)
    for i in 0..8 {
        a.addi(T0, 0, 0);
        a.sw(T0, TB, i * 4);
    }
    for p in 0..32u32 {
        // byte from mem[off+p]
        a.addi(T1, TC, p as i32);
        word_addr(&mut a, T2, RMEM, T1, T3);
        a.lw(T0, T2, 0); // byte
        let overall = 31 - p;
        let limb = overall / 4;
        let local = overall % 4;
        if local != 0 {
            a.slli(T0, T0, local * 8);
        }
        a.lw(T4, TB, (limb * 4) as i32);
        a.or(T4, T4, T0);
        a.sw(T4, TB, (limb * 4) as i32);
    }
    // sp unchanged (pop 1, push 1)
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // CALLDATALOAD: off=pop ; read 32 big-endian bytes from calldata -> push
    a.label("op_calldataload");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0);
    a.lw(TC, TA, 0); // off
    slot_addr(&mut a, TB, T8, T0);
    for i in 0..8 {
        a.addi(T0, 0, 0);
        a.sw(T0, TB, i * 4);
    }
    for p in 0..32u32 {
        // srcidx = off + p ; guard srcidx < calldata_size (both < 2^31):
        // lt = (srcidx - callN) >> 31.
        a.addi(T5, TC, p as i32); // srcidx
        a.sub(T9, T5, RCALLN);
        a.srli(T9, T9, 31); // T9 = lt (1 iff in-bounds)
        word_addr(&mut a, T2, RCALL, T5, T3);
        a.lw(T0, T2, 0); // byte (may be garbage if OOB)
        // mask = 0 - lt  (0 or 0xFFFFFFFF)
        a.sub(T4, 0, T9);
        a.and(T0, T0, T4);
        let overall = 31 - p;
        let limb = overall / 4;
        let local = overall % 4;
        if local != 0 {
            a.slli(T0, T0, local * 8);
        }
        a.lw(TD, TB, (limb * 4) as i32);
        a.or(TD, TD, T0);
        a.sw(TD, TB, (limb * 4) as i32);
    }
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // JUMP: dst=pop(low limb) ; pc = dst
    a.label("op_jump");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0);
    a.lw(PC, TA, 0);
    a.addi(SP, SP, -1);
    a.j("dispatch");

    // JUMPI: dst=pop, cond=pop ; if cond!=0 pc=dst else pc+=1
    a.label("op_jumpi");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0); // dst slot
    a.addi(T9, SP, -2);
    slot_addr(&mut a, TB, T9, T0); // cond slot
    // cond nonzero?
    a.addi(TC, 0, 0);
    for i in 0..8 {
        a.lw(T0, TB, i * 4);
        a.or(TC, TC, T0);
    }
    a.addi(SP, SP, -2);
    a.beq(TC, 0, "jumpi_nottaken");
    a.lw(PC, TA, 0); // pc = dst
    a.j("dispatch");
    a.label("jumpi_nottaken");
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // SLOAD: key=top ; scan storage entries for matching key ; result -> top slot
    a.label("op_sload");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0); // key slot (also result slot)
    // e = 0 ; entryaddr = STOR
    a.addi(T9, 0, 0); // e
    a.add(TB, RSTOR, 0); // entry ptr
    a.label("sload_loop");
    a.beq(T9, RSTORN, "sload_notfound");
    // compare key (8 limbs) at TA vs entry key at TB
    a.addi(TC, 0, 0); // diff acc
    for i in 0..8 {
        a.lw(T0, TA, i * 4);
        a.lw(T1, TB, i * 4);
        a.xor(T0, T0, T1);
        a.or(TC, TC, T0);
    }
    a.beq(TC, 0, "sload_found");
    a.addi(T9, T9, 1);
    a.addi(TB, TB, 64);
    a.j("sload_loop");
    a.label("sload_found");
    // copy val (entry+32) into key slot
    for i in 0..8 {
        a.lw(T0, TB, 32 + i * 4);
        a.sw(T0, TA, i * 4);
    }
    a.addi(PC, PC, 1);
    a.j("dispatch");
    a.label("sload_notfound");
    for i in 0..8 {
        a.addi(T0, 0, 0);
        a.sw(T0, TA, i * 4);
    }
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // SSTORE: key=pop, val=pop ; upsert into storage map
    a.label("op_sstore");
    a.addi(T8, SP, -1);
    slot_addr(&mut a, TA, T8, T0); // key slot
    a.addi(T9, SP, -2);
    slot_addr(&mut a, TB, T9, T0); // val slot
    // scan for key
    a.addi(TC, 0, 0); // e
    a.add(TD, RSTOR, 0); // entry ptr
    a.label("sstore_loop");
    a.beq(TC, RSTORN, "sstore_new");
    a.addi(TE, 0, 0); // diff
    for i in 0..8 {
        a.lw(T0, TA, i * 4);
        a.lw(T1, TD, i * 4);
        a.xor(T0, T0, T1);
        a.or(TE, TE, T0);
    }
    a.beq(TE, 0, "sstore_update");
    a.addi(TC, TC, 1);
    a.addi(TD, TD, 64);
    a.j("sstore_loop");
    a.label("sstore_new");
    // entry ptr TD points at STOR + storN*64 ; write key+val, storN++
    for i in 0..8 {
        a.lw(T0, TA, i * 4);
        a.sw(T0, TD, i * 4);
        a.lw(T1, TB, i * 4);
        a.sw(T1, TD, 32 + i * 4);
    }
    a.addi(RSTORN, RSTORN, 1);
    a.j("sstore_fin");
    a.label("sstore_update");
    for i in 0..8 {
        a.lw(T1, TB, i * 4);
        a.sw(T1, TD, 32 + i * 4);
    }
    a.label("sstore_fin");
    a.addi(SP, SP, -2);
    a.addi(PC, PC, 1);
    a.j("dispatch");

    // RETURN: off=pop, len=pop ; copy len bytes from EVM mem to RES ; halt=2
    a.label("op_return");
    emit_return(&mut a, 2);
    // REVERT: same copy, halt=3
    a.label("op_revert");
    emit_return(&mut a, 3);

    // ---- halt: compute a result-fold into x2/x3/RESULT_ADDR (the circuit's
    // public outputs), then spin (self-jump) so the trace pads to fixed length.
    a.label("halt");
    // x2 = fold(halt_code, return bytes RES[0..ret_len])
    a.li(T0, T1, 0x1000_0001);
    a.xor(2, T0, RHALT); // seed ^ halt_code
    a.addi(TA, 0, 0); // k
    a.label("fin_ret");
    a.beq(TA, RRETLEN, "fin_ret_done");
    word_addr(&mut a, T2, RRES, TA, T3);
    a.lw(T2, T2, 0); // byte
    a.addi(T2, T2, 0x9e);
    a.slli(T3, 2, 5);
    a.srli(T4, 2, 27);
    a.or(2, T3, T4); // rotl(x2,5)
    a.xor(2, 2, T2);
    a.addi(TA, TA, 1);
    a.j("fin_ret");
    a.label("fin_ret_done");
    // x3 = fold(storage_count, STOR words [0 .. storn*16))
    a.li(T0, T1, 0x2000_0002);
    a.xor(3, T0, RSTORN);
    a.slli(TB, RSTORN, 4); // storn*16 words
    a.addi(TA, 0, 0);
    a.label("fin_st");
    a.beq(TA, TB, "fin_st_done");
    word_addr(&mut a, T2, RSTOR, TA, T3);
    a.lw(T2, T2, 0);
    a.li(T5, T6, 0x85eb_ca6b);
    a.add(T2, T2, T5);
    a.slli(T3, 3, 7);
    a.srli(T4, 3, 25);
    a.or(3, T3, T4); // rotl(x3,7)
    a.xor(3, 3, T2);
    a.addi(TA, TA, 1);
    a.j("fin_st");
    a.label("fin_st_done");
    // combined = rotl(x2^x3, 11) ^ (ret_len + storn) -> RESULT_ADDR
    a.xor(T2, 2, 3);
    a.slli(T3, T2, 11);
    a.srli(T4, T2, 21);
    a.or(T2, T3, T4);
    a.add(T5, RRETLEN, RSTORN);
    a.xor(T2, T2, T5);
    a.li(T0, T1, RESULT_ADDR);
    a.sw(T2, T0, 0);
    a.label("spin");
    a.j("spin");

    a.assemble(PROG_BASE)
}

/// Store 1 (if acc==0) or 0 (if acc!=0) as a U256 into slot at `slot_base`.
fn emit_store_bool_iszero(a: &mut Asm, slot_base: u32, acc: u32) {
    let id = next_id();
    let one = format!("__iszero_one_{id}");
    let done = format!("__iszero_done_{id}");
    // limb0 = (acc==0)?1:0 ; other limbs 0
    a.addi(T0, 0, 0);
    for i in 1..8 {
        a.sw(T0, slot_base, i * 4);
    }
    a.beq(acc, 0, &one);
    // acc != 0 -> store 0
    a.addi(T0, 0, 0);
    a.sw(T0, slot_base, 0);
    a.j(&done);
    a.label(&one);
    a.addi(T0, 0, 1);
    a.sw(T0, slot_base, 0);
    a.label(&done);
}

fn next_id() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static C: AtomicUsize = AtomicUsize::new(0);
    C.fetch_add(1, Ordering::Relaxed)
}

/// Emit an unsigned 256-bit comparison handler. gt=false: a<b ; gt=true: a>b.
fn emit_cmp(a: &mut Asm, name: &'static str, gt: bool) {
    a.label(name);
    a.addi(T8, SP, -1);
    slot_addr(a, TA, T8, T0); // a slot
    a.addi(T9, SP, -2);
    slot_addr(a, TB, T9, T0); // b slot
    a.addi(TC, 0, 0); // borrow=0
    for i in 0..8 {
        if gt {
            // compute b - a ; borrow == (b<a) == (a>b)
            a.lw(T0, TB, i * 4);
            a.lw(T1, TA, i * 4);
        } else {
            a.lw(T0, TA, i * 4);
            a.lw(T1, TB, i * 4);
        }
        sbb32(a, T2, TD, T0, T1, TC, T3, T4, T5, T6);
        a.add(TC, TD, 0); // borrow chain
    }
    // result (TC = final borrow) -> b slot as U256
    a.addi(T0, 0, 0);
    for i in 1..8 {
        a.sw(T0, TB, i * 4);
    }
    a.sw(TC, TB, 0);
    a.addi(SP, SP, -1);
    a.addi(PC, PC, 1);
    a.j("dispatch");
}

/// RETURN/REVERT: off=pop, len=pop ; copy `len` bytes mem[off..off+len] to RES;
/// set return_len=len, halt=code, then jump to halt.
fn emit_return(a: &mut Asm, code: i32) {
    let tag = if code == 2 { "ret" } else { "rev" };
    a.addi(T8, SP, -1);
    slot_addr(a, TA, T8, T0);
    a.lw(TC, TA, 0); // off
    a.addi(T9, SP, -2);
    slot_addr(a, TB, T9, T0);
    a.lw(TD, TB, 0); // len
    a.add(RRETLEN, TD, 0);
    // copy loop: k=0..len : RES[k] = mem[off+k]
    a.addi(TE, 0, 0); // k
    a.label(&format!("{tag}_copy"));
    a.beq(TE, TD, &format!("{tag}_copydone"));
    a.add(T0, TC, TE); // off+k
    word_addr(a, T1, RMEM, T0, T2);
    a.lw(T1, T1, 0);
    word_addr(a, T2, RRES, TE, T3);
    a.sw(T1, T2, 0);
    a.addi(TE, TE, 1);
    a.j(&format!("{tag}_copy"));
    a.label(&format!("{tag}_copydone"));
    a.addi(RHALT, 0, code);
    a.j("halt");
}

// --------------------------- native driver (emulator) ----------------------

#[derive(Clone, Debug)]
pub struct EvmRun {
    pub output: Vec<u8>,
    pub storage: BTreeMap<U256, U256>,
    pub halt_code: u32, // 1 STOP, 2 RETURN, 3 REVERT, 4 ERR
    pub cycles: usize,
    /// distinct data-memory word addresses touched (loads+stores).
    pub touched: Vec<u32>,
    pub final_pc: u32,
    pub final_sp: u32,
    /// In-circuit public outputs = [x2, x3, mem[RESULT_ADDR]]: a deterministic
    /// fold of (halt_code, return data, final storage) the interpreter computes
    /// at halt. The GKR proof exposes exactly these three words.
    pub out_vals: [u32; 3],
    /// storage entry count as tracked by the RV32 program (insertion order).
    pub storage_count: u32,
    /// the full per-cycle trace (length == max_cycles passed to run_evm).
    pub trace: Vec<crate::emulator::StepRecord>,
    /// initial value at each `touched` address (parallel to `touched`), i.e. the
    /// memory content right after setup, before cycle 0 (grand-product seed).
    pub mem_init: Vec<u32>,
}

/// Set up initial RV32 memory for a run: header, code, calldata, pre-storage.
pub fn setup_memory(cpu: &mut Cpu, code: &[u8], calldata: &[u8], pre: &BTreeMap<U256, U256>) {
    // header @ 0x8000: storage_count, calldata_size, code_size
    cpu.mem.store(0x8000, pre.len() as u32);
    cpu.mem.store(0x8004, calldata.len() as u32);
    cpu.mem.store(0x8008, code.len() as u32);
    for (i, &b) in code.iter().enumerate() {
        cpu.mem.store(CODE_BASE + 4 * i as u32, b as u32);
    }
    for (i, &b) in calldata.iter().enumerate() {
        cpu.mem.store(CALL_BASE + 4 * i as u32, b as u32);
    }
    for (e, (k, v)) in pre.iter().enumerate() {
        let base = STOR_BASE + 64 * e as u32;
        for i in 0..8 {
            cpu.mem.store(base + 4 * i as u32, k.0[i]);
            cpu.mem.store(base + 32 + 4 * i as u32, v.0[i]);
        }
    }
}

/// Read a U256 (8 little-endian limbs) from memory at `base`.
fn read_u256(cpu: &Cpu, base: u32) -> U256 {
    let mut l = [0u32; 8];
    for i in 0..8 {
        l[i] = cpu.mem.load(base + 4 * i as u32);
    }
    U256(l)
}

/// Run `code`/`calldata`/`pre` through the RV32 EVM interpreter in the emulator.
pub fn run_evm(code: &[u8], calldata: &[u8], pre: &BTreeMap<U256, U256>, max_cycles: usize) -> EvmRun {
    let prog = assemble_interpreter();
    let prog_len = prog.len();
    if std::env::var("EVM_DBG").is_ok() {
        eprintln!("[dbg] interpreter program length = {prog_len} words ({} bytes)", prog_len * 4);
    }
    let mut cpu = Cpu::new(prog, PROG_BASE);
    setup_memory(&mut cpu, code, calldata, pre);
    let init_mem = cpu.mem.clone();
    if let Ok(v) = std::env::var("EVM_DBG") {
        let n = v.parse::<usize>().unwrap_or(80);
        for c in 0..n {
            if cpu.pc as usize >= prog_len * 4 {
                eprintln!("[{c:>4}] pc={:#x} OUT OF PROGRAM RANGE (len {} bytes) — bad jump", cpu.pc, prog_len * 4);
                break;
            }
            let r = cpu.step(c as u32);
            eprintln!(
                "[{c:>4}] pc={:#06x} insn={:#010x} op={:#04x} rd={} rs1={}={} rs2={}={} rd_val={:#x} next={:#06x}{}",
                r.pc, r.insn, r.opcode, r.rd_idx, r.rs1_idx, r.rs1_val, r.rs2_idx, r.rs2_val, r.rd_val, r.next_pc,
                if r.is_load {" LOAD"} else if r.is_store {" STORE"} else {""}
            );
        }
        std::process::exit(0);
    }
    let trace = cpu.run(max_cycles);

    // collect touched data-memory addresses.
    let mut touched = std::collections::BTreeSet::<u32>::new();
    let mut halt_cycle = max_cycles;
    for (c, r) in trace.iter().enumerate() {
        if r.is_load || r.is_store {
            touched.insert(r.mem_addr & !3);
        }
        // detect entry into the spin loop (halt): jal to self
        if r.opcode == crate::emulator::OPC_JAL && r.next_pc == r.pc && c < halt_cycle {
            halt_cycle = c;
        }
    }

    let halt_code = cpu.regs[RHALT as usize];
    let ret_len = cpu.regs[RRETLEN as usize] as usize;
    let mut output = vec![0u8; ret_len];
    for k in 0..ret_len {
        output[k] = (cpu.mem.load(RES_BASE + 4 * k as u32) & 0xff) as u8;
    }
    // reconstruct final storage
    let storn = cpu.regs[RSTORN as usize];
    let mut storage = BTreeMap::new();
    for e in 0..storn {
        let base = STOR_BASE + 64 * e;
        let k = read_u256(&cpu, base);
        let v = read_u256(&cpu, base + 32);
        storage.insert(k, v);
    }

    let out_vals = [
        cpu.regs[2],
        cpu.regs[3],
        cpu.mem.load(RESULT_ADDR),
    ];

    let touched: Vec<u32> = touched.into_iter().collect();
    let mem_init: Vec<u32> = touched.iter().map(|&a| init_mem.load(a)).collect();

    EvmRun {
        output,
        storage,
        halt_code,
        cycles: halt_cycle,
        touched,
        final_pc: cpu.pc,
        final_sp: cpu.regs[SP as usize],
        out_vals,
        storage_count: storn,
        trace,
        mem_init,
    }
}
