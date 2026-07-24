#![no_std]
#![no_main]

// The transaction rollup contract: a raw balance-transfer STF with a lightweight
// signature. Simple no_std Rust, compiled to riscv32im (opt-level=3), stack-free
// and multiply-free so it runs on the RV32I GKR circuit.
//
// Memory layout (byte addresses; the DA-committed input + state):
//   0x100         : N (number of transactions in this block)
//   0x104 + 16*i  : tx i = [sender(4), recipient(4), amount(4), sig(4)]
//   0x2000 + 4*a  : balance[account a]     (persistent rollup state)
//   0x3000 + 4*a  : key[account a]         (persistent; sender's secret)
//
// A tx applies iff sig == mac(key[sender], sender, recipient, amount) AND
// balance[sender] >= amount; then debit sender, credit recipient. The balance
// state is the rollup state whose root the proof transitions (pre -> post).

#[inline(always)]
fn rd(addr: u32) -> u32 { unsafe { core::ptr::read_volatile(addr as *const u32) } }
#[inline(always)]
fn wr(addr: u32, v: u32) { unsafe { core::ptr::write_volatile(addr as *mut u32, v) } }

// Multiply-free keyed MAC (ARX: rotate/shift/xor/add) standing in for a
// signature scheme. A genuine per-tx authorization keyed by the sender's secret;
// real ECDSA is ~1000x the cycles and infeasible on this prover.
#[inline(always)]
fn mac(key: u32, s: u32, r: u32, amt: u32) -> u32 {
    let mut h = key ^ s.rotate_left(7);
    h = h.wrapping_add(r).rotate_left(13);
    h ^= amt.rotate_left(17);
    h = h.wrapping_add(h << 5);
    h ^= h >> 11;
    h
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let n = rd(0x100);
    let mut i: u32 = 0;
    while i < n {
        let base = 0x104 + (i << 4);
        let s = rd(base);
        let r = rd(base + 4);
        let amt = rd(base + 8);
        let sig = rd(base + 12);
        if mac(rd(0x3000 + (s << 2)), s, r, amt) == sig {
            let bs = rd(0x2000 + (s << 2));
            if bs >= amt {
                wr(0x2000 + (s << 2), bs - amt);
                let br = rd(0x2000 + (r << 2));
                wr(0x2000 + (r << 2), br.wrapping_add(amt));
            }
        }
        i = i.wrapping_add(1);
    }
    loop {}
}

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
