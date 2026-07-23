#![no_std]
#![no_main]

// sum 1..=n : n is read from mem[0x100], result written to mem[0x200].
// Compiled to riscv32im-unknown-none-elf (opt-level=3) with no stack usage.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    let n = unsafe { core::ptr::read_volatile(0x100 as *const u32) };
    let mut acc: u32 = 0;
    let mut i: u32 = 1;
    while i <= n {
        acc = acc.wrapping_add(i);
        i += 1;
    }
    unsafe { core::ptr::write_volatile(0x200 as *mut u32, acc) };
    loop {}
}

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
