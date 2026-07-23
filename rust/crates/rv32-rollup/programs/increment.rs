#![no_std]
#![no_main]

// increment : a persistent counter at mem[0x200] (carried block to block).
// Compiled to riscv32im-unknown-none-elf (opt-level=3) with no stack usage.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    let c = unsafe { core::ptr::read_volatile(0x200 as *const u32) };
    unsafe { core::ptr::write_volatile(0x200 as *mut u32, c.wrapping_add(1)) };
    loop {}
}

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
