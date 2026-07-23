#![no_std]
#![no_main]

// add : mem[0x100] + mem[0x104] -> mem[0x200].
// Compiled to riscv32im-unknown-none-elf (opt-level=3) with no stack usage.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    let a = unsafe { core::ptr::read_volatile(0x100 as *const u32) };
    let b = unsafe { core::ptr::read_volatile(0x104 as *const u32) };
    unsafe { core::ptr::write_volatile(0x200 as *mut u32, a.wrapping_add(b)) };
    loop {}
}

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
