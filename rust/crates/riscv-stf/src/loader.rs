//! Minimal ELF32 (RISC-V, little-endian) loader for the SP1 guest.
//!
//! Hand-parses the ELF header + program headers (no external crate) and maps
//! every `PT_LOAD` segment into an emulator [`Memory`] image at its `p_vaddr`,
//! zero-filling BSS (`p_memsz > p_filesz`). Returns the entry PC.
//!
//! Target format (verified against the real ev-stf guest):
//!   ELFCLASS32, EM_RISCV (243), ET_EXEC, LSB. Image based at `STACK_TOP`
//!   (0x7800_0000), entry 0x7815_880c for this build.

use crate::emulator::Memory;

pub const PT_LOAD: u32 = 1;
pub const STACK_TOP: u32 = 0x7800_0000;

#[derive(Clone, Debug)]
pub struct LoadedElf {
    pub entry: u32,
    pub segments: Vec<Segment>,
}

#[derive(Clone, Debug)]
pub struct Segment {
    pub vaddr: u32,
    pub filesz: u32,
    pub memsz: u32,
    pub flags: u32,
}

fn rd_u16(d: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([d[off], d[off + 1]])
}
fn rd_u32(d: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([d[off], d[off + 1], d[off + 2], d[off + 3]])
}

/// Parse `elf` bytes and populate `mem` with all PT_LOAD segments.
/// Returns the loaded-ELF metadata (entry PC + segment table).
pub fn load_elf(elf: &[u8], mem: &mut Memory) -> LoadedElf {
    assert!(elf.len() >= 52, "ELF too small");
    assert_eq!(&elf[0..4], b"\x7fELF", "not an ELF file");
    assert_eq!(elf[4], 1, "expected ELFCLASS32");
    assert_eq!(elf[5], 1, "expected little-endian (ELFDATA2LSB)");

    let e_type = rd_u16(elf, 0x10);
    let e_machine = rd_u16(elf, 0x12);
    assert_eq!(e_type, 2, "expected ET_EXEC");
    assert_eq!(e_machine, 243, "expected EM_RISCV");

    let e_entry = rd_u32(elf, 0x18);
    let e_phoff = rd_u32(elf, 0x1c) as usize;
    let e_phentsize = rd_u16(elf, 0x2a) as usize;
    let e_phnum = rd_u16(elf, 0x2c) as usize;
    assert_eq!(e_entry & 3, 0, "entry PC must be 4-aligned");

    let mut segments = Vec::new();
    for i in 0..e_phnum {
        let ph = e_phoff + i * e_phentsize;
        let p_type = rd_u32(elf, ph);
        if p_type != PT_LOAD {
            continue;
        }
        let p_offset = rd_u32(elf, ph + 0x04) as usize;
        let p_vaddr = rd_u32(elf, ph + 0x08);
        let p_filesz = rd_u32(elf, ph + 0x10);
        let p_memsz = rd_u32(elf, ph + 0x14);
        let p_flags = rd_u32(elf, ph + 0x18);

        assert!(
            p_vaddr >= STACK_TOP,
            "PT_LOAD vaddr {p_vaddr:#x} below STACK_TOP {STACK_TOP:#x}"
        );

        let bytes = &elf[p_offset..p_offset + p_filesz as usize];
        mem.write_image(p_vaddr, bytes);
        // BSS (p_memsz > p_filesz) is implicitly zero: the sparse Memory returns 0
        // for any word never written, so no explicit zero-fill is required.

        segments.push(Segment {
            vaddr: p_vaddr,
            filesz: p_filesz,
            memsz: p_memsz,
            flags: p_flags,
        });
    }
    assert!(!segments.is_empty(), "no PT_LOAD segments found");

    LoadedElf { entry: e_entry, segments }
}
