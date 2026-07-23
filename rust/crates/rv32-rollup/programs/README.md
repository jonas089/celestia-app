# Sample rv32 programs (simple no_std Rust)

These are the programs the rollup executes each block. Each is simple
`#![no_std]` Rust compiled to `riscv32im-unknown-none-elf` at `opt-level=3`,
which produces **stack-free** code that matches the emulator/circuit ABI:

- input words are read from `mem[0x100]`, `mem[0x104]`, …
- the result is written to `mem[0x200]`
- execution ends in a self-loop (`loop {}`), the emulator's halt marker

The compiled instruction words are embedded in `src/main.rs` (`samples()`), and
the exact source is carried through the RPC so the explorer can show source +
bytecode + proof per block.

## Regenerate the bytecode

```sh
rustup target add riscv32im-unknown-none-elf
for p in sum_1_to_n add increment; do
  rustc --target riscv32im-unknown-none-elf --edition 2021 \
    -C panic=abort -C opt-level=3 --emit obj -o "/tmp/$p.o" "programs/$p.rs"
  # instruction words (u32, in order) from the _start section:
  "$(find "$(rustc --print sysroot)" -name llvm-objdump | head -1)" -d "/tmp/$p.o" \
    | sed -n '/<_start>:/,/^$/p' \
    | grep -oE '^[[:space:]]+[0-9a-f]+:[[:space:]]+[0-9a-f]{8}' \
    | awk '{print "0x"$2}'
done
```

Paste the printed words into the corresponding vector in `samples()`. Keep the
programs stack-free (verify the disassembly touches no `sp`) so they fit the
bounded committed-memory circuit.
