//! R4 — data-parallel batch keccak-256 over the committed DA block data.
//!
//! This is the FIRST rung built for the performance model (context.md): the AC
//! gain scales with committed-input size and is maximized for DATA-PARALLEL
//! circuits. The committed GKR input layer IS the real block-31 DA blob
//! (`testdata/block-31-blob.bin`, 5118 bytes) split into fixed 135-byte rows;
//! the circuit computes keccak-256 of EACH row as an independent, parallel
//! instance (no sequential state threading). rsema1d/DA is the sole polynomial
//! commitment, opened at the sumcheck point; the prover commits nothing else.
//!
//! keccak core (rc / keccak_f / rotate / xor_in / copy_out) is the proven GF2
//! gadget from the Phase-6 `stf-circuit` template, generalized here to a
//! single-block sponge for messages up to 135 bytes (rate 136 => exactly one
//! keccak_f per row).

use expander_compiler::frontend::*;

pub const ROW: usize = 135; // bytes per row (<= 135 => single-block keccak)
pub const ROW_BITS: usize = ROW * 8;
pub const N_ROWS: usize = 38; // ceil(5118 / 135)
pub const RATE: usize = 136; // keccak-256 rate in bytes

// --------------------------- keccak-f (GF2) --------------------------------

fn rc() -> Vec<u64> {
    vec![
        0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
        0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
        0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
    ]
}

fn xor<C: Config>(api: &mut impl RootAPI<C>, a: Vec<Variable>, b: Vec<Variable>) -> Vec<Variable> {
    (0..a.len()).map(|i| api.add(a[i], b[i])).collect()
}
fn and<C: Config>(api: &mut impl RootAPI<C>, a: Vec<Variable>, b: Vec<Variable>) -> Vec<Variable> {
    (0..a.len()).map(|i| api.mul(a[i], b[i])).collect()
}
fn not<C: Config>(api: &mut impl RootAPI<C>, a: Vec<Variable>) -> Vec<Variable> {
    (0..a.len()).map(|i| api.sub(1, a[i])).collect()
}
fn rotate_left(bits: &[Variable], k: usize) -> Vec<Variable> {
    let n = bits.len();
    let s = k & (n - 1);
    let mut nb = bits[n - s..].to_vec();
    nb.extend_from_slice(&bits[0..n - s]);
    nb
}

fn xor_in<C: Config>(api: &mut impl RootAPI<C>, mut s: Vec<Vec<Variable>>, buf: Vec<Vec<Variable>>) -> Vec<Vec<Variable>> {
    for y in 0..5 {
        for x in 0..5 {
            if x + 5 * y < buf.len() {
                s[5 * x + y] = xor(api, s[5 * x + y].clone(), buf[x + 5 * y].clone());
            }
        }
    }
    s
}

pub(crate) fn keccak_f<C: Config>(api: &mut impl RootAPI<C>, mut a: Vec<Vec<Variable>>) -> Vec<Vec<Variable>> {
    let mut b = vec![vec![api.constant(0); 64]; 25];
    let mut c = vec![vec![api.constant(0); 64]; 5];
    let mut d = vec![vec![api.constant(0); 64]; 5];
    let mut da = vec![vec![api.constant(0); 64]; 5];
    let rc = rc();
    for i in 0..24 {
        for j in 0..5 {
            let t1 = xor(api, a[j * 5 + 1].clone(), a[j * 5 + 2].clone());
            let t2 = xor(api, a[j * 5 + 3].clone(), a[j * 5 + 4].clone());
            c[j] = xor(api, t1, t2);
        }
        for j in 0..5 {
            d[j] = xor(api, c[(j + 4) % 5].clone(), rotate_left(&c[(j + 1) % 5], 1));
            da[j] = xor(api, a[((j + 4) % 5) * 5].clone(), rotate_left(&a[((j + 1) % 5) * 5], 1));
        }
        for j in 0..25 {
            let tmp = xor(api, da[j / 5].clone(), a[j].clone());
            a[j] = xor(api, tmp, d[j / 5].clone());
        }
        b[0] = a[0].clone();
        b[8] = rotate_left(&a[1], 36);
        b[11] = rotate_left(&a[2], 3);
        b[19] = rotate_left(&a[3], 41);
        b[22] = rotate_left(&a[4], 18);
        b[2] = rotate_left(&a[5], 1);
        b[5] = rotate_left(&a[6], 44);
        b[13] = rotate_left(&a[7], 10);
        b[16] = rotate_left(&a[8], 45);
        b[24] = rotate_left(&a[9], 2);
        b[4] = rotate_left(&a[10], 62);
        b[7] = rotate_left(&a[11], 6);
        b[10] = rotate_left(&a[12], 43);
        b[18] = rotate_left(&a[13], 15);
        b[21] = rotate_left(&a[14], 61);
        b[1] = rotate_left(&a[15], 28);
        b[9] = rotate_left(&a[16], 55);
        b[12] = rotate_left(&a[17], 25);
        b[15] = rotate_left(&a[18], 21);
        b[23] = rotate_left(&a[19], 56);
        b[3] = rotate_left(&a[20], 27);
        b[6] = rotate_left(&a[21], 20);
        b[14] = rotate_left(&a[22], 39);
        b[17] = rotate_left(&a[23], 8);
        b[20] = rotate_left(&a[24], 14);
        for j in 0..25 {
            let t = not(api, b[(j + 5) % 25].clone());
            let t = and(api, t, b[(j + 10) % 25].clone());
            a[j] = xor(api, b[j].clone(), t);
        }
        for j in 0..64 {
            if rc[i] >> j & 1 == 1 {
                a[0][j] = api.sub(1, a[0][j]);
            }
        }
    }
    a
}

fn copy_out_unaligned(s: Vec<Vec<Variable>>, rate: usize, output_len: usize) -> Vec<Variable> {
    let mut out = vec![];
    let w = 8;
    let mut b = 0;
    while b < output_len {
        for y in 0..5 {
            for x in 0..5 {
                if x + 5 * y < rate / w && b < output_len {
                    out.append(&mut s[5 * x + y].clone());
                    b += 8;
                }
            }
        }
    }
    out
}

/// keccak-256 of a single-block message of `nbytes` bytes (nbytes <= 135). `p`
/// is the message bits, LSB-first per byte (len == nbytes*8). Returns 256 output
/// bits. General over the message length (pad10*1 placed at nbytes..RATE).
pub fn keccak256_single_block<C: Config>(api: &mut impl RootAPI<C>, p: &[Variable]) -> Vec<Variable> {
    let nbytes = p.len() / 8;
    assert_eq!(p.len() % 8, 0);
    assert!(nbytes <= RATE - 1, "single-block keccak needs nbytes <= 135");
    let mut new_p = p.to_vec();
    // keccak pad10*1 over the rate: pad byte length = RATE - nbytes; first pad
    // byte |= 0x01, last pad byte |= 0x80 (coincide to 0x81 when they're one byte).
    let padlen = RATE - nbytes;
    let mut pad = vec![0u8; padlen];
    pad[0] |= 0x01;
    pad[padlen - 1] |= 0x80;
    for byte in pad {
        for j in 0..8 {
            new_p.push(api.constant(((byte >> j) & 1) as u32));
        }
    }
    // 17 lanes of 64 bits (rate = 1088 bits).
    let mut lanes = vec![vec![api.constant(0); 64]; 17];
    for i in 0..17 {
        for j in 0..64 {
            lanes[i][j] = new_p[i * 64 + j];
        }
    }
    let mut ss = vec![vec![api.constant(0); 64]; 25];
    ss = xor_in(api, ss, lanes);
    ss = keccak_f(api, ss);
    copy_out_unaligned(ss, RATE, 32)
}

/// keccak-256 of a full `ROW`-byte row (the R4 batch path). Thin wrapper.
pub fn keccak256_row<C: Config>(api: &mut impl RootAPI<C>, p: &[Variable]) -> Vec<Variable> {
    assert_eq!(p.len(), ROW_BITS);
    keccak256_single_block(api, p)
}

/// VARIABLE-LENGTH single-block keccak-256: `msg135` holds up to 135 message
/// bytes (LSB-first per byte; unused tail may be anything), `len_bits` is the
/// in-circuit byte length (<= 135) as a little-endian bit vector. The pad10*1 is
/// placed at the committed length via per-byte selects, so a single keccak_f
/// covers any message <= 135 bytes. Needed for MPT node hashing (node lengths
/// depend on the committed account values).
pub fn keccak256_varlen<C: Config>(api: &mut impl RootAPI<C>, msg135: &[Variable], len_bits: &[Variable]) -> Vec<Variable> {
    use crate::u256::{eq, lt};
    assert_eq!(msg135.len(), 135 * 8);
    let zero = api.constant(0);
    let one = api.constant(1);
    // Build the 136-byte rate block byte-by-byte.
    let mut block_bits = vec![zero; RATE * 8];
    for i in 0..RATE {
        // predicates on the byte index i vs the committed length.
        let i_bits: Vec<Variable> = (0..8).map(|b| api.constant(((i as u32) >> b) & 1)).collect();
        let is_msg = lt(api, &i_bits, len_bits); // i < len
        let is_pad0 = eq(api, &i_bits, len_bits); // i == len (first pad byte 0x01)
        for j in 0..8 {
            let msg_bit = if i < 135 { msg135[i * 8 + j] } else { zero };
            // first pad byte is 0x01 => only bit 0 set.
            let pad0_bit = if j == 0 { is_pad0 } else { zero };
            // select(is_msg, msg_bit, pad0_bit) == pad0_bit ^ is_msg*(msg_bit ^ pad0_bit)
            let diff = api.add(msg_bit, pad0_bit);
            let t = api.mul(is_msg, diff);
            block_bits[i * 8 + j] = api.add(pad0_bit, t);
        }
    }
    // Last rate byte gets |= 0x80 (bit 7 of byte 135). It is 0 for len<=134, and
    // 0x01 when len==135; XOR-in 0x80 gives 0x80 or 0x81 respectively.
    let b = 135 * 8 + 7;
    block_bits[b] = api.add(block_bits[b], one);
    // 17 lanes of 64 bits.
    let mut lanes = vec![vec![api.constant(0); 64]; 17];
    for i in 0..17 {
        for j in 0..64 {
            lanes[i][j] = block_bits[i * 64 + j];
        }
    }
    let mut ss = vec![vec![api.constant(0); 64]; 25];
    ss = xor_in(api, ss, lanes);
    ss = keccak_f(api, ss);
    copy_out_unaligned(ss, RATE, 32)
}

/// MULTI-BLOCK keccak-256 of the first `nbytes` bytes of `msg_bits` (LSB-first
/// per byte), with `nbytes` a COMPILE-TIME length. Absorbs `nbytes/136 + 1` rate
/// blocks through the shared `keccak_f`, so it hashes messages of ANY length —
/// this is what lets a dense 16-child MPT branch node (~532 bytes, > one rate
/// block) be hashed in-circuit. `msg_bits.len()` must be >= `nbytes * 8`; the
/// pad10*1 is placed at the compile-time-known message end. Matches tiny_keccak /
/// the Ethereum keccak for every length (verified in `multiblock_tests`).
///
/// This is the general form the R2 requirement calls `keccak256_multiblock`; the
/// length is a structural constant (like the MPT `child_off`/`value_off`), so the
/// fixed-length variant is both correct and the cheapest circuit. The existing
/// single-block `keccak256_single_block` / `keccak256_varlen` APIs are unchanged.
pub fn keccak256_fixed<C: Config>(api: &mut impl RootAPI<C>, msg_bits: &[Variable], nbytes: usize) -> Vec<Variable> {
    assert!(msg_bits.len() >= nbytes * 8, "keccak256_fixed: buffer shorter than nbytes");
    let zero = api.constant(0);
    let one = api.constant(1);
    let nblocks = nbytes / RATE + 1; // >= 1 pad byte, pad10*1 fills to a rate multiple
    let total = nblocks * RATE; // padded byte length
    // Padded bit buffer: message bytes copied in, tail (incl. pad region) zeroed.
    let mut buf: Vec<Variable> = vec![zero; total * 8];
    for i in 0..nbytes {
        for j in 0..8 {
            buf[i * 8 + j] = msg_bits[i * 8 + j];
        }
    }
    // pad10*1: bit 0 of the first pad byte (index `nbytes`), bit 7 of the last
    // byte. They coincide (=> 0x81) when the pad is a single byte.
    buf[nbytes * 8] = api.add(buf[nbytes * 8], one);
    let last = (total - 1) * 8 + 7;
    buf[last] = api.add(buf[last], one);
    // Absorb every rate block, then squeeze 256 bits.
    let mut ss = vec![vec![api.constant(0); 64]; 25];
    for b in 0..nblocks {
        let mut lanes = vec![vec![api.constant(0); 64]; 17];
        for i in 0..17 {
            for j in 0..64 {
                lanes[i][j] = buf[b * RATE * 8 + i * 64 + j];
            }
        }
        ss = xor_in(api, ss, lanes);
        ss = keccak_f(api, ss);
    }
    copy_out_unaligned(ss, RATE, 32)
}

// ------------------------------- circuit -----------------------------------

#[cfg(test)]
mod multiblock_tests {
    use super::*;
    use tiny_keccak::Hasher;

    fn kc(b: &[u8]) -> [u8; 32] {
        let mut h = tiny_keccak::Keccak::v256();
        h.update(b);
        let mut o = [0u8; 32];
        h.finalize(&mut o);
        o
    }

    // Capacity covers the largest tested input (532 B, a full 16-child branch).
    const MB_CAP: usize = 560;
    static mut MB_NBYTES: usize = 0;
    declare_circuit!(MBCircuit {
        msg: [Variable; MB_CAP * 8],
        out: [PublicVariable; 256],
    });
    impl Define<GF2Config> for MBCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let n = unsafe { MB_NBYTES };
            let h = keccak256_fixed(api, &self.msg.to_vec(), n);
            for i in 0..256 {
                api.assert_is_equal(h[i], self.out[i]);
            }
        }
    }

    #[test]
    fn multiblock_keccak_matches_tiny() {
        // 300 bytes (3 rate blocks) and 532 bytes (a dense 16-child branch, 4
        // rate blocks) — both exceed the 135-byte single-block limit.
        for &len in &[300usize, 532] {
            unsafe { MB_NBYTES = len };
            let msg: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(7)).collect();
            let expected = kc(&msg);
            let CompileResult { witness_solver, layered_circuit } =
                compile(&MBCircuit::default(), CompileOptions::default()).unwrap();
            let mut asg = MBCircuit::<GF2>::default();
            for i in 0..len {
                for j in 0..8 {
                    asg.msg[i * 8 + j] = (((msg[i] >> j) & 1) as u32).into();
                }
            }
            for i in 0..32 {
                for j in 0..8 {
                    asg.out[i * 8 + j] = (((expected[i] >> j) & 1) as u32).into();
                }
            }
            let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
            assert!(layered_circuit.run(&w).iter().all(|x| *x), "multiblock keccak wrong for len {len}");
            println!("R1 multiblock keccak OK len={len} digest=0x{}", expected.iter().map(|x| format!("{:02x}", x)).collect::<String>());
        }
    }
}

#[cfg(test)]
mod varlen_tests {
    use super::*;
    use tiny_keccak::Hasher;

    fn kc(b: &[u8]) -> [u8; 32] {
        let mut h = tiny_keccak::Keccak::v256();
        h.update(b);
        let mut o = [0u8; 32];
        h.finalize(&mut o);
        o
    }

    declare_circuit!(VarlenCircuit {
        msg: [Variable; 135 * 8],
        len: [Variable; 8],
        out: [PublicVariable; 256],
    });
    impl Define<GF2Config> for VarlenCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let h = keccak256_varlen(api, &self.msg.to_vec(), &self.len.to_vec());
            for i in 0..256 {
                api.assert_is_equal(h[i], self.out[i]);
            }
        }
    }

    #[test]
    fn varlen_keccak_matches_tiny() {
        let CompileResult { witness_solver, layered_circuit } =
            compile(&VarlenCircuit::default(), CompileOptions::default()).unwrap();
        for &len in &[1usize, 32, 75, 110, 134, 135] {
            let msg: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(7).wrapping_add(1)).collect();
            let expected = kc(&msg);
            let mut asg = VarlenCircuit::<GF2>::default();
            for i in 0..len {
                for j in 0..8 {
                    asg.msg[i * 8 + j] = (((msg[i] >> j) & 1) as u32).into();
                }
            }
            for b in 0..8 {
                asg.len[b] = (((len as u32) >> b) & 1).into();
            }
            for i in 0..32 {
                for j in 0..8 {
                    asg.out[i * 8 + j] = (((expected[i] >> j) & 1) as u32).into();
                }
            }
            let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
            assert!(layered_circuit.run(&w).iter().all(|x| *x), "varlen keccak wrong for len {len}");
        }
    }
}

declare_circuit!(BatchKeccakCircuit {
    // Committed GKR input layer = the block's data rows (bit-decomposed).
    rows: [[Variable; ROW_BITS]; N_ROWS],
    // Public per-row keccak-256 digests (NOT part of the committed layer).
    out: [[PublicVariable; 256]; N_ROWS],
});

fn set_bits(dst: &mut [GF2], bytes: &[u8]) {
    for (i, &byte) in bytes.iter().enumerate() {
        for j in 0..8 {
            dst[i * 8 + j] = (((byte >> j) & 1) as u32).into();
        }
    }
}

/// Build the assignment: `rows` = the committed block-data rows, `out` = the
/// expected per-row keccak digests (native reference bits).
pub fn build_assignment(rows: &[[u8; ROW]; N_ROWS], digests: &[[u8; 32]; N_ROWS]) -> BatchKeccakCircuit<GF2> {
    let mut a = BatchKeccakCircuit::<GF2>::default();
    for r in 0..N_ROWS {
        set_bits(&mut a.rows[r], &rows[r]);
        set_bits(&mut a.out[r], &digests[r]);
    }
    a
}

impl Define<GF2Config> for BatchKeccakCircuit<Variable> {
    fn define<Builder: RootAPI<GF2Config>>(&self, api: &mut Builder) {
        // Data-parallel: N_ROWS independent keccak instances, no cross-row state.
        for r in 0..N_ROWS {
            let digest = keccak256_row(api, &self.rows[r].to_vec());
            for j in 0..256 {
                api.assert_is_equal(digest[j], self.out[r][j]);
            }
        }
    }
}
