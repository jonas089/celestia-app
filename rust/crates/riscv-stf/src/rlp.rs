//! R6 (native reference) — EIP-1559 transaction RLP: encode (for constructing
//! test blocks), decode (extract fields), and the ECDSA signing hash `z`.
//!
//! ev-reth rollup txs are EIP-1559 (type 0x02). The committed DA block data is a
//! list of such tx byte-strings; the in-circuit parser (built next) must decode
//! exactly this layout to feed `ecrecover` (r, s, y_parity, z). This native
//! reference pins the byte layout and is the golden the circuit is checked
//! against (sender recovered here must equal the signer's address, and match
//! ev-reth's `recover_sender`).

use num_bigint::BigInt;
use num_traits::Zero;

// ------------------------------ RLP encoding -------------------------------

/// RLP-encode a byte string.
pub fn enc_bytes(b: &[u8]) -> Vec<u8> {
    if b.len() == 1 && b[0] < 0x80 {
        return b.to_vec();
    }
    let mut out = enc_len(b.len(), 0x80);
    out.extend_from_slice(b);
    out
}

/// RLP-encode a non-negative integer (minimal big-endian).
pub fn enc_uint(x: &BigInt) -> Vec<u8> {
    if x.is_zero() {
        return vec![0x80]; // empty string
    }
    let be = x.to_bytes_be().1;
    enc_bytes(&be)
}

/// RLP list header + concatenated items.
pub fn enc_list(items: &[Vec<u8>]) -> Vec<u8> {
    let body_len: usize = items.iter().map(|i| i.len()).sum();
    let mut out = enc_len(body_len, 0xc0);
    for i in items {
        out.extend_from_slice(i);
    }
    out
}

fn enc_len(len: usize, offset: u8) -> Vec<u8> {
    if len < 56 {
        vec![offset + len as u8]
    } else {
        let be = len.to_be_bytes();
        let be: Vec<u8> = be.iter().copied().skip_while(|&b| b == 0).collect();
        let mut out = vec![offset + 55 + be.len() as u8];
        out.extend_from_slice(&be);
        out
    }
}

// ------------------------------ RLP decoding -------------------------------

/// A decoded RLP item: a byte string or a list of items.
#[derive(Clone, Debug)]
pub enum Rlp {
    Str(Vec<u8>),
    List(Vec<Rlp>),
}

impl Rlp {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Rlp::Str(b) => b,
            _ => panic!("expected RLP string"),
        }
    }
    pub fn as_uint(&self) -> BigInt {
        BigInt::from_bytes_be(num_bigint::Sign::Plus, self.as_bytes())
    }
    pub fn as_list(&self) -> &[Rlp] {
        match self {
            Rlp::List(v) => v,
            _ => panic!("expected RLP list"),
        }
    }
}

/// Decode a single RLP item at `pos`; returns (item, next_pos).
pub fn decode_at(data: &[u8], pos: usize) -> (Rlp, usize) {
    let b = data[pos];
    if b < 0x80 {
        (Rlp::Str(vec![b]), pos + 1)
    } else if b < 0xb8 {
        let len = (b - 0x80) as usize;
        (Rlp::Str(data[pos + 1..pos + 1 + len].to_vec()), pos + 1 + len)
    } else if b < 0xc0 {
        let nlen = (b - 0xb7) as usize;
        let len = be_to_usize(&data[pos + 1..pos + 1 + nlen]);
        let s = pos + 1 + nlen;
        (Rlp::Str(data[s..s + len].to_vec()), s + len)
    } else if b < 0xf8 {
        let len = (b - 0xc0) as usize;
        let (items, _) = decode_list_body(data, pos + 1, pos + 1 + len);
        (Rlp::List(items), pos + 1 + len)
    } else {
        let nlen = (b - 0xf7) as usize;
        let len = be_to_usize(&data[pos + 1..pos + 1 + nlen]);
        let s = pos + 1 + nlen;
        let (items, _) = decode_list_body(data, s, s + len);
        (Rlp::List(items), s + len)
    }
}

fn decode_list_body(data: &[u8], mut pos: usize, end: usize) -> (Vec<Rlp>, usize) {
    let mut items = vec![];
    while pos < end {
        let (item, next) = decode_at(data, pos);
        items.push(item);
        pos = next;
    }
    (items, pos)
}

fn be_to_usize(b: &[u8]) -> usize {
    let mut v = 0usize;
    for &x in b {
        v = (v << 8) | x as usize;
    }
    v
}

// --------------------------- EIP-1559 transaction --------------------------

#[derive(Clone, Debug)]
pub struct Eip1559Tx {
    pub chain_id: u64,
    pub nonce: u64,
    pub max_priority_fee: BigInt,
    pub max_fee: BigInt,
    pub gas_limit: u64,
    pub to: [u8; 20],
    pub value: BigInt,
    pub data: Vec<u8>,
    // signature
    pub y_parity: u8,
    pub r: BigInt,
    pub s: BigInt,
}

impl Eip1559Tx {
    fn fields_unsigned(&self) -> Vec<Vec<u8>> {
        vec![
            enc_uint(&BigInt::from(self.chain_id)),
            enc_uint(&BigInt::from(self.nonce)),
            enc_uint(&self.max_priority_fee),
            enc_uint(&self.max_fee),
            enc_uint(&BigInt::from(self.gas_limit)),
            enc_bytes(&self.to),
            enc_uint(&self.value),
            enc_bytes(&self.data),
            enc_list(&[]), // empty access list
        ]
    }

    /// The signed tx bytes: 0x02 || rlp([...unsigned, y_parity, r, s]).
    pub fn encode_signed(&self) -> Vec<u8> {
        let mut fields = self.fields_unsigned();
        fields.push(enc_uint(&BigInt::from(self.y_parity)));
        fields.push(enc_uint(&self.r));
        fields.push(enc_uint(&self.s));
        let mut out = vec![0x02u8];
        out.extend_from_slice(&enc_list(&fields));
        out
    }

    /// The signing preimage: 0x02 || rlp([...unsigned]).
    pub fn signing_preimage(&self) -> Vec<u8> {
        let mut out = vec![0x02u8];
        out.extend_from_slice(&enc_list(&self.fields_unsigned()));
        out
    }
}

/// Decode a signed EIP-1559 tx from its bytes (0x02 || rlp-list).
pub fn decode_eip1559(bytes: &[u8]) -> Eip1559Tx {
    assert_eq!(bytes[0], 0x02, "not an EIP-1559 tx");
    let (rlp, _) = decode_at(bytes, 1);
    let f = rlp.as_list();
    let mut to = [0u8; 20];
    let tob = f[5].as_bytes();
    to[20 - tob.len()..].copy_from_slice(tob);
    Eip1559Tx {
        chain_id: be_to_usize(f[0].as_bytes()) as u64,
        nonce: be_to_usize(f[1].as_bytes()) as u64,
        max_priority_fee: f[2].as_uint(),
        max_fee: f[3].as_uint(),
        gas_limit: be_to_usize(f[4].as_bytes()) as u64,
        to,
        value: f[6].as_uint(),
        data: f[7].as_bytes().to_vec(),
        y_parity: be_to_usize(f[9].as_bytes()) as u8,
        r: f[10].as_uint(),
        s: f[11].as_uint(),
    }
}

/// In-circuit tx handling (R6b): committed input = the signing-preimage bytes
/// (+ y_parity, r, s). z = keccak(preimage) computed directly; sender recovered
/// via the R5 ecrecover. Deterministic; the committed preimage is the sole
/// source for both z and the tx fields (no desync possible).
pub mod circuit {
    use crate::batch_keccak::keccak256_single_block;
    use crate::secp256k1::circuit::ecrecover_pubkey;
    use crate::u256::BITS;
    use expander_compiler::frontend::*;

    /// z = keccak256(committed preimage) as 256 LSB-first-per-byte output bits.
    pub fn preimage_z<C: Config>(api: &mut impl RootAPI<C>, preimage_bits: &[Variable]) -> Vec<Variable> {
        keccak256_single_block(api, preimage_bits)
    }

    /// keccak output bits (LSB-first per byte) reinterpreted as a u256 LE bit
    /// vector (bit i = value 2^i). keccak byte b occupies output positions
    /// [b*8..b*8+8]; as a big-endian 32-byte integer, byte b has place value
    /// 2^(8*(31-b)). So u256 bit (8*(31-b)+j) = keccak_out[b*8+j].
    pub fn keccak_out_to_u256(h: &[Variable]) -> Vec<Variable> {
        let mut u = vec![h[0]; BITS];
        for b in 0..32 {
            for j in 0..8 {
                u[8 * (31 - b) + j] = h[b * 8 + j];
            }
        }
        u
    }

    /// A u256 LE bit vector (bit i = 2^i) laid out as big-endian message bytes,
    /// LSB-first per byte (keccak input order). Byte m (0..32) of the BE
    /// encoding = u256 byte (31-m); its bit j = u[(31-m)*8 + j].
    pub fn u256_to_be_msg_bits(u: &[Variable]) -> Vec<Variable> {
        let mut out = Vec::with_capacity(256);
        for m in 0..32 {
            for j in 0..8 {
                out.push(u[(31 - m) * 8 + j]);
            }
        }
        out
    }

    /// Full tx -> Ethereum address (recovered sender). Returns 160 address bits
    /// (keccak(pubkey)[12..32], LSB-first per byte). HEAVY (ecrecover).
    pub fn tx_to_address<C: Config>(
        api: &mut impl RootAPI<C>,
        preimage_bits: &[Variable],
        y_parity: Variable,
        r: &[Variable],
        s: &[Variable],
    ) -> Vec<Variable> {
        let z = preimage_z(api, preimage_bits);
        let z_u = keccak_out_to_u256(&z);
        let (qx, qy) = ecrecover_pubkey(api, r, s, y_parity, &z_u);
        // pubkey message = qx (BE 32B) || qy (BE 32B), LSB-first per byte.
        let mut msg = u256_to_be_msg_bits(&qx);
        msg.extend(u256_to_be_msg_bits(&qy));
        let h = keccak256_single_block(api, &msg); // 64-byte pubkey, single block
        // address = keccak(pubkey)[12..32] => output bits [12*8 .. 32*8].
        h[12 * 8..32 * 8].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secp256k1::{self, native};
    use num_bigint::BigInt;
    use num_traits::Num;
    use tiny_keccak::Hasher;

    fn keccak(b: &[u8]) -> [u8; 32] {
        let mut h = tiny_keccak::Keccak::v256();
        h.update(b);
        let mut o = [0u8; 32];
        h.finalize(&mut o);
        o
    }

    fn modn(x: BigInt) -> BigInt {
        let n = secp256k1::n();
        ((x % &n) + &n) % &n
    }

    use crate::u256::bigint_to_bits;
    use expander_compiler::frontend::*;

    const PRE_LEN: usize = 49; // demo transfer tx signing-preimage length

    fn bytes_bits(b: &[u8]) -> Vec<GF2> {
        let mut v = Vec::with_capacity(b.len() * 8);
        for &byte in b {
            for j in 0..8 {
                v.push((((byte >> j) & 1) as u32).into());
            }
        }
        v
    }
    fn u256_bits(x: &BigInt) -> Vec<GF2> {
        bigint_to_bits(x, 256).into_iter().map(|b| (b as u32).into()).collect()
    }

    /// Build + sign the demo transfer tx; return (tx, signer_address, z, preimage).
    fn demo_tx() -> (Eip1559Tx, Vec<u8>, BigInt, Vec<u8>) {
        let d = BigInt::from_str_radix("c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721", 16).unwrap();
        let pubq = native::scalar_mul(&d, &native::generator()).0.unwrap();
        let mut pub64 = [0u8; 64];
        pub64[32 - pubq.0.to_bytes_be().1.len()..32].copy_from_slice(&pubq.0.to_bytes_be().1);
        pub64[64 - pubq.1.to_bytes_be().1.len()..64].copy_from_slice(&pubq.1.to_bytes_be().1);
        let signer_addr = keccak(&pub64)[12..].to_vec();
        let mut tx = Eip1559Tx {
            chain_id: 1, nonce: 7,
            max_priority_fee: BigInt::from(2_000_000_000u64),
            max_fee: BigInt::from(20_000_000_000u64),
            gas_limit: 21000, to: [0xAB; 20],
            value: BigInt::from(1_000_000_000_000_000u64),
            data: vec![], y_parity: 0, r: BigInt::from(0), s: BigInt::from(0),
        };
        let pre = tx.signing_preimage();
        let z = modn(BigInt::from_bytes_be(num_bigint::Sign::Plus, &keccak(&pre)));
        let k = BigInt::from_str_radix("49a0d7b786ec9cde0d0721d72804befd06571c974b191efb42ecf322ba9ddd9a", 16).unwrap();
        let rpt = native::scalar_mul(&k, &native::generator()).0.unwrap();
        let r = modn(rpt.0.clone());
        let k_inv = k.modpow(&(secp256k1::n() - BigInt::from(2u32)), &secp256k1::n());
        let s = modn(&k_inv * (&z + &r * &d));
        tx.r = r; tx.s = s;
        tx.y_parity = (&rpt.1 & BigInt::from(1u32)).to_bytes_be().1.first().copied().unwrap_or(0) & 1;
        (tx, signer_addr, z, pre)
    }

    // Cheap: z = keccak(committed preimage) matches native. Verifies the
    // committed-tx -> signing-hash path in-circuit (single keccak block).
    declare_circuit!(TxZCircuit {
        pre: [Variable; PRE_LEN * 8],
        z: [PublicVariable; 256],
    });
    impl Define<GF2Config> for TxZCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let z = super::circuit::preimage_z(api, &self.pre.to_vec());
            for i in 0..256 {
                api.assert_is_equal(z[i], self.z[i]);
            }
        }
    }

    #[test]
    fn circuit_tx_z_matches_native() {
        let (_tx, _addr, _zmod, pre) = demo_tx();
        assert_eq!(pre.len(), PRE_LEN);
        let zraw = keccak(&pre); // raw 32-byte keccak (before mod n)
        let CompileResult { witness_solver, layered_circuit } =
            compile(&TxZCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = TxZCircuit::<GF2>::default();
        asg.pre.copy_from_slice(&bytes_bits(&pre));
        asg.z.copy_from_slice(&bytes_bits(&zraw));
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit z != keccak(preimage)");
    }

    // Full: committed tx preimage + (yparity,r,s) -> recovered sender address,
    // matches native. HEAVY (ecrecover ~hours). #[ignore]; run on demand.
    declare_circuit!(TxAddrCircuit {
        pre: [Variable; PRE_LEN * 8],
        yparity: Variable,
        r: [Variable; 256],
        s: [Variable; 256],
        addr: [PublicVariable; 160],
    });
    impl Define<GF2Config> for TxAddrCircuit<Variable> {
        fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
            let a = super::circuit::tx_to_address(api, &self.pre.to_vec(), self.yparity, &self.r.to_vec(), &self.s.to_vec());
            for i in 0..160 {
                api.assert_is_equal(a[i], self.addr[i]);
            }
        }
    }

    #[test]
    #[ignore = "full in-circuit tx->ecrecover->address: ~hours; run on demand"]
    fn circuit_tx_to_address_matches_native() {
        let (tx, addr, _zmod, pre) = demo_tx();
        let CompileResult { witness_solver, layered_circuit } =
            compile(&TxAddrCircuit::default(), CompileOptions::default()).unwrap();
        let mut asg = TxAddrCircuit::<GF2>::default();
        asg.pre.copy_from_slice(&bytes_bits(&pre));
        asg.yparity = (tx.y_parity as u32).into();
        asg.r.copy_from_slice(&u256_bits(&tx.r));
        asg.s.copy_from_slice(&u256_bits(&tx.s));
        asg.addr.copy_from_slice(&bytes_bits(&addr));
        let w = witness_solver.solve_witnesses(&vec![asg; 1]).unwrap();
        assert!(layered_circuit.run(&w).iter().all(|x| *x), "in-circuit address != native sender");
    }

    #[test]
    fn eip1559_roundtrip_and_sender_recovery() {
        // Signer key d, address = keccak(pubkey)[12:].
        let d = BigInt::from_str_radix("c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721", 16).unwrap();
        let pubq = native::scalar_mul(&d, &native::generator()).0.unwrap();
        let mut pub64 = [0u8; 64];
        pub64[32 - pubq.0.to_bytes_be().1.len()..32].copy_from_slice(&pubq.0.to_bytes_be().1);
        pub64[64 - pubq.1.to_bytes_be().1.len()..64].copy_from_slice(&pubq.1.to_bytes_be().1);
        let signer_addr = keccak(&pub64)[12..].to_vec();

        // Build an unsigned transfer tx.
        let mut tx = Eip1559Tx {
            chain_id: 1,
            nonce: 7,
            max_priority_fee: BigInt::from(2_000_000_000u64),
            max_fee: BigInt::from(20_000_000_000u64),
            gas_limit: 21000,
            to: [0xAB; 20],
            value: BigInt::from(1_000_000_000_000_000u64),
            data: vec![],
            y_parity: 0,
            r: BigInt::from(0),
            s: BigInt::from(0),
        };

        // Sign: z = keccak(signing_preimage) mod n; deterministic test nonce k.
        let z = modn(BigInt::from_bytes_be(num_bigint::Sign::Plus, &keccak(&tx.signing_preimage())));
        let k = BigInt::from_str_radix("49a0d7b786ec9cde0d0721d72804befd06571c974b191efb42ecf322ba9ddd9a", 16).unwrap();
        let rpt = native::scalar_mul(&k, &native::generator()).0.unwrap();
        let r = modn(rpt.0.clone());
        let k_inv = k.modpow(&(secp256k1::n() - BigInt::from(2u32)), &secp256k1::n());
        let s = modn(&k_inv * (&z + &r * &d));
        let y_parity = (&rpt.1 & BigInt::from(1u32)).to_bytes_be().1.first().copied().unwrap_or(0) & 1;
        tx.r = r;
        tx.s = s;
        tx.y_parity = y_parity;

        // Encode -> decode roundtrip.
        let bytes = tx.encode_signed();
        let pre = tx.signing_preimage();
        println!("R6 sizes: signed_len={} preimage_len={}", bytes.len(), pre.len());
        println!("R6 preimage hex={}", pre.iter().map(|b| format!("{:02x}", b)).collect::<String>());
        let dec = decode_eip1559(&bytes);
        assert_eq!(dec.nonce, 7);
        assert_eq!(dec.value, BigInt::from(1_000_000_000_000_000u64));
        assert_eq!(dec.to, [0xAB; 20]);
        assert_eq!(dec.r, tx.r);
        assert_eq!(dec.s, tx.s);
        assert_eq!(dec.y_parity, tx.y_parity);

        // Recompute z from the DECODED tx and recover the sender.
        let z2 = modn(BigInt::from_bytes_be(num_bigint::Sign::Plus, &keccak(&dec.signing_preimage())));
        assert_eq!(z2, z, "signing hash from decoded tx must match");
        let rec = native::recover(&dec.r, &dec.s, dec.y_parity, &z2).expect("recover");
        let rec_addr = keccak(&rec)[12..].to_vec();
        assert_eq!(rec_addr, signer_addr, "recovered sender != signer address");
        println!("R6 native: recovered sender 0x{}", rec_addr.iter().map(|b| format!("{:02x}", b)).collect::<String>());
    }
}
