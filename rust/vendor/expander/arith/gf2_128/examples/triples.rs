// Generates random (a, b, a*b) triples using the REAL Expander GF2_128 field,
// printing each as three space-separated 16-byte little-endian hex strings.
// Used to lock the Go reimplementation's representation against this crate.
use arith::Field;
use gf2_128::GF2_128;
use serdes::ExpSerde;
use rand::RngCore;

fn to_hex(x: &GF2_128) -> String {
    let mut buf = Vec::new();
    x.serialize_into(&mut buf).unwrap();
    buf.iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    let mut rng = rand::thread_rng();
    // A few fixed edge cases first.
    let fixed: [( [u8;16], [u8;16] ); 3] = [
        ([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0xff;16]),
        ([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
        ([0xde,0xad,0xbe,0xef,0,0,0,0,0,0,0,0,0,0,0,0], [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    ];
    for (ab, bb) in fixed.iter() {
        let a = GF2_128::from_uniform_bytes(ab);
        let b = GF2_128::from_uniform_bytes(bb);
        let p = a * b;
        println!("{} {} {}", to_hex(&a), to_hex(&b), to_hex(&p));
    }
    for _ in 0..2000 {
        let mut ab = [0u8; 16];
        let mut bb = [0u8; 16];
        rng.fill_bytes(&mut ab);
        rng.fill_bytes(&mut bb);
        let a = GF2_128::from_uniform_bytes(&ab);
        let b = GF2_128::from_uniform_bytes(&bb);
        let p = a * b;
        println!("{} {} {}", to_hex(&a), to_hex(&b), to_hex(&p));
    }
}
