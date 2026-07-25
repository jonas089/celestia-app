//! Cross-process hand-off FFI round-trip: encode_extended (DA side) ->
//! load_extended (prover side) must reproduce the exact commitment WITHOUT any
//! extra RS-encode, and the loaded handle must open+verify identically to a
//! freshly committed one.

use rsema1d_sys::{
    commit, encode_call_count, encode_extended, load_extended, open_at_full, verify_at_full,
    RowRange,
};

/// K=4, N=4, rowBytes=64: first K rows have last byte = i+1, parity zeroed.
fn build_square() -> Vec<Vec<u8>> {
    let mut rows = vec![vec![0u8; 64]; 8];
    for i in 0..4 {
        rows[i][63] = (i + 1) as u8;
    }
    rows
}

/// A full point: rCol = 5 GF128 (log2(32)), rRow = 2 GF128 (log2(4)), each 16 LE bytes.
fn full_point() -> (Vec<u8>, Vec<u8>) {
    let mut rcol = vec![0u8; 5 * 16];
    for (c, chunk) in rcol.chunks_mut(16).enumerate() {
        chunk[0] = (c + 1) as u8;
    }
    let mut rrow = vec![0u8; 2 * 16];
    for (c, chunk) in rrow.chunks_mut(16).enumerate() {
        chunk[0] = (10 + c) as u8;
    }
    (rcol, rrow)
}

#[test]
fn encode_then_load_matches_and_no_reencode() {
    let rows = build_square();
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();

    // DA side: one encode.
    let e0 = encode_call_count();
    let (root_enc, extended) = encode_extended(4, 4, &refs).expect("encode_extended");
    let e1 = encode_call_count();
    assert_eq!(e1 - e0, 1, "encode_extended must RS-encode exactly once");

    // Independent reference commit of the same rows (must match).
    let (root_commit, _h) = commit(4, 4, &refs).expect("commit");
    assert_eq!(root_enc, root_commit, "encode_extended root != commit root");

    // Prover side: load, NO encode.
    let e2 = encode_call_count();
    let (root_load, handle) = load_extended(&extended).expect("load_extended");
    let e3 = encode_call_count();
    assert_eq!(e3 - e2, 0, "load_extended must NOT RS-encode");
    assert_eq!(root_load, root_enc, "loaded root != DA root");

    // The reconstructed handle opens+verifies identically.
    let (rcol, rrow) = full_point();
    let proof = open_at_full(&handle, RowRange::new(0, 4), &rcol, &rrow, u32::MAX)
        .expect("open_at_full on loaded handle");
    let val = verify_at_full(4, 4, &root_load, &proof, &rcol, &rrow)
        .expect("verify_at_full on loaded proof");
    // Cross-check against a value opened from a freshly committed handle.
    let (_r, fresh) = commit(4, 4, &refs).expect("commit2");
    let proof2 = open_at_full(&fresh, RowRange::new(0, 4), &rcol, &rrow, u32::MAX).unwrap();
    let val2 = verify_at_full(4, 4, &root_enc, &proof2, &rcol, &rrow).unwrap();
    assert_eq!(val, val2, "loaded-handle opening value != fresh-handle value");
}
