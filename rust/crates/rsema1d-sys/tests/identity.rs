//! Commitment-identity gate + open/verify round-trip over the real Go PCS.
//!
//! The reference hex constants below are produced by the pure-Go oracle
//! `TestOracleReference` / `TestOracleReference2` in
//! `pkg/rsema1d/cshim/oracle_test.go` (run: `go test -v -run TestOracleReference
//! ./pkg/rsema1d/cshim`). Asserting the FFI output equals them proves the Rust
//! reverse-FFI commit is byte-identical to what pure Go produces.

use rsema1d_sys::{commit, open_at, verify_at, RowRange};

/// Pure-Go ORIGINAL `Encode` commitment for K=4, N=4, rowBytes=64 (row i's last
/// byte = i+1). From oracle_test.go: ORACLE_COMMITMENT. This is byte-identical
/// to `go run ./pkg/rsema1d/cmd/testvectors` vector 1 — the canonical DA/spec
/// commitment (ENCODE-ONCE).
const ORACLE_COMMITMENT_1: &str =
    "f57fdff87d54f71bc0c860808b046356c8d4850e67b923e08411208df08cb5ab";
/// Pure-Go opened+verified value (row-only legacy opening) at the fixed 2-GF128
/// point.
const ORACLE_VALUE_1: &str = "f4b02856ee9731834c337541fa2410fb";
/// Pure-Go ORIGINAL `Encode` commitment for K=4, N=12, rowBytes=256 (=
/// testvectors vector 2).
const ORACLE_COMMITMENT_2: &str =
    "8ac46440862f280346635eee5075f81ff04b659fb7a86c1e25a28f5f71c3f97e";

/// Builds K+N rows of `row_bytes`: first K have last byte = i+1, parity zeroed.
fn build_square(k: usize, n: usize, row_bytes: usize) -> Vec<Vec<u8>> {
    let mut rows = vec![vec![0u8; row_bytes]; k + n];
    for i in 0..k {
        rows[i][row_bytes - 1] = (i + 1) as u8;
    }
    rows
}

/// The fixed evaluation point: 2 GF128 challenges {1..8} and {9..16}, each an
/// 8-component GF(2^16) vector serialized little-endian (16 bytes each). Matches
/// oraclePoint() in oracle_test.go byte-for-byte.
fn oracle_point() -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    for chal in [[1u16, 2, 3, 4, 5, 6, 7, 8], [9u16, 10, 11, 12, 13, 14, 15, 16]] {
        for v in chal {
            out.push((v & 0xff) as u8);
            out.push((v >> 8) as u8);
        }
    }
    out
}

fn as_slices(rows: &[Vec<u8>]) -> Vec<&[u8]> {
    rows.iter().map(|r| r.as_slice()).collect()
}

#[test]
fn commitment_identity_case1() {
    let rows = build_square(4, 4, 64);
    let (commitment, _handle) = commit(4, 4, &as_slices(&rows)).expect("commit failed");
    assert_eq!(
        hex::encode(commitment),
        ORACLE_COMMITMENT_1,
        "FFI commitment must equal pure-Go Encode (testvectors) commitment"
    );
}

#[test]
fn commitment_identity_case2() {
    let rows = build_square(4, 12, 256);
    let (commitment, _handle) = commit(4, 12, &as_slices(&rows)).expect("commit failed");
    assert_eq!(hex::encode(commitment), ORACLE_COMMITMENT_2);
}

#[test]
fn open_verify_round_trip() {
    let rows = build_square(4, 4, 64);
    let (commitment, handle) = commit(4, 4, &as_slices(&rows)).expect("commit failed");
    assert_eq!(hex::encode(commitment), ORACLE_COMMITMENT_1);

    let point = oracle_point();
    let range = RowRange::new(0, 4);
    let proof = open_at(&handle, range, &point, 8).expect("open_at failed");
    assert!(!proof.is_empty(), "serialized proof must be non-empty");

    let value = verify_at(4, 4, &commitment, &proof, &point).expect("verify_at rejected the proof");
    assert_eq!(
        hex::encode(value),
        ORACLE_VALUE_1,
        "FFI verify value must equal Go's VerifyEvaluationAt value"
    );
}

#[test]
fn verify_rejects_tampered_proof() {
    let rows = build_square(4, 4, 64);
    let (commitment, handle) = commit(4, 4, &as_slices(&rows)).expect("commit failed");
    let point = oracle_point();
    let mut proof = open_at(&handle, RowRange::new(0, 4), &point, 8).expect("open_at failed");

    // Flip a byte in the serialized Value field (offset 8: after Range's two u32s).
    proof[8] ^= 0xFF;
    assert!(
        verify_at(4, 4, &commitment, &proof, &point).is_none(),
        "tampered proof must be rejected"
    );
}
