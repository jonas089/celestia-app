//! Commitment-identity gate over the real Go PCS.
//!
//! The reference hex constants below are produced by the pure-Go oracle
//! `TestOracleReference` / `TestOracleReference2` in
//! `pkg/rsema1d/cshim/oracle_test.go` (run: `go test -v -run TestOracleReference
//! ./pkg/rsema1d/cshim`). Asserting the FFI output equals them proves the Rust
//! reverse-FFI commit is byte-identical to what pure Go produces.

use rsema1d_sys::commit;

/// Pure-Go ORIGINAL `Encode` commitment for K=4, N=4, rowBytes=64 (row i's last
/// byte = i+1). From oracle_test.go: ORACLE_COMMITMENT. This is byte-identical
/// to `go run ./pkg/rsema1d/cmd/testvectors` vector 1 — the canonical DA/spec
/// commitment (ENCODE-ONCE).
const ORACLE_COMMITMENT_1: &str =
    "f57fdff87d54f71bc0c860808b046356c8d4850e67b923e08411208df08cb5ab";
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
