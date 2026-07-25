package main

import (
	"encoding/hex"
	"testing"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d"
)

// oracleRows builds the K=4,N=4,rowBytes=64 square used by cmd/testvectors
// (row i has its last byte = i+1, everything else zero, parity rows zeroed).
func oracleRows() [][]byte {
	const k, n, rowBytes = 4, 4, 64
	rows := make([][]byte, k+n)
	for i := range rows {
		rows[i] = make([]byte, rowBytes)
	}
	for i := 0; i < k; i++ {
		rows[i][rowBytes-1] = byte(i + 1)
	}
	return rows
}

// TestOracleReference prints the pure-Go commitment for the fixed vector. Run
// with -v; the Rust commitment-identity gate (rsema1d-sys/tests/identity.rs)
// asserts the FFI reproduces these exact bytes.
//
//	go test -v -run TestOracleReference ./pkg/rsema1d/cshim
func TestOracleReference(t *testing.T) {
	cfg := &rsema1d.Config{K: 4, N: 4, WorkerCount: 1}
	coder, err := rsema1d.NewCoder(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ed, err := coder.Encode(oracleRows())
	if err != nil {
		t.Fatal(err)
	}
	commitment := ed.Commitment()
	t.Logf("ORACLE_COMMITMENT=%s", hex.EncodeToString(commitment[:]))
}

// TestOracleReference2 is a second K/N/rowBytes case (matching cmd/testvectors
// vector 2 dimensions: K=4, N=12, rowBytes=256) for the commitment-identity
// gate.
func TestOracleReference2(t *testing.T) {
	const k, n, rowBytes = 4, 12, 256
	rows := make([][]byte, k+n)
	for i := range rows {
		rows[i] = make([]byte, rowBytes)
	}
	for i := 0; i < k; i++ {
		rows[i][rowBytes-1] = byte(i + 1)
	}
	cfg := &rsema1d.Config{K: k, N: n, WorkerCount: 1}
	coder, err := rsema1d.NewCoder(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ed, err := coder.Encode(rows)
	if err != nil {
		t.Fatal(err)
	}
	commitment := ed.Commitment()
	t.Logf("ORACLE2_COMMITMENT=%s", hex.EncodeToString(commitment[:]))
}
