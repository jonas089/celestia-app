package main

import (
	"encoding/hex"
	"testing"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
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

// oraclePoint is the fixed evaluation point (2 GF128 challenges, matching
// log2(rangeLen=4)) that the Rust round-trip test reuses byte-for-byte.
func oraclePoint() []field.GF128 {
	return []field.GF128{
		{1, 2, 3, 4, 5, 6, 7, 8},
		{9, 10, 11, 12, 13, 14, 15, 16},
	}
}

// TestOracleReference prints the pure-Go commitment and the opened/verified
// evaluation value for the fixed vector. Run with -v; the Rust FFI round-trip
// test asserts it reproduces these exact bytes.
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

	r := rsema1d.RowRange{Start: 0, Len: 4}
	const sampleCount = 8
	proof, err := ed.OpenAtLegacy(r, oraclePoint(), sampleCount)
	if err != nil {
		t.Fatal(err)
	}
	value, err := rsema1d.VerifyAtLegacy(cfg, commitment, proof, oraclePoint())
	if err != nil {
		t.Fatalf("pure-Go verify failed: %v", err)
	}
	var vbuf [field.GF128Size]byte
	field.EncodeGF128(vbuf[:], value)
	t.Logf("ORACLE_VALUE=%s", hex.EncodeToString(vbuf[:]))
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
