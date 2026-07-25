package rsema1d

import (
	"crypto/sha256"
	"encoding/binary"
	"testing"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
)

// buildSquare fills the K original rows of a K+N square with deterministic
// pseudo-random bytes derived from seed, leaving the parity rows zeroed (the
// contract Coder.Encode expects).
func buildSquare(t *testing.T, k, n, rowBytes int, seed byte) [][]byte {
	t.Helper()
	rows := make([][]byte, k+n)
	for i := range rows {
		rows[i] = make([]byte, rowBytes)
	}
	for i := 0; i < k; i++ {
		var ctr [8]byte
		ctr[0] = seed
		ctr[1] = byte(i)
		for off := 0; off < rowBytes; off += 32 {
			binary.LittleEndian.PutUint32(ctr[4:], uint32(off))
			d := sha256.Sum256(ctr[:])
			copy(rows[i][off:], d[:])
		}
	}
	return rows
}

// randChallenges deterministically derives n GF(2^128) challenges from a label,
// standing in for a protocol transcript's full evaluation point.
func randChallenges(label string, n int) []field.GF128 {
	out := make([]field.GF128, n)
	for i := 0; i < n; i++ {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], uint32(i))
		out[i] = field.HashToGF128(sha256.Sum256(append([]byte(label), b[:]...)))
	}
	return out
}

// evalMatrixFull is an INDEPENDENT reference for the full-point multilinear
// evaluation of the sub-matrix X_range. It uses the explicit eq-weight double
// sum — a different code path from foldGF128 / TensorCoefficients — so agreement
// is a genuine cross-check of the value produced by the PCS opening.
//
//	MLE_{X_range}(rRow, rCol) = Σ_{j',i} X[range.Start+j'][i]
//	                              * eqW(rRow, j') * eqW(rCol, i)
//
// eqW(chal, idx) = Π_b [ chal[b] if bit (len-1-b) of idx == 1 else (1 - chal[b]) ],
// i.e. chal[0] binds the most-significant index bit — the tensor convention.
func evalMatrixFull(rows [][]byte, r RowRange, rCol, rRow []field.GF128, numSymbols int) field.GF128 {
	eqW := func(chal []field.GF128, idx int) field.GF128 {
		w := field.One()
		m := len(chal)
		for b := 0; b < m; b++ {
			bit := (idx >> (m - 1 - b)) & 1
			if bit == 1 {
				w = field.MulFull(w, chal[b])
			} else {
				w = field.MulFull(w, field.Add128(field.One(), chal[b])) // 1 - chal[b]
			}
		}
		return w
	}

	acc := field.Zero()
	for jp := 0; jp < r.Len; jp++ {
		row := rows[r.Start+jp]
		wr := eqW(rRow, jp)
		for i := 0; i < numSymbols; i++ {
			sym := field.GF16FromLeopard(row, i)
			symGF := field.GF128{sym}
			term := field.MulFull(symGF, field.MulFull(wr, eqW(rCol, i)))
			acc = field.Add128(acc, term)
		}
	}
	return acc
}

// TestRowRangeValidation checks the per-namespace alignment invariants.
func TestRowRangeValidation(t *testing.T) {
	k := 16
	ok := []RowRange{{0, 1}, {0, 16}, {4, 4}, {8, 8}, {12, 4}}
	for _, r := range ok {
		if err := r.validate(k); err != nil {
			t.Errorf("range %+v should be valid: %v", r, err)
		}
	}
	bad := []RowRange{{0, 3}, {2, 4}, {4, 8}, {8, 16}, {-1, 4}, {0, 0}}
	for _, r := range bad {
		if err := r.validate(k); err == nil {
			t.Errorf("range %+v should be invalid", r)
		}
	}
}
