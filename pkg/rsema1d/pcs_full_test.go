package rsema1d

import (
	"crypto/sha256"
	"encoding/binary"
	"testing"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
)

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

func TestFullPointEvaluationRoundTrip(t *testing.T) {
	cfg := &Config{K: 8, N: 8, WorkerCount: 2}
	const rowBytes = 128 // numSymbols = 64 = 2^6 columns
	const numSymbols = rowBytes / 2
	logCols := 6

	rows := buildSquare(t, cfg.K, cfg.N, rowBytes, 0xC3)
	coder, err := NewCoder(cfg)
	if err != nil {
		t.Fatal(err)
	}
	sc, err := coder.EncodeStructured(rows)
	if err != nil {
		t.Fatal(err)
	}
	commit := sc.Commitment()

	rCol := randChallenges("FULL-COL", logCols)

	// Test the whole square and a proper sub-range.
	for _, r := range []RowRange{{Start: 0, Len: 8}, {Start: 4, Len: 4}, {Start: 0, Len: 1}} {
		logRows := 0
		for (1 << logRows) < r.Len {
			logRows++
		}
		rRow := randChallenges("FULL-ROW", logRows)

		proof, err := sc.OpenEvaluationAtFull(r, rCol, rRow, cfg.K+cfg.N)
		if err != nil {
			t.Fatalf("range %+v: open failed: %v", r, err)
		}
		val, err := VerifyEvaluationAtFull(cfg, commit, proof, rCol, rRow)
		if err != nil {
			t.Fatalf("range %+v: verify failed: %v", r, err)
		}
		if !field.Equal128(val, proof.Value) {
			t.Fatalf("range %+v: returned value != proof value", r)
		}
		want := evalMatrixFull(sc.ed.rows, r, rCol, rRow, numSymbols)
		if !field.Equal128(val, want) {
			t.Fatalf("range %+v: verified value %v != independent MLE %v", r, val, want)
		}

		// YrPrime must genuinely be the per-row evals at the supplied rCol.
		for j := 0; j < cfg.K; j++ {
			ref := rlc.EvalMultilinearRow(sc.ed.rows[j], rCol)
			if !field.Equal128(proof.YrPrime[j], ref) {
				t.Fatalf("YrPrime[%d] != EvalMultilinearRow at rCol", j)
			}
		}
	}
}

func TestFullPointEvaluationRejectsTampering(t *testing.T) {
	cfg := &Config{K: 8, N: 8, WorkerCount: 2}
	const rowBytes = 128
	logCols := 6
	logRows := 3

	rows := buildSquare(t, cfg.K, cfg.N, rowBytes, 0x2E)
	coder, _ := NewCoder(cfg)
	sc, err := coder.EncodeStructured(rows)
	if err != nil {
		t.Fatal(err)
	}
	commit := sc.Commitment()
	r := RowRange{Start: 0, Len: 8}
	rCol := randChallenges("TAMP-COL", logCols)
	rRow := randChallenges("TAMP-ROW", logRows)

	mustReject := func(name string, mutate func(p *EvalProofFull), vCol, vRow []field.GF128) {
		p, err := sc.OpenEvaluationAtFull(r, rCol, rRow, cfg.K+cfg.N)
		if err != nil {
			t.Fatalf("%s: open failed: %v", name, err)
		}
		mutate(p)
		if _, err := VerifyEvaluationAtFull(cfg, commit, p, vCol, vRow); err == nil {
			t.Fatalf("%s: expected rejection but verify accepted", name)
		}
	}

	// Sanity: an untouched proof verifies.
	base, _ := sc.OpenEvaluationAtFull(r, rCol, rRow, cfg.K+cfg.N)
	if _, err := VerifyEvaluationAtFull(cfg, commit, base, rCol, rRow); err != nil {
		t.Fatalf("honest proof must verify: %v", err)
	}

	mustReject("tampered value", func(p *EvalProofFull) { p.Value[0] ^= 0x01 }, rCol, rRow)
	mustReject("tampered sampled row", func(p *EvalProofFull) {
		bad := make([]byte, len(p.SampledRows[0].Row))
		copy(bad, p.SampledRows[0].Row)
		bad[0] ^= 0xFF
		p.SampledRows[0].Row = bad
	}, rCol, rRow)
	mustReject("tampered committed Yr", func(p *EvalProofFull) { p.Yr[1][0] ^= 0x01 }, rCol, rRow)
	mustReject("tampered YrPrime", func(p *EvalProofFull) { p.YrPrime[1][0] ^= 0x01 }, rCol, rRow)

	// Wrong column point: YrPrime was built for rCol; verifying against a
	// different column point breaks the (b) consistency check.
	wrongCol := make([]field.GF128, logCols)
	copy(wrongCol, rCol)
	wrongCol[0] = field.Add128(wrongCol[0], field.One())
	mustReject("wrong rCol", func(p *EvalProofFull) {}, wrongCol, rRow)

	// Wrong row point: the recomputed value no longer matches the claim.
	wrongRow := make([]field.GF128, logRows)
	copy(wrongRow, rRow)
	wrongRow[0] = field.Add128(wrongRow[0], field.One())
	mustReject("wrong rRow", func(p *EvalProofFull) {}, rCol, wrongRow)
}
