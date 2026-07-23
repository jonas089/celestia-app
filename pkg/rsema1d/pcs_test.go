package rsema1d

import (
	"crypto/sha256"
	"encoding/binary"
	"testing"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
)

// buildSquare returns K+N rows of rowBytes each: the first K filled
// deterministically from seed, the parity rows zeroed (ready for Encode).
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

// TestSubsetEvaluationRoundTrip is the end-to-end §5 accidental-computer proof:
// commit a shared square with the tensor RLC, open one rollup's aligned
// row-range as a multilinear evaluation, verify it against only the
// commitment, and cross-check the value against the independent multilinear
// evaluator so we know it is the real evaluation of that rollup's sub-matrix.
func TestSubsetEvaluationRoundTrip(t *testing.T) {
	cfg := &Config{K: 8, N: 8, WorkerCount: 2}
	const rowBytes = 128 // 64 symbols = 2^6 columns
	rows := buildSquare(t, cfg.K, cfg.N, rowBytes, 0xAB)

	coder, err := NewCoder(cfg)
	if err != nil {
		t.Fatal(err)
	}
	sc, err := coder.EncodeStructured(rows)
	if err != nil {
		t.Fatal(err)
	}

	// yr must be the multilinear partial evaluations of each row.
	for j := 0; j < cfg.K; j++ {
		want := rlc.EvalMultilinearRow(sc.ed.rows[j], sc.colChalls)
		if !field.Equal128(sc.ed.rlc[j], want) {
			t.Fatalf("row %d: committed yr != multilinear partial eval", j)
		}
	}

	// Rollup owns rows [4,8): aligned (4 % 4 == 0), power-of-two length.
	r := RowRange{Start: 4, Len: 4}
	proof, err := sc.OpenEvaluation(r, 8)
	if err != nil {
		t.Fatal(err)
	}

	val, err := VerifyEvaluation(cfg, sc.Commitment(), proof)
	if err != nil {
		t.Fatalf("verify failed: %v", err)
	}
	if !field.Equal128(val, proof.Value) {
		t.Fatalf("returned value != proof value")
	}

	// Independent cross-check: fold the rollup rows' own multilinear partial
	// evaluations with the re-derived row challenges.
	logRows := 2
	rRow := deriveEvalChallenges(sc.Commitment(), r, logRows)
	indep := make(rlc.Vector, r.Len)
	for i := 0; i < r.Len; i++ {
		indep[i] = rlc.EvalMultilinearRow(sc.ed.rows[r.Start+i], sc.colChalls)
	}
	expected := foldGF128(indep, rRow)
	if !field.Equal128(expected, val) {
		t.Fatalf("verified value %v != independent multilinear eval %v", val, expected)
	}
}

func TestSubsetEvaluationRejectsTampering(t *testing.T) {
	cfg := &Config{K: 8, N: 8, WorkerCount: 2}
	const rowBytes = 128
	rows := buildSquare(t, cfg.K, cfg.N, rowBytes, 0x11)
	coder, _ := NewCoder(cfg)
	sc, err := coder.EncodeStructured(rows)
	if err != nil {
		t.Fatal(err)
	}
	r := RowRange{Start: 0, Len: 4}
	commit := sc.Commitment()

	// (a) tampered sampled row (replace with a modified copy so ed is untouched).
	proof, _ := sc.OpenEvaluation(r, 8)
	bad := make([]byte, rowBytes)
	copy(bad, proof.SampledRows[0].Row)
	bad[0] ^= 0xFF
	proof.SampledRows[0].Row = bad
	if _, err := VerifyEvaluation(cfg, commit, proof); err == nil {
		t.Fatal("expected failure on tampered sampled row")
	}

	// (b) tampered claimed value.
	proof, _ = sc.OpenEvaluation(r, 8)
	proof.Value[0] ^= 0x01
	if _, err := VerifyEvaluation(cfg, commit, proof); err == nil {
		t.Fatal("expected failure on tampered value")
	}

	// (c) tampered partial-evaluation vector.
	proof, _ = sc.OpenEvaluation(r, 8)
	proof.Yr[1][0] ^= 0x01
	if _, err := VerifyEvaluation(cfg, commit, proof); err == nil {
		t.Fatal("expected failure on tampered yr vector")
	}
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
