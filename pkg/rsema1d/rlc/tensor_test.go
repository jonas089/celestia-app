package rlc

import (
	"crypto/sha256"
	"encoding/binary"
	"testing"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/merkle"
)

// deterministicRow fills a rowSize-byte Leopard-formatted row from a seed.
func deterministicRow(seed byte, rowSize int) []byte {
	row := make([]byte, rowSize)
	var ctr [8]byte
	ctr[0] = seed
	for off := 0; off < rowSize; off += 32 {
		binary.LittleEndian.PutUint32(ctr[1:], uint32(off))
		d := sha256.Sum256(ctr[:])
		copy(row[off:], d[:])
	}
	return row
}

// TestTensorFoldEqualsMultilinearEval is the core correctness property of the
// GKR-friendly RLC: folding a row's symbols with the tensor coefficients equals
// the row's multilinear extension evaluated at the challenge point. It checks
// this against an independent evaluator (EvalMultilinearRow) over several rows
// and several challenge points, plus the two boolean corners.
func TestTensorFoldEqualsMultilinearEval(t *testing.T) {
	const rowSize = 256 // 128 GF(2^16) symbols = 2^7
	const logCols = 7
	if (rowSize / 2) != (1 << logCols) {
		t.Fatalf("test setup: rowSize/2 must equal 2^logCols")
	}

	for trial := range 8 {
		var root merkle.Root
		root[0] = byte(trial + 1)
		challenges := DeriveTensorChallenges(root, logCols, rowSize)
		if len(challenges) != logCols {
			t.Fatalf("expected %d challenges, got %d", logCols, len(challenges))
		}
		coeffs := TensorCoefficients(challenges)
		if len(coeffs) != (1 << logCols) {
			t.Fatalf("expected %d coeffs, got %d", 1<<logCols, len(coeffs))
		}

		row := deterministicRow(byte(trial), rowSize)
		viaFold := ComputeRow(row, coeffs)
		viaEval := EvalMultilinearRow(row, challenges)
		if !field.Equal128(viaFold, viaEval) {
			t.Fatalf("trial %d: fold %v != multilinear eval %v", trial, viaFold, viaEval)
		}
	}
}

// TestTensorBooleanCorners checks that evaluating at boolean points selects the
// corresponding message symbol: all-zero challenges -> symbol 0, all-one
// challenges -> the last symbol. This anchors the coefficient/index ordering.
func TestTensorBooleanCorners(t *testing.T) {
	const rowSize = 256
	const logCols = 7
	row := deterministicRow(42, rowSize)

	zeros := make([]field.GF128, logCols)
	got := ComputeRow(row, TensorCoefficients(zeros))
	want := field.GF128{field.GF16FromLeopard(row, 0)}
	if !field.Equal128(got, want) {
		t.Fatalf("all-zero corner: got %v want symbol0 %v", got, want)
	}

	ones := make([]field.GF128, logCols)
	for i := range ones {
		ones[i] = field.One()
	}
	got = ComputeRow(row, TensorCoefficients(ones))
	want = field.GF128{field.GF16FromLeopard(row, (1<<logCols)-1)}
	if !field.Equal128(got, want) {
		t.Fatalf("all-one corner: got %v want last symbol %v", got, want)
	}
}

// TestTensorLinearity checks the fold is linear in the row (MLE is linear in
// its coefficients): eval(rowA XOR rowB) == eval(rowA) + eval(rowB), since the
// message symbols live in the GF(2^16) subfield and addition is XOR.
func TestTensorLinearity(t *testing.T) {
	const rowSize = 128 // 64 symbols = 2^6
	const logCols = 6
	var root merkle.Root
	root[0] = 7
	coeffs := TensorCoefficients(DeriveTensorChallenges(root, logCols, rowSize))

	a := deterministicRow(1, rowSize)
	b := deterministicRow(2, rowSize)
	ab := make([]byte, rowSize)
	for i := range ab {
		ab[i] = a[i] ^ b[i]
	}
	sum := field.Add128(ComputeRow(a, coeffs), ComputeRow(b, coeffs))
	got := ComputeRow(ab, coeffs)
	if !field.Equal128(got, sum) {
		t.Fatalf("linearity failed: eval(a^b)=%v, eval(a)+eval(b)=%v", got, sum)
	}
}
