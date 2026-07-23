package rsema1d

import (
	"testing"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
)

// TestAccidentalComputerWeld is the end-to-end demonstration that the
// data-availability encoding serves as the polynomial commitment for an
// interactive proof. A GKR-style sumcheck over a rollup's committed data
// reduces a global claim to a single evaluation f(r*) at a transcript-chosen
// point r*, and that evaluation is discharged by opening the rsema1d DA
// commitment at r* — no separate polynomial commitment is ever built.
func TestAccidentalComputerWeld(t *testing.T) {
	cfg := &Config{K: 8, N: 8, WorkerCount: 2}
	const rowBytes = 128
	rows := buildSquare(t, cfg.K, cfg.N, rowBytes, 0x5C)

	coder, err := NewCoder(cfg)
	if err != nil {
		t.Fatal(err)
	}
	sc, err := coder.EncodeStructured(rows)
	if err != nil {
		t.Fatal(err)
	}
	commit := sc.Commitment()

	// Rollup owns the whole K rows here (v = 3 sumcheck variables).
	r := RowRange{Start: 0, Len: 8}
	yrRange := make(rlc.Vector, r.Len)
	copy(yrRange, sc.ExtendedData().rlc[r.Start:r.Start+r.Len])

	// Prover: sumcheck over the rollup's committed partial-evaluation vector.
	proof, point, finalEval := ProveSum(yrRange)

	// The asserted sum is the real checksum of the committed data.
	var wantSum field.GF128
	for _, e := range yrRange {
		wantSum = field.Add128(wantSum, e)
	}
	if !field.Equal128(proof.Claim, wantSum) {
		t.Fatalf("sumcheck claim %v != actual sum %v", proof.Claim, wantSum)
	}

	// Verifier side of the sumcheck: re-derives the point and final claim.
	chals, finalClaim, err := VerifySum(proof)
	if err != nil {
		t.Fatalf("sumcheck verify failed: %v", err)
	}
	if len(chals) != len(point) {
		t.Fatalf("challenge length mismatch")
	}
	for i := range chals {
		if !field.Equal128(chals[i], point[i]) {
			t.Fatalf("verifier point[%d] != prover point", i)
		}
	}
	if !field.Equal128(finalClaim, finalEval) {
		t.Fatalf("verifier final claim %v != prover final eval %v", finalClaim, finalEval)
	}
	// Fold convention must line up with the PCS opening.
	if !field.Equal128(foldGF128(yrRange, point), finalEval) {
		t.Fatalf("foldGF128 at transcript point != sumcheck final eval")
	}

	// THE WELD: discharge the sumcheck's final evaluation by opening the DA
	// commitment at the transcript point — not via a separate PCS.
	evalProof, err := sc.OpenEvaluationAt(r, point, cfg.K+cfg.N)
	if err != nil {
		t.Fatal(err)
	}
	value, err := VerifyEvaluationAt(cfg, commit, evalProof, point)
	if err != nil {
		t.Fatalf("DA-commitment opening failed: %v", err)
	}
	if !field.Equal128(value, finalClaim) {
		t.Fatalf("DA opening value %v != sumcheck final claim %v — weld broken", value, finalClaim)
	}
	t.Logf("weld verified: sumcheck sum=%v discharged via DA commitment at r* (%d vars)", proof.Claim, len(point))
}

// TestSumcheckRejectsWrongSum ensures a lied-about sum is caught, and that an
// honest opening at the wrong point is rejected by the weld.
func TestSumcheckRejectsWrongSum(t *testing.T) {
	cfg := &Config{K: 8, N: 8, WorkerCount: 2}
	rows := buildSquare(t, cfg.K, cfg.N, 128, 0x77)
	coder, _ := NewCoder(cfg)
	sc, err := coder.EncodeStructured(rows)
	if err != nil {
		t.Fatal(err)
	}
	r := RowRange{Start: 0, Len: 8}
	yrRange := make(rlc.Vector, r.Len)
	copy(yrRange, sc.ExtendedData().rlc[r.Start:r.Start+r.Len])

	// (a) lie about the sum: verifier's round-0 check must fail.
	proof, _, _ := ProveSum(yrRange)
	proof.Claim[0] ^= 0x01
	if _, _, err := VerifySum(proof); err == nil {
		t.Fatal("expected sumcheck failure on tampered claim")
	}

	// (b) honest sumcheck, but open the DA commitment at a different point:
	// the opening's internal value check must reject it.
	proof2, point, _ := ProveSum(yrRange)
	_ = proof2
	wrong := make([]field.GF128, len(point))
	copy(wrong, point)
	wrong[0] = field.Add128(wrong[0], field.One())
	evalProof, _ := sc.OpenEvaluationAt(r, point, cfg.K+cfg.N)
	if _, err := VerifyEvaluationAt(cfg, sc.Commitment(), evalProof, wrong); err == nil {
		t.Fatal("expected opening failure at wrong point")
	}
}
