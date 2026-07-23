package rsema1d

import (
	"testing"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
)

// TestLegacyFullPointRoundTrip opens the ORIGINAL Encode commitment at a full
// point and cross-checks the verified value against the independent MLE
// reference (evalMatrixFull, defined in pcs_full_test.go). This proves the
// encode-once opening returns the genuine multilinear evaluation.
func TestLegacyFullPointRoundTrip(t *testing.T) {
	cfg := &Config{K: 8, N: 8, WorkerCount: 2}
	const rowBytes = 128 // numSymbols = 64 = 2^6 columns
	const numSymbols = rowBytes / 2
	logCols := 6

	rows := buildSquare(t, cfg.K, cfg.N, rowBytes, 0xC3)
	coder, err := NewCoder(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ed, err := coder.Encode(rows)
	if err != nil {
		t.Fatal(err)
	}
	commit := ed.Commitment()

	rCol := randChallenges("LEGACY-FULL-COL", logCols)

	for _, r := range []RowRange{{Start: 0, Len: 8}, {Start: 4, Len: 4}, {Start: 0, Len: 1}} {
		logRows := 0
		for (1 << logRows) < r.Len {
			logRows++
		}
		rRow := randChallenges("LEGACY-FULL-ROW", logRows)

		proof, err := ed.OpenAtFullLegacy(r, rCol, rRow, cfg.K+cfg.N)
		if err != nil {
			t.Fatalf("range %+v: open failed: %v", r, err)
		}
		val, err := VerifyAtFullLegacy(cfg, commit, proof, rCol, rRow)
		if err != nil {
			t.Fatalf("range %+v: verify failed: %v", r, err)
		}
		if !field.Equal128(val, proof.Value) {
			t.Fatalf("range %+v: returned value != proof value", r)
		}
		want := evalMatrixFull(ed.rows, r, rCol, rRow, numSymbols)
		if !field.Equal128(val, want) {
			t.Fatalf("range %+v: verified value %v != independent MLE %v", r, val, want)
		}
		for j := 0; j < cfg.K; j++ {
			ref := rlc.EvalMultilinearRow(ed.rows[j], rCol)
			if !field.Equal128(proof.YrPrime[j], ref) {
				t.Fatalf("YrPrime[%d] != EvalMultilinearRow at rCol", j)
			}
		}
	}
}

// TestLegacyFullPointRejectsTampering is gate (C) at the Go layer for the
// encode-once opening.
func TestLegacyFullPointRejectsTampering(t *testing.T) {
	cfg := &Config{K: 8, N: 8, WorkerCount: 2}
	const rowBytes = 128
	logCols := 6
	logRows := 3

	rows := buildSquare(t, cfg.K, cfg.N, rowBytes, 0x2E)
	coder, _ := NewCoder(cfg)
	ed, err := coder.Encode(rows)
	if err != nil {
		t.Fatal(err)
	}
	commit := ed.Commitment()
	r := RowRange{Start: 0, Len: 8}
	rCol := randChallenges("LTAMP-COL", logCols)
	rRow := randChallenges("LTAMP-ROW", logRows)

	base, _ := ed.OpenAtFullLegacy(r, rCol, rRow, cfg.K+cfg.N)
	if _, err := VerifyAtFullLegacy(cfg, commit, base, rCol, rRow); err != nil {
		t.Fatalf("honest proof must verify: %v", err)
	}

	mustReject := func(name string, mutate func(p *EvalProofFull), vCol, vRow []field.GF128) {
		p, err := ed.OpenAtFullLegacy(r, rCol, rRow, cfg.K+cfg.N)
		if err != nil {
			t.Fatalf("%s: open failed: %v", name, err)
		}
		mutate(p)
		if _, err := VerifyAtFullLegacy(cfg, commit, p, vCol, vRow); err == nil {
			t.Fatalf("%s: expected rejection but verify accepted", name)
		}
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

	wrongCol := make([]field.GF128, logCols)
	copy(wrongCol, rCol)
	wrongCol[0] = field.Add128(wrongCol[0], field.One())
	mustReject("wrong rCol", func(p *EvalProofFull) {}, wrongCol, rRow)

	wrongRow := make([]field.GF128, logRows)
	copy(wrongRow, rRow)
	wrongRow[0] = field.Add128(wrongRow[0], field.One())
	mustReject("wrong rRow", func(p *EvalProofFull) {}, rCol, wrongRow)
}

// TestEncodeOnceCommitmentSharedWithDASampler is gate (D): the SAME Encode
// commitment that the GKR full-point opening opens is also validated by the DA
// batched Verifier over sampled row proofs. One encoding, two consumers.
func TestEncodeOnceCommitmentSharedWithDASampler(t *testing.T) {
	cfg := &Config{K: 8, N: 8, WorkerCount: 2}
	const rowBytes = 128
	logCols := 6

	rows := buildSquare(t, cfg.K, cfg.N, rowBytes, 0x9A)
	coder, err := NewCoder(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ed, err := coder.Encode(rows)
	if err != nil {
		t.Fatal(err)
	}
	commit := ed.Commitment()

	// (1) GKR path opens this commitment.
	r := RowRange{Start: 0, Len: 8}
	rCol := randChallenges("DA-COL", logCols)
	rRow := randChallenges("DA-ROW", 3)
	proof, err := ed.OpenAtFullLegacy(r, rCol, rRow, cfg.K+cfg.N)
	if err != nil {
		t.Fatalf("GKR open failed: %v", err)
	}
	if _, err := VerifyAtFullLegacy(cfg, commit, proof, rCol, rRow); err != nil {
		t.Fatalf("GKR verify failed: %v", err)
	}

	// (2) The DA batched Verifier validates the SAME commitment over row proofs
	// plus the committed RLC vector — the unchanged production proximity check.
	verifier, err := NewVerifier(cfg)
	if err != nil {
		t.Fatal(err)
	}
	rowProofs := make([]*RowProof, cfg.K+cfg.N)
	for i := range rowProofs {
		p, err := ed.GenerateRowProof(i)
		if err != nil {
			t.Fatal(err)
		}
		rowProofs[i] = p
	}
	if err := verifier.Verify(commit, rowProofs, ed.RLC()); err != nil {
		t.Fatalf("DA sampler rejected the shared commitment: %v", err)
	}

	// A tampered row must be rejected by the DA sampler too.
	bad := make([]byte, rowBytes)
	copy(bad, rowProofs[0].Row)
	bad[0] ^= 0xFF
	rowProofs[0] = &RowProof{Index: rowProofs[0].Index, Row: bad, RowProof: rowProofs[0].RowProof}
	if err := verifier.Verify(commit, rowProofs, ed.RLC()); err == nil {
		t.Fatalf("DA sampler accepted a tampered row")
	}
}
