package rsema1d

import (
	"crypto/sha256"
	"fmt"
	"math/bits"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/merkle"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
)

// This file extends the structured multilinear PCS (pcs.go) to openings at a
// FULL evaluation point that spans BOTH the row and the column variables of the
// committed square, supplied externally by an interactive protocol (Expander's
// GKR). It is purely additive: the existing OpenEvaluationAt / VerifyEvaluationAt
// path (which derives the column point via Fiat-Shamir from the commitment and
// only accepts a row point) is untouched.
//
// -----------------------------------------------------------------------------
// Why a new path is needed
// -----------------------------------------------------------------------------
// EncodeStructured commits yr[j] = MLE(row_j)(rCol_FS), where rCol_FS is the
// tensor column challenge Fiat-Shamir-bound to the row Merkle root. The DA
// commitment can therefore only ever discharge an opening whose column point is
// rCol_FS. Expander's GKR, however, reduces the input layer to an evaluation at
// a transcript-chosen point that fixes EVERY input variable — both the row and
// the column coordinates — and that column coordinate is NOT rCol_FS. So the
// committed yr is at the wrong column point and cannot be folded directly.
//
// -----------------------------------------------------------------------------
// Construction (EvalProofFull) and soundness argument
// -----------------------------------------------------------------------------
// The prover reveals two length-K partial-evaluation vectors:
//
//	Yr[j]      = MLE(row_j)(rCol_FS)   — the committed vector (binds to rlcRoot)
//	YrPrime[j] = MLE(row_j)(rCol)      — the per-row evals at the SUPPLIED column
//	                                     point rCol (what GKR actually asked for)
//
// and the claimed value = fold_{rRow}(YrPrime[range]) = MLE_{X_range}(rRow,rCol).
// It also reveals sampleCount rows at indices Fiat-Shamir-bound to the
// commitment (originals or parity), with Merkle proofs. Verification:
//
//	(1) recover rowRoot from the sampled-row Merkle proofs;
//	(2) recompute rlcRoot from Yr and check SHA256(rowRoot||rlcRoot)==commitment.
//	    This binds Yr to the commitment: Yr is exactly the committed vector.
//	(3) re-derive rCol_FS from rowRoot (so it is unpredictable to the prover) and
//	    RS-extend both Yr and YrPrime to K+N with the same Leopard GF(2^16) code.
//	(4) for every sampled row index i, check BOTH
//	      (a) ComputeRow(row_i, tensor(rCol_FS)) == RS-ext(Yr)[i]
//	      (b) ComputeRow(row_i, tensor(rCol))    == RS-ext(YrPrime)[i]
//	(5) value == fold_{rRow}(YrPrime[range]).
//
// Soundness. Check (a) is the standard ZODA/Ligero proximity test: since rCol_FS
// is a GF(2^128) tensor challenge bound to rowRoot, any matrix whose rows are not
// (close to) the unique RS codeword folding to Yr fails (a) at a random sampled
// row except with probability ~ agreement-fraction per sample; over sampleCount
// samples the sampled rows are pinned to that unique codeword, which is bound to
// the commitment by (2). Check (b) reuses THOSE SAME pinned rows: because
// ComputeRow(row, tensor(rCol)) == EvalMultilinearRow(row, rCol), (b) says the
// codeword RS-ext(YrPrime) agrees, at every sampled index, with the codeword
// obtained by evaluating the true (pinned) rows at rCol. Both are RS codewords of
// dimension K; two distinct length-K codewords differ in ≥ N+1 of the K+N
// positions, so a random sampled index exposes any YrPrime ≠ (true per-row evals
// at rCol) with probability ≥ (N+1)/(K+N) per sample. Hence w.h.p. YrPrime is the
// genuine per-row evaluation vector at rCol, and value = fold_{rRow}(YrPrime[range])
// is the true multilinear evaluation of the sub-matrix X_range at the full point
// (rRow, rCol). Because the column challenge lives in GF(2^128), the proximity
// error is ~poly/2^128 (logarithmic randomness) rather than the ~1/2^16 an
// in-field GF(2^16) challenge would give. The sampling error is ((K-1)/(K+N))^s
// for s = sampleCount (dominated by ((K-1)/(K+N)) per drawn index).
//
// Tampering with the value, with rCol, with rRow, with Yr, with YrPrime, or with
// any sampled row is therefore rejected: a wrong value fails (5); a wrong Yr fails
// the commitment check (2) or (a); a wrong YrPrime fails (b) or yields a value
// that fails (5); a wrong rCol/rRow changes the recomputed value and fails (5)
// (and a wrong rCol additionally breaks (b) unless YrPrime is recomputed for it,
// which then no longer matches the pinned rows). See pcs_full_test.go.

// EvalProofFull is a full-point subset-evaluation proof. Unlike EvalProof it
// carries a SECOND partial-evaluation vector YrPrime at the externally supplied
// column point rCol, in addition to the committed Yr at the Fiat-Shamir column
// point. Both are needed to soundly bind the supplied column point to the same
// underlying (committed) rows.
type EvalProofFull struct {
	Range       RowRange
	Yr          rlc.Vector  // K committed partial evals yr[j]=MLE(row_j)(rCol_FS)
	YrPrime     rlc.Vector  // K partial evals yr'[j]=MLE(row_j)(rCol) at supplied rCol
	SampledRows []*RowProof // FS-random rows for the proximity/consistency check
	Value       field.GF128 // claimed MLE_{X_range}(rRow, rCol)
}

// OpenEvaluationAtFull produces a full-point subset-evaluation proof: it opens
// the sub-matrix X_range at the externally supplied point (rRow, rCol), where
// rCol has len == log2(numSymbols) (the column variables) and rRow has len ==
// log2(range.Len) (the row variables). This is the opening Expander's GKR needs:
// the transcript dictates the entire point, spanning both axes of the square.
//
// Sample indices remain Fiat-Shamir-bound to the commitment so a prover cannot
// steer the proximity check.
func (sc *StructuredCommitment) OpenEvaluationAtFull(r RowRange, rCol, rRow []field.GF128, sampleCount int) (*EvalProofFull, error) {
	k, n := sc.ed.config.K, sc.ed.config.N
	if err := r.validate(k); err != nil {
		return nil, err
	}
	if len(rCol) != sc.logCols {
		return nil, fmt.Errorf("expected %d column challenges, got %d", sc.logCols, len(rCol))
	}
	if want := bits.Len(uint(r.Len)) - 1; len(rRow) != want {
		return nil, fmt.Errorf("expected %d row challenges for range len %d, got %d", want, r.Len, len(rRow))
	}

	// Yr: the committed partial-eval vector (at the FS column point). Copied so
	// the proof owns its data independently of the live ExtendedData.
	yr := make(rlc.Vector, len(sc.ed.rlc))
	copy(yr, sc.ed.rlc)

	// YrPrime: per-row multilinear evaluation at the SUPPLIED column point rCol.
	yrPrime := make(rlc.Vector, k)
	for j := 0; j < k; j++ {
		yrPrime[j] = rlc.EvalMultilinearRow(sc.ed.rows[j], rCol)
	}

	value := foldGF128(yrPrime[r.Start:r.Start+r.Len], rRow)

	indices := deriveSampleIndices(sc.ed.commitment, r, sampleCount, k+n)
	sampled := make([]*RowProof, 0, len(indices))
	for _, idx := range indices {
		p, err := sc.ed.GenerateRowProof(idx)
		if err != nil {
			return nil, fmt.Errorf("sampling row %d: %w", idx, err)
		}
		sampled = append(sampled, p)
	}

	return &EvalProofFull{Range: r, Yr: yr, YrPrime: yrPrime, SampledRows: sampled, Value: value}, nil
}

// VerifyEvaluationAtFull checks a full-point subset-evaluation proof against the
// commitment at the externally supplied point (rRow, rCol) and returns the
// verified value. See the soundness argument at the top of this file.
func VerifyEvaluationAtFull(cfg *Config, commitment Commitment, proof *EvalProofFull, rCol, rRow []field.GF128) (field.GF128, error) {
	var zero field.GF128
	if err := cfg.Validate(); err != nil {
		return zero, fmt.Errorf("invalid config: %w", err)
	}
	if err := proof.Range.validate(cfg.K); err != nil {
		return zero, err
	}
	if len(proof.Yr) != cfg.K {
		return zero, fmt.Errorf("expected %d committed partial evaluations, got %d", cfg.K, len(proof.Yr))
	}
	if len(proof.YrPrime) != cfg.K {
		return zero, fmt.Errorf("expected %d supplied-point partial evaluations, got %d", cfg.K, len(proof.YrPrime))
	}
	if len(proof.SampledRows) == 0 {
		return zero, fmt.Errorf("no sampled rows in proof")
	}

	// (1) rowRoot from the sampled row Merkle proofs.
	proofInputs := make([]merkle.ProofInput, len(proof.SampledRows))
	rowBytes := len(proof.SampledRows[0].Row)
	for i, p := range proof.SampledRows {
		if p == nil {
			return zero, fmt.Errorf("nil sampled row proof")
		}
		if len(p.Row) != rowBytes {
			return zero, fmt.Errorf("sampled rows differ in size")
		}
		if p.Index < 0 || p.Index >= cfg.K+cfg.N {
			return zero, fmt.Errorf("sampled row index %d out of range", p.Index)
		}
		proofInputs[i] = merkle.ProofInput{Leaf: p.Row, Index: p.Index, Path: p.RowProof}
	}
	rowRoot, err := merkle.RootFromProofs(proofInputs, gomaxprocs)
	if err != nil {
		return zero, fmt.Errorf("verifying sampled row proofs: %w", err)
	}

	// (2) rlcRoot from the full committed Yr, then the commitment check. This is
	// what binds Yr to the commitment.
	rlcRoot := computeRLCRoot(proof.Yr, make([]byte, cfg.K*merkle.NodeSize), make([]byte, field.GF128Size))
	h := sha256.New()
	h.Write(rowRoot[:])
	h.Write(rlcRoot[:])
	var recomputed Commitment
	h.Sum(recomputed[:0])
	if recomputed != commitment {
		return zero, fmt.Errorf("commitment verification failed")
	}

	// (3) re-derive the FS column tensor challenges from rowRoot, and validate the
	// supplied column point length.
	numSymbols := rowBytes / 2
	if numSymbols <= 0 || numSymbols&(numSymbols-1) != 0 {
		return zero, fmt.Errorf("row bytes %d give non-power-of-two symbol count", rowBytes)
	}
	logCols := bits.Len(uint(numSymbols)) - 1
	if len(rCol) != logCols {
		return zero, fmt.Errorf("expected %d column challenges, got %d", logCols, len(rCol))
	}

	colChallsFS := rlc.DeriveTensorChallenges(rowRoot, logCols, rowBytes)
	coeffsFS := rlc.TensorCoefficients(colChallsFS)
	coeffsCol := rlc.TensorCoefficients(rCol)

	yrExt, err := rsExtendGF128(cfg, proof.Yr)
	if err != nil {
		return zero, fmt.Errorf("extending committed partial evaluations: %w", err)
	}
	yrPrimeExt, err := rsExtendGF128(cfg, proof.YrPrime)
	if err != nil {
		return zero, fmt.Errorf("extending supplied-point partial evaluations: %w", err)
	}

	// (4) proximity / consistency. Each sampled row must fold (a) under the FS
	// tensor to RS-ext(Yr) [binds the row to the committed vector, hence to the
	// commitment] AND (b) under the supplied-point tensor to RS-ext(YrPrime)
	// [binds YrPrime to that same row at the supplied column point].
	for _, p := range proof.SampledRows {
		gotFS := rlc.ComputeRow(p.Row, coeffsFS)
		if !field.Equal128(gotFS, yrExt[p.Index]) {
			return zero, fmt.Errorf("row %d: FS-tensor fold does not match committed partial evaluation", p.Index)
		}
		gotCol := rlc.ComputeRow(p.Row, coeffsCol)
		if !field.Equal128(gotCol, yrPrimeExt[p.Index]) {
			return zero, fmt.Errorf("row %d: supplied-point fold does not match YrPrime", p.Index)
		}
	}

	// (5) recompute the range evaluation from YrPrime at the supplied row point
	// and match the claimed value.
	if want := bits.Len(uint(proof.Range.Len)) - 1; len(rRow) != want {
		return zero, fmt.Errorf("expected %d row challenges, got %d", want, len(rRow))
	}
	yrPrimeRange := proof.YrPrime[proof.Range.Start : proof.Range.Start+proof.Range.Len]
	want := foldGF128(yrPrimeRange, rRow)
	if !field.Equal128(want, proof.Value) {
		return zero, fmt.Errorf("claimed evaluation does not match recomputed value")
	}
	return proof.Value, nil
}
