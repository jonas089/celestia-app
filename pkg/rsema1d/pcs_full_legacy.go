package rsema1d

import (
	"crypto/sha256"
	"fmt"
	"math/bits"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/merkle"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
)

// This file makes the GKR/DA path genuinely ENCODE-ONCE: it opens the ORIGINAL
// [Coder.Encode] commitment — the canonical DA/spec commitment
// SHA256(rowRoot||rlcRoot) whose rlcRoot is built from the legacy
// rlc.DeriveCoefficients RLC vector — at a full evaluation point supplied
// externally by an interactive protocol (Expander's GKR). It is purely
// additive: nothing in coder.go/derive.go (the original spec) or the
// EncodeStructured tensor path (pcs.go/pcs_full.go) is touched.
//
// -----------------------------------------------------------------------------
// Why this exists
// -----------------------------------------------------------------------------
// Encode commits the same square the celestia-app DA layer samples over. Its
// per-row RLC value Yr[j] = ComputeRow(row_j, DeriveCoefficients(rowRoot)) is a
// single random GF(2^128) linear combination of the row's symbols — a
// Ligero-style proximity digest, NOT a multilinear partial evaluation. So the
// committed vector cannot itself be folded into a polynomial evaluation the way
// the tensor-structured EncodeStructured vector can. Instead we reveal a SECOND
// per-row vector YrPrime[j] = MLE(row_j)(rCol) computed FRESH from the committed
// rows, and bind it to those same committed rows with a two-codeword argument.
// The result: one commitment, produced by the unchanged Encode, discharges both
// DA sampling and the GKR input-layer opening.
//
// -----------------------------------------------------------------------------
// Construction (EvalProofFull, reused from pcs_full.go) and soundness argument
// -----------------------------------------------------------------------------
// The prover reveals two length-K vectors:
//
//	Yr[j]      = ComputeRow(row_j, DeriveCoefficients(rowRoot))  — the LEGACY
//	             committed RLC vector (binds to rlcRoot, hence to the commitment)
//	YrPrime[j] = MLE(row_j)(rCol)                                — per-row evals
//	             at the SUPPLIED column point rCol (what GKR asked for)
//
// and value = fold_{rRow}(YrPrime[range]) = MLE_{X_range}(rRow, rCol). It also
// reveals sampleCount rows at indices Fiat-Shamir-bound to the commitment
// (originals or parity) with Merkle proofs. Verification:
//
//	(1) recover rowRoot from the sampled-row Merkle proofs;
//	(2) recompute rlcRoot from Yr and check SHA256(rowRoot||rlcRoot)==commitment.
//	    This is the EXACT original Encode commitment equation, so Yr is pinned to
//	    the committed legacy RLC vector and rowRoot to the committed rows.
//	(3) re-derive the LEGACY coefficients DeriveCoefficients(rowRoot,K,N,rowBytes)
//	    (unpredictable before rowRoot is fixed) and RS-extend both Yr and YrPrime
//	    to K+N with the same Leopard GF(2^16) code the rows use.
//	(4) for every sampled row index i, check BOTH
//	      (a) ComputeRow(row_i, DeriveCoefficients(rowRoot)) == RS-ext(Yr)[i]
//	      (b) ComputeRow(row_i, tensor(rCol))                == RS-ext(YrPrime)[i]
//	(5) value == fold_{rRow}(YrPrime[range]).
//
// Soundness. Check (a) is verbatim the production DA proximity / unique-decoding
// test (see Verifier.verify in verifier.go: it RS-extends the committed RLC
// vector and checks each sampled row's DeriveCoefficients-fold equals it). Since
// the coefficients are GF(2^128) elements bound to rowRoot, any matrix whose
// rows are not the unique RS codeword folding to Yr fails (a) at a random
// sampled row except with probability ~ agreement-fraction per sample; over
// sampleCount samples the sampled rows — INCLUDING the parity rows — are pinned
// to that unique codeword, which is bound to the commitment by (2). In
// particular (a) forces the parity rows to be the genuine per-column RS parity
// of the originals.
//
// Check (b) reuses THOSE SAME pinned rows. Because ComputeRow(row, tensor(rCol))
// == EvalMultilinearRow(row, rCol) and column-wise RS encoding is linear, the
// map row |-> MLE(row)(rCol) applied across all K+N rows yields a length-(K+N)
// RS codeword of dimension K whose first K entries are the true per-row evals
// T[j] = MLE(true_row_j)(rCol); i.e. RS-ext(T) equals that vector exactly (both
// are the unique degree-<K codeword through T[0..K)). Check (b) says
// RS-ext(YrPrime) agrees with RS-ext(T) at every sampled index. Two distinct
// length-K RS codewords differ in >= N+1 of the K+N positions, so a random
// sampled index exposes any YrPrime != T with probability >= (N+1)/(K+N) per
// sample. Hence w.h.p. YrPrime == T, and value = fold_{rRow}(YrPrime[range]) is
// the true multilinear evaluation of the sub-matrix X_range at (rRow, rCol).
// Because both challenge families live in GF(2^128), the proximity/evaluation
// error is ~poly/2^128 (logarithmic randomness), and the sampling error is
// ((K-1)/(K+N))^s for s = sampleCount.
//
// Tampering is rejected: a wrong value fails (5); a wrong Yr fails (2) or (a); a
// wrong YrPrime fails (b) or yields a value failing (5); a wrong rCol/rRow
// changes the recomputed value and fails (5) (a wrong rCol additionally breaks
// (b) unless YrPrime is recomputed for it, which then no longer matches the
// pinned rows); a tampered sampled row fails its Merkle proof in (1) or one of
// the folds in (4). See pcs_full_legacy_test.go.

// OpenAtFullLegacy opens the sub-matrix X_range of the ORIGINAL Encode
// commitment at the externally supplied full point (rCol, rRow): rCol has len ==
// log2(numSymbols) (column variables) and rRow has len == log2(range.Len) (row
// variables). This is the opening Expander's GKR needs, discharged directly by
// the DA/spec commitment (ed must come from Coder.Encode, not EncodeStructured).
//
// Sample indices are Fiat-Shamir-bound to the commitment so a prover cannot
// steer the proximity check.
func (ed *ExtendedData) OpenAtFullLegacy(r RowRange, rCol, rRow []field.GF128, sampleCount int) (*EvalProofFull, error) {
	k, n := ed.config.K, ed.config.N
	if err := r.validate(k); err != nil {
		return nil, err
	}
	rowBytes := len(ed.rows[0])
	numSymbols := rowBytes / 2
	if numSymbols <= 0 || numSymbols&(numSymbols-1) != 0 {
		return nil, fmt.Errorf("full-point opening needs a power-of-two symbol count; row bytes %d give %d symbols", rowBytes, numSymbols)
	}
	logCols := bits.Len(uint(numSymbols)) - 1
	if len(rCol) != logCols {
		return nil, fmt.Errorf("expected %d column challenges, got %d", logCols, len(rCol))
	}
	if want := bits.Len(uint(r.Len)) - 1; len(rRow) != want {
		return nil, fmt.Errorf("expected %d row challenges for range len %d, got %d", want, r.Len, len(rRow))
	}

	// Yr: the committed LEGACY RLC vector (copied so the proof owns its data).
	yr := make(rlc.Vector, len(ed.rlc))
	copy(yr, ed.rlc)

	// YrPrime: per-row multilinear evaluation at the supplied column point rCol.
	yrPrime := make(rlc.Vector, k)
	for j := 0; j < k; j++ {
		yrPrime[j] = rlc.EvalMultilinearRow(ed.rows[j], rCol)
	}

	value := foldGF128(yrPrime[r.Start:r.Start+r.Len], rRow)

	indices := deriveSampleIndices(ed.commitment, r, sampleCount, k+n)
	sampled := make([]*RowProof, 0, len(indices))
	for _, idx := range indices {
		p, err := ed.GenerateRowProof(idx)
		if err != nil {
			return nil, fmt.Errorf("sampling row %d: %w", idx, err)
		}
		sampled = append(sampled, p)
	}

	return &EvalProofFull{Range: r, Yr: yr, YrPrime: yrPrime, SampledRows: sampled, Value: value}, nil
}

// VerifyAtFullLegacy checks a full-point opening produced by OpenAtFullLegacy
// against the ORIGINAL Encode commitment at the supplied point (rCol, rRow) and
// returns the verified value. See the soundness argument at the top of the file.
func VerifyAtFullLegacy(cfg *Config, commitment Commitment, proof *EvalProofFull, rCol, rRow []field.GF128) (field.GF128, error) {
	var zero field.GF128
	if err := cfg.Validate(); err != nil {
		return zero, fmt.Errorf("invalid config: %w", err)
	}
	if err := proof.Range.validate(cfg.K); err != nil {
		return zero, err
	}
	if len(proof.Yr) != cfg.K {
		return zero, fmt.Errorf("expected %d committed RLC values, got %d", cfg.K, len(proof.Yr))
	}
	if len(proof.YrPrime) != cfg.K {
		return zero, fmt.Errorf("expected %d supplied-point partial evaluations, got %d", cfg.K, len(proof.YrPrime))
	}
	if len(proof.SampledRows) == 0 {
		return zero, fmt.Errorf("no sampled rows in proof")
	}

	// (1) rowRoot from the sampled row Merkle proofs.
	rowRoot, rowBytes, err := rowRootFromSamples(cfg, proof.SampledRows)
	if err != nil {
		return zero, err
	}

	// (2) rlcRoot from the committed LEGACY RLC vector Yr, then commitment check.
	// This is the exact original Encode commitment equation.
	if err := checkCommitment(commitment, rowRoot, proof.Yr, cfg); err != nil {
		return zero, err
	}

	// (3) legacy proximity coefficients (identical to the DA path) plus the
	// supplied-column tensor coefficients; RS-extend both revealed vectors.
	numSymbols := rowBytes / 2
	if numSymbols <= 0 || numSymbols&(numSymbols-1) != 0 {
		return zero, fmt.Errorf("row bytes %d give non-power-of-two symbol count", rowBytes)
	}
	logCols := bits.Len(uint(numSymbols)) - 1
	if len(rCol) != logCols {
		return zero, fmt.Errorf("expected %d column challenges, got %d", logCols, len(rCol))
	}

	legacyCoeffs := rlc.DeriveCoefficients(rowRoot, cfg.K, cfg.N, rowBytes, cfg.WorkerCount)
	coeffsCol := rlc.TensorCoefficients(rCol)

	yrExt, err := rsExtendGF128(cfg, proof.Yr)
	if err != nil {
		return zero, fmt.Errorf("extending committed RLC: %w", err)
	}
	yrPrimeExt, err := rsExtendGF128(cfg, proof.YrPrime)
	if err != nil {
		return zero, fmt.Errorf("extending supplied-point partial evaluations: %w", err)
	}

	// (4) proximity / consistency for every sampled row:
	//   (a) legacy fold == RS-ext(Yr)      — DA unique-decoding binding
	//   (b) tensor(rCol) fold == RS-ext(YrPrime) — binds YrPrime to pinned rows
	for _, p := range proof.SampledRows {
		gotLegacy := rlc.ComputeRow(p.Row, legacyCoeffs)
		if !field.Equal128(gotLegacy, yrExt[p.Index]) {
			return zero, fmt.Errorf("row %d: legacy RLC fold does not match committed value", p.Index)
		}
		gotCol := rlc.ComputeRow(p.Row, coeffsCol)
		if !field.Equal128(gotCol, yrPrimeExt[p.Index]) {
			return zero, fmt.Errorf("row %d: supplied-point fold does not match YrPrime", p.Index)
		}
	}

	// (5) value == fold_{rRow}(YrPrime[range]).
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

// OpenAtLegacy is the row-only counterpart of OpenAtFullLegacy: it proves that
// the committed LEGACY RLC vector, restricted to range, folds under rRow to
// Value. It opens the ORIGINAL Encode commitment (ed from Coder.Encode). Unlike
// the tensor OpenEvaluationAt this makes no multilinear claim about the column
// axis — it is the FS-column-free binding of the committed RLC vector, used by
// the row-only FFI smoke path. Sample indices are FS-bound to the commitment.
func (ed *ExtendedData) OpenAtLegacy(r RowRange, rRow []field.GF128, sampleCount int) (*EvalProof, error) {
	k, n := ed.config.K, ed.config.N
	if err := r.validate(k); err != nil {
		return nil, err
	}
	if want := bits.Len(uint(r.Len)) - 1; len(rRow) != want {
		return nil, fmt.Errorf("expected %d row challenges for range len %d, got %d", want, r.Len, len(rRow))
	}

	yr := make(rlc.Vector, len(ed.rlc))
	copy(yr, ed.rlc)
	value := foldGF128(yr[r.Start:r.Start+r.Len], rRow)

	indices := deriveSampleIndices(ed.commitment, r, sampleCount, k+n)
	sampled := make([]*RowProof, 0, len(indices))
	for _, idx := range indices {
		p, err := ed.GenerateRowProof(idx)
		if err != nil {
			return nil, fmt.Errorf("sampling row %d: %w", idx, err)
		}
		sampled = append(sampled, p)
	}
	return &EvalProof{Range: r, Yr: yr, SampledRows: sampled, Value: value}, nil
}

// VerifyAtLegacy checks a row-only opening produced by OpenAtLegacy against the
// ORIGINAL Encode commitment and returns the verified value. It runs steps
// (1),(2),(4a),(5) of the full-point argument (no column axis / YrPrime).
func VerifyAtLegacy(cfg *Config, commitment Commitment, proof *EvalProof, rRow []field.GF128) (field.GF128, error) {
	var zero field.GF128
	if err := cfg.Validate(); err != nil {
		return zero, fmt.Errorf("invalid config: %w", err)
	}
	if err := proof.Range.validate(cfg.K); err != nil {
		return zero, err
	}
	if len(proof.Yr) != cfg.K {
		return zero, fmt.Errorf("expected %d committed RLC values, got %d", cfg.K, len(proof.Yr))
	}
	if len(proof.SampledRows) == 0 {
		return zero, fmt.Errorf("no sampled rows in proof")
	}

	rowRoot, rowBytes, err := rowRootFromSamples(cfg, proof.SampledRows)
	if err != nil {
		return zero, err
	}
	if err := checkCommitment(commitment, rowRoot, proof.Yr, cfg); err != nil {
		return zero, err
	}

	legacyCoeffs := rlc.DeriveCoefficients(rowRoot, cfg.K, cfg.N, rowBytes, cfg.WorkerCount)
	yrExt, err := rsExtendGF128(cfg, proof.Yr)
	if err != nil {
		return zero, fmt.Errorf("extending committed RLC: %w", err)
	}
	for _, p := range proof.SampledRows {
		got := rlc.ComputeRow(p.Row, legacyCoeffs)
		if !field.Equal128(got, yrExt[p.Index]) {
			return zero, fmt.Errorf("row %d: legacy RLC fold does not match committed value", p.Index)
		}
	}

	if want := bits.Len(uint(proof.Range.Len)) - 1; len(rRow) != want {
		return zero, fmt.Errorf("expected %d row challenges, got %d", want, len(rRow))
	}
	yrRange := proof.Yr[proof.Range.Start : proof.Range.Start+proof.Range.Len]
	want := foldGF128(yrRange, rRow)
	if !field.Equal128(want, proof.Value) {
		return zero, fmt.Errorf("claimed evaluation does not match recomputed value")
	}
	return proof.Value, nil
}

// rowRootFromSamples recovers the row Merkle root from a batch of sampled row
// proofs and returns it together with the common row byte length.
func rowRootFromSamples(cfg *Config, samples []*RowProof) (merkle.Root, int, error) {
	proofInputs := make([]merkle.ProofInput, len(samples))
	rowBytes := len(samples[0].Row)
	for i, p := range samples {
		if p == nil {
			return merkle.Root{}, 0, fmt.Errorf("nil sampled row proof")
		}
		if len(p.Row) != rowBytes {
			return merkle.Root{}, 0, fmt.Errorf("sampled rows differ in size")
		}
		if p.Index < 0 || p.Index >= cfg.K+cfg.N {
			return merkle.Root{}, 0, fmt.Errorf("sampled row index %d out of range", p.Index)
		}
		proofInputs[i] = merkle.ProofInput{Leaf: p.Row, Index: p.Index, Path: p.RowProof}
	}
	rowRoot, err := merkle.RootFromProofs(proofInputs, gomaxprocs)
	if err != nil {
		return merkle.Root{}, 0, fmt.Errorf("verifying sampled row proofs: %w", err)
	}
	return rowRoot, rowBytes, nil
}

// checkCommitment recomputes rlcRoot from the revealed legacy RLC vector and
// verifies SHA256(rowRoot||rlcRoot) equals the commitment — the exact original
// Encode commitment equation.
func checkCommitment(commitment Commitment, rowRoot merkle.Root, yr rlc.Vector, cfg *Config) error {
	rlcRoot := computeRLCRoot(yr, make([]byte, cfg.K*merkle.NodeSize), make([]byte, field.GF128Size))
	h := sha256.New()
	h.Write(rowRoot[:])
	h.Write(rlcRoot[:])
	var recomputed Commitment
	h.Sum(recomputed[:0])
	if recomputed != commitment {
		return fmt.Errorf("commitment verification failed")
	}
	return nil
}
