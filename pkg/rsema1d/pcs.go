package rsema1d

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/bits"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/merkle"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
	"github.com/klauspost/reedsolomon"
)

// This file turns rsema1d into a multilinear polynomial commitment scheme with
// per-namespace subset openings — the "evaluation over a subset of rows"
// construction of the accidental-computer paper (§5). A shared square holds
// many rollups; each rollup owns a contiguous, power-of-two-aligned block of
// original rows (a RowRange). Because the square is committed with the
// tensor-structured RLC (rlc.TensorCoefficients), the per-row RLC value is the
// multilinear partial evaluation yr[j] = MLE(row_j)(r_col). A rollup can then
// prove that the multilinear extension of its own sub-matrix X_R evaluates to a
// value t at a random point, reusing the DA encoding as the commitment — no
// separate polynomial commitment is built.

// RowRange identifies a rollup's contiguous, power-of-two-aligned block of
// original rows within the shared square: rows [Start, Start+Len).
type RowRange struct {
	Start int
	Len   int
}

func (r RowRange) validate(k int) error {
	if r.Len <= 0 || r.Len&(r.Len-1) != 0 {
		return fmt.Errorf("range length must be a positive power of two, got %d", r.Len)
	}
	if r.Start < 0 || r.Start%r.Len != 0 {
		return fmt.Errorf("range start %d must be aligned to its length %d", r.Start, r.Len)
	}
	if r.Start+r.Len > k {
		return fmt.Errorf("range [%d,%d) exceeds K=%d original rows", r.Start, r.Start+r.Len, k)
	}
	return nil
}

// StructuredCommitment is an ExtendedData committed with tensor-structured RLC.
// Its rlc vector holds per-row multilinear partial evaluations yr[j] over the
// column challenges bound (Fiat-Shamir) to the row Merkle root.
type StructuredCommitment struct {
	ed        *ExtendedData
	colChalls []field.GF128
	logCols   int
}

// Commitment returns the SHA256(rowRoot || rlcRoot) commitment.
func (sc *StructuredCommitment) Commitment() Commitment { return sc.ed.commitment }

// ExtendedData exposes the underlying encoded data (rows, proofs).
func (sc *StructuredCommitment) ExtendedData() *ExtendedData { return sc.ed }

// EncodeStructured encodes rows exactly like Encode but derives the RLC
// coefficients as a tensor product ⊗(1-r_i, r_i) over Fiat-Shamir column
// challenges, so the committed RLC vector is the multilinear partial-evaluation
// vector yr. rows must have length K+N with parity rows zeroed, and the row
// byte length must give a power-of-two number of GF(2^16) symbols (len/2).
func (c *Coder) EncodeStructured(rows [][]byte) (*StructuredCommitment, error) {
	if err := c.validateRows(rows); err != nil {
		return nil, err
	}
	rowBytes := len(rows[0])
	numSymbols := rowBytes / 2
	if numSymbols <= 0 || numSymbols&(numSymbols-1) != 0 {
		return nil, fmt.Errorf("structured PCS needs a power-of-two symbol count per row; row bytes %d give %d symbols", rowBytes, numSymbols)
	}
	logCols := bits.Len(uint(numSymbols)) - 1

	if err := c.enc.Encode(rows); err != nil {
		return nil, fmt.Errorf("failed to encode: %w", err)
	}

	rowTreeBytes := merkle.TreeBufferSize(c.config.K + c.config.N)
	buf := make([]byte, rowTreeBytes+merkle.TreeBufferSize(c.config.K))

	rowTree := buildRowTree(rows, c.config, buf)
	rowRoot := rowTree.Root()

	colChalls := rlc.DeriveTensorChallenges(rowRoot, logCols, rowBytes)
	coeffs := rlc.TensorCoefficients(colChalls)
	rlcVec := rlc.Compute(rows[:c.config.K], coeffs, c.config.WorkerCount)

	rlcTree := buildRLCTree(rlcVec, c.config, buf[rowTreeBytes:])
	rlcRoot := rlcTree.Root()

	h := sha256.New()
	h.Write(rowRoot[:])
	h.Write(rlcRoot[:])
	var commitment Commitment
	h.Sum(commitment[:0])

	ed := &ExtendedData{
		config:     c.config,
		rows:       rows,
		rlc:        rlcVec,
		commitment: commitment,
		rowsTree:   rowTree,
		rlcTree:    rlcTree,
	}
	return &StructuredCommitment{ed: ed, colChalls: colChalls, logCols: logCols}, nil
}

// EvalProof is a subset-evaluation proof: it convinces a verifier that the
// multilinear extension of the sub-matrix X_R (rows Range of the committed
// square) evaluates to Value at the Fiat-Shamir point (r_row, r_col).
//
// Yr is the full per-row partial-evaluation vector (public in ZODA); it is
// bound to the commitment's rlcRoot by recomputation, and to the actual rows by
// the sampled-row proximity check. SampledRows are random rows (originals or
// parity) whose tensor-fold must match the RS-extension of Yr at their index.
type EvalProof struct {
	Range       RowRange
	Yr          rlc.Vector  // K partial evaluations yr[j] = MLE(row_j)(r_col)
	SampledRows []*RowProof // FS-random rows for the proximity/consistency check
	Value       field.GF128 // claimed MLE_{X_R}(r_row, r_col)
}

// OpenEvaluation produces a subset-evaluation proof for the rollup owning
// row-range r, sampling sampleCount rows for the proximity check. The row
// challenges are Fiat-Shamir-derived from the commitment so they cannot be
// chosen adversarially (standalone PCS use).
func (sc *StructuredCommitment) OpenEvaluation(r RowRange, sampleCount int) (*EvalProof, error) {
	if err := r.validate(sc.ed.config.K); err != nil {
		return nil, err
	}
	logRows := bits.Len(uint(r.Len)) - 1
	rRow := deriveEvalChallenges(sc.ed.commitment, r, logRows)
	return sc.OpenEvaluationAt(r, rRow, sampleCount)
}

// OpenEvaluationAt is like OpenEvaluation but evaluates the rollup's sub-matrix
// at an externally supplied row point rRow (len must be log2(r.Len)). This is
// the weld point for an interactive protocol (e.g. GKR/sumcheck): the protocol
// transcript dictates rRow, and the DA commitment discharges the opening at
// exactly that point. Sample indices remain Fiat-Shamir-bound to the
// commitment.
func (sc *StructuredCommitment) OpenEvaluationAt(r RowRange, rRow []field.GF128, sampleCount int) (*EvalProof, error) {
	k, n := sc.ed.config.K, sc.ed.config.N
	if err := r.validate(k); err != nil {
		return nil, err
	}
	if want := bits.Len(uint(r.Len)) - 1; len(rRow) != want {
		return nil, fmt.Errorf("expected %d row challenges for range len %d, got %d", want, r.Len, len(rRow))
	}

	yrRange := sc.ed.rlc[r.Start : r.Start+r.Len]
	value := foldGF128(yrRange, rRow)

	indices := deriveSampleIndices(sc.ed.commitment, r, sampleCount, k+n)
	sampled := make([]*RowProof, 0, len(indices))
	for _, idx := range indices {
		p, err := sc.ed.GenerateRowProof(idx)
		if err != nil {
			return nil, fmt.Errorf("sampling row %d: %w", idx, err)
		}
		sampled = append(sampled, p)
	}

	yrFull := make(rlc.Vector, len(sc.ed.rlc))
	copy(yrFull, sc.ed.rlc)

	return &EvalProof{Range: r, Yr: yrFull, SampledRows: sampled, Value: value}, nil
}

// VerifyEvaluation checks a subset-evaluation proof against commitment and
// returns the verified evaluation value. It (1) recovers rowRoot from the
// sampled row proofs and rlcRoot from Yr, (2) checks the commitment, (3)
// re-derives the column tensor challenges from rowRoot and RS-extends Yr, (4)
// checks each sampled row folds to the extended Yr value at its index
// (proximity/unique decoding), and (5) recomputes the range evaluation from Yr
// and the re-derived row challenges and matches it against the claimed Value.
func VerifyEvaluation(cfg *Config, commitment Commitment, proof *EvalProof) (field.GF128, error) {
	if err := proof.Range.validate(cfg.K); err != nil {
		return field.GF128{}, err
	}
	logRows := bits.Len(uint(proof.Range.Len)) - 1
	rRow := deriveEvalChallenges(commitment, proof.Range, logRows)
	return VerifyEvaluationAt(cfg, commitment, proof, rRow)
}

// VerifyEvaluationAt is like VerifyEvaluation but checks the opening at an
// externally supplied row point rRow (the weld point for a GKR/sumcheck
// transcript) rather than a Fiat-Shamir-derived one.
func VerifyEvaluationAt(cfg *Config, commitment Commitment, proof *EvalProof, rRow []field.GF128) (field.GF128, error) {
	var zero field.GF128
	if err := cfg.Validate(); err != nil {
		return zero, fmt.Errorf("invalid config: %w", err)
	}
	if err := proof.Range.validate(cfg.K); err != nil {
		return zero, err
	}
	if len(proof.Yr) != cfg.K {
		return zero, fmt.Errorf("expected %d partial evaluations, got %d", cfg.K, len(proof.Yr))
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

	// (2) rlcRoot from the full Yr vector, then the commitment check.
	rlcRoot := computeRLCRoot(proof.Yr, make([]byte, cfg.K*merkle.NodeSize), make([]byte, field.GF128Size))
	h := sha256.New()
	h.Write(rowRoot[:])
	h.Write(rlcRoot[:])
	var recomputed Commitment
	h.Sum(recomputed[:0])
	if recomputed != commitment {
		return zero, fmt.Errorf("commitment verification failed")
	}

	// (3) re-derive column tensor challenges from rowRoot and RS-extend Yr.
	numSymbols := rowBytes / 2
	if numSymbols <= 0 || numSymbols&(numSymbols-1) != 0 {
		return zero, fmt.Errorf("row bytes %d give non-power-of-two symbol count", rowBytes)
	}
	logCols := bits.Len(uint(numSymbols)) - 1
	colChalls := rlc.DeriveTensorChallenges(rowRoot, logCols, rowBytes)
	coeffs := rlc.TensorCoefficients(colChalls)

	yrExt, err := rsExtendGF128(cfg, proof.Yr)
	if err != nil {
		return zero, fmt.Errorf("extending partial evaluations: %w", err)
	}

	// (4) proximity / consistency: each sampled row must fold (via the same
	// column tensor coefficients) to the RS-extension of Yr at its index. By
	// linearity of both the RS code and the fold, this holds for parity rows
	// too, which is what binds Yr to a unique underlying matrix.
	for _, p := range proof.SampledRows {
		got := rlc.ComputeRow(p.Row, coeffs)
		if !field.Equal128(got, yrExt[p.Index]) {
			return zero, fmt.Errorf("row %d: fold does not match committed partial evaluation", p.Index)
		}
	}

	// (5) recompute the range evaluation at the supplied point and match the
	// claimed value.
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

// foldGF128 evaluates the multilinear extension of a GF(2^128) coefficient
// vector at the point (challenges) by the standard variable-by-variable fold,
// LSB (last challenge) first, matching rlc.EvalMultilinearRow's ordering.
// len(vals) must equal 2^len(challenges).
func foldGF128(vals rlc.Vector, challenges []field.GF128) field.GF128 {
	cur := make(rlc.Vector, len(vals))
	copy(cur, vals)
	for c := len(challenges) - 1; c >= 0; c-- {
		r := challenges[c]
		oneMinusR := field.Add128(field.One(), r)
		half := len(cur) / 2
		next := make(rlc.Vector, half)
		for i := range half {
			lo := cur[2*i]
			hi := cur[2*i+1]
			next[i] = field.Add128(field.MulFull(oneMinusR, lo), field.MulFull(r, hi))
		}
		cur = next
	}
	return cur[0]
}

// rsExtendGF128 RS-extends a length-K GF128 vector to length K+N using the same
// Leopard GF(2^16) code the rows are encoded with, mirroring the verifier's RLC
// extension path.
func rsExtendGF128(cfg *Config, yr rlc.Vector) (rlc.Vector, error) {
	enc, err := reedsolomon.New(cfg.K, cfg.N, reedsolomon.WithLeopardGF16(true))
	if err != nil {
		return nil, err
	}
	total := cfg.K + cfg.N
	shardsBuf := make([]byte, total*field.LeopardChunkSize)
	shards := make([][]byte, total)
	for i := range shards {
		shards[i] = shardsBuf[i*field.LeopardChunkSize : (i+1)*field.LeopardChunkSize]
	}
	for i := 0; i < cfg.K; i++ {
		field.GF128ToLeopard(yr[i], shards[i])
	}
	if err := enc.Encode(shards); err != nil {
		return nil, err
	}
	out := make(rlc.Vector, total)
	for i := range out {
		out[i] = field.GF128FromLeopard(shards[i])
	}
	return out, nil
}

// deriveEvalChallenges derives logRows GF(2^128) row challenges bound to
// (commitment, range) via Fiat-Shamir.
func deriveEvalChallenges(commitment Commitment, r RowRange, logRows int) []field.GF128 {
	seed := evalSeed(commitment, r, "EVAL")
	challenges := make([]field.GF128, logRows)
	var input [32 + 4]byte
	copy(input[:32], seed[:])
	for i := range logRows {
		binary.LittleEndian.PutUint32(input[32:], uint32(i))
		challenges[i] = field.HashToGF128(sha256.Sum256(input[:]))
	}
	return challenges
}

// deriveSampleIndices derives `count` distinct indices in [0,total) bound to
// (commitment, range) via Fiat-Shamir. count is clamped to total.
func deriveSampleIndices(commitment Commitment, r RowRange, count, total int) []int {
	if count > total {
		count = total
	}
	seed := evalSeed(commitment, r, "SAMP")
	seen := make(map[int]struct{}, count)
	indices := make([]int, 0, count)
	var input [32 + 4]byte
	copy(input[:32], seed[:])
	for ctr := uint32(0); len(indices) < count; ctr++ {
		binary.LittleEndian.PutUint32(input[32:], ctr)
		d := sha256.Sum256(input[:])
		idx := int(binary.LittleEndian.Uint32(d[:]) % uint32(total))
		if _, ok := seen[idx]; ok {
			continue
		}
		seen[idx] = struct{}{}
		indices = append(indices, idx)
	}
	return indices
}

func evalSeed(commitment Commitment, r RowRange, tag string) [32]byte {
	h := sha256.New()
	h.Write(commitment[:])
	var rb [8]byte
	binary.LittleEndian.PutUint32(rb[0:4], uint32(r.Start))
	binary.LittleEndian.PutUint32(rb[4:8], uint32(r.Len))
	h.Write(rb[:])
	h.Write([]byte(tag))
	var seed [32]byte
	h.Sum(seed[:0])
	return seed
}
