package rlc

import (
	"crypto/sha256"
	"encoding/binary"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/merkle"
)

// This file provides the *structured* (tensor / logarithmic-randomness) RLC
// used by the GKR-friendly, multilinear-PCS variant of rsema1d.
//
// Whereas DeriveCoefficients (derive.go) produces numSymbols independent
// Fiat-Shamir coefficients — enough for a Ligero-style proximity check — the
// tensor variant produces coefficients
//
//	coeff[i] = ⊗_{b=1..k} (1 - r_b, r_b) [i]      (paper eq. 1)
//
// so that folding a row's GF(2^16) symbols with them yields the *multilinear
// extension* of that row evaluated at the challenge point (r_1,...,r_k) over
// GF(2^128):
//
//	ComputeRow(row, TensorCoefficients(r)) == MLE(row)(r_1,...,r_k)
//
// This is exactly the partial-evaluation vector yr = X̃·ḡr of the ZODA paper:
// each row j of the data matrix contributes yr[j] = MLE(row_j)(r). Combining
// yr across rows with a second tensor gives the full multilinear evaluation an
// opened polynomial commitment needs. Because challenges are drawn from the
// large field GF(2^128), the soundness error of the associated proximity /
// evaluation check is ≈ k/2^128 (logarithmic randomness, DP24/AER24), rather
// than the k/2^16 an in-field GF(2^16) challenge would give.

// DeriveTensorChallenges derives k = logCols independent GF(2^128) challenges
// bound to (rowRoot, logCols, rowSize) via Fiat-Shamir. logCols is the number
// of multilinear variables, i.e. log2 of the number of GF(2^16) symbols per
// row (numSymbols = rowSize/2 must equal 2^logCols).
func DeriveTensorChallenges(rowRoot merkle.Root, logCols, rowSize int) []field.GF128 {
	h := sha256.New()
	h.Write(rowRoot[:])
	var params [12]byte
	binary.LittleEndian.PutUint32(params[0:4], uint32(logCols))
	binary.LittleEndian.PutUint32(params[4:8], uint32(rowSize))
	binary.LittleEndian.PutUint32(params[8:12], 0x54454e53) // "TENS" domain tag
	h.Write(params[:])
	var seed [32]byte
	h.Sum(seed[:0])

	challenges := make([]field.GF128, logCols)
	var input [32 + 4]byte
	copy(input[:32], seed[:])
	for i := range logCols {
		binary.LittleEndian.PutUint32(input[32:], uint32(i))
		challenges[i] = field.HashToGF128(sha256.Sum256(input[:]))
	}
	return challenges
}

// TensorCoefficients builds the tensor product ⊗_b (1 - r_b, r_b) as a flat
// Vector of length 2^len(challenges). The last challenge varies fastest: index
// i has bit b (LSB = last challenge) selecting r_b when set and (1 - r_b) when
// clear. This ordering is matched by EvalMultilinearRow so the two agree.
func TensorCoefficients(challenges []field.GF128) Vector {
	coeffs := Vector{field.One()}
	for _, r := range challenges {
		oneMinusR := field.Add128(field.One(), r) // 1 - r = 1 + r in char 2
		next := make(Vector, len(coeffs)*2)
		for j, c := range coeffs {
			next[2*j] = field.MulFull(c, oneMinusR) // bit = 0
			next[2*j+1] = field.MulFull(c, r)       // bit = 1
		}
		coeffs = next
	}
	return coeffs
}

// EvalMultilinearRow evaluates the multilinear extension of a row (its
// GF(2^16) symbols as coefficients over the boolean hypercube) at the point
// (challenges) in GF(2^128), by the standard variable-by-variable fold. It is
// an independent reference for the TensorCoefficients + ComputeRow path: both
// must return the same field element. numSymbols = len(row)/2 must be
// 2^len(challenges).
func EvalMultilinearRow(row []byte, challenges []field.GF128) field.GF128 {
	numSymbols := len(row) / 2
	// Lift the GF(2^16) symbols into GF(2^128) (as subfield constants).
	cur := make([]field.GF128, numSymbols)
	for i := range numSymbols {
		cur[i] = field.GF128{field.GF16FromLeopard(row, i)}
	}
	// Fold one variable per challenge, LSB (last challenge) first, so the
	// pairing matches TensorCoefficients' index bit ordering.
	for c := len(challenges) - 1; c >= 0; c-- {
		r := challenges[c]
		half := len(cur) / 2
		next := make([]field.GF128, half)
		for i := range half {
			lo := cur[2*i]   // bit = 0
			hi := cur[2*i+1] // bit = 1
			// (1-r)*lo + r*hi
			next[i] = field.Add128(
				field.MulFull(field.Add128(field.One(), r), lo),
				field.MulFull(r, hi),
			)
		}
		cur = next
	}
	return cur[0]
}
