package rsema1d

import (
	"crypto/sha256"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
)

// This file implements the interactive-proof layer that welds to the rsema1d
// multilinear commitment: a Fiat-Shamir sumcheck over GF(2^128) proving the
// value of Σ_{x∈{0,1}^v} f(x) for a multilinear polynomial f given by its
// evaluations on the hypercube. This is the shape of a GKR layer's final
// reduction: it collapses a global claim about f down to a single evaluation
// f(r*) at a transcript-chosen point r*, which is then discharged NOT by a
// dedicated polynomial commitment but by opening the DA commitment at r* via
// VerifyEvaluationAt. That reuse — the data-availability encoding standing in
// for the polynomial commitment the proof would otherwise need — is the
// "accidental computer".
//
// The sum being proven (Σ f) is a genuine, if simple, statement about the
// rollup's committed data (a checksum of its per-row partial evaluations); the
// point of this layer is to demonstrate a real sumcheck whose final opening is
// served by rsema1d. Richer GKR circuits (the lifted RETH STF) reduce to the
// same final-evaluation primitive.

// SumcheckProof is a transcript of a sum-check over a multilinear polynomial.
// Claim is the asserted Σ_x f(x); Rounds[i] = (g_i(0), g_i(1)) are the two
// evaluations of the i-th round's (degree-1) univariate.
type SumcheckProof struct {
	Claim  field.GF128
	Rounds [][2]field.GF128
}

// transcript is a minimal Fiat-Shamir transcript over GF(2^128): absorb field
// elements, squeeze challenges. Prover and verifier drive it identically.
type transcript struct{ state [32]byte }

func newTranscript(domain string) *transcript {
	t := &transcript{}
	t.state = sha256.Sum256([]byte("rsema1d/sumcheck/" + domain))
	return t
}

func (t *transcript) absorb(x field.GF128) {
	var buf [32 + field.GF128Size]byte
	copy(buf[:32], t.state[:])
	field.EncodeGF128(buf[32:], x)
	t.state = sha256.Sum256(buf[:])
}

func (t *transcript) challenge() field.GF128 {
	var buf [32 + 4]byte
	copy(buf[:32], t.state[:])
	copy(buf[32:], "CHAL")
	t.state = sha256.Sum256(buf[:])
	return field.HashToGF128(t.state)
}

// ProveSum runs the sum-check for Σ_x f(x) where f is the multilinear extension
// of evals (len must be a power of two). It returns the proof, the transcript
// point r* in the convention foldGF128/OpenEvaluationAt expect (so
// foldGF128(evals, point) == finalEval), and finalEval = f(r*).
func ProveSum(evals rlc.Vector) (*SumcheckProof, []field.GF128, field.GF128) {
	v := log2(len(evals))
	table := make(rlc.Vector, len(evals))
	copy(table, evals)

	claim := field.Zero()
	for _, e := range table {
		claim = field.Add128(claim, e)
	}

	tr := newTranscript("sum")
	tr.absorb(claim)

	proof := &SumcheckProof{Claim: claim, Rounds: make([][2]field.GF128, 0, v)}
	raw := make([]field.GF128, 0, v)
	for range v {
		half := len(table) / 2
		var g0, g1 field.GF128
		for i := range half {
			g0 = field.Add128(g0, table[2*i])   // x_lsb = 0 (even indices)
			g1 = field.Add128(g1, table[2*i+1]) // x_lsb = 1 (odd indices)
		}
		proof.Rounds = append(proof.Rounds, [2]field.GF128{g0, g1})
		tr.absorb(g0)
		tr.absorb(g1)
		r := tr.challenge()
		raw = append(raw, r)
		table = foldOneLSB(table, r)
	}
	return proof, reverseGF(raw), table[0]
}

// VerifySum re-derives the transcript, checks every round (g_i(0)+g_i(1) equals
// the running claim), and returns the transcript point r* (matching
// foldGF128/OpenEvaluationAt convention) and the final claim f(r*) that must be
// discharged by a commitment opening. err is non-nil if any round is
// inconsistent.
func VerifySum(proof *SumcheckProof) ([]field.GF128, field.GF128, error) {
	tr := newTranscript("sum")
	tr.absorb(proof.Claim)

	claim := proof.Claim
	raw := make([]field.GF128, 0, len(proof.Rounds))
	for i, round := range proof.Rounds {
		g0, g1 := round[0], round[1]
		if !field.Equal128(field.Add128(g0, g1), claim) {
			return nil, field.GF128{}, &sumcheckError{round: i}
		}
		tr.absorb(g0)
		tr.absorb(g1)
		r := tr.challenge()
		raw = append(raw, r)
		// next claim = g_i(r) = (1-r)*g0 + r*g1
		claim = field.Add128(
			field.MulFull(field.Add128(field.One(), r), g0),
			field.MulFull(r, g1),
		)
	}
	return reverseGF(raw), claim, nil
}

type sumcheckError struct{ round int }

func (e *sumcheckError) Error() string {
	return "sumcheck: round consistency check failed"
}

// foldOneLSB folds the least-significant variable of a multilinear table with
// challenge r: out[i] = (1-r)*table[2i] + r*table[2i+1]. Halves the length.
func foldOneLSB(table rlc.Vector, r field.GF128) rlc.Vector {
	oneMinusR := field.Add128(field.One(), r)
	half := len(table) / 2
	out := make(rlc.Vector, half)
	for i := range half {
		out[i] = field.Add128(
			field.MulFull(oneMinusR, table[2*i]),
			field.MulFull(r, table[2*i+1]),
		)
	}
	return out
}

func reverseGF(in []field.GF128) []field.GF128 {
	out := make([]field.GF128, len(in))
	for i := range in {
		out[len(in)-1-i] = in[i]
	}
	return out
}

func log2(n int) int {
	k := 0
	for 1<<k < n {
		k++
	}
	return k
}
