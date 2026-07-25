package rsema1d

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
	"github.com/klauspost/reedsolomon"
)

// This file holds the pieces shared by the multilinear-PCS layer built on top
// of the rsema1d DA encoding — the "evaluation over a subset of rows"
// construction of the accidental-computer paper (§5). A shared square holds
// many rollups; each rollup owns a contiguous, power-of-two-aligned block of
// original rows (a RowRange), and proves that the multilinear extension of its
// own sub-matrix evaluates to a value at a supplied point, reusing the DA
// encoding as the commitment. The opening itself lives in pcs_da.go.

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
