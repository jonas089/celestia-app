package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/merkle"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
)

// This file is the additive C-ABI surface for the FULL-POINT opening of the
// ORIGINAL Encode commitment (OpenAtFullLegacy / VerifyAtFullLegacy, see
// pkg/rsema1d/pcs_full_legacy.go).
// It mirrors cshim.go exactly, but the point is supplied as TWO little-endian
// GF128 blobs — the column point rCol and the row point rRow — because an
// Expander/GKR opening fixes both axes of the committed square. The existing
// rsema1d_open_at / rsema1d_verify_at exports (row-only, FS column point) are
// untouched.

// rsema1d_open_at_full opens the subset evaluation for [rangeStart,
// rangeStart+rangeLen) at the FULL point (rCol, rRow) and serializes the
// EvalProofFull into a freshly C.malloc'd buffer (*outProof, *outProofLen). The
// caller owns that buffer and must free it with rsema1d_free_buf.
//
// rcol is rcolLen bytes = 16 * log2(numSymbols) little-endian GF128 elements;
// rrow is rrowLen bytes = 16 * log2(rangeLen) little-endian GF128 elements.
//
// Return codes: 0 ok; 1 unknown handle; 2 rcolLen/rrowLen not a multiple of 16;
// 3 OpenEvaluationAtFull failed; 4 allocation failed.
//
//export rsema1d_open_at_full
func rsema1d_open_at_full(handle C.uint64_t, rangeStart, rangeLen C.uint32_t, rcol *C.uchar, rcolLen C.size_t, rrow *C.uchar, rrowLen C.size_t, sampleCount C.uint32_t, outProof **C.uchar, outProofLen *C.size_t) C.int {
	mu.Lock()
	ed := registry[uint64(handle)]
	mu.Unlock()
	if ed == nil {
		return 1
	}

	rCol, ok := decodePoint(rcol, int(rcolLen))
	if !ok {
		return 2
	}
	rRow, ok := decodePoint(rrow, int(rrowLen))
	if !ok {
		return 2
	}

	r := rsema1d.RowRange{Start: int(rangeStart), Len: int(rangeLen)}
	proof, err := ed.OpenAtFullLegacy(r, rCol, rRow, int(sampleCount))
	if err != nil {
		return 3
	}

	blob := serializeProofFull(proof)
	cbuf := C.malloc(C.size_t(len(blob)))
	if cbuf == nil {
		return 4
	}
	dst := unsafe.Slice((*byte)(cbuf), len(blob))
	copy(dst, blob)
	*outProof = (*C.uchar)(cbuf)
	*outProofLen = C.size_t(len(blob))
	return 0
}

// rsema1d_verify_at_full deserializes an EvalProofFull and verifies it against
// the 32-byte commitment at the full point (rCol, rRow), writing the verified
// 16-byte GF128 value into outValue.
//
// Return codes: 0 verified; 1 nil pointer arg; 2 proof deserialization failed;
// 3 rcolLen/rrowLen not a multiple of 16; 4 VerifyEvaluationAtFull rejected.
//
//export rsema1d_verify_at_full
func rsema1d_verify_at_full(k, n C.uint32_t, commitment *C.uchar, proof *C.uchar, proofLen C.size_t, rcol *C.uchar, rcolLen C.size_t, rrow *C.uchar, rrowLen C.size_t, outValue *C.uchar) C.int {
	if commitment == nil || proof == nil || outValue == nil {
		return 1
	}

	var commit rsema1d.Commitment
	copy(commit[:], C.GoBytes(unsafe.Pointer(commitment), commitmentSize))

	blob := C.GoBytes(unsafe.Pointer(proof), C.int(proofLen))
	ep, err := deserializeProofFull(blob)
	if err != nil {
		return 2
	}

	rCol, ok := decodePoint(rcol, int(rcolLen))
	if !ok {
		return 3
	}
	rRow, ok := decodePoint(rrow, int(rrowLen))
	if !ok {
		return 3
	}

	cfg := &rsema1d.Config{K: int(k), N: int(n), WorkerCount: 1}
	val, err := rsema1d.VerifyAtFullLegacy(cfg, commit, ep, rCol, rRow)
	if err != nil {
		return 4
	}

	out := unsafe.Slice((*byte)(unsafe.Pointer(outValue)), field.GF128Size)
	field.EncodeGF128(out, val)
	return 0
}

// --- EvalProofFull binary format (all integers little-endian) ---
//
//	Range.Start        u32
//	Range.Len          u32
//	Value              16 bytes (GF128, little-endian)
//	len(Yr)            u32
//	Yr[i]              16 bytes each
//	len(YrPrime)       u32
//	YrPrime[i]         16 bytes each
//	len(SampledRows)   u32
//	per sampled row:
//	  Index            u32
//	  RowLen           u32
//	  Row              RowLen bytes
//	  ProofDepth       u32
//	  node[j]          32 bytes each (merkle.NodeSize)

func serializeProofFull(p *rsema1d.EvalProofFull) []byte {
	var b bytes.Buffer
	var scratch [4]byte
	putU32 := func(v uint32) {
		binary.LittleEndian.PutUint32(scratch[:], v)
		b.Write(scratch[:])
	}
	var vbuf [field.GF128Size]byte
	putGF := func(g field.GF128) {
		field.EncodeGF128(vbuf[:], g)
		b.Write(vbuf[:])
	}

	putU32(uint32(p.Range.Start))
	putU32(uint32(p.Range.Len))
	putGF(p.Value)

	putU32(uint32(len(p.Yr)))
	for _, y := range p.Yr {
		putGF(y)
	}
	putU32(uint32(len(p.YrPrime)))
	for _, y := range p.YrPrime {
		putGF(y)
	}

	putU32(uint32(len(p.SampledRows)))
	for _, sr := range p.SampledRows {
		putU32(uint32(sr.Index))
		putU32(uint32(len(sr.Row)))
		b.Write(sr.Row)
		putU32(uint32(len(sr.RowProof)))
		for _, node := range sr.RowProof {
			b.Write(node)
		}
	}
	return b.Bytes()
}

func deserializeProofFull(blob []byte) (*rsema1d.EvalProofFull, error) {
	r := &reader{buf: blob}

	start, err := r.u32()
	if err != nil {
		return nil, err
	}
	length, err := r.u32()
	if err != nil {
		return nil, err
	}
	value, err := r.gf128()
	if err != nil {
		return nil, err
	}

	yrLen, err := r.u32()
	if err != nil {
		return nil, err
	}
	yr := make(rlc.Vector, yrLen)
	for i := range yr {
		if yr[i], err = r.gf128(); err != nil {
			return nil, err
		}
	}

	yrPrimeLen, err := r.u32()
	if err != nil {
		return nil, err
	}
	yrPrime := make(rlc.Vector, yrPrimeLen)
	for i := range yrPrime {
		if yrPrime[i], err = r.gf128(); err != nil {
			return nil, err
		}
	}

	nRows, err := r.u32()
	if err != nil {
		return nil, err
	}
	sampled := make([]*rsema1d.RowProof, nRows)
	for i := range sampled {
		idx, err := r.u32()
		if err != nil {
			return nil, err
		}
		rowLen, err := r.u32()
		if err != nil {
			return nil, err
		}
		row, err := r.bytes(int(rowLen))
		if err != nil {
			return nil, err
		}
		depth, err := r.u32()
		if err != nil {
			return nil, err
		}
		path := make([][]byte, depth)
		for j := range path {
			if path[j], err = r.bytes(merkle.NodeSize); err != nil {
				return nil, err
			}
		}
		sampled[i] = &rsema1d.RowProof{Index: int(idx), Row: row, RowProof: path}
	}

	if r.off != len(r.buf) {
		return nil, fmt.Errorf("trailing %d bytes in full proof blob", len(r.buf)-r.off)
	}

	return &rsema1d.EvalProofFull{
		Range:       rsema1d.RowRange{Start: int(start), Len: int(length)},
		Yr:          yr,
		YrPrime:     yrPrime,
		SampledRows: sampled,
		Value:       value,
	}, nil
}
