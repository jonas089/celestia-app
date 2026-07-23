// Package main is the C-ABI (c-shared) shim around the pure-Go rsema1d
// multilinear PCS. It lets non-Go callers (notably Rust, via the
// rust/crates/rsema1d-sys crate) drive the *real* Go prover/verifier over cgo,
// so commitments are byte-identical to the celestia-app DA layer with no
// reimplementation of the Leopard GF(2^16) encoder.
//
// Build:
//
//	go build -buildmode=c-shared -o librsema1d.dylib ./pkg/rsema1d/cshim
//
// This emits librsema1d.dylib and a matching librsema1d.h header.
//
// The API is handle-based: rsema1d_commit stores the StructuredCommitment in a
// mutex-guarded Go registry keyed by a monotonic uint64 handle and returns the
// 32-byte commitment. rsema1d_open_at looks the handle up and serializes an
// EvalProof into a C.malloc'd buffer the caller must release with
// rsema1d_free_buf. rsema1d_verify_at is stateless.
//
// cgo pointer rules: no Go pointer is ever handed to or retained by the caller.
// Every input is copied into Go memory on entry (C.GoBytes / unsafe.Slice read)
// and every output is copied into caller-owned C memory (the out* buffers, or a
// C.malloc'd buffer for the variable-length proof).
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
	"sync"
	"unsafe"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/merkle"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
)

// registry maps opaque uint64 handles to the live ExtendedData objects held on
// the Go heap. These come from the ORIGINAL Coder.Encode (the canonical DA/spec
// commitment), so FFI commitments are byte-identical to what the celestia-app
// DA layer and cmd/testvectors produce — the accidental-computer ENCODE-ONCE
// invariant. Guarded by mu; handles come from the monotonic counter nextHandle
// (starts at 1 so 0 is never a valid handle).
var (
	mu         sync.Mutex
	registry   = make(map[uint64]*rsema1d.ExtendedData)
	nextHandle uint64 = 1
)

// main is required by -buildmode=c-shared but never runs.
func main() {}

// commitmentSize is the byte length of a Commitment ([32]byte).
const commitmentSize = 32

// rsema1d_commit encodes a K+N row matrix with the ORIGINAL Coder.Encode (the
// canonical DA/spec commitment: rows RS-encoded -> rowRoot; legacy
// DeriveCoefficients RLC -> rlcRoot; SHA256(rowRoot||rlcRoot)), stores the
// resulting ExtendedData in the handle registry, and writes the 32-byte
// commitment. This is byte-identical to cmd/testvectors and to DA sampling.
//
// rows points at numRows contiguous rowLen-byte rows (the full K+N matrix with
// parity rows zeroed, matching Encode's contract). outCommitment must have room
// for 32 bytes; outHandle receives the registry handle.
//
// Return codes: 0 ok; 1 nil pointer arg; 2 numRows != k+n; 3 NewCoder failed
// (bad config); 4 Encode failed.
//
//export rsema1d_commit
func rsema1d_commit(k, n C.uint32_t, rows *C.uchar, rowLen, numRows C.size_t, outCommitment *C.uchar, outHandle *C.uint64_t) C.int {
	if rows == nil || outCommitment == nil || outHandle == nil {
		return 1
	}
	kk, nn := int(k), int(n)
	rl, nr := int(rowLen), int(numRows)
	if nr != kk+nn {
		return 2
	}

	// Copy the whole matrix into Go memory (never retain the caller's pointer).
	total := rl * nr
	src := C.GoBytes(unsafe.Pointer(rows), C.int(total))
	goRows := make([][]byte, nr)
	for i := 0; i < nr; i++ {
		// Fresh backing slices; Encode mutates parity rows in place.
		row := make([]byte, rl)
		copy(row, src[i*rl:(i+1)*rl])
		goRows[i] = row
	}

	cfg := &rsema1d.Config{K: kk, N: nn, WorkerCount: 1}
	coder, err := rsema1d.NewCoder(cfg)
	if err != nil {
		return 3
	}
	ed, err := coder.Encode(goRows)
	if err != nil {
		return 4
	}
	noteEncode() // count this RS-encode (see cshim_da.go: encodeCalls)

	commitment := ed.Commitment()
	out := unsafe.Slice((*byte)(unsafe.Pointer(outCommitment)), commitmentSize)
	copy(out, commitment[:])

	mu.Lock()
	h := nextHandle
	nextHandle++
	registry[h] = ed
	mu.Unlock()
	*outHandle = C.uint64_t(h)
	return 0
}

// rsema1d_open_at opens the subset evaluation for the row range [rangeStart,
// rangeStart+rangeLen) at the externally supplied point, sampling sampleCount
// rows for the proximity check, and serializes the EvalProof into a freshly
// C.malloc'd buffer (*outProof, *outProofLen). The caller owns that buffer and
// must free it with rsema1d_free_buf.
//
// point is pointLen bytes = 16 * log2(rangeLen) little-endian GF128 elements.
//
// Return codes: 0 ok; 1 unknown handle; 2 pointLen not a multiple of 16;
// 3 OpenAtLegacy failed; 4 allocation failed.
//
//export rsema1d_open_at
func rsema1d_open_at(handle C.uint64_t, rangeStart, rangeLen C.uint32_t, point *C.uchar, pointLen C.size_t, sampleCount C.uint32_t, outProof **C.uchar, outProofLen *C.size_t) C.int {
	mu.Lock()
	ed := registry[uint64(handle)]
	mu.Unlock()
	if ed == nil {
		return 1
	}

	rRow, ok := decodePoint(point, int(pointLen))
	if !ok {
		return 2
	}

	r := rsema1d.RowRange{Start: int(rangeStart), Len: int(rangeLen)}
	proof, err := ed.OpenAtLegacy(r, rRow, int(sampleCount))
	if err != nil {
		return 3
	}

	blob := serializeProof(proof)
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

// rsema1d_verify_at deserializes an EvalProof and verifies it against the
// 32-byte commitment at the supplied point, writing the verified 16-byte GF128
// value into outValue. rc 0 means verified.
//
// Return codes: 0 verified; 1 nil pointer arg; 2 proof deserialization failed;
// 3 pointLen not a multiple of 16; 4 VerifyAtLegacy rejected the proof.
//
//export rsema1d_verify_at
func rsema1d_verify_at(k, n C.uint32_t, commitment *C.uchar, proof *C.uchar, proofLen C.size_t, point *C.uchar, pointLen C.size_t, outValue *C.uchar) C.int {
	if commitment == nil || proof == nil || outValue == nil {
		return 1
	}

	var commit rsema1d.Commitment
	copy(commit[:], C.GoBytes(unsafe.Pointer(commitment), commitmentSize))

	blob := C.GoBytes(unsafe.Pointer(proof), C.int(proofLen))
	ep, err := deserializeProof(blob)
	if err != nil {
		return 2
	}

	rRow, ok := decodePoint(point, int(pointLen))
	if !ok {
		return 3
	}

	cfg := &rsema1d.Config{K: int(k), N: int(n), WorkerCount: 1}
	val, err := rsema1d.VerifyAtLegacy(cfg, commit, ep, rRow)
	if err != nil {
		return 4
	}

	out := unsafe.Slice((*byte)(unsafe.Pointer(outValue)), field.GF128Size)
	field.EncodeGF128(out, val)
	return 0
}

// rsema1d_free_handle drops the StructuredCommitment for handle from the
// registry, releasing it to the Go GC. No-op for unknown handles.
//
//export rsema1d_free_handle
func rsema1d_free_handle(handle C.uint64_t) {
	mu.Lock()
	delete(registry, uint64(handle))
	mu.Unlock()
}

// rsema1d_free_buf frees a buffer previously returned by rsema1d_open_at.
//
//export rsema1d_free_buf
func rsema1d_free_buf(buf *C.uchar) {
	if buf != nil {
		C.free(unsafe.Pointer(buf))
	}
}

// decodePoint copies pointLen bytes of little-endian GF128 elements out of C
// memory into a Go []field.GF128. ok is false if pointLen is not a multiple of
// the 16-byte GF128 encoding. A zero-length point yields an empty (non-nil)
// slice, which is valid for a range of length 1 (log2 = 0 challenges).
func decodePoint(point *C.uchar, pointLen int) ([]field.GF128, bool) {
	if pointLen%field.GF128Size != 0 {
		return nil, false
	}
	n := pointLen / field.GF128Size
	rRow := make([]field.GF128, n)
	if pointLen == 0 {
		return rRow, true
	}
	pb := C.GoBytes(unsafe.Pointer(point), C.int(pointLen))
	for i := 0; i < n; i++ {
		rRow[i] = field.DecodeGF128(pb[i*field.GF128Size:])
	}
	return rRow, true
}

// --- EvalProof binary format (all integers little-endian) ---
//
//	Range.Start        u32
//	Range.Len          u32
//	Value              16 bytes (GF128, little-endian)
//	len(Yr)            u32
//	Yr[i]              16 bytes each
//	len(SampledRows)   u32
//	per sampled row:
//	  Index            u32
//	  RowLen           u32
//	  Row              RowLen bytes
//	  ProofDepth       u32
//	  node[j]          32 bytes each (merkle.NodeSize)

func serializeProof(p *rsema1d.EvalProof) []byte {
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

	putU32(uint32(len(p.SampledRows)))
	for _, sr := range p.SampledRows {
		putU32(uint32(sr.Index))
		putU32(uint32(len(sr.Row)))
		b.Write(sr.Row)
		putU32(uint32(len(sr.RowProof)))
		for _, node := range sr.RowProof {
			b.Write(node) // merkle.NodeSize bytes each
		}
	}
	return b.Bytes()
}

func deserializeProof(blob []byte) (*rsema1d.EvalProof, error) {
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
		return nil, fmt.Errorf("trailing %d bytes in proof blob", len(r.buf)-r.off)
	}

	return &rsema1d.EvalProof{
		Range:       rsema1d.RowRange{Start: int(start), Len: int(length)},
		Yr:          yr,
		SampledRows: sampled,
		Value:       value,
	}, nil
}

// reader is a bounds-checked cursor over the proof blob.
type reader struct {
	buf []byte
	off int
}

func (r *reader) bytes(n int) ([]byte, error) {
	if n < 0 || r.off+n > len(r.buf) {
		return nil, fmt.Errorf("short read: want %d bytes at offset %d of %d", n, r.off, len(r.buf))
	}
	// Copy so the returned slice does not alias the (caller-owned) input blob.
	out := make([]byte, n)
	copy(out, r.buf[r.off:r.off+n])
	r.off += n
	return out, nil
}

func (r *reader) u32() (uint32, error) {
	b, err := r.bytes(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (r *reader) gf128() (field.GF128, error) {
	b, err := r.bytes(field.GF128Size)
	if err != nil {
		return field.GF128{}, err
	}
	return field.DecodeGF128(b), nil
}
