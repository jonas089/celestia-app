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
// The API is handle-based: rsema1d_commit stores the ExtendedData in a
// mutex-guarded Go registry keyed by a monotonic uint64 handle and returns the
// 32-byte commitment. rsema1d_open_at_full (cshim_full.go) looks the handle up
// and serializes an EvalProofFull into a C.malloc'd buffer the caller must
// release with rsema1d_free_buf. rsema1d_verify_at_full is stateless.
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
	"encoding/binary"
	"fmt"
	"sync"
	"unsafe"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
)

// registry maps opaque uint64 handles to the live ExtendedData objects held on
// the Go heap. These come from the ORIGINAL Coder.Encode (the canonical DA/spec
// commitment), so FFI commitments are byte-identical to what the celestia-app
// DA layer and cmd/testvectors produce — the accidental-computer ENCODE-ONCE
// invariant. Guarded by mu; handles come from the monotonic counter nextHandle
// (starts at 1 so 0 is never a valid handle).
var (
	mu         sync.Mutex
	registry          = make(map[uint64]*rsema1d.ExtendedData)
	nextHandle uint64 = 1
)

// main is required by -buildmode=c-shared but never runs.
func main() {}

// commitmentSize is the byte length of a Commitment ([32]byte).
const commitmentSize = 32

// rsema1d_commit encodes a K+N row matrix with the ORIGINAL Coder.Encode (the
// canonical DA/spec commitment: rows RS-encoded -> rowRoot; spec
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

// rsema1d_free_handle drops the ExtendedData for handle from the
// registry, releasing it to the Go GC. No-op for unknown handles.
//
//export rsema1d_free_handle
func rsema1d_free_handle(handle C.uint64_t) {
	mu.Lock()
	delete(registry, uint64(handle))
	mu.Unlock()
}

// rsema1d_free_buf frees a buffer previously returned by rsema1d_open_at_full.
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
