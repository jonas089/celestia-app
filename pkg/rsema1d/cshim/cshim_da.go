package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/binary"
	"sync/atomic"
	"unsafe"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d"
)

// This file is the additive C-ABI surface for the TRUE cross-process DA
// hand-off of the "accidental computer".
//
// The invariant it makes demonstrable: the Reed-Solomon (Leopard GF(2^16))
// polynomial encoding of the block-STF input layer happens EXACTLY ONCE, on the
// DA-encoder side (rsema1d_encode_extended -> Coder.Encode). A separate prover
// process then LOADS the already-extended K+N rows (rsema1d_load_extended ->
// NewExtendedDataFromEncoded), which rebuilds only the Merkle/RLC commitment
// structures and does NO RS-encoding. Both sides obtain the identical 32-byte
// commitment, so the GKR prover opens the DA commitment without re-encoding.
//
// Because the handle registry (see cshim.go) is an in-process Go map keyed by a
// monotonic uint64, a handle is meaningless across an OS-process boundary; what
// crosses the boundary is the serialized extended-row matrix, not the handle.
//
// encodeCalls counts every invocation of the RS encoder (Coder.Encode) in THIS
// process, across rsema1d_commit and rsema1d_encode_extended. A prover process
// that only ever calls rsema1d_load_extended observes encodeCalls == 0 — the
// machine-checkable statement of "the prover does not re-encode".
var encodeCalls uint64

// noteEncode records one RS-encode (Coder.Encode) in this process.
func noteEncode() { atomic.AddUint64(&encodeCalls, 1) }

// rsema1d_encode_call_count returns the number of RS encodes (Coder.Encode) run
// in this process so far. Used by callers to prove the prover side did zero.
//
//export rsema1d_encode_call_count
func rsema1d_encode_call_count() C.uint64_t {
	return C.uint64_t(atomic.LoadUint64(&encodeCalls))
}

// --- serialized extended-row matrix format (all integers little-endian) ---
//
//	K        u32
//	N        u32
//	rowLen   u32
//	rows     (K+N) * rowLen bytes, row-major (originals then RS parity)

// serializeExtended packs the full K+N extended row matrix into a byte blob.
func serializeExtended(k, n, rowLen int, rows [][]byte) []byte {
	blob := make([]byte, 12+(k+n)*rowLen)
	binary.LittleEndian.PutUint32(blob[0:4], uint32(k))
	binary.LittleEndian.PutUint32(blob[4:8], uint32(n))
	binary.LittleEndian.PutUint32(blob[8:12], uint32(rowLen))
	off := 12
	for _, r := range rows {
		copy(blob[off:off+rowLen], r)
		off += rowLen
	}
	return blob
}

// rsema1d_encode_extended is the DA-ENCODER entry point. It RS-encodes the K+N
// row matrix ONCE with the ORIGINAL Coder.Encode (byte-identical to DA sampling
// / cmd/testvectors), writes the 32-byte commitment, and serializes the full
// extended matrix (originals + genuine RS parity) into a freshly C.malloc'd
// buffer (*outExtended, *outExtendedLen) the caller must free with
// rsema1d_free_buf. It does NOT register a handle: the DA side only emits bytes.
//
// rows points at numRows contiguous rowLen-byte rows (K+N rows; parity rows may
// be zeroed — Encode fills them). outCommitment must have room for 32 bytes.
//
// Return codes: 0 ok; 1 nil pointer arg; 2 numRows != k+n; 3 NewCoder failed;
// 4 Encode failed; 5 allocation failed.
//
//export rsema1d_encode_extended
func rsema1d_encode_extended(k, n C.uint32_t, rows *C.uchar, rowLen, numRows C.size_t, outCommitment *C.uchar, outExtended **C.uchar, outExtendedLen *C.size_t) C.int {
	if rows == nil || outCommitment == nil || outExtended == nil || outExtendedLen == nil {
		return 1
	}
	kk, nn := int(k), int(n)
	rl, nr := int(rowLen), int(numRows)
	if nr != kk+nn {
		return 2
	}

	total := rl * nr
	src := C.GoBytes(unsafe.Pointer(rows), C.int(total))
	goRows := make([][]byte, nr)
	for i := 0; i < nr; i++ {
		row := make([]byte, rl)
		copy(row, src[i*rl:(i+1)*rl])
		goRows[i] = row
	}

	cfg := &rsema1d.Config{K: kk, N: nn, WorkerCount: 1}
	coder, err := rsema1d.NewCoder(cfg)
	if err != nil {
		return 3
	}
	ed, err := coder.Encode(goRows) // the ONE RS-encode, on the DA side
	if err != nil {
		return 4
	}
	noteEncode()

	commitment := ed.Commitment()
	out := unsafe.Slice((*byte)(unsafe.Pointer(outCommitment)), commitmentSize)
	copy(out, commitment[:])

	// Serialize the FULL extended matrix (originals + RS parity) so the prover
	// can reconstruct the committed square without re-encoding.
	blob := make([][]byte, nr)
	for i := 0; i < nr; i++ {
		blob[i] = ed.Row(i)
	}
	packed := serializeExtended(kk, nn, rl, blob)

	cbuf := C.malloc(C.size_t(len(packed)))
	if cbuf == nil {
		return 5
	}
	dst := unsafe.Slice((*byte)(cbuf), len(packed))
	copy(dst, packed)
	*outExtended = (*C.uchar)(cbuf)
	*outExtendedLen = C.size_t(len(packed))
	return 0
}

// rsema1d_load_extended is the PROVER entry point. It deserializes an extended
// matrix produced by rsema1d_encode_extended and reconstructs the committed
// square via NewExtendedDataFromEncoded — rebuilding ONLY the Merkle/RLC
// commitment structures, with NO RS-encoding (encodeCalls is NOT incremented).
// The reconstructed ExtendedData is stored in the handle registry (shared with
// cshim.go) so subsequent rsema1d_open_at_full calls reuse it. It writes the
// recomputed 32-byte commitment (which must match the DA side's) and the handle.
//
// Return codes: 0 ok; 1 nil pointer arg; 2 blob too short / malformed header;
// 3 blob length inconsistent with header; 4 NewExtendedDataFromEncoded failed.
//
//export rsema1d_load_extended
func rsema1d_load_extended(extended *C.uchar, extendedLen C.size_t, outCommitment *C.uchar, outHandle *C.uint64_t) C.int {
	if extended == nil || outCommitment == nil || outHandle == nil {
		return 1
	}
	blob := C.GoBytes(unsafe.Pointer(extended), C.int(extendedLen))
	if len(blob) < 12 {
		return 2
	}
	kk := int(binary.LittleEndian.Uint32(blob[0:4]))
	nn := int(binary.LittleEndian.Uint32(blob[4:8]))
	rl := int(binary.LittleEndian.Uint32(blob[8:12]))
	nr := kk + nn
	if kk <= 0 || nn <= 0 || rl <= 0 || len(blob) != 12+nr*rl {
		return 3
	}

	goRows := make([][]byte, nr)
	off := 12
	for i := 0; i < nr; i++ {
		row := make([]byte, rl)
		copy(row, blob[off:off+rl])
		goRows[i] = row
		off += rl
	}

	cfg := &rsema1d.Config{K: kk, N: nn, WorkerCount: 1}
	ed, err := rsema1d.NewExtendedDataFromEncoded(cfg, goRows) // NO RS-encode
	if err != nil {
		return 4
	}

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
