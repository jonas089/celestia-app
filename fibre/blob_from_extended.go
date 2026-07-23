package fibre

import (
	"fmt"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d"
)

// NewBlobFromExtendedData wraps an ALREADY-ENCODED [rsema1d.ExtendedData] into a
// [Blob] WITHOUT re-encoding it and WITHOUT injecting the 5-byte v0 blob header.
//
// This is the entry point used to publish an rsema1d square that was produced
// out-of-band — e.g. the rv32 GKR prover's input square from
// [rsema1d.EncodeGKRInputSquare]. Because the square's committed rows must stay
// byte-identical to what the prover opens its GKR proof against, we must NOT run
// them through [NewBlob] (which prepends blobHeaderV0 to row 0 and re-runs the
// Reed-Solomon encode, both of which change the committed bytes / commitment).
//
// The resulting blob therefore satisfies:
//
//	NewBlobFromExtendedData(ed).ID().Commitment() == Commitment(ed.Commitment())
//
// The blob's [BlobConfig] is derived entirely from ed so the client-side upload
// path (payment promise UploadSize, RLC length, shard assignment over K+N rows,
// row proofs) is self-consistent with the square's own K/N/rowLen:
//   - OriginalRows (K) = len(ed.RLC())         (one RLC entry per original row)
//   - rowLen             = len(ed.Row(0))
//   - TotalRows (K+N)    = probed via ed.GenerateRowProof (errors past K+N)
//
// NOTE on protocol shape: the blob is stamped BlobVersion 1 — the fibre "GKR
// square" version whose erasure shape (K=N, rowLen=one Leopard chunk, no blob
// header) matches an rsema1d square. The v1 server config
// ([DefaultBlobConfigV1]) is fixed at K=512, N=512, rowLen=64 (the constant rv32
// circuit shape, numVars=11), so a square of exactly that shape settles on-chain
// via MsgPayForFibre with BlobVersion=1. The K/N/rowLen recorded in the blob's
// [BlobConfig] are derived from ed itself (so the client's payment-promise
// UploadSize, RLC length and shard assignment are self-consistent with the
// square); for a numVars=11 square these equal DefaultBlobConfigV1, so client
// and server agree. The commitment binding is exact regardless of shape.
func NewBlobFromExtendedData(ed *rsema1d.ExtendedData) (*Blob, error) {
	if ed == nil {
		return nil, fmt.Errorf("extended data cannot be nil")
	}

	k := len(ed.RLC())
	if k <= 0 {
		return nil, fmt.Errorf("extended data has no original rows (empty RLC)")
	}
	rowLen := len(ed.Row(0))
	if rowLen <= 0 {
		return nil, fmt.Errorf("extended data row 0 is empty")
	}

	total, err := probeTotalRows(ed, k)
	if err != nil {
		return nil, err
	}
	n := total - k

	cfg := BlobConfig{
		BlobVersion:  1,
		OriginalRows: k,
		ParityRows:   n,
		// The square is already fully formed at rowLen bytes/row; the row size is
		// a fixed property of the square, independent of any "data length".
		RowSize:       func(int) int { return rowLen },
		MaxDataSize:   k * rowLen,
		MaxRowSize:    rowLen,
		CodingWorkers: 1,
		// Coder/Assembler/DataPool are intentionally nil: this blob is never
		// (re-)encoded and its row proofs come from ed's own merkle trees.
	}

	b := &Blob{
		cfg:          cfg,
		extendedData: ed,
		id:           NewBlobID(cfg.BlobVersion, Commitment(ed.Commitment())),
		// data is nil: no original payload is materialized here and no header is
		// injected. UploadSize/RowSize are driven by the fixed-size RowSize func.
		data: nil,
		// releaseFn is nil: ed's storage is owned by ed, not a pooled assembler.
	}
	b.refCount.Store(1)
	return b, nil
}

// probeTotalRows determines K+N for an ExtendedData using only its public API.
// ExtendedData exposes no row count, but GenerateRowProof returns an error for
// any index >= K+N. Since both K and K+N are powers of two and N >= 1, the total
// is one of 2K, 4K, 8K, ...; we test increasing candidates until the index is
// rejected, which pins the exact total.
func probeTotalRows(ed *rsema1d.ExtendedData, k int) (int, error) {
	const maxTotal = 1 << 16 // GF(2^16) field-size limit on K+N
	for total := 2 * k; total <= maxTotal; total *= 2 {
		// Index `total` is valid iff the matrix has more than `total` rows.
		if _, err := ed.GenerateRowProof(total); err != nil {
			return total, nil
		}
	}
	return 0, fmt.Errorf("could not determine row count: exceeds field-size limit %d", maxTotal)
}
