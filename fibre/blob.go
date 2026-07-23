package fibre

import (
	"encoding/binary"
	"errors"
	"fmt"
	"runtime"
	"sync/atomic"

	"github.com/celestiaorg/celestia-app/v10/fibre/internal/row"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d"
	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/rlc"
	"github.com/klauspost/reedsolomon"
)

// ErrBlobTooLarge is returned when the blob size exceeds BlobConfig.MaxDataSize.
var ErrBlobTooLarge = errors.New("blob size exceeds maximum allowed size")

// BlobConfig contains configuration parameters for blob encoding and decoding.
type BlobConfig struct {
	// BlobVersion is the version of the row format.
	BlobVersion uint8
	// OriginalRows is the number of original rows before erasure coding (K in rsema1d).
	OriginalRows int
	// ParityRows is the number of parity rows added by erasure coding (N in rsema1d).
	// Total rows = OriginalRows + ParityRows.
	ParityRows int
	// RowSize computes row size given the data length.
	RowSize func(dataLen int) int
	// MaxDataSize is the maximum data size that can be passed to NewBlob.
	MaxDataSize int
	// MaxRowSize is the maximum allowed size of a single row in bytes. It bounds
	// the DataPool/Assembler allocations, so rows larger than this must be
	// rejected before they reach the pool (which would otherwise panic).
	MaxRowSize int
	// CodingWorkers is the number of workers to use for encoding and decoding rsema1d.
	CodingWorkers int

	// Coder is a cached rsema1d encoder/decoder for blob encoding and reconstruction.
	Coder *rsema1d.Coder
	// Assembler builds pooled row layouts for encoding blobs.
	Assembler *row.Assembler
	// DataPool pools blob downloads allocations.
	DataPool *row.Pool
}

// defaultBlobConfigV0 is the shared default config, created at init time.
var defaultBlobConfigV0 = func() BlobConfig {
	cfg, err := NewBlobConfigFromParams(0, DefaultProtocolParams)
	if err != nil {
		panic(fmt.Sprintf("creating default blob config v0: %v", err))
	}
	return cfg
}()

// DefaultBlobConfigV0 returns a [BlobConfig] with default values for version 0.
// The config is created once at init and shared across all callers.
func DefaultBlobConfigV0() BlobConfig {
	return defaultBlobConfigV0
}

// v1 (rv32 GKR square) erasure shape. Unlike v0, the committed rows ARE the
// rsema1d square produced by [rsema1d.EncodeGKRInputSquare]: the rv32 GKR
// prover commits an rsema1d square with K=N (num_vars=11 => K=512, N=512) and a
// fixed row length of one Leopard chunk (64 bytes), with NO 5-byte v0 blob
// header. v1 therefore must not re-encode or prepend a header — the on-chain
// commitment must equal the prover's exact commitment. The rv32 circuit shape
// is constant, so v1 is a single fixed config.
const (
	// blobV1OriginalRows is K for the rv32 GKR square (1<<(numVars-2), numVars=11).
	blobV1OriginalRows = 512
	// blobV1ParityRows is N for the rv32 GKR square (K=N).
	blobV1ParityRows = 512
	// blobV1RowLen is the fixed row length in bytes (one Leopard chunk, gkrRowBytes).
	blobV1RowLen = 64
)

// protocolParamsV1 describes the v1 erasure shape (K=N=512, EncodingRatio 0.5).
// It is used only to derive the codec work-buffer sizing (CodecWorkRows) and
// parity count; v1 row size is a fixed constant and does NOT go through
// [ProtocolParams.RowSize] (which is version-0 only).
var protocolParamsV1 = ProtocolParams{
	Rows:                       blobV1OriginalRows,
	EncodingRatio:              0.5, // K=N => half original, half parity
	MaxValidatorCount:          DefaultProtocolParams.MaxValidatorCount,
	UniqueDecodingSecurityBits: DefaultProtocolParams.UniqueDecodingSecurityBits,
	SafetyThreshold:            DefaultProtocolParams.SafetyThreshold,
	LivenessThreshold:          DefaultProtocolParams.LivenessThreshold,
	MaxBlobSize:                blobV1OriginalRows * blobV1RowLen,
	MinRowSize:                 blobV1RowLen,
}

// defaultBlobConfigV1 is the shared v1 config, created at init time.
var defaultBlobConfigV1 = func() BlobConfig {
	cfg, err := newBlobConfigV1()
	if err != nil {
		panic(fmt.Sprintf("creating default blob config v1: %v", err))
	}
	return cfg
}()

// DefaultBlobConfigV1 returns the [BlobConfig] for version 1 (the rv32 GKR
// square shape: K=512, N=512, rowLen=64, no header). Created once at init and
// shared across all callers.
func DefaultBlobConfigV1() BlobConfig {
	return defaultBlobConfigV1
}

// BlobConfigForVersion returns the [BlobConfig] for the given blob version.
// Returns an error if the version is not supported.
func BlobConfigForVersion(version uint8) (BlobConfig, error) {
	switch version {
	case 0:
		return DefaultBlobConfigV0(), nil
	case 1:
		return DefaultBlobConfigV1(), nil
	default:
		return BlobConfig{}, fmt.Errorf("unsupported blob version: %d", version)
	}
}

// newBlobConfigV1 builds the fixed v1 config (K=512, N=512, rowLen=64, no
// header). It reproduces EXACTLY what [rsema1d.EncodeGKRInputSquare] produces
// for numVars=11 so that a blob built via [NewBlobFromExtendedData] for a v1
// square has blob.ID().Commitment() == ed.Commitment() and the server can
// reconstruct/verify the same K=512,N=512 layout.
func newBlobConfigV1() (BlobConfig, error) {
	const blobVersion = 1
	k := protocolParamsV1.Rows
	n := protocolParamsV1.ParityRows()
	rowLen := blobV1RowLen
	maxRowSize := blobV1RowLen

	codecCfg := &rsema1d.Config{
		K:           k,
		N:           n,
		WorkerCount: runtime.GOMAXPROCS(0),
	}

	assembler, err := row.NewAssembler(k, n, maxRowSize, codecCfg.TreeBufferSize())
	if err != nil {
		return BlobConfig{}, fmt.Errorf("creating v1 row assembler: %w", err)
	}

	workPool := row.NewPool(maxRowSize, protocolParamsV1.CodecWorkRows())
	coder, err := rsema1d.NewCoder(codecCfg, reedsolomon.WithWorkAllocator(workPool))
	if err != nil {
		return BlobConfig{}, fmt.Errorf("creating v1 rsema1d coder: %w", err)
	}
	dataPool := row.NewPool(maxRowSize, k)

	return BlobConfig{
		BlobVersion:  blobVersion,
		OriginalRows: k,
		ParityRows:   n,
		// v1 rows are the fixed-size rsema1d square rows; there is NO blob header,
		// so row size is a constant independent of any data length.
		RowSize:       func(int) int { return rowLen },
		MaxDataSize:   k * rowLen,
		MaxRowSize:    maxRowSize,
		CodingWorkers: runtime.GOMAXPROCS(0),
		Coder:         coder,
		Assembler:     assembler,
		DataPool:      dataPool,
	}, nil
}

// NewBlobConfigFromParams creates a [BlobConfig] with values derived from the given [ProtocolParams].
// Use this when you need a config with non-default protocol parameters (e.g., for testing).
// Version 1 has a fixed shape (K=512, N=512, rowLen=64) and ignores params.
func NewBlobConfigFromParams(blobVersion uint8, params ProtocolParams) (BlobConfig, error) {
	if blobVersion == 1 {
		return newBlobConfigV1()
	}
	if blobVersion != 0 {
		return BlobConfig{}, fmt.Errorf("unsupported blob version: %d", blobVersion)
	}

	maxRowSize := params.MaxRowSize(blobVersion)
	codecCfg := &rsema1d.Config{
		K:           params.Rows,
		N:           params.ParityRows(),
		WorkerCount: runtime.GOMAXPROCS(0),
	}

	assembler, err := row.NewAssembler(params.Rows, params.ParityRows(), maxRowSize, codecCfg.TreeBufferSize())
	if err != nil {
		return BlobConfig{}, fmt.Errorf("creating row assembler: %w", err)
	}

	workPool := row.NewPool(maxRowSize, params.CodecWorkRows())
	coder, err := rsema1d.NewCoder(codecCfg, reedsolomon.WithWorkAllocator(workPool))
	if err != nil {
		return BlobConfig{}, fmt.Errorf("creating rsema1d coder: %w", err)
	}
	dataPool := row.NewPool(maxRowSize, params.Rows)

	return BlobConfig{
		BlobVersion:  blobVersion,
		OriginalRows: params.Rows,
		ParityRows:   params.ParityRows(),
		RowSize: func(dataLen int) int {
			return params.RowSize(blobVersion, dataLen+blobHeaderLen)
		},
		MaxDataSize:   params.MaxBlobSize - blobHeaderLen,
		MaxRowSize:    maxRowSize,
		CodingWorkers: runtime.GOMAXPROCS(0),
		Coder:         coder,
		Assembler:     assembler,
		DataPool:      dataPool,
	}, nil
}

// TotalRows returns the total number of rows (OriginalRows + ParityRows).
func (c BlobConfig) TotalRows() int {
	return c.OriginalRows + c.ParityRows
}

// UploadSize calculates size of blob data with padding and w/o parity.
// This is the size included in the [PaymentPromise] and the one actually paid for.
func (c BlobConfig) UploadSize(dataLen int) int {
	return c.RowSize(dataLen) * c.OriginalRows
}

// Blob represents encoded data with Reed-Solomon erasure coding.
// NOTE: The Blob currently embeds the versioned header. The long-term intention is to keep the Blob struct version independent,
// while the respective header+config combination is versioned and produce general Blob.
// Once the new version is introduced, we can consider restricting the Blob to be general, i.e. without keeping the header of a particular version.
type Blob struct {
	cfg BlobConfig

	extendedData *rsema1d.ExtendedData
	id           BlobID

	// holds meta fields about the blob
	header blobHeaderV0
	// data holds the decoded original data (without header).
	data []byte

	refCount     atomic.Int32
	userReleased atomic.Bool
	releaseFn    func()
}

// NewBlob creates a new [Blob] instance by encoding the data.
// It takes ownership of data and may reuse it directly as backing storage for
// original rows. Callers must not modify data after calling NewBlob.
//
// The data is prefixed with a header containing the blob version and data size.
// Returns [ErrBlobTooLarge] if the data size exceeds BlobConfig.MaxDataSize.
//
// The returned blob holds pooled row buffers from the [row.Assembler]. The
// caller is responsible for calling [Blob.Free] once they are done with the
// blob — when handing the blob to [Client.Upload], the standard pattern is
// `defer blob.Free()` at the call site.
func NewBlob(data []byte, cfg BlobConfig) (*Blob, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("data cannot be empty")
	}
	if len(data) > cfg.MaxDataSize {
		return nil, fmt.Errorf("%w: data size %d exceeds maximum %d", ErrBlobTooLarge, len(data), cfg.MaxDataSize)
	}

	header := newBlobHeaderV0(len(data))
	extendedData, asm, err := header.encode(data, cfg)
	if err != nil {
		return nil, err
	}

	b := &Blob{
		cfg:          cfg,
		extendedData: extendedData,
		id:           NewBlobID(cfg.BlobVersion, extendedData.Commitment()),
		header:       header,
		data:         data,
		releaseFn:    asm.Free,
	}
	b.refCount.Store(1)
	return b, nil
}

// Free releases the caller's reference. When all references (including any
// internal references taken by [Client.Upload]) have been released, the
// blob's pool-backed storage — assembler slab on the upload side, slab
// alias on the download side — is returned to its pool. Idempotent against
// repeated calls from the same caller; safe on a nil receiver. Reading
// [Blob.Data] after Free is undefined behavior — for download blobs the
// bytes may have been recycled by another consumer of the pool.
func (d *Blob) Free() {
	if d == nil {
		return
	}
	if d.userReleased.Swap(true) {
		return
	}
	d.release()
}

// ID returns the BlobID of this blob.
func (d *Blob) ID() BlobID {
	return d.id
}

// Config returns the blob's configuration.
func (d *Blob) Config() BlobConfig {
	return d.cfg
}

// RLC returns the computed random linear combination vector for the original rows.
func (d *Blob) RLC() rlc.Vector {
	if d.extendedData == nil {
		return nil
	}
	return d.extendedData.RLC()
}

// RowSize returns the size of each row in bytes.
// Returns 0 if no original data available to determine row size.
func (d *Blob) RowSize() int {
	return d.cfg.RowSize(len(d.data))
}

// DataSize returns the size of the original data (without header) by reading from the blob header.
// Returns 0 if no original data available to determine its size.
func (d *Blob) DataSize() int {
	return len(d.data)
}

// UploadSize calculates size of the [Blob] data with padding and w/o parity.
// This is the size included in the [PaymentPromise] and the one actually paid for.
func (d *Blob) UploadSize() int {
	return d.cfg.UploadSize(d.DataSize())
}

// Data returns the cached original data (without header).
func (d *Blob) Data() []byte {
	return d.data
}

// RowProofs yields the row data and Merkle proof for each index (see
// [rsema1d.ExtendedData.RowProofs]). row and proof alias pooled storage valid
// until the blob is released; returns an error after release.
func (d *Blob) RowProofs(indices []int, yield func(index int, row []byte, proof [][]byte)) error {
	if d.extendedData == nil {
		return fmt.Errorf("no extended data available")
	}
	if d.released() {
		return fmt.Errorf("storage has been released")
	}
	return d.extendedData.RowProofs(indices, yield)
}

// retain bumps the refcount for an additional non-user owner — used by
// [Client.Upload] to keep the asm slab alive across background goroutines
// that read the blob's pool-backed rows. It returns false if the storage has
// already been released (refcount 0).
// Each successful retain must pair with a later [Blob.release].
func (d *Blob) retain() bool {
	for {
		n := d.refCount.Load()
		if n == 0 {
			return false
		}
		if d.refCount.CompareAndSwap(n, n+1) {
			return true
		}
	}
}

// release decrements the refcount; when it hits zero the releaseFn fires.
// Called by internal owners (e.g., Upload's terminal goroutine) after they
// finish using the blob.
func (d *Blob) release() {
	if d.refCount.Add(-1) == 0 && d.releaseFn != nil {
		d.releaseFn()
		d.releaseFn = nil
	}
}

func (d *Blob) released() bool {
	return d.refCount.Load() == 0
}

const (
	// blobVersionLen is the length of the version field in bytes.
	blobVersionLen = 1
	// blobDataSizeLen is the length of the data size field in bytes.
	blobDataSizeLen = 4
	// blobHeaderLen is the total length of the blob header in bytes.
	// Format: 1 byte version + 4 bytes data size
	blobHeaderLen = blobVersionLen + blobDataSizeLen
)

// blobHeaderV0 represents the version 0 blob header at the start of the first row.
// Format: 1 byte version (uint8, always 0) + 4 bytes data size (uint32)
type blobHeaderV0 struct {
	dataSize uint32
}

// newBlobHeaderV0 creates a new version 0 blob header with the given data size.
// The version field is implicitly 0 for this header type.
func newBlobHeaderV0(dataSize int) blobHeaderV0 {
	return blobHeaderV0{
		dataSize: uint32(dataSize),
	}
}

// encode assembles rows from data, writes the blob header, and produces the
// commitment via the [rsema1d.Coder]. Returns the extended data and an
// [Assembly] that owns the pooled row storage for the blob's lifetime.
func (h blobHeaderV0) encode(data []byte, cfg BlobConfig) (*rsema1d.ExtendedData, *row.Assembly, error) {
	rowSize := cfg.RowSize(len(data))
	asm := cfg.Assembler.Assemble(data, rowSize, blobHeaderLen)
	rows, treeBuf := asm.Buffers()
	h.marshalTo(rows[0])

	extData, err := cfg.Coder.EncodeWithTree(rows, treeBuf)
	if err != nil {
		asm.Free()
		return nil, nil, fmt.Errorf("encoding data: %w", err)
	}
	return extData, asm, nil
}

// decode runs Reed-Solomon data-shard recovery when the reconstructor has
// enough unique rows, parses the blob header from the contiguous bytes
// view, and returns the data sub-slice (header stripped). The returned
// slice aliases bytes — no copy.
//
// decode parses the v0 header at the start of bytes and returns the sub-slice
// spanning just the payload (no header). bytes is typically the K*rowSize
// contiguous region of a download's row slab. The returned slice aliases
// bytes — no copy.
func (h *blobHeaderV0) decode(bytes []byte, cfg BlobConfig) ([]byte, error) {
	if len(bytes) < blobHeaderLen {
		return nil, fmt.Errorf("bytes too small for header: got %d, need at least %d", len(bytes), blobHeaderLen)
	}
	if err := h.unmarshalFrom(bytes); err != nil {
		return nil, fmt.Errorf("decoding header: %w", err)
	}
	dataSize := int(h.dataSize)
	if dataSize == 0 {
		return nil, fmt.Errorf("invalid blob size in header: must be greater than 0")
	}
	if dataSize > cfg.MaxDataSize {
		return nil, fmt.Errorf("blob size in header (%d bytes) exceeds maximum allowed size (%d bytes)", dataSize, cfg.MaxDataSize)
	}
	if blobHeaderLen+dataSize > len(bytes) {
		return nil, fmt.Errorf("data size in header (%d bytes) exceeds available bytes (%d bytes)", dataSize, len(bytes)-blobHeaderLen)
	}
	return bytes[blobHeaderLen : blobHeaderLen+dataSize], nil
}

// marshalTo writes the version 0 blob header into the provided buffer.
// The buffer must be at least blobHeaderLen bytes long.
// Always writes version byte as 0.
func (h blobHeaderV0) marshalTo(buf []byte) {
	buf[0] = 0 // version 0
	binary.BigEndian.PutUint32(buf[blobVersionLen:blobHeaderLen], h.dataSize)
}

// unmarshalFrom reads the blob header from the provided buffer.
// The buffer must be at least blobHeaderLen bytes long.
// Returns an error if the version byte is not 0.
func (h *blobHeaderV0) unmarshalFrom(buf []byte) error {
	if buf[0] != 0 {
		return fmt.Errorf("invalid blob version: expected 0, got %d", buf[0])
	}
	h.dataSize = binary.BigEndian.Uint32(buf[blobVersionLen:blobHeaderLen])
	return nil
}
