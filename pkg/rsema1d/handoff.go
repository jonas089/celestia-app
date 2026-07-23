package rsema1d

import "fmt"

// NewExtendedDataFromEncoded rebuilds an [ExtendedData] from an ALREADY
// RS-encoded K+N row matrix WITHOUT running the Reed-Solomon encoder.
//
// This is the prover side of the cross-process "accidental-computer" hand-off:
// the DA encoder pays the Reed-Solomon (Leopard GF(2^16)) encoding cost exactly
// once, wherever the extended rows were originally produced (see
// [Coder.Encode]); a different process then transports those already-extended
// rows and reconstructs the committed square here. Only the commitment
// structures are rebuilt — the row Merkle tree, the legacy DeriveCoefficients
// RLC digest, and the SHA256(rowRoot||rlcRoot) commitment — which are hashing /
// linear-digest work, NOT polynomial encoding. No [reedsolomon.Encoder] is even
// instantiated, so RS-encoding cannot happen on this path.
//
// The reconstructed commitment is byte-identical to the one Coder.Encode
// produced for the same rows, so a GKR prover can open it at any point exactly
// as if it had encoded the rows itself.
//
// extendedRows must be the full K+N matrix (originals in [0,K), the genuine RS
// parity in [K,K+N)), each row equal length, matching cfg.
func NewExtendedDataFromEncoded(cfg *Config, extendedRows [][]byte) (*ExtendedData, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	if len(extendedRows) != cfg.K+cfg.N {
		return nil, fmt.Errorf("expected %d extended rows, got %d", cfg.K+cfg.N, len(extendedRows))
	}
	rowLen := len(extendedRows[0])
	for i, r := range extendedRows {
		if len(r) != rowLen {
			return nil, fmt.Errorf("row %d has length %d, want %d", i, len(r), rowLen)
		}
	}
	// A Coder holding only config: commit() never touches the RS encoder (enc),
	// it only builds the Merkle/RLC trees and the commitment. So no re-encode.
	c := &Coder{config: cfg}
	return c.commit(extendedRows, nil), nil
}
