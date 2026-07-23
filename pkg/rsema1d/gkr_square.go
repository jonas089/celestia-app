package rsema1d

import "fmt"

// GKR square layout constants, byte-identical to the Rust reference
// (rust/crates/rsema1d-pcs/src/lib.rs: LOG_COLS/NUM_SYMBOLS/ROW_BYTES and
// build_rows). NUM_SYMBOLS=32 GF(2^16) symbols per row => ROW_BYTES=64
// (one Leopard chunk: 32 low bytes followed by 32 high bytes).
const (
	gkrNumSymbols = 32
	gkrRowBytes   = 2 * gkrNumSymbols // 64
)

// EncodeGKRInputSquare reproduces, byte-identically, the rsema1d square that the
// Rust GKR prover commits to via rsema1d-pcs `build_rows` /
// `install_da_commitment`. A fibre blob built from the returned rows therefore
// has the SAME rsema1d commitment root the GKR proof opens against.
//
// inputVals is the input multilinear layer's evaluation (hypercube-basis)
// vector in the SAME representation Rust build_rows consumes: one byte per
// GF2x8 element (length 2^numVars), where bit s of the byte is SIMD lane s
// (matching GF2x8::unpack, which returns lane s = (v>>s)&1). This is the
// per-element GF2x8 byte layout.
//
// Layout (identical to build_rows):
//   - K = 1<<(numVars-2), N = K.
//   - The square has K+N rows of ROW_BYTES=64 bytes each. The first K rows hold
//     the Leopard-formatted {0,1} GF(2^16) symbols; the trailing N parity rows
//     start zeroed and are filled by the Reed-Solomon (Leopard GF(2^16))
//     encoder inside Coder.Encode.
//   - For data row j in [0,K) and symbol i in [0,32): flat bit address
//     a = j*32 + i maps to hypercube index g = a>>3 and SIMD lane s = a&7. The
//     symbol's low byte is row[j][i] = (inputVals[g] >> s) & 1; its high byte
//     row[j][32+i] stays zero (the symbol is a single bit in {0,1}).
//
// The returned ExtendedData.Commitment() equals the Rust commitment root
// (SHA256(rowRoot || rlcRoot)) for the same inputVals/numVars.
func EncodeGKRInputSquare(inputVals []byte, numVars int) (*ExtendedData, error) {
	if numVars < 2 {
		return nil, fmt.Errorf("numVars=%d too small: need numVars >= 2", numVars)
	}
	wantLen := 1 << uint(numVars)
	if len(inputVals) != wantLen {
		return nil, fmt.Errorf("inputVals must have 2^numVars = %d bytes, got %d", wantLen, len(inputVals))
	}

	k := 1 << uint(numVars-2)
	n := k

	// K data rows followed by N zeroed parity rows (total K+N), each ROW_BYTES.
	rows := make([][]byte, k+n)
	for r := range rows {
		rows[r] = make([]byte, gkrRowBytes)
	}
	for j := 0; j < k; j++ {
		row := rows[j]
		for i := 0; i < gkrNumSymbols; i++ {
			a := j*gkrNumSymbols + i
			g := a >> 3
			s := a & 7
			// Leopard chunk: low byte at [i], high byte at [32+i]. The symbol
			// is a bit in {0,1}, so the high byte stays zero.
			row[i] = (inputVals[g] >> uint(s)) & 1
		}
	}

	cfg := &Config{K: k, N: n, WorkerCount: 1}
	coder, err := NewCoder(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create coder: %w", err)
	}
	return coder.Encode(rows)
}
