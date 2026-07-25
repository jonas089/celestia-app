package rsema1d

import (
	"bytes"
	"math/rand"
	"testing"
)

// TestNewExtendedDataFromEncoded proves the cross-process hand-off is sound at
// the Go level: reconstructing an ExtendedData from the already-extended rows
// yields a commitment byte-identical to the one Coder.Encode produced, without
// re-running the Reed-Solomon encoder.
func TestNewExtendedDataFromEncoded(t *testing.T) {
	const k, n, rowLen = 8, 8, 64
	cfg := &Config{K: k, N: n, WorkerCount: 1}
	coder, err := NewCoder(cfg)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(42))
	rows := make([][]byte, k+n)
	for i := range rows {
		rows[i] = make([]byte, rowLen)
	}
	for i := 0; i < k; i++ {
		rng.Read(rows[i]) // parity rows [k,k+n) stay zeroed for Encode
	}

	ed, err := coder.Encode(rows) // DA-side RS-encode
	if err != nil {
		t.Fatal(err)
	}
	want := ed.Commitment()

	// Extract the now-extended rows (originals + genuine RS parity).
	extended := make([][]byte, k+n)
	for i := range extended {
		extended[i] = append([]byte(nil), ed.Row(i)...)
	}

	// Prover-side reconstruction: no RS-encode.
	rebuilt, err := NewExtendedDataFromEncoded(cfg, extended)
	if err != nil {
		t.Fatal(err)
	}
	got := rebuilt.Commitment()
	if !bytes.Equal(want[:], got[:]) {
		t.Fatalf("commitment mismatch:\n encode: %x\n reload: %x", want, got)
	}

	// The reconstructed square must serve openings identically: same RLC vector.
	if len(rebuilt.RLC()) != len(ed.RLC()) {
		t.Fatalf("rlc len mismatch")
	}
	for i := range ed.RLC() {
		a, b := ed.RLC()[i], rebuilt.RLC()[i]
		if a != b {
			t.Fatalf("rlc[%d] mismatch", i)
		}
	}
}
