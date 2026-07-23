package rsema1d

import (
	"encoding/hex"
	"strings"
	"testing"
)

// kind of deterministic input; must match the Rust reference
// (rust/crates/rsema1d-pcs/tests/gkr_square_roots.rs).
type gkrKind int

const (
	gkrZeros gkrKind = iota
	gkrOnes
	gkrPattern
)

type gkrCase struct {
	label    string
	numVars  int
	kind     gkrKind
	mul, add int // used when kind == gkrPattern
	// expectedRoot is the commitment root hex printed by the Rust reference
	// helper `gkr_square_roots` in rsema1d-pcs (see that file). Produced by:
	//   cargo +nightly-2025-05-17 test -p rsema1d-pcs --release gkr_square -- --nocapture
	expectedRoot string
}

// genInput is byte-identical to the Rust reference `gen_input`.
// inputVals[i] is the GF2x8 element for hypercube index i; bit s of the byte is
// SIMD lane s.
func genInput(numVars int, kind gkrKind, mul, add int) []byte {
	n := 1 << uint(numVars)
	out := make([]byte, n)
	for i := 0; i < n; i++ {
		switch kind {
		case gkrZeros:
			out[i] = 0x00
		case gkrOnes:
			out[i] = 0xFF
		case gkrPattern:
			out[i] = byte((i*mul + add) & 0xFF)
		}
	}
	return out
}

// gkrCases is the IDENTICAL list of deterministic inputs used by the Rust
// reference helper. expectedRoot values are baked in from that helper's output.
var gkrCases = []gkrCase{
	{label: "nv8_zeros", numVars: 8, kind: gkrZeros, expectedRoot: "d3125a1ef180532e3af3384ff8d5f91146709a04d4b9f07ab7be333f643c7834"},
	{label: "nv8_ones", numVars: 8, kind: gkrOnes, expectedRoot: "9b1ba5f4cfb420859e2e0df895316274b0826a7887cb62d6302ab1e34cdea278"},
	{label: "nv8_patA", numVars: 8, kind: gkrPattern, mul: 37, add: 11, expectedRoot: "b31455c104c111c08a3a38c4736c0bc651e08e7c1e41ed753c603aef67c64c4d"},
	{label: "nv10_patB", numVars: 10, kind: gkrPattern, mul: 89, add: 5, expectedRoot: "0172c84c67ec5d42afa0a21785e267e3f14a3e8b0631528ce8173b1c30e9f016"},
	{label: "nv11_patC", numVars: 11, kind: gkrPattern, mul: 197, add: 3, expectedRoot: "1a248a5e6088b2f0377fe2b6db50c3f5fb7e8bbefff45d2891d196e4a6c2260d"},
	{label: "nv12_patD", numVars: 12, kind: gkrPattern, mul: 7, add: 123, expectedRoot: "7a1e7fb4d86105f14f8fc6b216ecdd5e5dde4ded1e8270e9b7cd3a8441207806"},
}

func TestGKRSquareRoots(t *testing.T) {
	for _, tc := range gkrCases {
		tc := tc
		t.Run(tc.label, func(t *testing.T) {
			input := genInput(tc.numVars, tc.kind, tc.mul, tc.add)
			ed, err := EncodeGKRInputSquare(input, tc.numVars)
			if err != nil {
				t.Fatalf("EncodeGKRInputSquare: %v", err)
			}
			c := ed.Commitment()
			got := hex.EncodeToString(c[:])
			t.Logf("%s num_vars=%d root=%s", tc.label, tc.numVars, got)
			if strings.HasPrefix(tc.expectedRoot, "REPLACE") {
				t.Fatalf("expected root not set for %s; got %s (fill in from Rust reference)", tc.label, got)
			}
			if got != tc.expectedRoot {
				t.Fatalf("%s: Go root %s != Rust root %s", tc.label, got, tc.expectedRoot)
			}
		})
	}
}
