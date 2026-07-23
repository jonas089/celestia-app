package main

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
)

func randU64() uint64 {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return binary.LittleEndian.Uint64(b[:])
}

// u128FromLanes builds a value from four little-endian u32 lanes
// (lane0 = bits 0..31, lane3 = bits 96..127).
func u128FromLanes(l0, l1, l2, l3 uint32) u128 {
	return u128{
		lo: uint64(l0) | uint64(l1)<<32,
		hi: uint64(l2) | uint64(l3)<<32,
	}
}

func u128FromBytes(b [16]byte) u128 {
	return u128{
		lo: binary.LittleEndian.Uint64(b[0:8]),
		hi: binary.LittleEndian.Uint64(b[8:16]),
	}
}

// runExpanderKATs checks expMul against the real crate's known-answer tests
// (arith/gf2_128/src/tests.rs::test_gf_mul_kat, cross-checked with AVX).
func runExpanderKATs() bool {
	type kat struct {
		name    string
		a, b, p u128
	}
	rep := func(v byte) [16]byte {
		var out [16]byte
		for i := range out {
			out[i] = v
		}
		return out
	}
	a5 := rep(7)
	b5 := rep(5)
	a6 := rep(6)
	a6[8] = 0
	b6 := rep(5)
	b6[4] = 1

	kats := []kat{
		{
			name: "one*a",
			a:    u128{1, 0},
			b:    u128{5, 3},
			p:    u128FromLanes(5, 0, 3, 0),
		},
		{
			name: "a*b small",
			a:    u128{5, 3}, // (3<<64)+5
			b:    u128{7, 1}, // (1<<64)+7
			p:    u128FromLanes(402, 0, 12, 0),
		},
		{
			name: "b*c reduction",
			a:    u128{7, 1},
			b:    u128FromLanes(1, 1, 1, 1), // (1<<96)+(1<<64)+(1<<32)+1
			p:    u128FromLanes(128, 128, 6, 6),
		},
		{
			name: "7^16 * 5^16",
			a:    u128FromBytes(a5),
			b:    u128FromBytes(b5),
			p:    u128FromLanes(232394202, 232394202, 232394202, 232394202),
		},
		{
			name: "mixed bytes",
			a:    u128FromBytes(a6),
			b:    u128FromBytes(b6),
			p:    u128FromLanes(508894806, 1107902981, 155322701, 155322714),
		},
	}

	ok := true
	for _, k := range kats {
		got := expMul(k.a, k.b)
		status := "PASS"
		if got != k.p {
			status = fmt.Sprintf("FAIL got=%s want=%s", hexU128LE(got), hexU128LE(k.p))
			ok = false
		}
		fmt.Printf("  KAT %-16s %s\n", k.name, status)
	}
	return ok
}
