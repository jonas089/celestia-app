package main

import (
	"encoding/binary"
	"math/rand"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
)

// btaRNG is a deterministically-seeded PRNG used only for Berlekamp-trace
// element selection, so the discovered root beta (and hence the exported
// matrices) are reproducible across runs.
var btaRNG = rand.New(rand.NewSource(0x6673656d613164)) // "fsema1d"

// Thin helpers over the real rsema1d GF128 field.

func rAdd(a, b field.GF128) field.GF128 { return field.Add128(a, b) }
func rMul(a, b field.GF128) field.GF128 { return field.MulFull(a, b) }
func rSqr(a field.GF128) field.GF128    { return field.MulFull(a, a) }

func rEqual(a, b field.GF128) bool { return field.Equal128(a, b) }
func rIsZero(a field.GF128) bool   { return field.IsZero(a) }

// rInv computes the multiplicative inverse via Fermat: a^(2^128 - 2).
// Exponent 2^128-2 has bits 1..127 set (bit 0 clear).
func rInv(a field.GF128) field.GF128 {
	result := field.One()
	base := a
	for i := 0; i < 128; i++ {
		if i >= 1 { // bit i of (2^128-2) is set for i in 1..127
			result = rMul(result, base)
		}
		base = rSqr(base)
	}
	return result
}

func randR() field.GF128 {
	var b [field.GF128Size]byte
	binary.LittleEndian.PutUint64(b[0:8], btaRNG.Uint64())
	binary.LittleEndian.PutUint64(b[8:16], btaRNG.Uint64())
	return field.DecodeGF128(b[:])
}

// ---- polynomials over R (coeff index == degree) ----

type rpoly []field.GF128

func (p rpoly) deg() int {
	for i := len(p) - 1; i >= 0; i-- {
		if !rIsZero(p[i]) {
			return i
		}
	}
	return -1
}

func (p rpoly) trim() rpoly {
	d := p.deg()
	if d < 0 {
		return rpoly{}
	}
	return p[:d+1]
}

func (p rpoly) clone() rpoly {
	q := make(rpoly, len(p))
	copy(q, p)
	return q
}

func rpolyAdd(a, b rpoly) rpoly {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	out := make(rpoly, n)
	for i := 0; i < n; i++ {
		var av, bv field.GF128
		if i < len(a) {
			av = a[i]
		}
		if i < len(b) {
			bv = b[i]
		}
		out[i] = rAdd(av, bv)
	}
	return out.trim()
}

// rpolyMul multiplies two polynomials over R (schoolbook).
func rpolyMul(a, b rpoly) rpoly {
	a = a.trim()
	b = b.trim()
	if a.deg() < 0 || b.deg() < 0 {
		return rpoly{}
	}
	out := make(rpoly, len(a)+len(b)-1)
	for i := range a {
		if rIsZero(a[i]) {
			continue
		}
		for j := range b {
			if rIsZero(b[j]) {
				continue
			}
			out[i+j] = rAdd(out[i+j], rMul(a[i], b[j]))
		}
	}
	return out.trim()
}

// rpolyMakeMonic scales p so its leading coefficient is 1.
func rpolyMakeMonic(p rpoly) rpoly {
	p = p.trim()
	d := p.deg()
	if d < 0 {
		return p
	}
	lc := p[d]
	if rEqual(lc, field.One()) {
		return p
	}
	inv := rInv(lc)
	out := make(rpoly, len(p))
	for i := range p {
		out[i] = rMul(p[i], inv)
	}
	return out
}

// rpolyDivMod returns quotient and remainder of a / b (b monic or not).
func rpolyDivMod(a, b rpoly) (q, r rpoly) {
	a = a.trim()
	b = b.trim()
	db := b.deg()
	if db < 0 {
		panic("divide by zero polynomial")
	}
	lcInv := rInv(b[db])
	rem := a.clone().trim()
	da := rem.deg()
	if da < db {
		return rpoly{}, rem
	}
	quo := make(rpoly, da-db+1)
	for {
		dr := rem.deg()
		if dr < db {
			break
		}
		shift := dr - db
		coef := rMul(rem[dr], lcInv)
		quo[shift] = coef
		// rem -= coef * x^shift * b
		for i := 0; i <= db; i++ {
			rem[shift+i] = rAdd(rem[shift+i], rMul(coef, b[i]))
		}
		rem = rem.trim()
		if rem.deg() >= dr { // safety: degree must strictly drop
			panic("division did not reduce degree")
		}
	}
	return quo.trim(), rem.trim()
}

func rpolyMod(a, b rpoly) rpoly {
	_, r := rpolyDivMod(a, b)
	return r
}

// rpolyMulMod = (a*b) mod m.
func rpolyMulMod(a, b, m rpoly) rpoly {
	return rpolyMod(rpolyMul(a, b), m)
}

// rpolyGCD returns the monic gcd of a and b.
func rpolyGCD(a, b rpoly) rpoly {
	a = a.trim()
	b = b.trim()
	for b.deg() >= 0 {
		r := rpolyMod(a, b)
		a, b = b, r
	}
	return rpolyMakeMonic(a)
}
