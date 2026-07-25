package field

// This file gives GF128 a genuine field structure so it can be used as the
// large challenge field for the structured (tensor) RLC used by the
// GKR-friendly / multilinear-PCS variant of rsema1d.
//
// GF128 is modelled as GF(2^16)[X] / (f(X)) where f is a monic irreducible
// polynomial of degree 8 over GF(2^16) (reedsolomon's GF(2^16)). An element
//
//	a = a[0] + a[1]*X + ... + a[7]*X^7
//
// is stored little-endian in the 8 uint16 components, matching the existing
// GF128 layout. The GF(2^16) subfield sits as the degree-0 coefficients, so
// the pre-existing subfield action Mul128(scalar, vec) (component-wise
// GF(2^16) scaling) coincides with field multiplication by a constant — the
// two representations are consistent.
//
// reductionPoly holds the low 8 coefficients (r_0..r_7) of the irreducible
// f(X) = X^8 + r_7 X^7 + ... + r_1 X + r_0, i.e. the reduction rule
// X^8 ≡ r_7 X^7 + ... + r_1 X + r_0. It is verified irreducible by
// TestReductionPolyIsIrreducible (a Rabin-style Frobenius test) so the
// structure really is a field (no zero divisors), which is what the
// Schwartz–Zippel soundness argument relies on.
//
// This value was found and certified by TestFindIrreducible over reedsolomon's
// GF(2^16) (Cantor/Leopard) basis. Trinomials X^8+cX+c0 are systematically
// reducible in this basis, so a full-coefficient irreducible is used.
var reductionPoly = [GF128Width]uint16{0x9f9d, 0x0529, 0xa627, 0x62be, 0x8f6a, 0x8559, 0x6eb2, 0xb219}

// reductionPolySet guards MulFull so we never silently multiply in a non-field.
const reductionPolySet = true

// One returns the multiplicative identity of GF128.
func One() GF128 {
	return GF128{1}
}

// IsZero reports whether g is the additive identity.
func IsZero(g GF128) bool {
	for i := range GF128Width {
		if g[i] != 0 {
			return false
		}
	}
	return true
}

// MulFull multiplies two GF128 elements in GF(2^16)[X]/(f). Unlike Mul128
// (which is the GF(2^16)-subfield scalar action) this is the full field
// product and is required to build tensor products of challenge elements.
func MulFull(a, b GF128) GF128 {
	if !reductionPolySet {
		panic("field: GF128 reduction polynomial not set; call setReductionPoly with a verified irreducible")
	}
	return mulFullMod(a, b, reductionPoly)
}

// mulFullMod computes a*b mod (X^8 + Σ red[j] X^j). Kept parameterized on the
// reduction so the irreducibility search/test can evaluate candidate polys
// before one is committed to the package-level reductionPoly.
func mulFullMod(a, b GF128, red [GF128Width]uint16) GF128 {
	// Schoolbook multiply into a degree-≤14 product (15 coefficients).
	var p [2*GF128Width - 1]uint16
	for i := range GF128Width {
		ai := a[i]
		if ai == 0 {
			continue
		}
		for j := range GF128Width {
			bj := b[j]
			if bj == 0 {
				continue
			}
			p[i+j] ^= ll.GF16Mul(ai, bj)
		}
	}
	// Reduce top coefficients: X^d ≡ X^(d-8) * (Σ red[j] X^j) for d ≥ 8.
	// Processing d from high to low keeps every folded contribution at an
	// index < d, so a single downward pass fully reduces the product.
	for d := 2*GF128Width - 2; d >= GF128Width; d-- {
		c := p[d]
		if c == 0 {
			continue
		}
		p[d] = 0
		base := d - GF128Width
		for j := range GF128Width {
			if red[j] == 0 {
				continue
			}
			p[base+j] ^= ll.GF16Mul(c, red[j])
		}
	}
	var out GF128
	copy(out[:], p[:GF128Width])
	return out
}

// squareMod returns a*a mod (X^8 + Σ red[j] X^j).
func squareMod(a GF128, red [GF128Width]uint16) GF128 {
	return mulFullMod(a, a, red)
}
