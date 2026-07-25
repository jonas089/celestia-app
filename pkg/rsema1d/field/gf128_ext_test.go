package field

import (
	"crypto/sha256"
	"encoding/binary"
	"testing"
)

// polyX is the field element X (degree-1 monomial), used as the generator of
// the residue ring for the Frobenius irreducibility test.
func polyX() GF128 { return GF128{0, 1} }

// frobenius raises a to the q-th power (q = 2^16) modulo the candidate
// reduction poly, i.e. applies the GF(2^16) Frobenius once: 16 squarings.
func frobenius(a GF128, red [GF128Width]uint16) GF128 {
	for range 16 {
		a = squareMod(a, red)
	}
	return a
}

// isIrreducible tests whether f(X) = X^8 + Σ red[j] X^j is irreducible over
// GF(2^16) using the standard criterion for degree d=8: f is irreducible iff
// X^(q^d) ≡ X (mod f) and X^(q^(d/p)) ≢ X (mod f) for every prime p | d.
// Here d=8 so the only prime is p=2, giving the single proper divisor d/2=4.
func isIrreducible(red [GF128Width]uint16) bool {
	const d = GF128Width // 8
	// e_d = X^(q^d): apply Frobenius d times to X.
	ed := polyX()
	for range d {
		ed = frobenius(ed, red)
	}
	if !Equal128(ed, polyX()) {
		return false
	}
	// e_half = X^(q^(d/2)): apply Frobenius d/2 times to X. Must differ from X.
	eh := polyX()
	for range d / 2 {
		eh = frobenius(eh, red)
	}
	return !Equal128(eh, polyX())
}

// TestFindIrreducible searches pseudo-random full degree-8 candidate reduction
// polys (all 8 low coefficients derived from a counter hash) and logs the first
// several irreducible ones. Run with -v to read off a poly to wire into
// reductionPoly. Trinomials of the form X^8+cX+c0 turn out to be systematically
// reducible in reedsolomon's GF(2^16) basis, so we sweep the full coefficient
// space instead.
func TestFindIrreducible(t *testing.T) {
	found := 0
	for n := uint32(1); n < 200000 && found < 5; n++ {
		var red [GF128Width]uint16
		var seed [4]byte
		binary.LittleEndian.PutUint32(seed[:], n)
		h := sha256.Sum256(seed[:])
		for j := range GF128Width {
			red[j] = binary.LittleEndian.Uint16(h[j*2:])
		}
		if red[0] == 0 { // f must have nonzero constant term to be irreducible
			continue
		}
		if isIrreducible(red) {
			t.Logf("IRREDUCIBLE #%d (n=%d): reductionPoly = %#v", found, n, red)
			found++
		}
	}
	if found == 0 {
		t.Fatal("no irreducible found — criterion or arithmetic bug")
	}
}

// TestReductionPolyIsIrreducible pins the committed reductionPoly: once wired
// in, it must pass the irreducibility test so MulFull really operates in a
// field. Skips while the poly is still the zero placeholder.
func TestReductionPolyIsIrreducible(t *testing.T) {
	if !reductionPolySet {
		t.Skip("reductionPoly not yet committed; run TestFindIrreducible first")
	}
	if !isIrreducible(reductionPoly) {
		t.Fatalf("committed reductionPoly %v is NOT irreducible — GF128 would not be a field", reductionPoly)
	}
}

// TestFieldAxioms exercises multiplicative identity, commutativity,
// associativity, distributivity over addition, and absence of zero divisors on
// a handful of elements, giving confidence MulFull is a real field operation.
func TestFieldAxioms(t *testing.T) {
	if !reductionPolySet {
		t.Skip("reductionPoly not yet committed")
	}
	elems := []GF128{
		One(),
		{1, 2, 3, 4, 5, 6, 7, 8},
		{0xffff, 0, 0xabcd, 0, 0x1234, 0, 0x5678, 0},
		{9, 0, 0, 0, 0, 0, 0, 1},
		{0, 0x4321, 0, 0, 0, 0x9999, 0, 0},
	}
	for _, a := range elems {
		if !Equal128(MulFull(a, One()), a) {
			t.Fatalf("identity failed for %v", a)
		}
		for _, b := range elems {
			if !Equal128(MulFull(a, b), MulFull(b, a)) {
				t.Fatalf("commutativity failed: %v * %v", a, b)
			}
			if !IsZero(a) && !IsZero(b) && IsZero(MulFull(a, b)) {
				t.Fatalf("zero divisor found: %v * %v = 0", a, b)
			}
			for _, c := range elems {
				// associativity
				l := MulFull(MulFull(a, b), c)
				r := MulFull(a, MulFull(b, c))
				if !Equal128(l, r) {
					t.Fatalf("associativity failed: %v %v %v", a, b, c)
				}
				// distributivity: a*(b+c) = a*b + a*c
				lhs := MulFull(a, Add128(b, c))
				rhs := Add128(MulFull(a, b), MulFull(a, c))
				if !Equal128(lhs, rhs) {
					t.Fatalf("distributivity failed: %v %v %v", a, b, c)
				}
			}
		}
	}
}
