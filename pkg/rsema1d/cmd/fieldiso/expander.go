package main

// Expander GF2_128 arithmetic implemented in Go.
//
// Representation (confirmed against the real crate, see tests.rs KATs and the
// generated Rust triples): an element is a 128-bit value V packed little-endian
// as [16]byte. The coefficient of x^i is bit i of V (byte i/8, bit i%8, LSB
// first). one = 1, X = 2. Multiplication is a carry-less 128x128 product
// reduced modulo p(x) = x^128 + x^7 + x^2 + x + 1 (the constant 0x87 folds the
// high half back down, matching mul_by_x's reduction).

// u128 is a little-endian 128-bit value; bit i lives in lo/hi.
type u128 struct {
	lo, hi uint64
}

func (a u128) xor(b u128) u128 { return u128{a.lo ^ b.lo, a.hi ^ b.hi} }

func (a u128) bit(i int) uint64 {
	if i < 64 {
		return (a.lo >> uint(i)) & 1
	}
	return (a.hi >> uint(i-64)) & 1
}

func (a *u128) setBit(i int) {
	if i < 64 {
		a.lo |= 1 << uint(i)
	} else {
		a.hi |= 1 << uint(i-64)
	}
}

func (a u128) isZero() bool { return a.lo == 0 && a.hi == 0 }

// clmul64 computes the carry-less product of two 64-bit polynomials, returning
// the 128-bit result as (hi, lo).
func clmul64(x, y uint64) (hi, lo uint64) {
	for i := 0; i < 64; i++ {
		if (y>>uint(i))&1 == 1 {
			// XOR (x << i) into the 128-bit accumulator.
			if i == 0 {
				lo ^= x
			} else {
				lo ^= x << uint(i)
				hi ^= x >> uint(64-i)
			}
		}
	}
	return hi, lo
}

// expMul multiplies two Expander GF2_128 elements.
func expMul(a, b u128) u128 {
	// 256-bit carry-less product in words w0..w3 (w0 = bits 0..63).
	// a = a1:a0, b = b1:b0
	h00, l00 := clmul64(a.lo, b.lo)
	h11, l11 := clmul64(a.hi, b.hi)
	h01, l01 := clmul64(a.lo, b.hi)
	h10, l10 := clmul64(a.hi, b.lo)

	// middle = a0*b1 + a1*b0 (128-bit), placed at offset 64.
	mLo := l01 ^ l10
	mHi := h01 ^ h10

	w0 := l00
	w1 := h00 ^ mLo
	w2 := l11 ^ mHi
	w3 := h11

	return reduce256(w0, w1, w2, w3)
}

// reduce256 reduces a 256-bit polynomial (w0..w3, little-endian words) modulo
// p(x) = x^128 + x^7 + x^2 + x + 1.
func reduce256(w0, w1, w2, w3 uint64) u128 {
	// High half H (bits 128..255) held in words (w2, w3) as a 128-bit value.
	hLo, hHi := w2, w3

	// Fold H down: H * x^128 == H * (x^7 + x^2 + x + 1) == H*0x87.
	// Compute H ^ (H<<1) ^ (H<<2) ^ (H<<7) as a 128-bit low part plus a small
	// overflow of coefficients >= x^128.
	var foldLo0, foldLo1, over uint64
	for _, n := range [4]uint{0, 1, 2, 7} {
		var sl0, sl1, ov uint64
		if n == 0 {
			sl0, sl1, ov = hLo, hHi, 0
		} else {
			sl0 = hLo << n
			sl1 = (hHi << n) | (hLo >> (64 - n))
			ov = hHi >> (64 - n)
		}
		foldLo0 ^= sl0
		foldLo1 ^= sl1
		over ^= ov
	}

	r0 := w0 ^ foldLo0
	r1 := w1 ^ foldLo1

	// The overflow bits represent coefficients of x^(128 + k). Fold them once
	// more: they are < 8 bits, so overflow*0x87 lands entirely below x^128.
	if over != 0 {
		for i := 0; i < 8; i++ {
			if (over>>uint(i))&1 == 1 {
				// 0x87 << i, i < 8 so it fits in the low word (max bit 14).
				r0 ^= 0x87 << uint(i)
			}
		}
	}
	return u128{r0, r1}
}

var expOne = u128{1, 0}
