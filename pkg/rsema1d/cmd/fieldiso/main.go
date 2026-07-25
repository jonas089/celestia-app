package main

import (
	"bufio"
	"encoding/hex"
	"fmt"
	"math/bits"
	"os"
	"strings"

	"github.com/celestiaorg/celestia-app/v10/pkg/rsema1d/field"
)

// pExpander returns p(x) = x^128 + x^7 + x^2 + x + 1 as a polynomial over R
// (coefficients 0 or 1).
func pExpander() rpoly {
	p := make(rpoly, 129)
	one := field.One()
	for _, i := range []int{0, 1, 2, 7, 128} {
		p[i] = one
	}
	return p
}

// traceMod computes Tr(s*X) mod f = sum_{i=0}^{127} (s*X)^(2^i) mod f, with X
// the residue class of the indeterminate in R[X]/(f).
func traceMod(s field.GF128, f rpoly) rpoly {
	// u = s*X  (degree 1, already reduced since deg f > 1)
	u := rpoly{field.Zero(), s}.trim()
	acc := u.clone()
	cur := u.clone()
	for i := 1; i < 128; i++ {
		cur = rpolyMulMod(cur, cur, f) // square mod f
		acc = rpolyAdd(acc, cur)
	}
	return rpolyMod(acc, f)
}

// findRoot returns one root of the completely-split, squarefree monic poly f
// over R using the (randomised) Berlekamp trace algorithm.
func findRoot(f rpoly) (field.GF128, error) {
	f = rpolyMakeMonic(f.trim())
	for f.deg() > 1 {
		var g rpoly
		found := false
		for attempt := 0; attempt < 64; attempt++ {
			s := randR()
			if rIsZero(s) {
				continue
			}
			t := traceMod(s, f)
			cand := rpolyGCD(f, t)
			d := cand.deg()
			if d > 0 && d < f.deg() {
				g = cand
				found = true
				break
			}
		}
		if !found {
			return field.GF128{}, fmt.Errorf("BTA failed to split factor of degree %d", f.deg())
		}
		// Recurse into the smaller factor.
		other, _ := rpolyDivMod(f, g)
		other = rpolyMakeMonic(other)
		if g.deg() <= other.deg() {
			f = g
		} else {
			f = other
		}
	}
	if f.deg() != 1 {
		return field.GF128{}, fmt.Errorf("root finding ended at degree %d", f.deg())
	}
	// f = x + c0 (monic) -> root = c0 (char 2).
	return f[0], nil
}

// ---- 128x128 GF(2) matrix, rows are u128 masks over the input bits ----

type matrix struct {
	row [128]u128 // out bit r = parity(row[r] & inputVector)
}

func matVec(m *matrix, v u128) u128 {
	var out u128
	for r := 0; r < 128; r++ {
		p := bits.OnesCount64(m.row[r].lo&v.lo) + bits.OnesCount64(m.row[r].hi&v.hi)
		if p&1 == 1 {
			out.setBit(r)
		}
	}
	return out
}

// encodeBits returns the 128-bit little-endian encoding of an R element as a
// u128 (bit b = bit b%8 of encoding byte b/8).
func encodeBits(g field.GF128) u128 {
	var b [field.GF128Size]byte
	field.EncodeGF128(b[:], g)
	var v u128
	for i := 0; i < 8; i++ {
		v.lo |= uint64(b[i]) << (8 * uint(i))
	}
	for i := 0; i < 8; i++ {
		v.hi |= uint64(b[8+i]) << (8 * uint(i))
	}
	return v
}

func decodeBits(v u128) field.GF128 {
	var b [field.GF128Size]byte
	for i := 0; i < 8; i++ {
		b[i] = byte(v.lo >> (8 * uint(i)))
		b[8+i] = byte(v.hi >> (8 * uint(i)))
	}
	return field.DecodeGF128(b[:])
}

// buildMatrix constructs M with column i = encodeBits(beta^i). Since output =
// XOR of columns selected by the input bits, we set M.row[r].bit(i) = bit r of
// column i.
func buildMatrix(beta field.GF128) *matrix {
	m := &matrix{}
	cur := field.One() // beta^0
	for i := 0; i < 128; i++ {
		col := encodeBits(cur)
		for r := 0; r < 128; r++ {
			if col.bit(r) == 1 {
				m.row[r].setBit(i)
			}
		}
		cur = rMul(cur, beta)
	}
	return m
}

// invertMatrix computes the GF(2) inverse via Gaussian elimination.
func invertMatrix(m *matrix) (*matrix, error) {
	a := *m
	inv := &matrix{}
	for i := 0; i < 128; i++ {
		inv.row[i].setBit(i) // identity
	}
	for col := 0; col < 128; col++ {
		// find pivot row >= col with bit col set
		piv := -1
		for r := col; r < 128; r++ {
			if a.row[r].bit(col) == 1 {
				piv = r
				break
			}
		}
		if piv < 0 {
			return nil, fmt.Errorf("matrix singular at column %d", col)
		}
		a.row[col], a.row[piv] = a.row[piv], a.row[col]
		inv.row[col], inv.row[piv] = inv.row[piv], inv.row[col]
		for r := 0; r < 128; r++ {
			if r != col && a.row[r].bit(col) == 1 {
				a.row[r] = a.row[r].xor(a.row[col])
				inv.row[r] = inv.row[r].xor(inv.row[col])
			}
		}
	}
	return inv, nil
}

// phi maps an Expander element to an R element.
func phi(m *matrix, e u128) field.GF128 {
	return decodeBits(matVec(m, e))
}

func hexU128LE(v u128) string {
	var b [16]byte
	for i := 0; i < 8; i++ {
		b[i] = byte(v.lo >> (8 * uint(i)))
		b[8+i] = byte(v.hi >> (8 * uint(i)))
	}
	return hex.EncodeToString(b[:])
}

func main() {
	failFatal := func(msg string) {
		fmt.Println("FAILED:", msg)
		os.Exit(1)
	}

	// -------- Step 4a: lock representation against in-crate KATs --------
	fmt.Println("== Expander GF2_128 KAT check (values from real crate tests.rs) ==")
	if !runExpanderKATs() {
		failFatal("Expander GF2_128 KATs did not match the real crate")
	}
	// Optionally cross-check against freshly generated Rust triples.
	if triples, ok := loadRustTriples(); ok {
		bad := 0
		for _, t := range triples {
			got := expMul(t.a, t.b)
			if got != t.prod {
				bad++
			}
		}
		if bad != 0 {
			failFatal(fmt.Sprintf("%d/%d Rust-generated triples mismatched", bad, len(triples)))
		}
		fmt.Printf("Rust-generated triples: %d/%d byte-identical\n", len(triples), len(triples))
	} else {
		fmt.Println("(no Rust triple file present; relying on in-crate KATs)")
	}

	// -------- Step 2: root finding via BTA --------
	fmt.Println("\n== Root finding (Berlekamp trace) ==")
	p := pExpander()
	beta, err := findRoot(p)
	if err != nil {
		failFatal(err.Error())
	}
	var bb [field.GF128Size]byte
	field.EncodeGF128(bb[:], beta)
	fmt.Printf("beta (R encoding, hex LE) = %s\n", hex.EncodeToString(bb[:]))

	// Verify p(beta) = 0 in R: beta^128 == beta^7 + beta^2 + beta + 1.
	pow := make([]field.GF128, 129)
	pow[0] = field.One()
	for i := 1; i <= 128; i++ {
		pow[i] = rMul(pow[i-1], beta)
	}
	rhs := rAdd(rAdd(rAdd(pow[7], pow[2]), pow[1]), pow[0])
	if !rEqual(pow[128], rhs) {
		failFatal("p(beta) != 0")
	}
	fmt.Println("p(beta) = 0 confirmed: beta^128 == beta^7 + beta^2 + beta + 1")

	// Confirm beta generates degree 128 (1, beta, ..., beta^127 independent):
	// M invertibility below is the definitive check.

	// -------- Step 3: build M and M^{-1} --------
	fmt.Println("\n== Building isomorphism matrices ==")
	M := buildMatrix(beta)
	Minv, err := invertMatrix(M)
	if err != nil {
		failFatal("M not invertible => beta does not generate: " + err.Error())
	}
	fmt.Println("M invertible (beta generates GF(2^128) over GF(2))")

	// -------- Step 5: homomorphism validation --------
	fmt.Println("\n== Homomorphism validation ==")
	const N = 10000
	mulPass, addPass, rtPass := 0, 0, 0
	// one_E -> one_R
	if rEqual(phi(M, expOne), field.One()) {
		fmt.Println("phi(one_E) == one_R: PASS")
	} else {
		failFatal("phi(one_E) != one_R")
	}
	for i := 0; i < N; i++ {
		a := randExpander()
		b := randExpander()
		// multiplicative
		lhs := phi(M, expMul(a, b))
		rhs := rMul(phi(M, a), phi(M, b))
		if rEqual(lhs, rhs) {
			mulPass++
		}
		// additive
		la := phi(M, a.xor(b))
		ra := rAdd(phi(M, a), phi(M, b))
		if rEqual(la, ra) {
			addPass++
		}
		// round trip M^{-1} . M = I  (via vectors)
		if matVec(Minv, matVec(M, a)) == a {
			rtPass++
		}
	}
	fmt.Printf("phi(a*b)==phi(a)*phi(b):   %d/%d\n", mulPass, N)
	fmt.Printf("phi(a+b)==phi(a)+phi(b):   %d/%d\n", addPass, N)
	fmt.Printf("M^{-1}.M = I (round trip): %d/%d\n", rtPass, N)

	// Also verify M^{-1}.M = I as an identity matrix directly.
	identOK := true
	for i := 0; i < 128; i++ {
		e := u128{}
		e.setBit(i)
		if matVec(Minv, matVec(M, e)) != e {
			identOK = false
			break
		}
	}
	if !identOK {
		failFatal("M^{-1}.M != I on basis vectors")
	}
	fmt.Println("M^{-1}.M = I on all 128 basis vectors: PASS")

	if mulPass != N || addPass != N || rtPass != N {
		failFatal("homomorphism validation did not reach 100%")
	}

	// -------- Step 6: export --------
	fmt.Println("\n== Exporting matrices ==")
	outDir := os.Getenv("FIELDISO_OUT")
	if outDir == "" {
		outDir = "."
	}
	if err := exportRust(outDir, beta, M, Minv); err != nil {
		failFatal("export: " + err.Error())
	}
	fmt.Printf("wrote %s/field_iso.rs and %s/README.md\n", outDir, outDir)

	fmt.Println("\nALL CHECKS PASSED")
}

// exportRust writes the matrices as a Rust source file plus a README.
func exportRust(dir string, beta field.GF128, M, Minv *matrix) error {
	var sb strings.Builder
	sb.WriteString("// AUTO-GENERATED by pkg/rsema1d/cmd/fieldiso. Do not edit by hand.\n")
	sb.WriteString("//\n")
	sb.WriteString("// GF(2^128) field isomorphism between Expander's GF2_128 and rsema1d's GF128.\n")
	sb.WriteString("//\n")
	sb.WriteString("// Convention: a field element is a 128-bit vector; bit i is the coefficient\n")
	sb.WriteString("// of x^i for the Expander (domain) side, and bit i of the little-endian\n")
	sb.WriteString("// 16-byte rsema1d encoding for the codomain side. Both sides pack bit i into\n")
	sb.WriteString("// byte i/8, bit i%8 (LSB first) -> the same layout as a little-endian u128.\n")
	sb.WriteString("//\n")
	sb.WriteString("// The maps are GF(2)-linear. ISO_E_TO_R applies to an Expander element e:\n")
	sb.WriteString("//   let r = 0u128; for i in 0..128 { if (e>>i)&1==1 { r ^= ISO_E_TO_R[i]; } }\n")
	sb.WriteString("// i.e. ISO_E_TO_R[i] is column i of M = encode(beta^i). r is the rsema1d\n")
	sb.WriteString("// encoding as a little-endian u128. ISO_R_TO_E is the inverse (columns of\n")
	sb.WriteString("// M^{-1}); apply it the same way to recover the Expander element.\n")
	sb.WriteString("//\n")
	var bb [field.GF128Size]byte
	field.EncodeGF128(bb[:], beta)
	sb.WriteString("// beta (rsema1d encoding, hex LE): " + hex.EncodeToString(bb[:]) + "\n")
	sb.WriteString("// p(x) = x^128 + x^7 + x^2 + x + 1\n\n")

	writeCols := func(name string, m *matrix) {
		// column i = the u128 whose bit r = m.row[r].bit(i)
		fmt.Fprintf(&sb, "pub const %s: [u128; 128] = [\n", name)
		for i := 0; i < 128; i++ {
			var col u128
			for r := 0; r < 128; r++ {
				if m.row[r].bit(i) == 1 {
					col.setBit(r)
				}
			}
			fmt.Fprintf(&sb, "    0x%016x_%016xu128,\n", col.hi, col.lo)
		}
		sb.WriteString("];\n\n")
	}
	writeCols("ISO_E_TO_R", M)
	writeCols("ISO_R_TO_E", Minv)

	if err := os.WriteFile(dir+"/field_iso.rs", []byte(sb.String()), 0o644); err != nil {
		return err
	}

	readme := `# GF(2^128) field isomorphism: Expander GF2_128 <-> rsema1d GF128

Generated by ` + "`pkg/rsema1d/cmd/fieldiso`" + `.

## Fields
- Expander GF2_128: GF(2)[x]/(x^128 + x^7 + x^2 + x + 1). Element = [u8;16]
  little-endian; coefficient of x^i is bit i (byte i/8, bit i%8, LSB first).
  one = 1, X = 2. This matches NeonGF2_128 / AVXGF2_128 raw serialization.
- rsema1d GF128: GF(2^16)[X]/(f) (klauspost Cantor basis), 16-byte little-endian
  encoding via EncodeGF128/DecodeGF128.

## Convention
A field element is a 128-bit vector v with bit i in byte i/8, bit i%8 (LSB
first) -- identical to a little-endian u128. On the Expander side bit i = coeff
of x^i; on the rsema1d side bit i = bit i of the 16-byte encoding.

## Matrices
Both maps are GF(2)-linear. Each constant is stored as columns of the transform.

    // Expander element e (u128) -> rsema1d encoding r (u128)
    let mut r = 0u128;
    for i in 0..128 { if (e >> i) & 1 == 1 { r ^= ISO_E_TO_R[i]; } }

    // rsema1d encoding r (u128) -> Expander element e (u128)
    let mut e = 0u128;
    for i in 0..128 { if (r >> i) & 1 == 1 { e ^= ISO_R_TO_E[i]; } }

ISO_E_TO_R[i] = encode(beta^i); ISO_R_TO_E = inverse. beta is a root of p(x) in
rsema1d GF128, found via the Berlekamp trace algorithm.

## Validation
phi is a verified field isomorphism: phi(a*b)=phi(a)*phi(b),
phi(a+b)=phi(a)+phi(b), phi(1)=1, and M^{-1}.M = I, all at 10000/10000 over
random inputs, with Expander multiplication byte-matched to the real Rust crate.
`
	return os.WriteFile(dir+"/README.md", []byte(readme), 0o644)
}

func randExpander() u128 {
	return u128{randU64(), randU64()}
}

// ---- Rust triple loading (optional) ----

type triple struct {
	a, b, prod u128
}

func loadRustTriples() ([]triple, bool) {
	path := os.Getenv("FIELDISO_TRIPLES")
	if path == "" {
		return nil, false
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, false
	}
	defer f.Close()
	var out []triple
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) != 3 {
			continue
		}
		out = append(out, triple{
			a:    hexLEToU128(parts[0]),
			b:    hexLEToU128(parts[1]),
			prod: hexLEToU128(parts[2]),
		})
	}
	return out, len(out) > 0
}

func hexLEToU128(s string) u128 {
	b, err := hex.DecodeString(s)
	if err != nil || len(b) != 16 {
		panic("bad triple hex: " + s)
	}
	var v u128
	for i := 0; i < 8; i++ {
		v.lo |= uint64(b[i]) << (8 * uint(i))
		v.hi |= uint64(b[8+i]) << (8 * uint(i))
	}
	return v
}
