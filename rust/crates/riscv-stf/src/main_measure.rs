//! Memory/size measurement (compile only, no prove) for the ecrecover building
//! blocks, so circuit size can be sized against the 64 GB budget BEFORE any heavy
//! run. Prints ECC's totalCost; run under a watchdog + `/usr/bin/time -l` for peak
//! RSS. Arg selects what to compile: `smul1` (one 256-bit scalar-mul), `ecrec`
//! (full ecrecover_pubkey).
use expander_compiler::frontend::*;
use riscv_stf::secp256k1::circuit as ec;
use riscv_stf::u256::BITS;

declare_circuit!(Smul1 {
    k: [Variable; BITS],
    bx: [Variable; BITS],
    by: [Variable; BITS],
    ox: [PublicVariable; BITS],
    oy: [PublicVariable; BITS],
});
impl Define<GF2Config> for Smul1<Variable> {
    fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
        let j = ec::scalar_mul(api, &self.k.to_vec(), &self.bx.to_vec(), &self.by.to_vec());
        let (x, y) = ec::to_affine(api, &j);
        for i in 0..BITS { api.assert_is_equal(x[i], self.ox[i]); api.assert_is_equal(y[i], self.oy[i]); }
    }
}

declare_circuit!(Ecrec {
    r: [Variable; BITS], s: [Variable; BITS], v: Variable, z: [Variable; BITS],
    qx: [PublicVariable; BITS], qy: [PublicVariable; BITS],
});
impl Define<GF2Config> for Ecrec<Variable> {
    fn define<B: RootAPI<GF2Config>>(&self, api: &mut B) {
        let (x, y) = ec::ecrecover_pubkey(api, &self.r.to_vec(), &self.s.to_vec(), self.v, &self.z.to_vec());
        for i in 0..BITS { api.assert_is_equal(x[i], self.qx[i]); api.assert_is_equal(y[i], self.qy[i]); }
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "smul1".into());
    println!("[measure] compiling '{which}' (no prove) ...");
    let t = std::time::Instant::now();
    match which.as_str() {
        "smul1" => { let _ = compile(&Smul1::default(), CompileOptions::default()).unwrap(); }
        "ecrec" => { let _ = compile(&Ecrec::default(), CompileOptions::default()).unwrap(); }
        _ => { eprintln!("unknown target"); std::process::exit(2); }
    }
    println!("[measure] compiled '{which}' in {:?}", t.elapsed());
}
