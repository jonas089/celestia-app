//! Adds a runtime rpath so the rsema1d-pcs test/bin artifacts locate
//! `librsema1d.dylib` (built by rust/crates/rsema1d-sys/build-dylib.sh).
//!
//! The rsema1d-sys build script emits the same rpath, but `cargo:rustc-link-arg`
//! only affects the crate the build script belongs to — it does not propagate to
//! downstream binaries. So this crate re-emits the rpath for its own artifacts.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = env::var("RSEMA1D_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("../rsema1d-sys/lib"));
    let lib_dir = lib_dir.canonicalize().unwrap_or(lib_dir);

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RSEMA1D_LIB_DIR");
}
