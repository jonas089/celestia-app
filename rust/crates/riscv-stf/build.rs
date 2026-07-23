//! Bakes an rpath to librsema1d.dylib into the final `keccak_pipeline` binary.
//!
//! `rsema1d-sys`'s own build script emits `rustc-link-search` / `rustc-link-lib`
//! (which propagate to dependents so the symbols resolve at link time) but its
//! `-rpath` is a `rustc-link-arg`, which does NOT propagate to a downstream
//! binary in another crate. We therefore re-emit the rpath here, where the build
//! script belongs to the crate that actually links the binary.

use std::path::PathBuf;

fn main() {
    // rust/crates/stf-circuit -> rust/crates/rsema1d-sys/lib
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = manifest
        .parent()
        .unwrap()
        .join("rsema1d-sys")
        .join("lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
}
