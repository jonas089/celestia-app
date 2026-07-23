//! Bakes an rpath to librsema1d.dylib into the final cdylib so the transitively
//! linked Go DA encoder resolves at load time. Mirrors rsema1d-sys/build.rs's
//! rpath emission (rsema1d-sys emits the link-search / link-lib that propagate to
//! dependents at link time, but its rpath is a `rustc-link-arg` that does NOT
//! propagate to a downstream artifact — so we re-emit it here, in the crate that
//! actually produces the final linked library).

use std::path::PathBuf;

fn main() {
    // rust/crates/libaccprover -> rust/crates/rsema1d-sys/lib
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = manifest
        .parent()
        .unwrap()
        .join("rsema1d-sys")
        .join("lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    // Also allow overriding at runtime via the standard loader path.
    println!("cargo:rerun-if-changed=build.rs");
}
