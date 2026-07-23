//! Bake an rpath to the rsema1d DA-encoder dylib into the rollup binary so it
//! resolves `@rpath/librsema1d.dylib` without relying on DYLD_LIBRARY_PATH
//! (which macOS SIP strips). On Linux the .so is found via ldconfig/LD_LIBRARY_PATH.
fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let lib = format!("{manifest}/../rsema1d-sys/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rerun-if-changed=build.rs");
}
