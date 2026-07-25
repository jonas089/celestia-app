//! Links the prebuilt Go c-shared library `librsema1d.dylib`.
//!
//! The dylib + header are expected under `$CARGO_MANIFEST_DIR/lib` (produced by
//! `build-dylib.sh`, which runs `go build -buildmode=c-shared`). Override the
//! directory with `RSEMA1D_LIB_DIR`.
//!
//! An rpath entry pointing at the lib directory is added so `cargo test` finds
//! the dylib at runtime; the dylib's install-name is set to
//! `@rpath/librsema1d.dylib` by build-dylib.sh, so the rpath resolves it.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = env::var("RSEMA1D_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("lib"));

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=rsema1d");

    // rpath so the test/bin artifacts locate librsema1d.dylib at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // The Go runtime's c-shared library needs these system frameworks on macOS
    // (mirrors rust/goffi/goffi.go's `#cgo darwin LDFLAGS`).
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Security");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RSEMA1D_LIB_DIR");
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("librsema1d.dylib").display()
    );
}
