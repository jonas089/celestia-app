#!/usr/bin/env bash
# Builds the Go c-shared library that rsema1d-sys links against.
#
#   go build -buildmode=c-shared -o librsema1d.dylib ./pkg/rsema1d/cshim
#
# The dylib's install-name is set to @rpath/librsema1d.dylib so build.rs's rpath
# entry resolves it at test/run time. Emits librsema1d.dylib + librsema1d.h into
# rust/crates/rsema1d-sys/lib/.
set -euo pipefail

CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# rust/crates/rsema1d-sys -> repo root
REPO_ROOT="$(cd "${CRATE_DIR}/../../.." && pwd)"
OUT_DIR="${CRATE_DIR}/lib"
mkdir -p "${OUT_DIR}"

echo ">> Building librsema1d.dylib (real Go rsema1d PCS) ..."
cd "${REPO_ROOT}"
CGO_ENABLED=1 go build \
    -buildmode=c-shared \
    -ldflags "-extldflags '-Wl,-install_name,@rpath/librsema1d.dylib'" \
    -o "${OUT_DIR}/librsema1d.dylib" \
    ./pkg/rsema1d/cshim

echo ">> Wrote ${OUT_DIR}/librsema1d.dylib and librsema1d.h"
