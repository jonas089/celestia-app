# Multi-stage build for the "accidental rv32i computer" sovereign rollup
# (rust/crates/rv32-rollup). The rollup executes committed rv32i programs over
# persistent VM state, posts program+input+output+trace+state to Celestia DA,
# and GKR-proves each block by REUSING the rsema1d/DA commitment as the sole
# polynomial commitment (zero prover re-encode).
#
# Build context MUST be the celestia-app repo root: it needs both the Go DA
# encoder (pkg/rsema1d/cshim -> librsema1d.so) and the Rust prover + rollup
# (rust/). The vendored Expander lives at rust/vendor and is wired via the
# crate's [patch] block; the ECC frontend + rsmpi + halo2curves are git deps
# fetched from GitHub at build time (needs network + git).
#
#   docker build -f docker/rv32-rollup.Dockerfile -t rv32-rollup:accidental-local .

# ---------------------------------------------------------------------------
# Stage 1 — builder: Go 1.26 (celestia-app go.mod needs >=1.26; the `tool`
# directive breaks on 1.23) + nightly Rust pinned to the prover's toolchain.
# ---------------------------------------------------------------------------
FROM golang:1.26-bookworm AS builder

# System toolchain: C/C++ + clang (cgo + rust link), OpenMPI (rsmpi links
# libmpi), OpenSSL + pkg-config (git deps), git + curl + CA certs (rustup,
# cargo git fetches).
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        clang \
        libclang-dev \
        curl \
        ca-certificates \
        git \
        libopenmpi-dev \
        openmpi-bin \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Nightly Rust pinned to the toolchain the prover was validated with.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain nightly-2025-05-17 \
    && rustc --version && cargo --version

WORKDIR /build/celestia-app
COPY . .

# Build the Go c-shared DA encoder that rsema1d-sys links against. Emit it as a
# real Linux .so (with a librsema1d.so SONAME) into the exact directory the
# rsema1d-sys crate's build.rs searches (rustc-link-search=<manifest>/lib +
# rustc-link-lib=dylib=rsema1d).
RUN CGO_ENABLED=1 go build \
        -buildmode=c-shared \
        -ldflags "-extldflags '-Wl,-soname,librsema1d.so'" \
        -o rust/crates/rsema1d-sys/lib/librsema1d.so \
        ./pkg/rsema1d/cshim \
    && ls -l rust/crates/rsema1d-sys/lib/

# Build the fibre uploader tool (tools/rv32-fibre-upload). In FIBRE-REUSE DA mode
# the rollup execs this to encode the rsema1d input square ONCE, upload it to the
# fibre server, and settle its commitment on-chain via MsgPayForFibre. Pure-Go
# (no cgo), so build it statically.
RUN CGO_ENABLED=0 go build -o /usr/local/bin/rv32-fibre-upload ./tools/rv32-fibre-upload \
    && /usr/local/bin/rv32-fibre-upload --help 2>&1 | head -1 || true

# Build the rollup. The crate is its own workspace root (own Cargo.lock +
# [patch]->rust/vendor). Fetch git deps over the git CLI (CARGO_NET_GIT_FETCH_
# WITH_CLI). RUSTFLAGS make the final link resolve librsema1d.so and libmpi at
# link time (the -soname/SONAME references need -rpath-link on Linux; runtime
# resolution is handled by LD_LIBRARY_PATH in stage 2).
# Circuit dims for the deployed transaction contract (read by rustc via
# option_env!): STATE_SLOTS=16 = 8 accounts x (balance+key); MEM_SLOTS=64 holds
# the state + a per-block tx batch; PROG_LEN=64 fits the 45-word contract.
ENV RV32_MEM_SLOTS=64 \
    RV32_STATE_SLOTS=16 \
    RV32_PROG_LEN=64
RUN set -eux; \
    RSEMA1D_DIR=/build/celestia-app/rust/crates/rsema1d-sys/lib; \
    MPI_LIBDIR="$(mpicc --showme:libdirs | awk '{print $1}')"; \
    export CARGO_NET_GIT_FETCH_WITH_CLI=true; \
    export RUSTFLAGS="-L native=${MPI_LIBDIR} -C link-arg=-Wl,-rpath-link,${MPI_LIBDIR} -C link-arg=-Wl,-rpath-link,${RSEMA1D_DIR}"; \
    cd rust/crates/rv32-rollup; \
    cargo +nightly-2025-05-17 build --release; \
    ls -l target/release/rv32-rollup

# ---------------------------------------------------------------------------
# Stage 2 — runtime: slim Ubuntu with just the OpenMPI runtime + the two
# artifacts (rollup binary + librsema1d.so).
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libopenmpi3 \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*

# The DA encoder dylib, the rollup binary, and the fibre uploader tool.
COPY --from=builder /build/celestia-app/rust/crates/rsema1d-sys/lib/librsema1d.so /usr/local/lib/librsema1d.so
COPY --from=builder /build/celestia-app/rust/crates/rv32-rollup/target/release/rv32-rollup /usr/local/bin/rv32-rollup
COPY --from=builder /usr/local/bin/rv32-fibre-upload /usr/local/bin/rv32-fibre-upload
RUN ldconfig

# Linux resolves librsema1d.so via LD_LIBRARY_PATH/ldconfig (the crate's
# build.rs only bakes a macOS @rpath). OMPI_* let the MPI runtime start as root
# inside the container without oversubscription errors.
ENV LD_LIBRARY_PATH=/usr/local/lib \
    OMPI_MCA_rmaps_base_oversubscribe=1 \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    RV32_ADDR=0.0.0.0:8545

EXPOSE 8545
ENTRYPOINT ["tini", "--", "rv32-rollup"]
