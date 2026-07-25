# Transaction-throughput benchmark image for the accidental-computer rv32 rollup.
# Builds the `tx_bench` binary with build-tunable circuit dims (PROG_LEN /
# MEM_SLOTS / STATE_SLOTS via option_env!), so a sweep can size the committed
# circuit per transaction-batch. Run: `docker run --rm tx-bench:local <N>`.
#
#   docker build -f docker/tx-bench.Dockerfile \
#     --build-arg RV32_MEM_SLOTS=1024 --build-arg RV32_STATE_SLOTS=32 \
#     --build-arg RV32_PROG_LEN=64 -t tx-bench:local .

FROM golang:1.26-bookworm AS builder

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config clang libclang-dev curl ca-certificates \
        git libopenmpi-dev openmpi-bin libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain nightly-2025-05-17 \
    && rustc --version && cargo --version

WORKDIR /build/celestia-app
COPY . .

RUN CGO_ENABLED=1 go build \
        -buildmode=c-shared \
        -ldflags "-extldflags '-Wl,-soname,librsema1d.so'" \
        -o rust/crates/rsema1d-sys/lib/librsema1d.so \
        ./pkg/rsema1d/cshim

# Circuit dims fixed at build time (read by rustc via option_env!). Sized per
# benchmark point so the committed square fits the transaction batch.
ARG RV32_MEM_SLOTS=1024
ARG RV32_STATE_SLOTS=32
ARG RV32_PROG_LEN=64
ENV RV32_MEM_SLOTS=$RV32_MEM_SLOTS \
    RV32_STATE_SLOTS=$RV32_STATE_SLOTS \
    RV32_PROG_LEN=$RV32_PROG_LEN

RUN set -eux; \
    RSEMA1D_DIR=/build/celestia-app/rust/crates/rsema1d-sys/lib; \
    MPI_LIBDIR="$(mpicc --showme:libdirs | awk '{print $1}')"; \
    export CARGO_NET_GIT_FETCH_WITH_CLI=true; \
    export RUSTFLAGS="-L native=${MPI_LIBDIR} -C link-arg=-Wl,-rpath-link,${MPI_LIBDIR} -C link-arg=-Wl,-rpath-link,${RSEMA1D_DIR}"; \
    cd rust/crates/rv32-rollup; \
    cargo +nightly-2025-05-17 build --release --bin tx_bench; \
    ls -l target/release/tx_bench

FROM ubuntu:24.04 AS runtime
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libopenmpi3 tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/celestia-app/rust/crates/rsema1d-sys/lib/librsema1d.so /usr/local/lib/librsema1d.so
COPY --from=builder /build/celestia-app/rust/crates/rv32-rollup/target/release/tx_bench /usr/local/bin/tx_bench
RUN ldconfig
ENV LD_LIBRARY_PATH=/usr/local/lib \
    OMPI_MCA_rmaps_base_oversubscribe=1 \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENTRYPOINT ["tini", "--", "tx_bench"]
