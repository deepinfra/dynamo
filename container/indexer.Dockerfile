# syntax=docker/dockerfile:1.7
# =============================================================================
# Standalone KV-cache indexer image  ->  `python -m dynamo.indexer`
#
# Used by DeepInfra's KV-cache measurement pod (backend kubernetes/9X-kv-measurer.yaml):
# three copies of this image run side-by-side under different policy flags
#   8090  --ignore-evictions --merge-shards   (PERFECT      : no evictions, one global shard)
#   8091  --ignore-evictions                  (ROUTING_COST : no evictions, per-shard)
#   8092  (no flags)                          (REALITY      : real evictions, per-shard)
# Each subscribes over ZMQ to the engine shards' KV-event tee and serves the
# HTTP API: /register /unregister /query /workers /dump /health (+ /metrics).
#
# Self-contained: compiles the Rust python bindings (dynamo._core) from source,
# then installs ONLY that wheel into a slim runtime. Both stages are ubuntu:24.04,
# so the runtime libstdc++ is identical to what the extension was linked against
# -> no LD_PRELOAD shim needed (that hack was only for the mismatched conda env).
#
# The build context is a `git archive` of the dynamo source (committed files
# only), so the multi-GB local target/ never enters the build. The indexer is
# fully standalone: no etcd / NATS, just HTTP + ZMQ. See build-indexer-image.sh.
# =============================================================================

# ---------- builder: compile the wheel -------------------------------------
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        build-essential pkg-config cmake \
        clang libclang-dev \
        protobuf-compiler \
        libzmq3-dev \
        python3 python3-dev python3-venv \
 && rm -rf /var/lib/apt/lists/*

# Rust: rustup with NO default toolchain. dynamo's rust-toolchain.toml pins
# 1.93.1 (edition 2024), which rustup auto-fetches on the first cargo run in /src.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --default-toolchain none --profile minimal
ENV PATH=/root/.cargo/bin:$PATH

# maturin in an isolated venv (Ubuntu 24.04 system python is externally-managed).
RUN python3 -m venv /opt/maturin && /opt/maturin/bin/pip install --no-cache-dir maturin
ENV PATH=/opt/maturin/bin:$PATH

# Source tarball (git archive), auto-extracted by ADD. rust-toolchain.toml lands
# at /src so rustup resolves 1.93.1 for the whole workspace.
ADD dynamo-src.tar /src

# Build the release wheel. kv-indexer-metrics adds the `dynamo.indexer` binary
# + Prometheus /metrics. nixl-sys' build script runs bindgen, so point it at
# libclang (resolved dynamically to survive llvm version bumps). Cache mounts
# keep the crate registry + compiled artifacts warm across rebuilds.
WORKDIR /src/lib/bindings/python
ENV CARGO_TARGET_DIR=/cargo-target
# Cap parallelism: this compiles on a shared ~288-core services node. The default
# (one rustc per core) both starves prod neighbours and exhausts the build's
# file-descriptor limit (hundreds of parallel rustc -> "too many open files").
# 16 jobs is plenty and well-behaved; build-indexer-image.sh also raises nofile.
ENV CARGO_BUILD_JOBS=16
# Debug build: fast to compile and correct for measurement/validation. Switch to
# `--release` only when you need real indexer throughput. kv-indexer-metrics adds
# the `dynamo.indexer` binary + Prometheus /metrics. nixl-sys' build script runs
# bindgen, so point it at libclang (resolved dynamically to survive llvm bumps).
# Cache mounts keep the crate registry + compiled artifacts warm across rebuilds.
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/cargo-target \
    export LIBCLANG_PATH="$(dirname "$(find /usr/lib/llvm-* -name 'libclang.so*' 2>/dev/null | head -1)")" \
 && maturin build --features kv-indexer-metrics --out /wheels

# ---------- runtime: install just the wheel --------------------------------
FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libzmq5 \
        python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

# venv with only the indexer wheel and its deps (pydantic, uvloop -> manylinux
# wheels from PyPI, no compiler needed here).
COPY --from=builder /wheels/ /wheels/
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir /wheels/ai_dynamo_runtime*.whl \
 && rm -rf /wheels
ENV PATH=/opt/venv/bin:$PATH

# Smoke-test the import + CLI at build time so a broken wheel fails the build.
RUN python -m dynamo.indexer --help >/dev/null

EXPOSE 8090
ENTRYPOINT ["python", "-m", "dynamo.indexer"]
CMD ["--port", "8090"]
