#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint for the FastVideo runtime image.
#
# Pre-warms the Linux page cache for the compile-cache directory and the
# model-weights directory before launching the worker (or whatever command
# was passed to `docker run`). This trades ~5-15s of container start-up time
# for ~50-100s saved on the first inference per shape -- the .so files and
# weight tensors are already in RAM when torch.compile / VideoGenerator try
# to read them.
#
# After warming, exec the passed command so it runs as the container's main
# process (PID 1, proper signal handling).

set -uo pipefail

# Warm the page cache for one directory in parallel. Survives missing dirs
# and read errors; we don't want a warming hiccup to crash the container.
warm() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    echo "[entrypoint] $label: $path not present, skipping"
    return 0
  fi
  local t0
  t0=$(date +%s)
  local size_mb
  size_mb=$(du -sm "$path" 2>/dev/null | awk '{print $1}')
  echo "[entrypoint] $label: warming $path (${size_mb} MB) ..."
  # -P 8 = 8 parallel cat workers; -n 50 = each handles 50 files at a time.
  # NVMe queue depth handles this comfortably; tuned for small-file-heavy
  # workloads like the inductor cache (hundreds of small .so files).
  find "$path" -type f -print0 2>/dev/null \
    | xargs -0 -P 8 -n 50 cat 2>/dev/null > /dev/null || true
  local t1
  t1=$(date +%s)
  echo "[entrypoint] $label: warmed in $((t1 - t0))s"
}

echo "[entrypoint] starting page-cache warming"
T_START=$(date +%s)

# Run warmers in parallel; the kernel handles concurrent reads with no
# contention and we hit NVMe throughput rather than syscall overhead.
warm "compile cache (inductor + triton)" /cache &
warm "model weights (k8s mount)"         /data/default &
warm "model weights (HF cache)"          /root/.cache/huggingface &
wait

T_END=$(date +%s)
echo "[entrypoint] page-cache warming done in $((T_END - T_START))s; launching: $*"

# Blackwell (B200/GB200): unset LD_LIBRARY_PATH before launching the worker so
# torch loads its own bundled cuBLAS (cu130) instead of the system CUDA libs on
# the LD path (/usr/local/cuda/lib64 from the CUDA base image). FastVideo's
# LTX-2.3 reference flags this as mandatory -- the system-vs-torch cuBLAS
# mismatch otherwise fails every GEMM, and the refine path in particular crashes
# (pad_mm INVALID_VALUE). The cache bake runs with this unset too; serving must
# match or the baked cache's first refine GEMM dies. CUDA driver libs still
# resolve via the nvidia container runtime's ldconfig (verified: the bake loads
# FlashAttention-4 and compiles with LD_LIBRARY_PATH unset).
unset LD_LIBRARY_PATH

# exec preserves PID 1 semantics so docker stop / k8s SIGTERM go to the worker.
exec "$@"
