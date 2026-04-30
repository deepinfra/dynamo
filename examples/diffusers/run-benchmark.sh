#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run benchmark.py inside a baked FastVideo ship image, pinned to a
# specific GPU UUID. Hides the docker invocation so engineers don't
# need to remember --gpus quoting, --ipc, --shm-size, volume mounts,
# or working directory.
#
# Usage:
#   ./run-benchmark.sh <image-tag> <gpu-uuid> [output-dir]
#
# Example:
#   ./run-benchmark.sh \
#     localhost:30500/fastvideo-runtime:2.1.0-ltx2-c3266d71 \
#     GPU-d1062f6e-0195-5a0e-3872-b6ca86579cad
#
# To run unattended (long benchmarks take ~2 hours):
#   nohup ./run-benchmark.sh <image-tag> <gpu-uuid> \
#     > ~/benchmark.log 2>&1 &
#
# The image must be a FastVideo build that contains benchmark.py at
# /opt/app/benchmark.py and warmup_shapes.json next to it (i.e. the
# script bind-mounts the local examples/diffusers/ directory over
# /opt/app inside the container, so whatever is on disk here is what
# the container sees).

set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<USAGE
Usage: $0 <image-tag> <gpu-uuid> [output-dir] [extra benchmark.py flags...]

Run benchmark.py inside the given image, pinned to the given GPU UUID.
Run this on the SAME host that has the GPU reservation (not your laptop).

Arguments:
  image-tag    Full registry/image:tag, e.g.
               localhost:30500/fastvideo-runtime:2.1.0-ltx2-c3266d71
  gpu-uuid     GPU UUID, e.g. GPU-d1062f6e-0195-5a0e-3872-b6ca86579cad
               (find with: nvidia-smi -L)
  output-dir   Where to write MP4s and timings.csv.
               Default: /tmp/benchmark-outputs
  extra ...    Any further args are forwarded to benchmark.py.
               Useful for --prompt-major (forces shape-switch between
               consecutive generations to test in-memory cache eviction).

Typical invocation (unattended, ~2 hour run):
  nohup $0 <image-tag> <gpu-uuid> > ~/benchmark.log 2>&1 &

Production-like access pattern (every shape switch incurs whatever
torch.compile in-memory state cost it normally would):
  nohup $0 <image-tag> <gpu-uuid> /tmp/benchmark-outputs-pm \\
    --prompt-major > ~/benchmark-pm.log 2>&1 &

Monitor from another shell on the SAME host (not your laptop):
  tail -f ~/benchmark.log | grep -F '[benchmark]'
  # or a live dashboard:
  watch -n 30 "tail -20 ~/benchmark.log; echo; \\
    echo done: \\\$(grep -cF '-> ' ~/benchmark.log) / 30"
USAGE
  exit 1
fi

IMAGE="$1"
GPU_UUID="$2"
OUTPUT_DIR="${3:-/tmp/benchmark-outputs}"
shift 3 2>/dev/null || shift $#  # consume the three positional args we used
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"

echo "[run-benchmark] image=$IMAGE"
echo "[run-benchmark] gpu=$GPU_UUID"
echo "[run-benchmark] output_dir=$OUTPUT_DIR"
echo "[run-benchmark] script_dir=$SCRIPT_DIR (mounted at /opt/app)"
echo

exec docker run --rm \
  --gpus "device=$GPU_UUID" \
  --ipc=host \
  --shm-size=16g \
  -v "$SCRIPT_DIR:/opt/app" \
  -v "$OUTPUT_DIR:$OUTPUT_DIR" \
  -w /opt/app \
  "$IMAGE" \
  python3 benchmark.py \
    --shapes warmup_shapes.json \
    --output-dir "$OUTPUT_DIR" \
    --csv "$OUTPUT_DIR/timings.csv" \
    --gpu-uuid "$GPU_UUID" \
    "${EXTRA_ARGS[@]}"
