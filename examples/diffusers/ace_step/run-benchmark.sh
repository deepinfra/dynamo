#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run benchmark.py inside a baked ACE-Step ship image, pinned to a
# specific GPU UUID. Hides the docker invocation so engineers don't
# need to remember --gpus quoting, --ipc, --shm-size, volume mounts,
# or working directory.
#
# Mirrors examples/diffusers/run-benchmark.sh (fastvideo).
#
# Usage:
#   ./run-benchmark.sh <image-tag> <gpu-uuid> [output-dir] [extra benchmark.py flags...]
#
# Example:
#   ./run-benchmark.sh \
#     localhost:30500/ace-step-runtime:0.1.0 \
#     GPU-f58493b8-7d8b-698e-2e19-20b84b1e27c7
#
# To run unattended (the bigger LM tier may take several minutes per clip on first load):
#   nohup ./run-benchmark.sh <image-tag> <gpu-uuid> \
#     > ~/ace-step-benchmark.log 2>&1 &
#
# The image must contain benchmark.py at /opt/app/benchmark.py. The
# script bind-mounts the local examples/diffusers/ace_step/ over
# /opt/app inside the container, so whatever is on disk is what runs.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<USAGE
Usage: $0 <image-tag> <gpu-uuid> [output-dir] [extra benchmark.py flags...]

Run benchmark.py inside the given image, pinned to the given GPU UUID.
Run this on the SAME host that has the GPU reservation (not your laptop).

Arguments:
  image-tag    Full registry/image:tag, e.g.
               localhost:30500/ace-step-runtime:0.1.0
  gpu-uuid     GPU UUID, e.g. GPU-f58493b8-7d8b-698e-2e19-20b84b1e27c7
               (find with: nvidia-smi -L)
  output-dir   Where to write audio files and timings.csv.
               Default: /tmp/ace-step-outputs
  extra ...    Any further args are forwarded to benchmark.py
               (e.g. --duration 30 --dit-config <path> --lm-model <name>).

Typical invocation (unattended):
  nohup $0 <image-tag> <gpu-uuid> > ~/ace-step-benchmark.log 2>&1 &

Monitor from another shell on the SAME host:
  tail -f ~/ace-step-benchmark.log | grep -F '[benchmark]'
USAGE
  exit 1
fi

IMAGE="$1"
GPU_UUID="$2"
OUTPUT_DIR="${3:-/tmp/ace-step-outputs}"
shift 3 2>/dev/null || shift $#
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"

echo "[run-benchmark] image=$IMAGE"
echo "[run-benchmark] gpu=$GPU_UUID"
echo "[run-benchmark] output_dir=$OUTPUT_DIR"
echo "[run-benchmark] script_dir=$SCRIPT_DIR (mounted at /opt/app)"
echo

# HF cache is bind-mounted from the host so weights survive container restarts.
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

exec docker run --rm \
  --gpus "device=$GPU_UUID" \
  --ipc=host \
  --shm-size=16g \
  -v "$SCRIPT_DIR:/opt/app" \
  -v "$OUTPUT_DIR:$OUTPUT_DIR" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  -w /opt/app \
  "$IMAGE" \
  python3 benchmark.py \
    --gpu-uuid "$GPU_UUID" \
    --output-dir "$OUTPUT_DIR" \
    --csv "$OUTPUT_DIR/timings.csv" \
    "${EXTRA_ARGS[@]}"
