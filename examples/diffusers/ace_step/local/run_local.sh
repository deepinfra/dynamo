#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${PYTHON_BIN:=python3}"
: "${MODEL:=ACE-Step/acestep-v15-xl}"
: "${DIT_CONFIG:=acestep-v15-xl-sft}"
: "${LM_MODEL:=acestep-5Hz-lm-4B}"
: "${LM_BACKEND:=vllm}"
: "${CHECKPOINT_DIR:=${ACESTEP_CHECKPOINT_DIR:-/models/acestep}}"
: "${PROJECT_ROOT:=${ACESTEP_PROJECT_ROOT:-/opt/ACE-Step-1.5}}"
: "${NUM_GPUS:=1}"
: "${HTTP_PORT:=8000}"
: "${DISCOVERY_DIR:=${SCRIPT_DIR}/.runtime/discovery}"
: "${LOG_DIR:=${SCRIPT_DIR}/.runtime/logs}"
: "${WORKER_EXTRA_ARGS:=}"
: "${FRONTEND_EXTRA_ARGS:=}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found"
  exit 1
fi

mkdir -p "${DISCOVERY_DIR}" "${LOG_DIR}"

export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV="${DYN_FILE_KV:-${DISCOVERY_DIR}}"

cd "${EXAMPLE_DIR}"

worker_cmd=(
  "${PYTHON_BIN}" worker.py
  --model "${MODEL}"
  --dit-config "${DIT_CONFIG}"
  --lm-model "${LM_MODEL}"
  --lm-backend "${LM_BACKEND}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --project-root "${PROJECT_ROOT}"
  --num-gpus "${NUM_GPUS}"
)
if [[ -n "${WORKER_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  worker_extra=( ${WORKER_EXTRA_ARGS} )
  worker_cmd+=("${worker_extra[@]}")
fi

frontend_cmd=("${PYTHON_BIN}" -m dynamo.frontend --http-port "${HTTP_PORT}" --discovery-backend file)
if [[ -n "${FRONTEND_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  frontend_extra=( ${FRONTEND_EXTRA_ARGS} )
  frontend_cmd+=("${frontend_extra[@]}")
fi

cleanup() {
  echo
  echo "Stopping local processes..."
  kill "${frontend_pid:-}" "${worker_pid:-}" 2>/dev/null || true
  wait "${frontend_pid:-}" "${worker_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting worker: ${worker_cmd[*]}"
"${worker_cmd[@]}" >"${LOG_DIR}/worker.log" 2>&1 &
worker_pid=$!

echo "Starting frontend: ${frontend_cmd[*]}"
"${frontend_cmd[@]}" >"${LOG_DIR}/frontend.log" 2>&1 &
frontend_pid=$!

echo ""
echo "Worker log:   ${LOG_DIR}/worker.log"
echo "Frontend log: ${LOG_DIR}/frontend.log"
echo ""
echo "The worker is registered as model='${MODEL}'."
echo "Until /v1/audio/generations is wired in the Rust HTTP frontend,"
echo "call the worker directly via the Dynamo runtime — see"
echo "examples/diffusers/ace_step/test/smoke_test.py."
echo ""

wait -n "${worker_pid}" "${frontend_pid}"
