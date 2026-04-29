<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ACE-Step Music Pipeline Runbook

Operational procedures for evaluating ACE-Step 1.5 as a Dynamo example.

Mirrors the format of `examples/diffusers/RUNBOOK.md` (fastvideo). The engineering loop here is the same: build a ship image on a reserved GPU node, run `benchmark.py` inside it via the wrapper. The Dynamo HTTP frontend (`worker.py`) is a separate, future production-wiring step.

---

## Quick reference

| Task | Section |
|---|---|
| Build a ship image | [Producing a ship image](#producing-a-ship-image) |
| Run the smoke benchmark | [Running the benchmark](#running-the-benchmark) |
| Diagnose a failing run | [Diagnosing failures](#diagnosing-failures) |
| Promote to production wiring | [Production wiring (TODO)](#production-wiring-todo) |

---

## File map

| File | Purpose |
|---|---|
| `Dockerfile` | ACE-Step runtime image (CUDA + ai-dynamo + ace-step from upstream) |
| `benchmark.py` | In-process eval: loads the pipeline once, runs N prompts, logs timings + saves audio |
| `run-benchmark.sh` | Wrapper hiding the `docker run` invocation |
| `worker.py` | Dynamo backend endpoint (production wiring — not used in the engineering loop) |
| `local/run_local.sh` | Spawns dynamo frontend + worker for HTTP testing (production-only) |
| `test/smoke_test.py` | HTTP-frontend smoke test (production-only) |

---

## Image tag convention

```
ace-step-runtime:<base-version>-<sha8>
```

Where `<sha8>` is the first 8 hex chars of the source git commit. Build locally on the reserved GPU node into the per-host registry `localhost:30500` — same pattern as fastvideo. We don't push to a shared registry yet.

---

## Producing a ship image

**Prerequisites**:
- 1× A100-80GB host with a current GPU reservation
- Docker access on that host (the host's `docker` CLI must be functional — some nodes have a broken nvidia-container-toolkit hook; verify with `docker run --rm --gpus '"device=<UUID>"' ubuntu:22.04 nvidia-smi -L` before building)
- Free disk: ≥80 GB in Docker's storage dir for the build
- Local registry write access (`localhost:30500`)

### Step 1: clone and check out

```bash
cd ~ && mkdir -p work/ace-step && cd work/ace-step
git clone <dynamo-remote> dynamo
cd dynamo
git checkout claude/evaluate-ace-step-WPI3K
```

### Step 2: build the image

```bash
SHA=$(git rev-parse --short=8 HEAD)
docker build \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0 9.0 9.0a" \
  -t localhost:30500/ace-step-runtime:0.1.0-${SHA} \
  -f examples/diffusers/ace_step/Dockerfile \
  examples/diffusers/ace_step/
```

Build time: ~30–60 min (flash-attn + vLLM + ACE-Step source install). Subsequent builds re-use Docker's layer cache.

### Step 3: pre-pull weights (one-time, ~25–35 GB)

The benchmark wrapper bind-mounts `~/.cache/huggingface` into the container, so this pull happens once per host:

```bash
mkdir -p ~/.cache/huggingface
# Run a throwaway container with HF login if your account is gated:
docker run --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  localhost:30500/ace-step-runtime:0.1.0-${SHA} \
  bash -c 'huggingface-cli download ACE-Step/acestep-v15-xl-sft && \
           huggingface-cli download ACE-Step/acestep-5Hz-lm-4B'
```

---

## Running the benchmark

```bash
cd ~/work/ace-step/dynamo/examples/diffusers/ace_step
./run-benchmark.sh \
  localhost:30500/ace-step-runtime:0.1.0-${SHA} \
  GPU-<your-reserved-uuid> \
  /tmp/ace-step-outputs
```

For long unattended runs:

```bash
nohup ./run-benchmark.sh \
  localhost:30500/ace-step-runtime:0.1.0-${SHA} \
  GPU-<your-reserved-uuid> \
  > ~/ace-step-benchmark.log 2>&1 &
```

### Pass / fail

The wrapper exits 0 only if **all 3 prompts** generate non-silent audio within ±2s of the requested duration. Per-prompt details land in `/tmp/ace-step-outputs/timings.csv`:

```
prompt_id,seed,requested_duration_s,actual_duration_s,sample_rate_hz,peak_amplitude,generation_time_s,audio_file,error
lofi,42,15.0,15.04,44100,0.812,8.34,/tmp/ace-step-outputs/lofi_seed42.flac,
synthwave,42,15.0,14.99,44100,0.794,8.21,/tmp/ace-step-outputs/synthwave_seed42.flac,
cinematic,42,15.0,15.07,44100,0.683,8.55,/tmp/ace-step-outputs/cinematic_seed42.flac,
```

### Manual sanity listen

```bash
scp <node>:/tmp/ace-step-outputs/lofi_seed42.flac ~/Desktop/
# Open in any audio player. If it sounds like music (any music), the
# integration works. Quality assessment is a model-card concern, not
# an integration concern.
```

---

## Diagnosing failures

| Symptom | Likely cause |
|---|---|
| `OCI runtime create failed: ... error running ... hook ... enable-cuda-compat: unknown` | Host `docker` CLI hook is broken (nvidia-container-toolkit version mismatch). Move to a different node. |
| `OutOfMemoryError` at LM or DiT load | 4B+4B XL tier needs ~30–50 GB. Confirm GPU UUID resolves to an A100 80GB and not a 40GB. Drop to `--lm-model acestep-5Hz-lm-1.7B` to isolate. |
| `AceStepHandler.initialize_service` raises | Upstream config name vs. path ambiguity. Try the absolute path under `/opt/ACE-Step-1.5/configs/<dit-config>` (the source clone landing point in the Dockerfile). |
| LM checkpoint not found | HF cache layout mismatch. Pass `--checkpoint-dir` pointing at the actual snapshot dir under `~/.cache/huggingface/hub/`. |
| Smoke decode raises "audio appears silent" | Model loaded but produced silence. Check the saved file with `soundfile`/`librosa` and verify the prompt isn't being filtered. |

---

## Production wiring (TODO)

These are the steps to expose the worker through Dynamo's HTTP frontend at `/v1/audio/generations`. Not yet validated; same gap exists for fastvideo per the team's runbook.

1. Wire `/v1/audio/generations` routing in the Rust frontend (`lib/llm/src/protocols/openai/`).
2. Run `local/run_local.sh` to spawn frontend + `worker.py` together.
3. Hit it from `test/smoke_test.py`, which talks to the worker via `DistributedRuntime`.
4. `i model-add` flow + image push to a shared registry (when there is one).
