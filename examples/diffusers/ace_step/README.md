<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ACE-Step Music Generation Example

A Dynamo example for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), an open-source (MIT) music generation model. Two-stage pipeline (LM planner → Diffusion Transformer + VAE) running inside one worker process.

Mirrors the structure of `examples/diffusers/` (fastvideo): a baked container image + a `benchmark.py` that exercises the model in-process for engineering iteration, plus a Dynamo `worker.py` for the eventual production HTTP path.

## Quick links

- **[RUNBOOK.md](RUNBOOK.md)** — operational procedures (build, run, diagnose).
- **[ACE-Step 1.5 upstream](https://github.com/ace-step/ACE-Step-1.5)** — model docs and weights.

## Two paths through this example

### 1. Engineering loop (active)

`Dockerfile` + `benchmark.py` + `run-benchmark.sh`. Build a ship image on a reserved GPU node, run the benchmark via the wrapper. The benchmark imports `acestep` directly and calls `generate_music()` in-process — **no Dynamo runtime, no HTTP frontend**. Same pattern as `examples/diffusers/benchmark.py` (fastvideo).

```bash
# On a reserved A100 80GB host with working Docker
docker build -t localhost:30500/ace-step-runtime:0.1.0 \
  -f examples/diffusers/ace_step/Dockerfile examples/diffusers/ace_step/

cd examples/diffusers/ace_step
./run-benchmark.sh localhost:30500/ace-step-runtime:0.1.0 GPU-<your-uuid>
```

Pass criterion: 3/3 prompts generate non-silent audio within ±2s of requested duration. See [RUNBOOK.md](RUNBOOK.md#running-the-benchmark) for details and `timings.csv` schema.

### 2. Production wiring (future)

`worker.py` + `local/run_local.sh` + `test/smoke_test.py`. Registers a Dynamo backend endpoint, spawns the dynamo HTTP frontend locally, hits it via the runtime client. **Not yet validated end-to-end** — same gap exists for fastvideo. Tracked in [RUNBOOK.md → Production wiring (TODO)](RUNBOOK.md#production-wiring-todo).

## Default tier

| Component  | Default                     | VRAM (fp16)  |
| ---------- | --------------------------- | ------------ |
| DiT        | `acestep-v15-xl-sft` (4B)   | ~8–10 GB     |
| LM planner | `acestep-5Hz-lm-4B`         | ~8 GB        |
| VAE + activations + KV cache | —         | ~10–20 GB    |
| **Target GPU** | **A100 80GB**            | comfortable headroom |

A100 80GB is the v1 SKU. H100 / B200 also work and are faster but more expensive; latency is already <10s/song on A100 per upstream, so cost-per-song dominates.

## File map

```
examples/diffusers/ace_step/
├── Dockerfile             # CUDA + ai-dynamo + ACE-Step from upstream
├── benchmark.py           # In-process eval (active)
├── run-benchmark.sh       # docker run wrapper (active)
├── RUNBOOK.md             # operational procedures
├── README.md              # this file
├── worker.py              # Dynamo backend endpoint (future production wiring)
├── local/run_local.sh     # frontend + worker spawn (future)
└── test/smoke_test.py     # HTTP-frontend smoke test (future)
```

The request/response contract lives in `components/src/dynamo/common/protocols/music_protocol.py` (`NvCreateMusicRequest`, `NvMusicResponse`) and is what `worker.py` accepts. `benchmark.py` doesn't use these — it bypasses the protocol layer entirely and calls the upstream pipeline directly.

## Testing without music expertise

`benchmark.py` answers all the questions an integration owner needs without any musical-theory knowledge:

- **Did the pipeline accept the request and return a valid result?** — `acestep.inference.generate_music` returns a `GenerationResult` we check for `success`.
- **Is the returned file a valid audio?** — `soundfile.SoundFile()` decode.
- **Is it actually audio, not silence?** — peak-amplitude check (`> 0.001`).
- **Does duration match what was requested?** — within ±2s.
- **Sanity listen** — the saved `.flac` files; anyone can hear "music vs. static" in five seconds.

Musical *quality* (does it sound good?) is a model-card concern, not an integration concern.

## Known gaps before production

- [ ] **Validate `--dit-config` / `--lm-model` argument shape** against an actual upstream checkout — upstream docs are ambiguous about whether these are config names or file paths.
- [ ] **Wire `/v1/audio/generations`** routing in the Rust frontend — see RUNBOOK § "Production wiring (TODO)".
- [ ] **Image push** to a shared registry (when one exists). For now, images live in per-host `localhost:30500`.
- [ ] **Bench cost-per-song and end-to-end latency** on A100 80GB at the 4B+4B XL tier.
- [ ] **Decide response transport for long clips** (>1 min) — base64 is fine up to a point; beyond that, an object-store URL is healthier.
