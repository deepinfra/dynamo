<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ACE-Step Music Generation Example

A Dynamo worker that wraps [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) — an open-source (MIT) music generation model that turns text prompts (and optional lyrics, BPM, key, reference audio) into 10s–10min audio clips.

ACE-Step is a two-stage pipeline run inside a single worker process:

1. **LM planner** (vLLM-backed) turns the user query into a structured song blueprint (caption metadata + lyrics with timestamps).
2. **Diffusion Transformer + VAE** synthesizes the audio from the blueprint.

This mirrors the FastVideo example pattern (`../worker.py`): one Python worker that owns its full pipeline and exposes a Dynamo endpoint.

## Status

**Scaffold — not yet validated end-to-end on hardware.** The worker, protocol types, Dockerfile, local runner, and smoke test are all in place. Before deploying:

- Confirm that the upstream config strings (`acestep-v15-xl-sft`, `acestep-5Hz-lm-4B`) match what your weights checkout exposes — they may need to be paths rather than names. Both are CLI-overridable.
- Native HTTP routing for `/v1/audio/generations` is **not yet wired** in the Rust frontend. Until that lands, hit the worker through the Dynamo runtime directly (`test/smoke_test.py`).

## Default tier (best quality)

| Component  | Default                     | VRAM (fp16)  |
| ---------- | --------------------------- | ------------ |
| DiT        | `acestep-v15-xl-sft` (4B)   | ~8–10 GB     |
| LM planner | `acestep-5Hz-lm-4B`         | ~8 GB        |
| VAE + activations + KV cache | —         | ~10–20 GB    |
| **Total recommended GPU** | **A100 80GB** | comfortable headroom |

A100 80GB is the v1 SKU. H100 is faster but ~2× the cost; latency is already <10s/song on A100, so cost-per-song wins.

Smaller tiers (0.6B / 1.7B LM × 2B DiT) can be selected via `--lm-model` and `--dit-config` and will fit on smaller GPUs.

## Files

```
examples/diffusers/ace_step/
├── worker.py              # Dynamo backend (mirrors FastVideo worker.py)
├── Dockerfile             # CUDA 13.1 + ai-dynamo + ACE-Step from upstream
├── README.md              # this file
├── local/
│   └── run_local.sh       # local frontend + worker spawn
└── test/
    └── smoke_test.py      # functional verification (no music expertise required)
```

The request/response contract lives in `components/src/dynamo/common/protocols/music_protocol.py` (`NvCreateMusicRequest`, `NvMusicResponse`).

## Local run

```bash
cd examples/diffusers/ace_step

# Defaults target the 4B XL tier; override with env vars:
#   MODEL, DIT_CONFIG, LM_MODEL, LM_BACKEND, CHECKPOINT_DIR, PROJECT_ROOT
./local/run_local.sh
```

Then in a second terminal:

```bash
python test/smoke_test.py \
    --prompt "upbeat lo-fi hip hop with warm vinyl crackle" \
    --duration 15 \
    --seed 42 \
    --out /tmp/sample.flac
```

The smoke test validates response shape, decodes the audio with `soundfile`, and asserts the clip is non-silent and within ±2s of the requested duration. No musical-theory knowledge is required to interpret pass/fail.

## Testing without music expertise

The smoke test answers all the questions an integration owner needs to answer:

- **Did the worker accept the request and return a valid response shape?** — assertion on the response Pydantic shape.
- **Is the returned payload a valid audio file?** — `soundfile.SoundFile(io.BytesIO(...))` decode.
- **Is it actually audio, not silence?** — peak-amplitude check (`> 0.001`).
- **Does duration match what was requested?** — within configurable tolerance.
- **Is generation deterministic with a fixed seed?** — `--check-determinism` runs twice, compares SHA-256 of the output bytes.
- **Sanity listen** — `--out /tmp/sample.flac` saves the clip; anyone can hear "music vs. static" in five seconds.

Musical *quality* (does it sound good?) is a model-card concern, not an integration concern.

## Request shape

```json
POST /v1/audio/generations  (once frontend routing lands)
{
  "model": "ACE-Step/acestep-v15-xl",
  "prompt": "energetic synthwave with arpeggiated leads",
  "lyrics": "[Instrumental]",
  "duration": 30,
  "response_format": "flac",
  "nvext": {
    "bpm": 120,
    "keyscale": "C minor",
    "num_inference_steps": 30,
    "guidance_scale": 7.0,
    "seed": 42,
    "thinking": true
  }
}
```

Response is `NvMusicResponse` with `data[0].b64_json` carrying the encoded audio.

## TODO before production

- [ ] Wire `/v1/audio/generations` routing in the Rust frontend (`lib/llm/src/protocols/openai/`) so clients don't have to use the runtime directly.
- [ ] Validate the `--dit-config` / `--lm-model` argument shape against an actual upstream checkout — the docs are ambiguous about whether these are config names or paths.
- [ ] Add deploy YAML (`deploy/agg.yaml`) once the SKU and PVC story for ACE-Step weights is decided.
- [ ] Bench cost-per-song and end-to-end latency on A100 80GB at the 4B+4B XL tier.
- [ ] Decide response transport for long clips (>1 min) — base64 is fine up to a point, beyond which an object-store URL is healthier.
