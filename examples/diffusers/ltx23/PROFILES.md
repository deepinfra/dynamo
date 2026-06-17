# LTX-2.3 generation profiles

Two configs, each a **faithful mirror of a specific FastVideo recipe**. Pick with
the `LTX23_PROFILE` env var (`quality` is the default). Source of truth:
`ltx23/config.py` (load-time kwargs) + `ltx23/factory.py` (quant + Inductor knobs).
CI: `ltx23/test_config.py` pins both profiles.

Both profiles share: model `FastVideo/LTX-2.3-Distilled-Diffusers` (`fast-ltx23`),
**1920×1088 landscape**, 121 frames @ 24fps, guidance 1.0, negative `""`, refine
upsampler = `<model>/spatial_upscaler`, `FLASH_ATTN`, `vae_tiling=False`, the
Blackwell Inductor knobs (`shape_padding=False`, `conv_1x1_as_mm`,
`coordinate_descent_tuning`, `coordinate_descent_check_all_directions`,
`epilogue_fusion=False`), `LD_LIBRARY_PATH` unset, the cu128/CUDA-12.9 image
(`2.1.5-ltx23-cu128`).

## QUALITY (default) — mirrors `examples/inference/basic/basic_ltx2_3_distilled_i2v_typed.py`

| setting | value | FastVideo source |
|---|---|---|
| quant | **bf16** (no NVFP4) | example: `quant_config = None` |
| denoise steps | **8** | example sampling `num_inference_steps=8` |
| refine | enabled, **3** steps, gs 1.0, add_noise | example `preset_overrides.refine` |
| compile | DiT + text_encoder + **VAE**, inductor, fullgraph, **mode=default**, dynamic=false | example `CompileConfig` |
| offload | all False | example `OffloadConfig` |

Result: **vivid, crisp**; slower (no 4-bit speedup). The "looks good" path.

## SPEED — mirrors `apps/dreamverse/serve_configs/streaming_demo.yaml` (the 4.55s/1080p deploy)

| setting | value | FastVideo source |
|---|---|---|
| quant | **NVFP4** | yaml `engine.quantization.transformer_quant: NVFP4` |
| denoise steps | **5** | yaml `default_request.sampling.num_inference_steps: 5` |
| refine | enabled, **2** steps, gs 1.0, add_noise | yaml `preset_overrides.refine` |
| compile | DiT + text_encoder (**NO VAE**), inductor, fullgraph, **max-autotune-no-cudagraphs**, dynamic=false | yaml `engine.compile` |
| offload | all False, **pin_cpu_memory=true** | yaml `engine.offload` |

Result: **fast** (FastVideo's 4.55s path), but **NVFP4 desaturates ~40%**
(measured: dog bf16 SATAVG 15.6 vs NVFP4 9.6; clown 25 vs 16) — that is
FastVideo's own fast-path tradeoff, NOT a bug. max-autotune bake ~70 min.

## How the three parts are applied

Each profile = quant + denoise-steps + load-time-kwargs. They are applied in
three different places (keep them consistent — `config.py` documents the
canonical values via `QUALITY_DENOISE_STEPS`/`SPEED_DENOISE_STEPS`):

1. **quant** → `factory.py` (`profile_uses_nvfp4`): SPEED sets `NVFP4Config()`,
   QUALITY leaves bf16. NVFP4 is gated on `enable_optimizations` + Blackwell.
2. **denoise steps** → the shapes file at bake time / `num_inference_steps`
   per request at serve time. NOTE: torch.compile's cache is keyed on the graph,
   NOT the step count, so 5-vs-8 does not need separate caches — but it changes
   output + speed, so bake/serve each profile with its own step count.
3. **load-time kwargs** (refine steps, compile mode, offload, VAE compile) →
   `config.py` `profile_kwargs(profile)`, passed to `VideoGenerator.from_pretrained`.

## Bake & serve

```
# QUALITY bake (bf16, 8 steps, mode=default):
docker run ... -e LTX23_PROFILE=quality ... warmup.py --shapes <8-step shapes>
# SPEED bake (NVFP4, 5 steps, max-autotune ~70min):
docker run ... -e LTX23_PROFILE=speed   ... warmup.py --shapes <5-step shapes>
```
Serving: set `LTX23_PROFILE` on the model's k8s env; the pool subprocess inherits
it and `factory.load_model` builds the matching recipe.

## Known deltas vs FastVideo (intentional / N/A)

- We run **LTX-2.3** (`fast-ltx23`); streaming's *default* is LTX-2.0
  (`fast-ltx2`). 2.3 is what the 4.55s/1080p blog used.
- **Prompt enhancement** (their Dreamverse demo rewrites prompts via an LLM,
  `cerebras gpt-oss-120b`) is NOT wired here — a separate quality lever, not a
  model setting.
- Streaming-only knobs (`conditioning_num_frames`, `stream_mode`,
  `generation_segment_cap`, boot `warmup`) are for interactive continuation; we
  do single 5s batch gens.
- Output mp4 still needs BT.709 color tagging in the serving encode.
