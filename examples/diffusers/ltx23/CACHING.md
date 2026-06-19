# LTX-2.3 compile-cache & cold-start strategy

How we make a freshly-booted pod fast, what is and isn't cacheable, and how to
bake the caches into the serving image. Full investigation + measurements:
`~/ltx23_cache_investigation_report.md` (and memory `reference_ltx_compile_cache_portability`).

## TL;DR
- Cold first-gen (compile + generate) ≈ **1103 s** (QUALITY 1080p) / ~1980 s (SPEED max-autotune).
- **Mega-Cache** (torch `save/load_cache_artifacts`) halves it to **~560 s** by restoring the inductor
  back-end. This is the one real win — wire it.
- The remaining **~512 s is the torch.compile FRONT-END** (dynamo trace + AOTAutograd) which re-runs every
  process *to produce the cache key* — NOT cacheable short of AOTInductor (a future project). Regional
  compilation is already in use (per-`transformer_block`); `caching_precompile`/DynamoCache crashes on
  reload; the implicit on-disk inductor cache does NOT port across processes. None of these help further.
- **Production model: Mega-Cache + resident pool + boot-warm.** A pod pays ~560 s ONCE at boot behind a
  readiness gate, then serves warm (~24 s QUALITY / ~15 s SPEED per clip). No on-the-fly recompile.

## What's wired (code)
- **Mega-Cache** — `lib/pool.py` exports `LTX_MEGACACHE_BLOB=<dir>/<shape_key>.megacache.bin` (from
  `LTX_MEGACACHE_DIR`) before building the generator; `fastvideo/worker/gpu_worker.py` (fork patch) LOADS
  the blob before the first forward and SAVES it after the first forward. Per-shape (one blob per shape).
  Best-effort: a missing/incompatible blob silently falls back to a cold compile (never wedges a pod).
- **CuTe / CUDA / QuACK env caches** — `lib/pool.py` sets `CUTE_DSL_CACHE_DIR=/cache/cutedsl`,
  `CUDA_CACHE_PATH=/cache/cuda`, `QUACK_CACHE_DIR=/cache/quack`, `QUACK_CACHE_AUTOTUNING=1`. These are
  **hygiene only** — measurement showed they're negligible for the residual (the front-end dominates). Keep
  them (cheap, correct) but don't expect a cold-start win from them.

## Production: bake the Mega-Cache blobs into the image
The blob is keyed by (profile + shape + torch version + GPU arch + code/image), so **bake per profile** and
**rebake on any image change** (a mismatch just falls back to cold compile, so it's safe, just slow).

1. **Bake** on a Blackwell box, `LTX_MEGACACHE_DIR` pointed at a writable scratch (one run does all shapes
   in `shapes.json`; the worker saves a blob per shape):
   ```
   docker run --gpus '"device=<UUID>"' \
     -e LTX23_PROFILE=quality -e VIDEO_MODEL_FAMILY=ltx23 -e LTX_MEGACACHE_DIR=/cache/megacache \
     -v <weights>:/data/default:ro -v /scratch/ltx23-bake:/cache \
     <image> python3 ltx23/warmup.py --shapes ltx23/shapes.json --output-dir /cache/out --per-shape-timeout 3600
   # -> /scratch/ltx23-bake/megacache/<shape_key>.megacache.bin  (one per shape, ~300 MB each)
   ```
2. **Bake into the image** (tar + ADD, like the LTX-2 cache bake):
   ```
   tar -cf megacache.tar -C /scratch/ltx23-bake megacache
   # Dockerfile: ADD megacache.tar /opt/app/    -> /opt/app/megacache/<shape_key>.megacache.bin
   ```
3. **Serve**: set `LTX_MEGACACHE_DIR=/opt/app/megacache` in the LTX-2.3 model's `extra_env` (per-model, so
   LTX-2 is unaffected). The per-shape pool worker loads its shape's blob before the first compile.
4. **Boot-warm gate**: have the pod route one dummy request per shape at startup (the ~560 s cost) before
   marking ready, so no real request ever hits a cold compile. The resident pool then serves warm.

## Caveats / limits (don't relitigate — measured)
- **One process serves ≤ ~8 distinct shapes** before dynamo's per-code-location `cache_size_limit` (=8)
  bites and it stops recompiling that location. Fine for our 2-shape menu; bump
  `torch._dynamo.config.cache_size_limit` if a single process must serve more.
- **One blob per shape — NOT one multi-shape blob.** A blob saved from a process that compiled multiple
  shapes (via `LTX_MEGACACHE_SAVE_EVERY`) fails to hit the *first* shape on reload (its key gets polluted by
  later compilation). Keep the per-shape pool + per-shape blobs (the proven path).
- **Implicit inductor cache (`TORCHINDUCTOR_CACHE_DIR`) does NOT port** across processes — don't ship/rely
  on it for cold-start (LTX-2's tar+ADD bake only "worked" because of the di-slc-35 host-pin).
- **Profile-specific**: a QUALITY (`mode=default`) blob will not serve the SPEED (`max-autotune`) profile.

## Future (not now): kill the ~512 s front-end
Only **AOTInductor** (export submodules to `.so`, no runtime dynamo) can remove the front-end. It's a
multi-week R&D effort with brittle per-model/per-version export maintenance — pursue only if ~560 s/pod
cold-start is too slow for the scaling pattern. See the report's "Reduction options" section.
