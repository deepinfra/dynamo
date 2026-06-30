# LTX-2 Video Pipeline Runbook

Operational procedures for the LTX-2 video generation pipeline served via Dynamo.

This file is the canonical procedure document. Error messages from the pipeline (CI checks, runtime asserts, admission checks) link here.

---

## Quick reference

| Task | Section |
|---|---|
| Add a shape to the menu | [Adding a shape](#adding-a-shape) |
| Remove a shape from the menu | [Removing a shape](#removing-a-shape) |
| Build a new ship image | [Producing a ship image](#producing-a-ship-image) |
| Roll back to a previous image | [Rollback](#rollback) |
| Diagnose a failing worker | [Diagnosing failures](#diagnosing-failures) |
| Diagnose CI drift error | [CI drift](#ci-drift) |
| Identify FastVideo version in a running image | [Updating FastVideo](#updating-fastvideo) |
| Bump FastVideo to a newer upstream | [Updating FastVideo](#updating-fastvideo) |

---

## File map

| File | Repo | Purpose |
|---|---|---|
| `examples/diffusers/shapes.json` | dynamo | Shape menu (one source of truth on dynamo side) |
| `examples/diffusers/ltx2/warmup.py` | dynamo | Pre-compiles torch.compile cache for every menu shape |
| `examples/diffusers/ltx2/benchmark.py` | dynamo | Validates a baked image: 30 generations + timings |
| `examples/diffusers/run-benchmark.sh` | dynamo | Wrapper hiding the docker invocation |
| `examples/diffusers/Dockerfile` | dynamo | Base FastVideo runtime image |
| `examples/diffusers/worker.py` (top-level shim) + `examples/diffusers/ltx2/worker.py` | dynamo | Dynamo backend endpoint |
| `deepinfra/deepapi/inf_models/ltx.py` | backend | API admission: `SUPPORTED_SHAPES` + `_validate_shape_in_menu` |

`SUPPORTED_SHAPES` and `shapes.json` must contain the same set of `(width, height, num_frames)` tuples. They live in different repos by necessity (API enforcement vs. image build); CI verifies they match.

---

## Image tag convention

Production images are tagged:
```
fastvideo-runtime:<base-version>-ltx2-<sha8>
```

`<sha8>` is the first 8 hex chars of:
```
sha256(json.dumps(sorted(shape_tuples), separators=(',', ':')))
```

Tag's hash binds the image to a specific menu. If `shapes.json` changes, the hash changes, and the old image's tag no longer matches the menu. Layered safety checks (admission, runtime invariant) reject mismatches.

---

## Producing a ship image

End-to-end procedure for building a new ship image. Run when you change shapes, upgrade FastVideo/torch/CUDA, or otherwise need a fresh build.

**Prerequisites**:
- B200 host with at least one GPU reservation (currently `di-slc-111`)
- Docker access on that host
- Local registry write access (`localhost:30500`)

### Step 1: bring the menu in sync

If you changed `shapes.json`, also edit `SUPPORTED_SHAPES` in backend's `ltx.py`. Both must contain the same set of tuples. If you only added/removed entries on one side, CI will fail; see [CI drift](#ci-drift).

### Step 2: build the base image

On the B200 host:

```bash
cd ~/dynamo
git fetch origin <your-branch> && git checkout <your-branch>
docker build -t localhost:30500/fastvideo-runtime:<version>-ltx2-warmupbase examples/diffusers/
```

30-60 min build (flash-attn + fastvideo-kernel compile from source). Verify env:

```bash
docker image inspect localhost:30500/fastvideo-runtime:<version>-ltx2-warmupbase \
  --format '{{range .Config.Env}}{{println .}}{{end}}' | \
  grep -iE "TORCH_CUDA_ARCH_LIST|CMAKE_ARGS"
```

Expect `TORCH_CUDA_ARCH_LIST=10.0a` and `CMAKE_ARGS=...FASTVIDEO_KERNEL_BUILD_TK=OFF`.

### Step 3: warmup (populate `/cache`)

Each shape is routed through `lib.pool.SubprocessPool` so the
cache-building subprocess is byte-identical in code path to the runtime
serving subprocess. Cache keys produced here match what production reads
at serve time.

By default a shape failure is logged and the run continues to the next
shape (pass `--fail-fast` to stop on first failure). See
`claude_plans/2026-05-14-ltx2-phase2-subprocess-pool.md` "What we tried
that was wrong" for the Phase 3A regression that motivated the
pool-routed path.

```bash
sudo mkdir -p /cache/per-shape
sudo chown -R $USER:$USER /cache

nohup docker run --rm \
  --gpus '"device=<GPU-UUID>"' \
  --ipc=host --shm-size=16g \
  -v /cache:/cache \
  -v "$HOME/dynamo/examples/diffusers:/opt/app" \
  -v /tmp/warmup-outputs:/tmp/warmup-outputs \
  -v <path-to-weights>:/data/default:ro \
  -e HF_HUB_OFFLINE=1 \
  -w /opt/app \
  localhost:30500/fastvideo-runtime:<version>-ltx2-warmupbase \
  python3 ltx2/warmup.py \
    --shapes ltx2/shapes.json \
    --output-dir /tmp/warmup-outputs \
    --model /data/default \
  > ~/warmup.log 2>&1 &
```

**FP4 (fast / 10s) ship image:** add `--enable-optimizations` to the
`warmup.py` invocation above. This bakes the FP4 + `max-autotune-no-cudagraphs`
+ `fullgraph` compile cache instead of the standard bf16 one. The flag MUST
match the serving worker's `--enable-optimizations` — the two recipes produce
different torch.compile keys, so a bf16-baked image serving FP4 (or vice versa)
misses the cache and cold-compiles 10-15 min per shape. Keep the bf16 and FP4
images on distinct tags (e.g. `<version>-ltx2-$HASH` vs `<version>-ltx2-fp4-$HASH`)
so they're never conflated; `IMAGE_SHAPE_HASH` only encodes the shape menu, not
the recipe.

Monitor:
```bash
tail -f ~/warmup.log | grep -aF '[warmup]'
```

Wall-clock: 2-3 hours. Look for `[warmup] done. success=N/N failures=[]`. Any failures, see [Diagnosing failures](#diagnosing-failures).

### Step 4: bake the cache into a ship image

```bash
HASH=$(python3 -c "import hashlib, json; shapes = json.load(open('$HOME/dynamo/examples/diffusers/shapes.json'))['shapes']; canonical = json.dumps(sorted([(s['width'], s['height'], s['num_frames']) for s in shapes]), separators=(',',':')); print(hashlib.sha256(canonical.encode()).hexdigest()[:8])")
echo "Hash: $HASH"

BAKE=/tmp/warmup-bake && rm -rf $BAKE && mkdir -p $BAKE && cd $BAKE
tar -cf cache.tar -C / cache
printf 'FROM localhost:30500/fastvideo-runtime:<version>-ltx2-warmupbase\nADD cache.tar /\nENV IMAGE_SHAPE_HASH=%s\n' "$HASH" > Dockerfile
docker build -t localhost:30500/fastvideo-runtime:<version>-ltx2-$HASH .
```

Replace `<version>` with the actual base version. ~3-5 min total. Verify cache landed:

```bash
docker run --rm localhost:30500/fastvideo-runtime:<version>-ltx2-$HASH \
  du -sh /cache/torchinductor /cache/triton
```

Should report sizes matching the host's `/cache`.

### Step 5: validate with `benchmark.py`

```bash
nohup ./examples/diffusers/run-benchmark.sh \
  localhost:30500/fastvideo-runtime:<version>-ltx2-$HASH \
  <GPU-UUID> > ~/benchmark.log 2>&1 &

tail -f ~/benchmark.log | grep -aF '[benchmark]'
```

~2 hours. Check `/tmp/benchmark-outputs/timings.csv`:
- Steady-state generations should be roughly equivalent across shapes (currently ~50s per 1080p 5s video)
- First-request-per-shape may take 200-600s — expected (in-memory torch.compile state warming)
- Visually inspect the 30 MP4s for any obvious quality regression

### Step 6: push the image

```bash
docker push localhost:30500/fastvideo-runtime:<version>-ltx2-$HASH
```

### Step 7: register with `i model-add`

(Backend-side step — see backend's separate model-management procedures.)

---

## Adding a shape

1. Edit `examples/diffusers/shapes.json` to add the tuple. Width and height must be multiples of 32 (LTX-2 VAE constraint). Verify with `(w + 31) // 32 * 32 == w`.
2. Edit `deepinfra/deepapi/inf_models/ltx.py SUPPORTED_SHAPES` to match.
3. Open one PR per file (or one PR touching both repos if your tooling supports it). CI in backend will fail if the lists drift.
4. Once both PRs are reviewed and ready, do not merge yet. First, follow [Producing a ship image](#producing-a-ship-image) to build a new image whose tag-hash matches the new menu.
5. Push the image to the registry.
6. Merge both PRs. Then `i model-add` (or update model-config) to point production at the new tag.

---

## Removing a shape

1. Same as adding, but delete tuple from both files.
2. **No need to rebuild the image** — the existing baked cache is a superset; removing a shape just narrows admission.
3. However: the tag-hash will change. Re-tag the existing image:

```bash
NEW_HASH=$(python3 -c "...same hash code as Step 4 of producing...")
docker tag localhost:30500/fastvideo-runtime:<old-tag> localhost:30500/fastvideo-runtime:<version>-ltx2-$NEW_HASH
docker push localhost:30500/fastvideo-runtime:<version>-ltx2-$NEW_HASH
```

4. Merge the PR; update model config to the new tag.

---

## Rollback

If a deployment goes bad:

1. Identify the previous good image tag (e.g. `fastvideo-runtime:2.1.0-ltx2-<old-sha8>`).
2. Update model config to point back at it.
3. Update `SUPPORTED_SHAPES` in `ltx.py` to match the menu that previous tag was built for. Push as a hotfix PR.
4. The runtime invariant will re-validate on pod restart.

---

## Diagnosing failures

### Worker boot-asserts: "shape-hash mismatch"

The worker has detected that the cache baked in its image doesn't match the menu in `SUPPORTED_SHAPES`.

**Cause**: someone changed `SUPPORTED_SHAPES` without producing a matching image, or a stale image tag was deployed.

**Fix**:
- Quick: rollback the model config to the last image whose tag-hash matches `SUPPORTED_SHAPES`.
- Forward: produce a new image with the current menu (Producing a ship image).

### CI drift

CI fails: "shapes.json and SUPPORTED_SHAPES are out of sync".

**Cause**: someone edited one file without the other.

**Fix**: edit the lagging file so they match. If shapes were added/removed, you also need a new image (Adding/Removing a shape).

### `i model-add` rejected my image

Admission check found a tag-hash mismatch.

**Cause**: image was built for a different menu than what's currently in `SUPPORTED_SHAPES`.

**Fix**: either build a new image for the current menu, or revert `SUPPORTED_SHAPES` to whatever the existing image was built for.

### Warmup failed at shape N

Look at the actual error in `~/warmup.log`. Common modes:

- **`ValueError: Height and width must be divisible by 32`** — bad shape entry; round to a multiple of 32.
- **`RuntimeError: input tensor must fit into 32-bit index math`** — VAE int32 limit; the shape is too large for default VAE tiling. See [Known limits](#known-limits).
- **CUDA out of memory** — the shape is too large for the GPU you're warming on. Verify you're on B200 (180 GiB), not H200 (80 GiB). Check `nvidia-smi -L`.
- **`Forward execution thread failed.` / `BrokenPipeError`** — the in-process `VideoGenerator` died. Subprocess isolation should prevent this from killing the whole batch, but the shape that crashed needs investigation.

### Benchmark timings look wrong

- **Steady-state >100s on a small shape** — cache may not have been baked correctly. Verify with `docker run --rm <image> du -sh /cache/torchinductor /cache/triton` — should be hundreds of MB to ~1 GB total. If <10 MB, the bake step failed silently.
- **First-request-per-shape >1000s** — the cache isn't being read at all. Check `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` env vars are set inside the container.

---

## Known limits

- **`1920x1088@241f` and `1088x1920@241f` are excluded** from the menu because the VAE decoder's intermediate activation exceeds 2³¹ elements at 1080p × 241 frames, triggering `RuntimeError: input tensor must fit into 32-bit index math` in `F.pad`. 10-second video is offered at 720p and below.
- **"Nominal 720p" is served as 1280×736 / 736×1280**, not 1280×720. The LTX-2 VAE has `spatial_compression_ratio=32`, so output dims must be multiples of 32. The field-validator in `ltx.py` rounds 720 → 736 automatically.
- **First-request-per-shape after pod startup takes 200-600s** even with a baked cache. After that first request, subsequent same-shape requests are ~50s. Pod-startup latency is therefore ~5-10 min × number of distinct shapes the pod sees before reaching steady state. Future mitigation: a preflight loop in `ltx2/worker.py` that runs one tiny generation per shape on boot.

---

## Metrics

Pool-internal metrics (subprocess pool size, spawns, evictions, request counts/latencies by shape, subprocess failures by reason) are emitted via Dynamo's worker-side `system_status_server`. This is a **separate HTTP endpoint** from the frontend's `/metrics` — the frontend does not aggregate worker metrics.

### Required deployment environment

Set on the worker pod:

- `DYN_SYSTEM_PORT` — port for the metrics HTTP server (default: `-1`, disabled). Recommended: `9090`.
- `DYN_SYSTEM_HOST` — bind address (default: `127.0.0.1`). Set to `0.0.0.0` if the scrape needs to reach across pod-network boundaries.
- A `containerPort: 9090` declaration in the worker's k8s manifest.
- A Prometheus scrape config targeting this port on worker pods (separate from the frontend scrape on `:8000`).

Without these, `ltx2/worker.py` logs a WARNING at startup:

```
DYN_SYSTEM_PORT is not set (or is '-1'); pool metrics will NOT be exposed.
```

### Metrics emitted

All metrics prefixed `video_pool_`, auto-injected with Dynamo hierarchy labels (`dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, `model`, `model_name`):

| Metric | Type | Labels | Use |
|---|---|---|---|
| `video_pool_size` | Gauge | — | Current live subprocess count |
| `video_pool_spawn_total` | Counter | `shape_key` | Pool churn rate; high values per shape suggest LRU thrash |
| `video_pool_eviction_total` | Counter | `shape_key` | LRU evictions; matches spawn rate if pool is saturated |
| `video_pool_request_total` | Counter | `shape_key`, `status` | Throughput; status ∈ {DONE, ERROR, FATAL} |
| `video_pool_request_latency_seconds` | Histogram | `shape_key` | End-to-end pool latency (route entry → DONE response) |
| `video_pool_cold_spawn_seconds` | Histogram | `shape_key` | Time from fork to READY message |
| `video_pool_subprocess_failure_total` | Counter | `reason` | Failure breakdown; reason ∈ {cuda_fault, parent_disconnect, spawn_timeout, spawn_eof, spawn_parse_error, gen_timeout, gen_eof, desync, parse_error, send_failed} |

### Recommended alerts

- `rate(video_pool_eviction_total[5m]) > 0.1` — pool is thrashing; consider bumping `VIDEO_POOL_MAX_SIZE` or pruning the shape menu.
- `histogram_quantile(0.95, rate(video_pool_cold_spawn_seconds_bucket[10m])) > 60` — cold-spawn latency degraded.
- `rate(video_pool_subprocess_failure_total{reason="cuda_fault"}[5m]) > 0` — CUDA faults observed; investigate.
- `rate(video_pool_request_total{status="FATAL"}[5m]) > 0` — same as above, surfaced from the request path.

---

## What's in the cache

The baked `/cache` contains:

- **`/cache/torchinductor/`**: torch.compile output. Per-shape compiled kernels for the DiT forward pass. Largest part (~1 GB).
- **`/cache/triton/`**: triton kernel binaries. CUDA cubins for the inductor-generated kernels (~120 MB).

The cache is keyed by a hash of (tensor shape, dtype, device, kernel source). When a request comes in with a shape the cache covers, torch.compile loads the pre-built kernel instead of re-compiling — savings of ~10-15 minutes per first request.

The cache does not eliminate **all** first-request cost: torch's in-memory compile-state machine still does some setup work per shape, even with disk hits. That's why first-request-per-shape is ~200-600s instead of ~10s.

---

## Future improvements

- **Preflight loop in worker.py**: run one small generation per shape on pod startup, so the first customer request is steady-state fast.
- **CI-driven warmup** ("Option A"): instead of manual warmup on a B200 host, have CI on a GPU runner do the bake on every shape-menu change. Removes the manual `i model-add` step and the human-in-the-loop. Requires B200 capacity in CI; not blocking for soft launch.
- **Hash check in `lib/menu.py` (invoked from `ltx2/worker.py`)**: boot-asserts the baked cache matches the running shape menu. Currently planned, not yet implemented.
- **`i model-add` admission check**: refuses images whose tag-hash doesn't match `SUPPORTED_SHAPES`. Currently planned, not yet implemented.

---

## Updating FastVideo

FastVideo upstream is pinned to a specific SHA in `Dockerfile`'s `FASTVIDEO_SHA` ARG. Reproducible builds depend on this. Bumping it is a deliberate act because FastVideo's API surface (kwargs to `from_pretrained`, `generate_video` signature, available compile flags) keys our compile cache — drift breaks the cache invariant or the worker.

### Identify the FastVideo SHA in a running container

The pin is stamped into the runtime env, so:

```bash
# on any host where the image runs (di-slc-111, customer pod, etc.)
docker run --rm <image-tag> printenv FASTVIDEO_SHA
# or against a running container:
docker exec <container> printenv FASTVIDEO_SHA
```

If the env var isn't set, the image was built before this pinning landed; fall back to:

```bash
docker run --rm <image-tag> bash -c 'cd /tmp/FastVideo && git rev-parse HEAD'
```

### Decide: rebuild or upgrade?

- **Rebuild byte-identically (no FastVideo change)**: keep the existing `FASTVIDEO_SHA` ARG. The image will hash to the same compile-cache contents as the previous build, modulo torch/CUDA-driver patch bumps in the base layer.
- **Deliberate FastVideo upgrade**: pick a new SHA from `https://github.com/hao-ai-lab/FastVideo/commits/main`, edit the ARG, then re-bake the cache (procedure: [Producing a ship image](#producing-a-ship-image)). The new image MUST be re-tagged with a fresh `<sha8>` if the warmup_shapes hash changed; if shapes are unchanged, keep the existing `<sha8>` but bump the `<base-version>` so old and new aren't conflated.

### What to check before bumping the SHA

A non-exhaustive checklist — the actual surface depends on what changed upstream:

1. Did `VideoGenerator.from_pretrained` kwargs change? Compare ours in `examples/diffusers/config.py` against the upstream signature.
2. Did `generate_video` parameter names change (e.g., `num_frames`, `num_inference_steps`, `guidance_scale`)?
3. Did the FastVideo MP-executor or VAE tiling behavior change in a way that breaks our `expandable_segments`-incompatible runtime constraint?
4. Did any test in `tests/test_config.py` or `tests/test_warmup_shapes.py` start failing after the bump?

If any of (1)-(3), update `config.py` along with the SHA bump, then re-bake.

### Pinned SHA history

| Date | SHA | Image tag baked from this SHA | Notes |
|---|---|---|---|
| 2026-04-30 | `70ee5d23` | `2.1.3-ltx2-c3266d71` | Initial pin (was unpinned `main` tip at build time) |

Add a row when bumping. Keep at least the last two for rollback context.

---

## FAQ

**Q: Why 736 instead of 720?**
A: LTX-2's VAE has spatial_compression_ratio=32; height and width must be multiples of 32. 720 is not. The field-validator in `ltx.py` rounds customer requests of 720 up to 736 automatically. See [Known limits](#known-limits).

**Q: Can I do a 10-second 1080p video?**
A: Not at soft launch. The VAE hits a PyTorch int32 limit at 1080p × 241 frames. 10-second video is available at 720p and below. Future: smaller VAE tile size (~10-30% slower across the board) or upstream FastVideo/PyTorch fix.

**Q: Why does adding a shape need a rebuild?**
A: torch.compile produces shape-specific kernels. A new shape that isn't in the baked cache will hit a 10-15 minute cold compile on the first request — too slow for production. The cache must be re-baked.

**Q: How often will we need to rebuild?**
A: Whenever shapes change, or when we adopt a new FastVideo / torch / CUDA version. Initial expectation: maybe monthly during ramp, less after. Each rebuild is a few hours of mostly-unattended GPU time on slc-111.

**Q: Who has B200 access?**
A: The two-GPU reservation on di-slc-111 (UUIDs `GPU-d1062f6e-...` and `GPU-db888802-...`). Coordinate with the team before kicking off long builds.

**Q: How does warmup produce a cache that hits at serve time?**
A: Warmup routes each shape through the same `lib.pool.SubprocessPool` code path the runtime serving worker uses (spawning `worker.py --pool-worker` per shape). The cache-building subprocess IS the cache-reading subprocess by construction, so compile cache keys produced at warmup time match what runtime asks for at serve time. The load-bearing property: torch.compile / inductor fxgraph cache keys are sensitive to invocation context (__main__ identity, argv, sys.modules layout) in ways that diverge across invocation paths — using the production code path sidesteps the divergence. Validated 2026-05-16 di-slc-39: per-shape caches built this way produce byte-identical fxgraph keys to what runtime asks for (`fx_graph_cache hit` logs on byte-identical keys). See `claude_plans/2026-05-14-ltx2-phase2-subprocess-pool.md` "What we tried that was wrong" for the Phase 3A regression that motivated this design.
