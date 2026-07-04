# Dynamo Learning Plan — Upstream Foundations + DeepInfra Additions

A structured plan for getting productive on **Dynamo** (the open-source,
datacenter-scale inference stack from `ai-dynamo/dynamo`) and on **what
DeepInfra has added on top of upstream** in this fork.

> **Where the customizations live.** This branch (`main` /
> `claude/dynamo-learning-upstream-48j1i3`) tracks a clean mirror of upstream
> Dynamo (currently **v1.2.0**). DeepInfra's production customizations live on
> the **`deep-main-v1.1.1-videogen`** branch (based on upstream **v1.1.1**).
> To read the custom code, check out that branch:
>
> ```bash
> git fetch origin deep-main-v1.1.1-videogen
> git switch deep-main-v1.1.1-videogen
> ```
>
> File paths in **Part 2** below refer to that branch.

---

## What Dynamo is (one paragraph)

Dynamo is **the orchestration layer above inference engines** — it does not
replace SGLang, TensorRT-LLM, or vLLM; it turns them into a coordinated
multi-node inference system. Its headline capabilities are **disaggregated
serving** (independently scaling prefill and decode), **KV-aware routing**
(avoiding redundant prefill), **multi-tier KV caching (KVBM)**, and an
**SLA-based planner** for autoscaling. It is built in **Rust** for the
performance-critical runtime and **Python** for engine integration and
extensibility.

---

# Part 1 — Learn Upstream Dynamo

Goal: understand the architecture well enough to know *where* a change like
"add a video-generation backend" plugs in, and *why*.

### Phase 0 — Orientation (½ day)

- Read the top-level [`README.md`](README.md) — "When to use Dynamo", the
  backend feature matrix, and the architecture diagram.
- Skim the docs site map under [`docs/`](docs/) and
  [`fern/`](fern/) (the Fern-built docs site).
- Map the repo at a glance:

  | Path | What lives here |
  |---|---|
  | `lib/` | **Rust core.** `runtime/` (distributed runtime, request/response planes), `llm/` (OpenAI protocols, backend, migration, KV routing glue), `parsers/` (tool-call parsers), `kv-router/`, `kvbm-*` (KV block manager tiers), `protocols/`, `tokenizers/`, `bindings/` (Python bindings). |
  | `components/src/dynamo/` | **Python engine integrations & services.** `frontend/`, `vllm/`, `sglang/`, `trtllm/`, `router/`, `planner/`, `global_planner/`, `global_router/`, `profiler/`, `common/`. |
  | `examples/` | Deployable reference workers (per backend + multimodal + diffusion). |
  | `deploy/` | Kubernetes operator, CRDs, Helm charts. |
  | `recipes/` | Tested end-to-end model deployment recipes. |
  | `benchmarks/`, `tests/` | Perf harnesses and integration tests. |

### Phase 1 — The distributed runtime (2–3 days)

- **Concepts:** namespaces, components, endpoints, the **request plane** vs
  the **event plane** (note the local-launch flags
  `--request-plane tcp --event-plane zmq`), service discovery via **etcd**,
  and messaging via **NATS**.
- Read `lib/runtime/` and how a worker registers an endpoint and a model.
- Trace one request end-to-end: frontend (`components/src/dynamo/frontend/`)
  → router → worker endpoint → engine → streamed response.
- **Exercise:** run a single-backend example locally (vLLM or SGLang) from
  `examples/backends/` and hit `/v1/chat/completions`.

### Phase 2 — The LLM protocol & serving layer (2–3 days)

- `lib/llm/src/protocols/openai/` — the OpenAI-compatible request/response
  types: `chat_completions/`, `completions/`, and **`nvext.rs`** (the
  Dynamo extension envelope; important for Part 2).
- `lib/llm/src/backend.rs` and `lib/llm/src/protocols/common/llm_backend.rs`
  — the backend abstraction and streamed delta generation.
- `lib/llm/src/migration.rs` — request migration between workers.
- Tool calling: `lib/parsers/` and `docs/tool-calling/`.

### Phase 3 — Routing, disaggregation, KV cache (3–4 days)

- **KV-aware routing:** `lib/kv-router/`, `lib/kv-hashing/`,
  `components/src/dynamo/router/`.
- **Disaggregated serving:** how prefill and decode are split and scheduled
  (design docs in `docs/design-docs/`).
- **KVBM (KV Block Manager):** the `lib/kvbm-*` crates — logical/physical
  block tiers, consolidator, engine, kernels.
- **Planner / autoscaling:** `components/src/dynamo/planner/` and
  `global_planner/` — SLA-driven scaling (TTFT/ITL targets).

### Phase 4 — Engine integrations & deployment (2–3 days)

- Compare how `vllm/`, `sglang/`, and `trtllm/` under
  `components/src/dynamo/` each implement the worker contract — this is the
  pattern any new backend (including a video backend) must follow.
- **Kubernetes:** the operator and CRDs under `deploy/`; walk a recipe in
  `recipes/`.
- **Observability:** Prometheus metrics surface, OTel tracing
  (recent upstream work), and the `DYN_SYSTEM_PORT` metrics endpoint.

**Milestone for Part 1:** you can deploy a backend, route a request through
disaggregated workers, and explain where engine integration, protocols, and
metrics each live.

---

# Part 2 — What DeepInfra Added on Top of Upstream

All paths below are on **`deep-main-v1.1.1-videogen`** (a moving branch —
re-fetch before relying on specifics). There are three buckets: a large
video-generation feature (2A), small Rust runtime/protocol extensions (2B),
and in-flight work not yet merged (2C).

## 2A. Video Generation Backend — a multi-model family (the big one)

DeepInfra added a **FastVideo-backed video-generation backend** for Dynamo,
serving through the runtime's `/v1/videos` endpoint. What started as a single
LTX-2 worker has grown into a **multi-model video family**: generic
infrastructure in `lib/` plus per-model packages that each plug a factory
function into the same pool / IPC / metrics machinery, with **no duplication**
of that infrastructure. As of the current `deep-main-v1.1.1-videogen` there
are **three model families shipping**, and more are in flight (see 2C).

| Family | Model | Notes |
|---|---|---|
| `ltx2/` | LTX-2 (~19B) | The original text-to-video worker. Fully documented `ARCHITECTURE.md` + `RUNBOOK.md`. |
| `ltx23/` | LTX-2.3-Distilled | Text-to-video **and image-to-video (i2v)**, refine upsampler, QUALITY/SPEED profiles, Mega-Cache cold-start. The most sophisticated family. |
| `fastwan/` | FastWan-QAD-FP8-1.3B | Smaller FP8 model, TAEHV decode, portrait shapes. |

**Start here (read in this order):**

1. [`examples/diffusers/README.md`](examples/diffusers/README.md) — layout
   and entry points.
2. [`examples/diffusers/ltx2/ARCHITECTURE.md`](examples/diffusers/ltx2/ARCHITECTURE.md)
   — **the most important document.** It explains *why* the worker is shaped
   the way it is, and records the dead ends that were tried and rejected. The
   design decisions here (§ below) apply to every family.
3. [`examples/diffusers/ltx23/CACHING.md`](examples/diffusers/ltx23/CACHING.md)
   — **the second most important document.** The Mega-Cache cold-start
   strategy, with measured numbers (§ below).
4. [`examples/diffusers/ltx23/PROFILES.md`](examples/diffusers/ltx23/PROFILES.md)
   — the QUALITY vs SPEED profiles and how each mirrors a FastVideo recipe.
5. [`examples/diffusers/ltx2/RUNBOOK.md`](examples/diffusers/ltx2/RUNBOOK.md)
   and [`examples/diffusers/ltx23/RUNBOOK.md`](examples/diffusers/ltx23/RUNBOOK.md)
   — operational procedures: adding a shape, baking an image, rollback, CI
   drift, diagnosing failures, updating FastVideo.

**Code map (current):**

```
examples/diffusers/
├── worker.py            top-level shim: dispatches --pool-worker invocations
│                        into lib.pool BEFORE importing the model worker
├── lib/                 GENERIC video-pipeline infrastructure (model-agnostic)
│   ├── pool.py            SubprocessPool, Connection-based IPC wire protocol,
│   │                      _pool_worker_main, PR_SET_PDEATHSIG, Mega-Cache
│   │                      blob env wiring
│   ├── backend.py         GenericVideoBackend: Dynamo endpoint, legacy
│   │                      in-process path + pool routing path
│   ├── i2v_input.py       image-to-video input handling (decode + validation)
│   ├── metrics.py         video_pool_* Prometheus series (per-model label)
│   ├── models.py          Pydantic request/response models
│   ├── menu.py            shape-menu hash + boot assertion
│   └── dynamo_wiring.py   get_worker_namespace, register_model
├── ltx2/                LTX-2 family (worker/factory/config/shapes/warmup +
│                        ARCHITECTURE.md, RUNBOOK.md, benchmark, tests)
├── ltx23/               LTX-2.3 family (worker/factory/config/shapes/warmup +
│                        ARCHITECTURE.md, CACHING.md, PROFILES.md, RUNBOOK.md,
│                        bake_bench.py, prompt_extension_system_prompt.md,
│                        streaming_speed.yaml, tests)
├── fastwan/             FastWan family (worker/factory/shapes/warmup + tests)
├── patches/             FastVideo fork patches baked into the image:
│                        ltx23_gpu_worker_megacache.patch, x264-threads-cap.patch
├── Dockerfile,          the standard image, plus
│   Dockerfile.dreamverse  the dreamverse-based image for the distilled/speed path
├── deploy/, local/      k8s manifests / docker-compose dev loop
```

Every family package exposes the same surface — `worker.py` (`main_cli`),
`factory.py` (`load_model()`, the shared factory used by both the legacy and
pool paths), `config.py` (canonical cache-keying kwargs), `shapes.json` (the
supported `(width, height, num_frames)` menu), `warmup.py` (per-shape
compile-cache producer), and `test_shapes.py`/`test_config.py` (pin the menu
hash and ship-path kwargs). To onboard a new video model you add a new package
in this shape; you do not touch `lib/`.

Also touched in the core tree (diffusion plumbing across backends):
`components/src/dynamo/common/protocols/video_protocol.py`,
`.../common/multimodal/video_loader.py`, `.../common/utils/video_utils.py`,
plus diffusion handlers/engines under `components/src/dynamo/sglang/` and
`components/src/dynamo/trtllm/`, and docs under `docs/features/diffusion/`.

### The five ideas you must understand

These are the load-bearing design decisions. Each was learned the hard way
(see ARCHITECTURE.md "What we tried that was wrong").

1. **Per-shape subprocess isolation (the compile-cache contract).**
   `torch.compile`/inductor/Triton cache keys fold in *in-process accumulated
   state*, not just shape params, so compiling shapes in different orders
   produces different on-disk keys — a fresh production worker would miss the
   cache and pay an 80–150s recompile per shape. `_dynamo.reset()` is *not*
   sufficient (Triton/CUDA state survives it). The fix: compile each shape in
   a **fresh subprocess with isolated cache dirs**
   (`TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` under
   `/cache/per-shape/<shape_key>/`). `warmup.py` produces these at image-bake
   time *through the same pool code path* the worker uses at runtime.

2. **Persistent SubprocessPool, not subprocess-per-request.** Cold start is
   ~3 minutes (interpreter + torch import + 30s model load + ~150–180s
   torch.compile retrace); steady-state inference is 5–50s. The pool keeps
   subprocesses alive, each **pinned to one shape** for its lifetime.

3. **K=2 on B200, sized empirically.** FastVideo's internal
   `multiproc_executor` forks a Worker child holding the model on GPU
   (~70 GiB resident), so each pool slot costs ~71 GiB — far more than the
   model card suggests. K=2 fits a B200 (178 GiB); K=3 OOMs.
   `VIDEO_POOL_MAX_SIZE` is env-overridable. **LRU eviction**
   (SIGTERM → 30s grace → SIGKILL) handles the shape menu exceeding pool
   capacity.

4. **IPC over `multiprocessing.Connection` on a dedicated fd, not text on
   stdout.** FastVideo and its child processes spew unstructured/ANSI log
   lines to stdout, which corrupts any text-line protocol (this failed twice
   in one day). The protocol is pickled dicts (`READY`/`REQUEST`/`DONE`/
   `ERROR`/`FATAL` + `request_id`) over a `Pipe` passed via `pass_fds=`;
   stdout/stderr become pure diagnostic ring buffers. (Pickle is OK here:
   intra-pod, same-user trust boundary.)

5. **`VIDEO_POOL_MODE` opt-in flag + shared factory.** Pool mode ships
   *off* so the image can roll with pool code present-but-inactive, then be
   flipped via config (instantly revertable), then defaulted-on in a
   follow-up. Both the legacy in-process path and the pool path call the
   **same** `ltx2/factory.py:load_model` so compile-cache keys match
   byte-for-byte — do not split this factory.

**Operational gotchas worth knowing** (from ARCHITECTURE.md): first request
per fresh subprocess is always slow (compile retrace); LRU eviction creates a
cold tail; FastVideo's `multiproc_executor` does **not** survive its own
input-validation errors (it kills our pool subprocess as a side effect → soft-
DoS risk, mitigated by parent-side validation + the deepapi admission layer);
and orphan risk on non-SIGTERM parent crashes (mitigated by
`PR_SET_PDEATHSIG`).

### LTX-2.3 additions you must understand (`ltx23/`)

LTX-2.3 is the most involved family and introduces concepts the LTX-2 docs
don't cover. Read `ltx23/CACHING.md` and `ltx23/PROFILES.md` in full.

- **Mega-Cache cold-start strategy.** Cold first-gen (compile + generate) is
  ~1103s (QUALITY 1080p). **Mega-Cache** — torch's
  `save_cache_artifacts` / `load_cache_artifacts` — halves it to ~560s by
  restoring the **inductor back-end**. The remaining ~512s is the
  torch.compile **front-end** (dynamo trace + AOTAutograd), which re-runs in
  every process to *produce the cache key* and is **not** cacheable short of
  AOTInductor. So the production model is **Mega-Cache + resident pool +
  boot-warm behind a readiness gate**: a pod pays ~560s once at boot, then
  serves warm at **~24s/clip** (QUALITY) with no on-the-fly recompile. The
  blob is per-shape, loaded before the first forward and saved after — wired
  via a FastVideo fork patch (`patches/ltx23_gpu_worker_megacache.patch`) and
  `lib/pool.py` env export. Correctness is validated (PSNR: blob-served output
  is as correct as cold).
- **1080p-only is a memory constraint, not a preference.** One resident
  process holding both t2v and i2v modes (16 compiled graphs) peaks at
  ~119 GB and fits a B200 (180 GB). Two *resolutions* would need two resident
  pool processes (~238 GB) → OOM. So the shape menu is deliberately 1080p-only.
- **Image-to-video (i2v).** LTX-2.3 serves i2v as a **separate compile** from
  t2v (both must be boot-warmed). Image input is routed via the top-level
  `input_reference` and handled by `lib/i2v_input.py`, which validates the
  image is decodable *before* handing off to the GPU subprocess.
- **QUALITY vs SPEED profiles.** Selected by the `LTX23_PROFILE` env var
  (`quality` default). Each faithfully mirrors a specific FastVideo recipe;
  they differ in quant (bf16 vs NVFP4), denoise steps, and refine settings.
  `ltx23/config.py` is the source of truth and `test_config.py` pins both.
- **Refine upsampler path** and Blackwell-specific Inductor knobs
  (`shape_padding=False`, `conv_1x1_as_mm`, `coordinate_descent_tuning`, …),
  `FLASH_ATTN` attention backend, and the cu128/CUDA-12.9 image are all wired
  in `factory.py`.

### FastWan additions (`fastwan/`)

A smaller **FastWan-QAD-FP8-1.3B** family with TAEHV decode (width/height/
num_frames plumbed through for correct portrait output) and a portrait
480×832 shape in the warm menu. Good example of a *lightweight* family that
reuses all of `lib/` with minimal per-model code.

### Suggested exercises for 2A

- Read `lib/pool.py` and trace one request from `backend.py` → pool → pinned
  subprocess → Connection round-trip → MP4-as-base64 response.
- Read `warmup.py` and `factory.py` together (pick one family); explain how
  the bake-time cache and the runtime cache stay key-compatible.
- Trace the shape-menu hash through `lib/menu.py`, `shapes.json`, the bake
  step, and the boot assertion — explain what drift the hash guards against.
- Read `ltx23/CACHING.md` end to end and explain, in one sentence each, what
  Mega-Cache *does* fix and what it *cannot* fix (the front-end residual).
- Locally: `examples/diffusers/local/` has a `docker-compose.yml` and
  `run_local.sh` for a non-B200 dev loop.

## 2B. Rust runtime / protocol extensions (small, surgical)

Two targeted changes in `lib/llm/` (and runtime) that support DeepInfra's
serving stack:

1. **`engine_data` opaque field on the `nvext` response** *(commit
   `1c9412d27`, "feat(lpu): add opaque engine_data field…")*. Adds an
   opaque, per-request **opt-in** field (via `extra_fields`) so the engine
   can return passthrough data on the response. Touches
   `lib/llm/src/protocols/openai/nvext.rs`, the chat/completions `delta.rs`
   stream builders, `backend.rs`, `migration.rs`, and
   `protocols/common/llm_backend.rs`.

2. **Configurable response-stream server port/host** *(commit `3acdc0f86`,
   "feat(runtime): add DYN_TCP_RESPONSE_STREAM_PORT/HOST…")*. New env vars
   `DYN_TCP_RESPONSE_STREAM_PORT` / `DYN_TCP_RESPONSE_STREAM_HOST` to make the
   TCP response-stream server bindable, for deployment behind DeepInfra's
   networking.

### Other operational deltas on the branch

Small but important production hardening, mostly around the video worker:
subprocess Prometheus metrics (spawns, evictions, requests, latencies,
failures), `PR_SET_PDEATHSIG` on pool subprocesses, defensive cleanup on
spawn failure / graceful exit on parent disconnect, a `DYN_SYSTEM_PORT`
warning, and the local-launch flag note
(`--request-plane tcp --event-plane zmq`, since the v1.1.1 default changed).
More recent hardening around the video families: **x264 encode-thread cap**
(`FASTVIDEO_X264_THREADS`, default 32, via `patches/x264-threads-cap.patch`),
**prompt no longer logged** in `create_video` (privacy parity), and
i2v image decodability validation before the GPU subprocess.

## 2C. In-flight work (branches not yet merged into `deep-main`)

At the time of writing these `johan/*` branches carry work ahead of
`deep-main-v1.1.1-videogen` — useful to know what's coming:

- **`johan/ltx23`** — SSRF guard on the i2v image-URL fetch (server-side
  request forgery protection for user-supplied image references).
- **`johan/megacache-rename`** — rename `LTX_MEGACACHE_*` env vars to a
  generic `DI_MEGACACHE_*` (backward-compatible), reflecting that the cache
  strategy is no longer LTX-specific.
- **`johan/cosmos-encode-threads`** — cap CPU video-encoder threads in
  `dynamo.common` `video_utils`; the "cosmos" name hints a **Cosmos** video
  family may be the next model onboarded.

(`johan/fastwan`, `johan/ltx23-speed-bake`, and `johan/x264-threads-cap` have
already been merged into `deep-main` and are described above.)

---

## How to find these changes yourself

```bash
# DeepInfra-authored commits = those WITHOUT an upstream (#NNNN) PR suffix:
git log --oneline origin/deep-main-v1.1.1-videogen | grep -vE '\(#[0-9]+\)'

# The whole video backend (all families):
git ls-tree -r --name-only origin/deep-main-v1.1.1-videogen \
  | grep -iE 'diffus|ltx|fastwan|video'

# In-flight work ahead of deep-main:
for b in $(git branch -r | grep 'origin/johan/'); do
  echo "== $b =="; git log --oneline origin/deep-main-v1.1.1-videogen..$b
done

# Inspect a specific runtime change:
git show 1c9412d27   # engine_data nvext field
git show 3acdc0f86   # DYN_TCP_RESPONSE_STREAM_* env vars
```

> Historical design narratives referenced by `ARCHITECTURE.md` live under
> `deepinfra/backend/claude_plans/` (dated plan docs covering the
> cache-order-dependence discovery, the per-shape cache fix, and the Phase-2
> subprocess-pool redesign). Read them if you need the long-form "why".

---

## Suggested overall sequence

1. **Part 1, Phases 0–2** — get the runtime + protocol mental model
   (about a week).
2. **Part 2A, docs first** (`README` → `ARCHITECTURE.md` → `RUNBOOK.md`),
   then the `lib/` infra, then `ltx2/` specifics.
3. **Part 2B** — read the two Rust commits in the context of Phase-2 protocol
   knowledge.
4. **Part 1, Phases 3–4** — depth on routing/KVBM/planner and k8s deployment,
   as needed for your work.
