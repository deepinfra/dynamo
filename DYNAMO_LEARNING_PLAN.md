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

All paths below are on **`deep-main-v1.1.1-videogen`**. There are two
distinct buckets.

## 2A. LTX-2 Video Generation Backend (the big one)

DeepInfra added a **FastVideo-backed text-to-video backend** for Dynamo,
serving the ~19B-param **LTX-2** model through the runtime's `/v1/videos`
endpoint. It is structured so additional video models plug in as new
per-model packages without duplicating the pool / IPC / metrics
infrastructure.

**Start here (read in this order):**

1. [`examples/diffusers/README.md`](examples/diffusers/README.md) — layout
   and entry points.
2. [`examples/diffusers/ltx2/ARCHITECTURE.md`](examples/diffusers/ltx2/ARCHITECTURE.md)
   — **the most important document.** It explains *why* the worker is shaped
   the way it is, and records the dead ends that were tried and rejected.
3. [`examples/diffusers/ltx2/RUNBOOK.md`](examples/diffusers/ltx2/RUNBOOK.md)
   — operational procedures: adding a shape, baking an image, rollback, CI
   drift, diagnosing failures, updating FastVideo.

**Code map:**

```
examples/diffusers/
├── worker.py            top-level shim: dispatches --pool-worker invocations
│                        into lib.pool BEFORE importing ltx2.worker
├── lib/                 GENERIC video-pipeline infrastructure (model-agnostic)
│   ├── pool.py            SubprocessPool, Connection-based IPC wire protocol,
│   │                      _pool_worker_main, PR_SET_PDEATHSIG handling
│   ├── backend.py         GenericVideoBackend: Dynamo endpoint, legacy
│   │                      in-process path + pool routing path
│   ├── metrics.py         video_pool_* Prometheus series (per-model label)
│   ├── models.py          Pydantic request/response models
│   ├── menu.py            shape-menu hash + boot assertion
│   └── dynamo_wiring.py   get_worker_namespace, register_model
└── ltx2/                LTX-2-SPECIFIC code
    ├── worker.py           main_cli: CLI parse, backend setup, registration
    ├── factory.py          load_model(): the shared VideoGenerator factory
    ├── config.py           canonical kwargs (cache-keying)
    ├── shapes.json         the supported (width, height, num_frames) menu
    ├── warmup.py           per-shape compile-cache producer (bake time)
    ├── benchmark.py        post-bake validation harness
    └── ARCHITECTURE.md / RUNBOOK.md / tests
```

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

### Suggested exercises for 2A

- Read `lib/pool.py` and trace one request from `backend.py` → pool → pinned
  subprocess → Connection round-trip → MP4-as-base64 response.
- Read `warmup.py` and `factory.py` together; explain how the bake-time cache
  and the runtime cache stay key-compatible.
- Trace the shape-menu hash through `lib/menu.py`, `shapes.json`, the bake
  step, and the boot assertion — explain what drift the hash guards against.
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

---

## How to find these changes yourself

```bash
# DeepInfra-authored commits = those WITHOUT an upstream (#NNNN) PR suffix:
git log --oneline origin/deep-main-v1.1.1-videogen | grep -vE '\(#[0-9]+\)'

# The whole video backend:
git ls-tree -r --name-only origin/deep-main-v1.1.1-videogen \
  | grep -iE 'diffus|ltx|video'

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
