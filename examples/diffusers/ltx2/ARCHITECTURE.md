# LTX-2 Worker Architecture

This document explains *why* the LTX-2 video-generation worker is shaped the
way it is. For operational procedures (adding a shape, baking an image,
rollback, diagnosing failures) see [`RUNBOOK.md`](RUNBOOK.md). For the
historical journey that produced this design, see the dated plan documents
under `deepinfra/backend/claude_plans/` referenced at the end of this file.

**Code layout.** Generic video-pipeline infrastructure (subprocess pool,
Connection-based IPC, Pydantic request/response models, Prometheus metrics,
menu-hash check, Dynamo registration helpers) lives in the sibling
[`../lib/`](../lib/) package. This document covers LTX-2-specific decisions
and operational characteristics; the patterns described here apply to any
future video model that plugs a factory function into the same
infrastructure.

If you are reading the worker code (the LTX-2 entry in
[`worker.py`](worker.py), the pool / IPC in
[`../lib/pool.py`](../lib/pool.py), or the backend / endpoint in
[`../lib/backend.py`](../lib/backend.py)) and wondering why the IPC plumbing
is so elaborate, or why subprocess management exists at all, this file is
the answer. It is written so a future maintainer (human or AI) can read it
cold and understand the constraints before changing anything.

## What this code serves

LTX-2 is a FastVideo-backed text-to-video model (~19B parameters) served
through the Dynamo runtime's `/v1/videos` endpoint. The pipeline accepts a
fixed menu of ~10 supported `(width, height, num_frames)` shapes
(`shapes.json`), runs ~5 diffusion inference steps per request, and
returns an MP4 as base64. Production hardware is B200 (178 GiB GPU memory).
A request takes 5-50 seconds of steady-state inference and produces a
0.3-2 MB MP4.

## Why subprocess isolation per shape

The torch.compile / inductor / Triton cache key for a compiled graph folds
in **in-process accumulated state**, not just the input shape parameters.
When a single Python process compiles shapes A → B → C in different orders,
the on-disk keys it writes for shape C differ depending on which shapes
preceded it. `torch._dynamo.reset()` clears Dynamo's graph cache but does
not clear Triton autotuner state, CUDA module state, or other accumulated
mutable state that the cache key incorporates.

The practical consequence: a "warmup" process that pre-compiles every shape
in one Python interpreter produces on-disk caches whose keys are unique to
that compilation order. A fresh production worker booting and serving the
same shape computes a different key, misses the cache, and recompiles from
scratch — a 80-150 second cost per shape per pod, exactly the cost the
warmup was supposed to eliminate.

The fix is to compile each shape in a **fresh Python subprocess** with its
own dedicated cache directories:

```
TORCHINDUCTOR_CACHE_DIR = /cache/per-shape/<shape_key>/torchinductor
TRITON_CACHE_DIR        = /cache/per-shape/<shape_key>/triton
```

Fresh process + isolated cache directories = deterministic cache key for
that shape. A production worker pointed at the same per-shape directory
sees the same fresh-state key, finds the cached compiled kernels, and
loads them rather than recompiling.

[`warmup.py`](warmup.py) produces these per-shape caches at image-bake time
by routing each shape through `lib.pool.SubprocessPool` — the same code
path the runtime worker uses. The runtime worker (this directory's
[`worker.py`](worker.py) + the generic pool in
[`../lib/pool.py`](../lib/pool.py)) consumes them at runtime by spawning a
subprocess per shape and pointing its env at the matching per-shape
directory.

This is the **per-shape compile cache contract**. The cache layout under
`/cache/per-shape/<shape_key>/` is the interface between the warmup
producer and the worker consumer; do not change one side without the
other.

## Why a persistent pool, not subprocess-per-request

Spawning a subprocess for every customer request would pay the full cold
cost on every request: Python interpreter startup (~3s), torch import (~5s),
model load from `/data/default` (~30s), per-shape cache hydration (~5s),
and crucially **torch.compile / dynamo retrace** on the first
`generate_video` call against a fresh interpreter (~150-180s). End-to-end
cold start is ~3 minutes. Steady-state inference is 5-50 seconds. A
subprocess-per-request architecture would pay 30-40x overhead on every
customer call.

The pool keeps subprocesses alive between requests. Each subprocess is
**pinned to one shape** (chosen at spawn time and immutable for the
subprocess's lifetime — the per-shape cache directories are set via
`env=` before exec, and torch's cache lookup happens on first inductor
use which is before any subsequent shape change would take effect). After
the first request per subprocess pays the compile retrace, every
subsequent request to that subprocess for that shape is steady-state.

The pool's job is to map an incoming request's `shape_key` to a live
subprocess, spawn a new one if absent, and evict old ones when pool
capacity is exceeded.

## Why K=2 on B200, not larger

Per-shape memory cost is dominated by FastVideo's internal multiprocess
architecture, not by our pool. FastVideo's `multiproc_executor` forks a
**Worker child** during model load that holds the actual model on GPU
(weights + workspace + compiled kernels). The Worker child is ~70 GiB
resident for LTX-2. The outer pool subprocess that owns the Worker child
is only ~600 MiB — it is essentially a shepherd, not a holder.

This means **each pool slot costs ~71 GiB of GPU memory**, not the
~25-35 GiB one might naively estimate from the model card. On a B200
(178 GiB) the math works out to K=2 fitting comfortably (~142 GiB
resident, leaves ~36 GiB for activation tensors and CUDA workspace).
K=3 OOMs deterministically: the third subprocess's model load fails
because there is only a few MiB of GPU memory left.

`VIDEO_POOL_MAX_SIZE` is env-overridable for larger hardware. H100
(80 GiB) cannot run pool mode for this model at all — K=1 leaves no
headroom for activations. If LTX-2 is ever scheduled onto non-B200 GPUs,
operationally either lower `VIDEO_POOL_MAX_SIZE` to match the actual
per-slot fit or disable pool mode entirely.

## Why LRU eviction

A pod can see all menu shapes over its lifetime (typically 10 shapes for
the LTX-2 menu) but the pool is bounded at K=2 by memory. Some eviction
policy is needed. LRU is a natural choice for shape-routing because
real-world customer traffic has temporal locality (a customer typically
generates several videos in the same shape in a session). The least
recently used shape is the best eviction candidate: it is least likely
to be the next request.

Eviction is `SIGTERM` → 30 second grace → `SIGKILL` if needed. The
30 second grace covers ordinary Python interpreter shutdown; SIGKILL
is the safety net for stuck subprocesses.

## Why `multiprocessing.Connection` for IPC, not text lines on stdout

FastVideo (and its internal Worker child, and vllm-flavored loggers
deeper in the stack) write log lines to stdout in unstructured format,
including ANSI-colored progress output from forked subprocesses. Any
text-line protocol layered on stdout will be corrupted by these writes:
the parser cannot distinguish a real protocol message from a library
log line that happens to start with the same token.

This was attempted (a text protocol with `READY` / `DONE` / `ERROR` /
`FATAL` lines tagged by request id) and failed twice in one day —
once for the outer pool subprocess's stdout (FastVideo's logger
desynced the parser on the first request) and once for FastVideo's
Worker child stdout (it inherits the outer subprocess's pre-isolation
fd 1 and bypasses any muzzling we apply later). The lesson is that
**any text protocol on stdout is whack-a-mole**: every library or
library-of-library that spawns a subprocess before the muzzle is a
new failure waiting to surface.

The current design routes the protocol through `multiprocessing.Pipe`
to a duplex `Connection` on a **dedicated fd no library has a handle
to**. The parent calls `Pipe(duplex=True)`, passes the child end's fd
via `pass_fds=` on the Popen, and immediately closes its copy of the
child end so EOF is observable if the subprocess dies. The child
reconstructs a `Connection` from the inherited fd. Messages are
pickled Python dicts with a `kind` discriminator (`READY`, `REQUEST`,
`DONE`, `ERROR`, `FATAL`) and a `request_id` for correlation.
stdout and stderr are pure diagnostic channels; library noise on
either is drained into a per-handle ring buffer for crash forensics
but cannot corrupt the protocol.

**Trust boundary on pickle:** both ends of the Connection are the same
codebase running on the same host as the same user. This is an
intra-pod, intra-trust-boundary channel; pickle is fine here. Any
future cross-tenant use of this pattern would need to switch to
JSON-over-Connection.

## Opt-in via `VIDEO_POOL_MODE`

Pool mode is **opt-in** for the first release: `VIDEO_POOL_MODE=1`
enables the pool path, anything else (default unset) keeps the legacy
in-process generator. This lets us:

1. Roll the image with pool code present but inactive — zero behavior
   change for the legacy in-process path.
2. Flip `VIDEO_POOL_MODE=1` in the model config as a separate,
   instantly-revertable change.
3. After a soft-launch period without incidents, a follow-up commit
   flips the default and deletes the legacy in-process path.

Until that follow-up lands, both paths remain in
[`../lib/backend.py`](../lib/backend.py). The shared
[`factory.load_model(model_name, num_gpus, enable_optimizations)`](factory.py)
factory is used by both (legacy path: called directly from
``GenericVideoBackend.initialize_model``; pool path: imported in pool
subprocesses via the ``--model-factory ltx2.factory:load_model`` dotted
reference), so compile-cache keys match byte-for-byte across the two paths
(do not split this factory; if you do, one path will produce caches the
other cannot hit).

## Operational costs to know

These are inherent to the architecture, not bugs:

- **First-request-per-fresh-subprocess is slow.** Subprocess READY
  fires fast (~20-25 seconds against a hot per-shape cache: Python
  startup + torch import + model load + cache hydrate). But the first
  `generate_video` call in a fresh subprocess still pays
  ~3 minutes of torch.compile / dynamo retrace before reaching
  steady-state. This cost applies on **every** subprocess spawn —
  pod startup, LRU eviction → respawn, FastVideo-side error → respawn —
  not just on pod startup.
- **Subsequent same-shape requests are 5-50 seconds steady-state.**
  This is the customer's typical experience for hot shapes.
- **LRU eviction triggers a cold tail.** If a customer hits a shape
  that was just evicted, they pay the ~3-minute first-request cost on
  the respawn. Mitigations on the followup list (eager-spawn top-N
  shapes by traffic; parent-side request validation).
- **FastVideo's `multiproc_executor` does not survive its own
  validation errors.** If FastVideo's `InputValidationStage` rejects
  a request (e.g., `num_inference_steps=0`), the Worker child crashes
  and FastVideo terminates its parent (= our outer pool subprocess)
  as a side effect. The pool's ERROR code path is exercised correctly
  but the subprocess dies anyway. The next request for that shape
  pays the ~3-minute cold respawn. This means "subprocess stays alive
  after a soft error" only applies to errors that don't go through
  FastVideo's Worker — for example, an output-path-save failure
  after a successful generation, or a malformed protocol message
  detected by our code before reaching FastVideo. There is a
  soft-DoS risk worth knowing about: a customer streaming malformed
  requests can keep one pool slot in permanent respawn at the cost
  of ~3 minutes per bad request. Mitigated by parent-side input
  validation in `FastVideoBackend.create_video` (followup) and by
  the deepapi admission layer's existing shape validation.
- **Subprocess orphan risk on parent crash.** Subprocesses inherit
  the parent's process group, so a k8s `SIGTERM` to the pod kills
  the whole group cleanly. If the parent crashes (not SIGTERM),
  subprocesses can become orphaned and rely on container teardown
  for reaping. Acceptable for containerized deployment;
  `PR_SET_PDEATHSIG` deferred as belt-and-suspenders.

## What we tried that was wrong

Each of these was a real attempt that produced real bugs. They are
recorded here so future maintainers and AI sessions don't repeat them.
Full narrative for each is in the dated plan documents at
`deepinfra/backend/claude_plans/`.

- **Default-on pool mode in the first release.** Tempting (cleaner
  diff once the legacy path is gone), but the pool is untested IPC
  code on day one. Default-off → flip-flag deploy bisects deployment
  risk from behavior-change risk; default-on couples them.
- **`sys.exit(1)` on every subprocess exception.** Defeats persistence
  for soft failures. Distinguish CUDA-context-corrupting errors
  (exit so the parent respawns fresh) from validation / per-request
  errors (stay alive, the persistent subprocess is the whole point
  of the pool).
- **In-process `_dynamo.reset()` as the fix for cache-order
  dependence.** `_dynamo.reset()` clears Dynamo's compilation graph
  cache, but Triton autotuner / CUDA module / other in-process
  kernel state survives the reset and still feeds into compile-cache
  keys. Subprocess isolation is necessary; the reset alone is not
  sufficient.
- **Trusting FastVideo to keep stdout clean after READY.** FastVideo's
  vllm-style logger writes log lines to stdout from inside
  `generator.generate_video(...)`. A text-line protocol on stdout
  was the original Phase 2 design; the first library log line
  post-READY desynced it on the first request.
- **Text-line protocol on stdout with discipline-based isolation.**
  An attempt to keep the text protocol by muzzling library stdout
  via fd redirection. The outer pool subprocess's stdout could be
  muzzled, but FastVideo's `multiproc_executor` forks a Worker child
  during model load that inherits the outer's pre-muzzle fd 1.
  Whack-a-mole. The substrate (text on stdout) is wrong, not the
  patches; switched to `multiprocessing.Connection` on a dedicated
  fd no library has a handle to.
- **Estimating per-subprocess GPU memory from the model card.** The
  visible model size (~30 GiB weights) is not the per-subprocess
  resident cost. FastVideo's internal Worker child holds the model
  plus workspace plus compiled kernels at ~70 GiB, and the outer
  pool subprocess that owns it is a separate process holding
  ~600 MiB more. Measure empirically before choosing K.
- **Assuming FastVideo survives its own input-validation errors.**
  The plan-doc semantics for soft errors ("ERROR responses keep
  the subprocess alive") assumed `StageVerificationError` and
  similar generator-side validation failures would be confined to
  the request. They are not: FastVideo's `multiproc_executor`
  terminates its parent when its Worker child dies on validation
  failure. The pool's ERROR code path is correct and exercised;
  the subprocess dies anyway as a FastVideo-internal side effect.

## References

Operational procedures:
- [`RUNBOOK.md`](RUNBOOK.md) — adding shapes, baking images, rollback,
  CI drift, diagnosing failures, updating FastVideo.

Code (LTX-2-specific, this directory):
- [`worker.py`](worker.py) — LTX-2 worker entry; CLI parse, namespace
  resolution, endpoint registration, factory wiring.
- [`factory.py`](factory.py) — `load_model()`: the shared
  ``VideoGenerator.from_pretrained`` factory used by both the legacy
  in-process path and pool subprocesses.
- [`warmup.py`](warmup.py) — per-shape compile-cache producer.
- [`benchmark.py`](benchmark.py) — post-bake validation harness.
- [`config.py`](config.py) — canonical kwargs shared
  between warmup, benchmark, and worker.
- [`shapes.json`](shapes.json) — the shape menu.

Code (generic infrastructure, in sibling `../lib/`):
- [`../lib/pool.py`](../lib/pool.py) — `SubprocessPool`, the
  Connection-based wire protocol, `_pool_worker_main`, the dispatch
  entry point invoked by the top-level shim with `--pool-worker`,
  `_set_parent_death_signal`.
- [`../lib/backend.py`](../lib/backend.py) — `GenericVideoBackend`:
  the Dynamo endpoint, the global serialization lock, the legacy
  in-process path, and the pool routing path.
- [`../lib/metrics.py`](../lib/metrics.py) — `video_pool_*`
  Prometheus series with the `model` label.
- [`../lib/models.py`](../lib/models.py) — Pydantic request/response
  models.
- [`../lib/menu.py`](../lib/menu.py) — `compute_menu_hash` +
  `assert_shape_menu_hash_matches`; the algorithm that the IMAGE
  bake-time hash, this worker's boot-assertion, and
  ``backend/tests/test_ltx_shape_menu.py`` must all agree on.
- [`../lib/dynamo_wiring.py`](../lib/dynamo_wiring.py) —
  ``get_worker_namespace`` and ``register_model`` helpers.
- [`../worker.py`](../worker.py) — top-level shim; dispatches
  `--pool-worker` subprocesses into `lib.pool` BEFORE importing
  `ltx2.worker`, so pool cold-start skips the parent-side import tree.

Historical narrative (in `deepinfra/backend`):
- `claude_plans/2026-05-13-ltx2-cache-order-dependence.md` — the
  original order-dependence discovery and the first (insufficient)
  `_dynamo.reset()` fix.
- `claude_plans/2026-05-14-ltx2-per-shape-cache.md` — Phase 1: the
  subprocess-isolation + per-shape cache directories fix.
- `claude_plans/2026-05-14-ltx2-phase2-subprocess-pool.md` — Phase 2:
  the persistent pool, the IPC redesign (text protocol → Connection),
  and the K sizing correction. The amnesia anchor and "what we tried
  that was wrong" sections in this document are the authoritative
  long-form versions of the same lessons summarized above.
- `claude_plans/2026-05-07-ltx2-public-frontend.md` — `/v1/videos`
  public frontend gap analysis.
