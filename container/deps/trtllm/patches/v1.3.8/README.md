# TRT-LLM v1.3.8 — DeepInfra runtime patches

Four files COPYed into the docker image on top of stock TRT-LLM v1.3.8 +
dynamo. Together they implement the **drain protocol** and the
**py_executor idle-timer fix**.

| file in this dir | COPY destination inside the image |
|---|---|
| `publisher.py` | `/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm/publisher.py` |
| `handler_base.py` | `/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm/request_handlers/handler_base.py` |
| `llm_worker.py` | `/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm/workers/llm_worker.py` |
| `py_executor.py` | `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/pyexecutor/py_executor.py` |

`Dockerfile.v6` is the layered Dockerfile we used to build the current
image (`localhost:30500/dynamo-di-trtllm:v1.3.8-drain-v6`). It builds
`FROM v1.3.8-drain-v5` which itself builds on prior layers — see
`[[dynamo-drain-protocol]]` for the v1-v6 history. A non-layered single
Dockerfile that applies all four files in one stage is the cleaner
long-term shape.

## What each patch does

### `publisher.py` — FPM publisher shutdown / restart on drain
Adds `pause_fpm()` / `resume_fpm()` methods that fully shut down the Rust
`FpmDirectPublisher` on drain and recreate it on wake. Without this, the
planner's `FpmEventSubscriber` keeps reading stale `latest_stats` from
drained pods.

Init args are stashed in `self._fpm_publisher_init_args` so wake can
recreate the publisher identically.

### `handler_base.py` — model unregister / reregister on drain
`release_memory_occupation` now also calls `unregister_model(endpoint)`;
`resume_memory_occupation` calls a stashed `_reregister_model_fn` closure
on wake. The planner's `FpmEventSubscriber` only prunes its DashMap when
Task 2 sees a `Removed` event on the `ComponentModels` discovery watch,
which requires the model card itself (not just the endpoint instance) to
leave the MDC. v5 alone was insufficient — the Rust publisher was off but
the planner cache was still hot.

### `llm_worker.py` — wire the reregister closure
Captures the original `register_model(...)` arguments in a closure
`_do_register_model` and binds it onto the handler via
`set_reregister_model_fn(...)` right after the initial registration.
Also registers `/sleep` and `/wake_up` route aliases so the trtllm worker
responds to the same orchestrator calls model-manager already uses for
vLLM workers.

### `py_executor.py` — move iter timer past blocking queue fetch (3 sites)
Three `_executor_loop_*` variants in TRT-LLM previously captured
`iter_start_time = time.time()` BEFORE
`_fetch_and_activate_new_requests()`, which calls a blocking
`get_from_request_queue(...)`. On an idle engine that call blocks until a
request arrives; the blocked time was being attributed to
`iter_latency_ms` on the next iter.

On disagg-prefill workloads this contaminated the OLS regression
`wall_time = a·sum_prefill_tokens + b` with sleep time on the y-axis,
dragging the intercept negative and ultimately producing the
`engine_rps = 1e6` pathology. See `[[trtllm-iter-timer-patch]]`.

Each site moves the timer to AFTER the fetch, inside an
`if self.enable_iter_perf_stats:` block. Patch sites in the current file:
lines 1940, 2749, 3031.

## What was deliberately NOT kept

These were investigation artifacts that landed in earlier image
iterations and were stripped here:

- **`trtllm/llm_engine.py` — DI_TIMING per-token logs**: probe used to
  diagnose the 250ms-ITL vs 5ms-TPOT gap. Root-caused via stream_interval=5.
- **`trtllm/publisher.py` — DI_ITER per-iter log + faster poll backoff**:
  probe for the post-stream-interval wedge investigation. No longer needed.
- **`trtllm/metrics.py` + wiring — disagg state gauges**
  (`trtllm_decode_stuck_on_kv`, `trtllm_prefill_holding_kv`): dashboard
  observability, not consumed by the planner.
- **`py_executor.py` — per-LlmRequestState gauges**
  (`trtllm_state_*` family): same — dashboard observability,
  not consumed by the planner.

If you want any of these back, see the prior file states under
`/tmp/trtllm-drain-v6/` or earlier image iterations.

## Engine-config dependencies (deployment-side, not in this dir)

The drain protocol assumes the worker exposes `:9090/system_status_server`
and the deployment manifest sets `--max-num-tokens=131072` etc. See the
deepinfra deploy tooling for the `override-engine-args` block; the
recommended values are summarized in `[[dynamo-drain-protocol]]`.
