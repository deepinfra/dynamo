# Standalone indexer — notes

## Per-worker gap recovery (`GET /kv_recover`)

Each registered `(worker_id, dp_rank)` runs a ZMQ SUB listener that tracks a
watermark and detects gaps on the event `seq` (== `event_id`). On a gap the
listener issues `GET <recover_endpoint>/kv_recover?start=<expected_seq>&end=<got>`
against the worker (an HTTP base URL supplied at registration via
`recover_endpoint`, or by pod discovery via `--watch-recover-port` →
`http://<pod-ip>:<port>`). The response is a `WorkerKvQueryResponse`
(externally-tagged JSON):

- **`Events`** — apply each event, then set the watermark to `last_event_id`.
- **`TreeDump`** — `remove_worker_dp_rank(worker_id, dp_rank)` first (drop stale
  state for *this* logical unit, leaving sibling dp_ranks intact), apply all
  events, then set the watermark to `last_event_id`. The dump's event ids are
  synthetic 0-based; `last_event_id` is the real resume point.
- **`TooNew`** — no-op (consumer is ahead).
- **`InvalidRange`** / **`Error`** — logged, no-op.

Recovery is at-least-once: the watermark is driven by `last_event_id`, never by
counting events, so re-applied live batches stay idempotent. The worker's
self-reported `worker_id`/`dp_rank` in returned events are informational — they
are rewritten to the consumer-assigned `(worker_id, dp_rank)` before applying.
If no `recover_endpoint` is configured, gaps are logged and the dropped batches
are lost. Implementation lives in `listener.rs` (`recover_gap`,
`apply_recover_response`, `apply_recovered_events`).

The SUB sockets run with an unbounded receive queue (`ZMQ_RCVHWM = 0`,
`zmq.rs`): a listener blocked in recovery must never HWM-stop its pipe, because
libzmq 4.3.4 aborts on `_input_stopped` when a heartbeating peer restarts such
a pipe (zeromq/libzmq#3596). Seen in prod as a crash loop of the h24 indexer
while 66 startup TreeDumps were being applied under the single h24 mutex.

A download runs under a process-wide gate (`--recover-concurrency`, default 8)
with a total timeout of `--recover-timeout-secs` (default 120), and a failed
download is retried up to 3 times with 2/4 s backoff before the gap is given up
(`"kv_recover request failed; giving up, batches lost"`). A large engine's
TreeDump is tens of MB serialized inside the engine process; with the previous
10 s timeout and no gate, a fleet-wide (re)subscription lost about a third of
its recoveries on 39 DP=2 pods.

## Data-parallel engines (`--watch-dp-size`)

A vLLM engine with `--data-parallel-size N` runs N ranks per pod, each with its
own KV cache and its own event stream: rank `r` publishes on `zmq_port + r` and
serves `/kv_recover` on `kv_recover_port + r`. Pod discovery registers one
listener per rank under the pod's instance (`--watch-dp-size N`, default 1),
with `dp_rank = r` and the per-rank endpoints, so `/workers` shows N
`listeners` per pod. With the default on a DP engine only rank 0's cache is
indexed and every prefill scheduled on another rank is invisible to `/query`
while the engine still hits it (`pod_watcher.rs`, `rank_endpoints`).

## Audit logging (`--enable-logging`)

Pass `--enable-logging` to `python -m dynamo.indexer` to turn on verbose audit
logging. Off by default; zero cost when off. All lines use the `kv_audit`
tracing target — isolate them with `RUST_LOG=kv_audit=info`.

When enabled, two things get logged:

**On every query** (`/query` and `/query_by_hash`), one line *before* the
response goes back to the client:
- `kind="QUERY"`, `ts_ms`, `model_name`, `tenant_id`, HTTP `status`
- `block_hashes` — the block hashes of the user's tokens
- `response` — the entire JSON response (includes `longest_matched`, per-tier
  `instances` breakdown, and legacy `scores`)

**On every KV event ingested from the engine** (both the live SUB stream over
ZMQ, `source="live"`, and events pulled in by HTTP gap recovery via
`GET /kv_recover`, `source="recover"`). Logged *before* the
`--keep-evictions` filter, so it reflects what the engine actually published
(real `worker_id`, evictions not parked):
- `kind` — `STORE` (Stored) / `EVICT` (Removed) / `CLEAR` (Cleared). There is
  no "modify" event; the protocol only has these three.
- `ts_ms`, `worker_id`, `dp_rank`, `storage_tier`, `event_id`, `seq`
- STORE: `token_block_hashes` (local `tokens_hash`, same space as `/query`) and
  `sequence_block_hashes` (chained `block_hash`)
- EVICT: `sequence_block_hashes`

Implementation: flag in `mod.rs` (`ENABLE_LOGGING` / `logging_enabled()`),
query log in `server.rs` (`run_tiered_query`), event log in `listener.rs`
(`audit_log_event`).
