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
`--ignore-evictions` filter, so it reflects what the engine actually published
(real `worker_id`, evictions not dropped):
- `kind` — `STORE` (Stored) / `EVICT` (Removed) / `CLEAR` (Cleared). There is
  no "modify" event; the protocol only has these three.
- `ts_ms`, `worker_id`, `dp_rank`, `storage_tier`, `event_id`, `seq`
- STORE: `token_block_hashes` (local `tokens_hash`, same space as `/query`) and
  `sequence_block_hashes` (chained `block_hash`)
- EVICT: `sequence_block_hashes`

Implementation: flag in `mod.rs` (`ENABLE_LOGGING` / `logging_enabled()`),
query log in `server.rs` (`run_tiered_query`), event log in `listener.rs`
(`audit_log_event`).
