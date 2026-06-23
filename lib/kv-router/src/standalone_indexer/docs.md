# Standalone indexer — notes

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

**On every KV event ingested from the engine** over ZMQ (both the live SUB
stream, `source="live"`, and replayed batches, `source="replay"`). Logged
*before* the `--ignore-evictions` filter, so it reflects
what the engine actually published (real `worker_id`, evictions not dropped):
- `kind` — `STORE` (Stored) / `EVICT` (Removed) / `CLEAR` (Cleared). There is
  no "modify" event; the protocol only has these three.
- `ts_ms`, `worker_id`, `dp_rank`, `storage_tier`, `event_id`, `seq`
- STORE: `token_block_hashes` (local `tokens_hash`, same space as `/query`) and
  `sequence_block_hashes` (chained `block_hash`)
- EVICT: `sequence_block_hashes`

Implementation: flag in `mod.rs` (`ENABLE_LOGGING` / `logging_enabled()`),
query log in `server.rs` (`run_tiered_query`), event log in `listener.rs`
(`audit_log_event`).
