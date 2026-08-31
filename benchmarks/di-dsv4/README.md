# DSv4-Flash benchmark harness

Scripts used to evaluate dynamo disagg / KV migration / engine configs against
the production fleet.

| script | purpose |
| --- | --- |
| `migtest.sh` | aiperf sweep vs a second fleet; warms both endpoints first |
| `mirror_h2h.sh` | mirror the same prod shards to two fleets, compare N windows |
| `compare.sh` | one fleet vs prod, N windows |
| `h2h.sh` | synthetic 7v7 aiperf |
| `kvxfer_force.py` | force a KV migration and check output determinism at temp=0 |
| `kvxfer_cross.py` | migration-vs-cold-prefill crossover sweep |
| `summarize.py` | median-per-window summary of compare/mirror logs |

## Hard-won rules

1. **Warm before the first measured point.** The first aiperf concurrency level
   after a fleet boot reads ~5.8x slow (1.53 vs 8.90 req/s on identical config).
   This produced three false "engine regression" verdicts.
2. **Never trust pod logs for rates.** The container log holds only ~900 lines,
   so `--since=1h` and `--since=1m` return nearly the same count. Use
   `dynamo_frontend_requests_total{status,error_type}` instead.
3. **Preflight both endpoints.** Port-forwards die when a pod rolls, and the
   harness silently records `req/s=0.00` rather than failing.
4. **Give each synthetic prompt a distinct first 256 tokens**, or they collide
   on one session id and the seed request 400s.
5. **Mirror the same shards to both candidates.** Synthetic traffic that is too
   uniform/cacheable flatters KV-aware routing; a shard-sliced mirror is a
   load-based sample, so both sides must get the identical slice.
