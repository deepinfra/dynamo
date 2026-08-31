set -u
T=/home/pernekhan/.claude/jobs/3aed5ed4/tmp
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/di-main/bin/python
TOKDIR=$T/ds4tok/tokenizers/di--deepseek-ai--DeepSeek-V4-Flash-0731--D9jmKKxS
KEY=$(cat $T/.tok)
OUT=$T/migtest; mkdir -p $OUT
NS=deepinfra
D='deepseek-ai/DeepSeek-V4-Flash-0731-roce-disagg'
V='Pernekhan/DeepSeek-V4-Flash-0731-test'

# Prompts must clear the migration gate: overlap_blocks*256 >= 16384 tokens.
# 12 shared prefixes x 24k tokens, ~2k unique tail => ~26k prompts, ~24k cacheable.
PREFIX_N=12
PREFIX_LEN=24000
IN_MEAN=2000
OSL=300

fe() { kubectl -n $NS get pods --no-headers | grep 'roce-disagg-fron' | grep -v Terminating | head -1 | awk '{print $1}'; }

run() {
  NAME=$1; MODEL=$2; URL=$3; C=$4; N=$5; shift 5
  DIR=$OUT/${NAME}_c${C}; rm -rf $DIR
  timeout 3600 conda run -n di-main --no-capture-output aiperf profile \
    -m "$MODEL" --url "$URL" --endpoint-type chat --streaming \
    --tokenizer "$TOKDIR" --tokenizer-trust-remote-code \
    --num-prefix-prompts $PREFIX_N --prefix-prompt-length $PREFIX_LEN \
    --synthetic-input-tokens-mean $IN_MEAN --synthetic-input-tokens-stddev 500 \
    --output-tokens-mean $OSL --output-tokens-stddev 0 \
    --concurrency "$C" --request-count "$N" --num-warmup-requests 6 \
    --random-seed 4242 --output-artifact-dir "$DIR" "$@" >/dev/null 2>&1
  $PY - "$DIR/profile_export_aiperf.json" "$NAME" "$C" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception:
    print(f"  {sys.argv[2]:<9} c={sys.argv[3]:<4} FAILED"); raise SystemExit
g=lambda k,f='avg': (d.get(k) or {}).get(f) or 0
print(f"  {sys.argv[2]:<9} c={sys.argv[3]:<4} req/s={g('request_throughput'):6.2f} "
      f"TTFT p50={g('time_to_first_token','p50'):7.0f} p90={g('time_to_first_token','p90'):7.0f} p99={g('time_to_first_token','p99'):8.0f}ms "
      f"ITL={g('inter_token_latency','p50'):5.1f}ms tok/s={g('output_token_throughput'):7.0f}")
PY
}

# Warm both fleets before the first measured point. Without this the first
# concurrency level absorbs all CUDA-graph/JIT warmup and reads ~5x slow --
# it produced three bogus "engine regression" verdicts before it was caught.
warmup() {
  timeout 900 conda run -n di-main --no-capture-output aiperf profile \
    -m "$2" --url "$3" --endpoint-type chat --streaming \
    --tokenizer "$TOKDIR" --tokenizer-trust-remote-code \
    --num-prefix-prompts $PREFIX_N --prefix-prompt-length $PREFIX_LEN \
    --synthetic-input-tokens-mean $IN_MEAN --output-tokens-mean $OSL \
    --concurrency 16 --request-count 64 --num-warmup-requests 4 \
    --random-seed 4242 --output-artifact-dir "$OUT/warm_$1" "${@:4}" >/dev/null 2>&1
  echo "    warmed $1"
}
warmup dynamo7 "$D" "http://localhost:18041"
warmup vllm7 "$V" "http://localhost:18042" --custom-endpoint /v1/openai/chat/completions --api-key "$KEY"
echo "WARMUP_DONE"

echo "=== 7 dynamo GPU vs 7 standalone vLLM GPU, no mirror, seed 4242 ==="
echo "    prompts: ${PREFIX_N} shared prefixes x ${PREFIX_LEN} tok + ~${IN_MEAN} tail (clears the 16384 gate)"
echo ""

for C in 24 48 128; do
  N=$(( C * 8 )); [ $N -lt 150 ] && N=150

  # capture the dynamo frontend log for the whole dynamo run so migration
  # counts are complete (container log holds only ~900 lines)
  FP=$(fe)
  LOG=$OUT/fe_c${C}.log
  ( kubectl -n $NS logs -f "$FP" --since=5s > "$LOG" 2>/dev/null ) &
  TAILPID=$!
  sleep 3
  run dynamo7 "$D" "http://localhost:18041" $C $N
  sleep 5
  kill $TAILPID 2>/dev/null; wait $TAILPID 2>/dev/null

  SP=$(grep -c 'KVMIGRATE_SPILL' "$LOG" 2>/dev/null || echo 0)
  HS=$(grep -c 'handshake obtained' "$LOG" 2>/dev/null || echo 0)
  MG=$(grep -c 'KVMIGRATE: decoding on target' "$LOG" 2>/dev/null || echo 0)
  PF=$(grep -c 'producer dispatch failed' "$LOG" 2>/dev/null || echo 0)
  echo "    migration@c${C}: spills=$SP handshakes=$HS migrated=$MG producer_failed=$PF (of $N requests)"
  grep -oE 'migratable_tokens=[0-9]+' "$LOG" 2>/dev/null | head -3 | sed 's/^/      sample /'

  run vllm7 "$V" "http://localhost:18042" $C $N \
      --custom-endpoint /v1/openai/chat/completions --api-key "$KEY"
  echo ""
done
echo MIGTEST_COMPLETE
