set -u
T=/home/pernekhan/.claude/jobs/3aed5ed4/tmp
export PATH=$HOME/miniconda3/bin:$PATH
TOKDIR=$T/ds4tok/tokenizers/di--deepseek-ai--DeepSeek-V4-Flash-0731--D9jmKKxS
KEY=$(cat $T/.tok)
OUT=$T/h2h; mkdir -p $OUT

# prod-shaped: long shared prefixes (cacheable) + unique tail, ~300 output tokens
PREFIX_N=64
PREFIX_LEN=6000
IN_MEAN=3000
IN_STD=2000
OSL=300

run() {
  NAME=$1; MODEL=$2; URL=$3; C=$4; N=$5; shift 5
  D=$OUT/${NAME}_c${C}
  rm -rf $D
  timeout 2400 conda run -n di-main --no-capture-output aiperf profile \
    -m "$MODEL" --url "$URL" --endpoint-type chat --streaming \
    --tokenizer "$TOKDIR" --tokenizer-trust-remote-code \
    --num-prefix-prompts $PREFIX_N --prefix-prompt-length $PREFIX_LEN \
    --synthetic-input-tokens-mean $IN_MEAN --synthetic-input-tokens-stddev $IN_STD \
    --output-tokens-mean $OSL --output-tokens-stddev 0 \
    --concurrency "$C" --request-count "$N" --num-warmup-requests 8 \
    --random-seed 8800 --output-artifact-dir "$D" "$@" >/dev/null 2>&1
  python3 - "$D/profile_export_aiperf.json" "$NAME" "$C" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception:
    print(f"  {sys.argv[2]:<8} c={sys.argv[3]:<4} FAILED"); raise SystemExit
g=lambda k,f='avg': (d.get(k) or {}).get(f) or 0
print(f"  {sys.argv[2]:<8} c={sys.argv[3]:<4} req/s={g('request_throughput'):6.2f} "
      f"TTFT p50={g('time_to_first_token','p50'):7.0f} p90={g('time_to_first_token','p90'):7.0f} p99={g('time_to_first_token','p99'):8.0f}ms "
      f"ITL p50={g('inter_token_latency','p50'):5.1f}ms out_tok/s={g('output_token_throughput'):7.0f}")
PY
}

echo "=== 7 GPU vs 7 GPU, identical synthetic traffic (seed 8800) ==="
echo "    prefix pool=$PREFIX_N x ${PREFIX_LEN}tok, unique tail mean=$IN_MEAN, out=$OSL"
for C in 8 24 48 96; do
  N=$(( C * 12 )); [ $N -lt 200 ] && N=200
  run vllm7 "Pernekhan/DeepSeek-V4-Flash-0731-test" "http://localhost:18002" $C $N \
      --custom-endpoint /v1/openai/chat/completions --api-key "$KEY"
  run dynamo7 "deepseek-ai/DeepSeek-V4-Flash-0731-roce-disagg" "http://localhost:18001" $C $N
  echo ""
done
echo H2H_COMPLETE
