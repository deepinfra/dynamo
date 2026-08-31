set -u
LABEL="$1"; WINDOWS="${2:-7}"; GAP="${3:-300}"
cd /data/home/pernekhan/backend
export PATH=$HOME/miniconda3/bin:$PATH
PY=$HOME/miniconda3/envs/di-main/bin/python
D='deepseek-ai/DeepSeek-V4-Flash-0731-roce-disagg'
P='deepseek-ai/DeepSeek-V4-Flash-0731'

q() { $PY -m scripts.cli vm-query --instant --label='' --query "$1" 2>/dev/null | tail -1 | awk -F, '{print $NF+0}'; }
tq() { q "histogram_quantile($1, sum(rate(vllm:time_to_first_token_seconds_bucket{model_name=\"$2\"}[5m])) by (le))"; }
hit() { q "sum(rate(vllm:prefix_cache_hits_total{model_name=\"$1\"$2}[5m]))/sum(rate(vllm:prefix_cache_queries_total{model_name=\"$1\"$2}[5m]))"; }
gen() { q "sum(rate(vllm:generation_tokens_total{model_name=\"$1\"}[5m]))"; }
req() { q "sum(rate(vllm:request_success_total{model_name=\"$1\"}[5m]))"; }

echo "CONFIG=$LABEL windows=$WINDOWS gap=${GAP}s"
for i in $(seq 1 "$WINDOWS"); do
  echo "W$i t=$(date -u +%H:%M)" \
    "hitD=$(hit "$D" ',dynamo_component="backend"')" \
    "hitP=$(hit "$P" '')" \
    "t50D=$(tq 0.50 "$D") t90D=$(tq 0.90 "$D") t99D=$(tq 0.99 "$D")" \
    "t50P=$(tq 0.50 "$P") t90P=$(tq 0.90 "$P") t99P=$(tq 0.99 "$P")" \
    "genD=$(gen "$D") genP=$(gen "$P")" \
    "reqD=$(req "$D") reqP=$(req "$P")" \
    "eng=$(q "count(vllm:num_requests_running{model_name=\"$P\"})")"
  [ "$i" -lt "$WINDOWS" ] && sleep "$GAP"
done
echo "DONE_$LABEL"
