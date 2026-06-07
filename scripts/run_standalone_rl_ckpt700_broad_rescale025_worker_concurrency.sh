#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
cd "$REPO_ROOT"

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
unset RAY_ADDRESS RAY_NAMESPACE

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for judged evals}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set for judged evals}"

PYTHON_BIN="${PYTHON_BIN:-/home/ywxzml3j/ywxzml3juser40/.conda/envs/vllm-latest/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"
export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/home/ywxzml3j/ywxzml3juser40/mms1_rl/ckpts/insight_doc/insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams/global_step_700__actor_merged_hf}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/workspace/rl_ckpt700_broad_rescale025_global_queue_worker_concurrency_${WORKER_CONCURRENCY:-1}_${RUN_ID}}"
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/cache"

VAL_FILES="${VAL_FILES:-[\"/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet\",\"/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet\",\"/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet\",\"/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet\",\"/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet\"]}"
AGENT_CONFIG="${AGENT_CONFIG:-standalone_eval/agent_configs/insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-$OUTPUT_DIR/model_config.yaml}"
SERVER_MANIFEST="${SERVER_MANIFEST:-$OUTPUT_DIR/ray_vllm_server_manifest.json}"
HEARTBEAT_PATH="${HEARTBEAT_PATH:-$OUTPUT_DIR/ray_vllm_server.heartbeat}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rvllm_${RUN_ID:0:32}}"

GPUS="${GPUS:-0,1,2,3}"
AGENT_WORKER_PROCESSES="${AGENT_WORKER_PROCESSES:-8}"
WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-1}"
CONCURRENCY="${CONCURRENCY:-$((AGENT_WORKER_PROCESSES * WORKER_CONCURRENCY))}"
PROCESSOR_CONCURRENCY="${PROCESSOR_CONCURRENCY:-4}"
JUDGE_WORKERS="${JUDGE_WORKERS:-32}"

echo "output_dir=$OUTPUT_DIR"
echo "gpus=$GPUS concurrency=$CONCURRENCY agent_worker_processes=$AGENT_WORKER_PROCESSES worker_concurrency=$WORKER_CONCURRENCY"

cat > "$MODEL_CONFIG" <<EOF
model: $MODEL_PATH
backend: ray_vllm

generation:
  max_tokens_after_initial_prompt: 16384

ray_vllm:
  num_replicas: 4
  gpus_per_replica: 1
  max_model_len: 262144
  max_num_seqs: 1024
  max_num_batched_tokens: 32768
  gpu_memory_utilization: 0.9
  enable_prefix_caching: true
  enable_chunked_prefill: true
  enforce_eager: true
  sampling:
    temperature: 0.7
    top_p: 0.8
    top_k: 20
    presence_penalty: 1.5
    repetition_penalty: 1.0
EOF

cleanup() {
  if [[ -f "$SERVER_MANIFEST" ]]; then
    "$PYTHON_BIN" scripts/stop_ray_vllm.py --server-manifest "$SERVER_MANIFEST" >> "$OUTPUT_DIR/logs/stop_ray_vllm.log" 2>&1 || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -u scripts/serve_ray_vllm.py \
  --model-config "$MODEL_CONFIG" \
  --server-manifest "$SERVER_MANIFEST" \
  --heartbeat-path "$HEARTBEAT_PATH" \
  --ray-temp-dir "$RAY_TMPDIR" \
  --ray-namespace "standalone_worker_concurrency_${WORKER_CONCURRENCY}_${RUN_ID}" \
  > "$OUTPUT_DIR/logs/serve_ray_vllm.log" 2>&1 &
serve_pid=$!

for _ in $(seq 1 180); do
  if [[ -f "$SERVER_MANIFEST" ]]; then
    break
  fi
  if ! kill -0 "$serve_pid" 2>/dev/null; then
    echo "Ray/vLLM server exited before manifest was written" >&2
    wait "$serve_pid" || true
    exit 1
  fi
  sleep 2
done
if [[ ! -f "$SERVER_MANIFEST" ]]; then
  echo "Timed out waiting for Ray/vLLM server manifest: $SERVER_MANIFEST" >&2
  exit 1
fi

set +e
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -u standalone_eval/rollout.py \
  --model-config "$MODEL_CONFIG" \
  --val-files "$VAL_FILES" \
  --output-dir "$OUTPUT_DIR" \
  --agent-config "$AGENT_CONFIG" \
  --agent-worker-processes "$AGENT_WORKER_PROCESSES" \
  --worker-concurrency "$WORKER_CONCURRENCY" \
  --ray-server-manifest "$SERVER_MANIFEST"
status=$?
set -e
echo "rollout.py exited with status=$status"
if [[ "$status" -eq 0 ]]; then
  "$PYTHON_BIN" -u standalone_eval/judge.py \
    --rollout-dir "$OUTPUT_DIR" \
    --judge-model gpt-5-nano \
    --judge-workers "$JUDGE_WORKERS"
  status=$?
  echo "judge.py exited with status=$status"
fi
exit "$status"
