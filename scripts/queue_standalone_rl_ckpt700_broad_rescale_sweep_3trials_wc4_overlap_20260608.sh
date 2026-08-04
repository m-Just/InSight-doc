#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
cd "$REPO_ROOT"

# Keep local Ray/vLLM traffic off external proxies. Judge subprocesses map
# API_HTTP_PROXY/API_HTTPS_PROXY to standard proxy env vars explicitly.
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
unset RAY_ADDRESS RAY_NAMESPACE

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for judged evals}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set for judged evals}"
: "${API_HTTP_PROXY:?API_HTTP_PROXY must be set for judge API proxy}"
: "${API_HTTPS_PROXY:?API_HTTPS_PROXY must be set for judge API proxy}"

PYTHON_BIN="${PYTHON_BIN:-/home/ywxzml3j/ywxzml3juser40/.conda/envs/vllm-latest/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"
export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export ENSURE_API_LOGGER="${ENSURE_API_LOGGER:-1}"
export API_LOGGER_PROJECT_NAME="${API_LOGGER_PROJECT_NAME:-standalone_eval_judge}"

MODEL_PATH="${MODEL_PATH:-/home/ywxzml3j/ywxzml3juser40/mms1_rl/ckpts/insight_doc/insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams/global_step_700__actor_merged_hf}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/workspace/rl_ckpt700_broad_standalone_rescale025_035_05_3trials_wc4_overlap_${RUN_ID}}"
VAL_FILES="${VAL_FILES:-[\"/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet\",\"/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet\",\"/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet\",\"/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet\",\"/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet\"]}"
AGENT_CONFIG="${AGENT_CONFIG:-standalone_eval/agent_configs/insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml}"

AGENT_WORKER_PROCESSES="${AGENT_WORKER_PROCESSES:-8}"
WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-4}"
JUDGE_WORKERS="${JUDGE_WORKERS:-32}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"

mkdir -p "$OUTPUT_ROOT/logs"

write_model_config() {
  local model_config="$1"
  cat > "$model_config" <<EOF
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
}

rescale_slug() {
  local value="$1"
  printf '%s\n' "${value/./}"
}

run_judge_follow() {
  local output_dir="$1"
  HTTP_PROXY="$API_HTTP_PROXY" \
  HTTPS_PROXY="$API_HTTPS_PROXY" \
  http_proxy="$API_HTTP_PROXY" \
  https_proxy="$API_HTTPS_PROXY" \
  "$PYTHON_BIN" -u standalone_eval/judge.py \
    --rollout-dir "$output_dir" \
    --judge-model "$JUDGE_MODEL" \
    --judge-workers "$JUDGE_WORKERS"
}

run_eval() {
  local gpus="$1"
  local server_manifest="$2"
  local model_config="$3"
  local rescale="$4"
  local trial="$5"
  local slug
  slug="$(rescale_slug "$rescale")"
  local output_dir="$OUTPUT_ROOT/rescale${slug}_trial${trial}"
  mkdir -p "$output_dir/logs"
  echo "[$(date -u +'%F %T')] start rescale=$rescale trial=$trial output=$output_dir"
  {
    echo "started=$(date -u +'%F %T')"
    echo "judge_model=$JUDGE_MODEL judge_workers=$JUDGE_WORKERS"
    echo "openai_base_url=$OPENAI_BASE_URL"
    echo "api_proxy_configured=1"
  } > "$output_dir/logs/judge.env"

  run_judge_follow "$output_dir" > "$output_dir/logs/judge.log" 2>&1 &
  local judge_pid=$!

  set +e
  CUDA_VISIBLE_DEVICES="$gpus" "$PYTHON_BIN" -u standalone_eval/rollout.py \
    --model-config "$model_config" \
    --val-files "$VAL_FILES" \
    --output-dir "$output_dir" \
    --agent-config "$AGENT_CONFIG" \
    --agent-config-override "images.initial_rescale=$rescale" \
    --agent-config-override "limits.max_tool_response_length=256" \
    --agent-worker-processes "$AGENT_WORKER_PROCESSES" \
    --worker-concurrency "$WORKER_CONCURRENCY" \
    --ray-server-manifest "$server_manifest" \
    > "$output_dir/logs/rollout.log" 2>&1
  local rollout_status=$?
  if [[ "$rollout_status" -ne 0 ]]; then
    echo "[$(date -u +'%F %T')] rollout failed status=$rollout_status; stopping judge pid=$judge_pid" \
      >> "$output_dir/logs/judge.log"
    kill "$judge_pid" >/dev/null 2>&1 || true
    wait "$judge_pid" >/dev/null 2>&1 || true
    set -e
    return "$rollout_status"
  fi

  wait "$judge_pid"
  local judge_status=$?
  set -e
  if [[ "$judge_status" -ne 0 ]]; then
    echo "[$(date -u +'%F %T')] judge failed status=$judge_status" >&2
    return "$judge_status"
  fi
  echo "[$(date -u +'%F %T')] done rescale=$rescale trial=$trial"
}

run_lane() {
  local lane_name="$1"
  local gpus="$2"
  shift 2
  local lane_dir="$OUTPUT_ROOT/$lane_name"
  local model_config="$lane_dir/model_config.yaml"
  local server_manifest="$lane_dir/ray_vllm_server_manifest.json"
  local heartbeat_path="$lane_dir/ray_vllm_server.heartbeat"
  local ray_tmpdir="/tmp/rv_${lane_name}_$$"
  mkdir -p "$lane_dir/logs"
  write_model_config "$model_config"

  cleanup_lane() {
    if [[ -f "$server_manifest" ]]; then
      "$PYTHON_BIN" scripts/stop_ray_vllm.py --server-manifest "$server_manifest" \
        >> "$lane_dir/logs/stop_ray_vllm.log" 2>&1 || true
    fi
  }
  trap cleanup_lane EXIT

  echo "[$(date -u +'%F %T')] lane=$lane_name gpus=$gpus starting Ray/vLLM server"
  CUDA_VISIBLE_DEVICES="$gpus" "$PYTHON_BIN" -u scripts/serve_ray_vllm.py \
    --model-config "$model_config" \
    --server-manifest "$server_manifest" \
    --heartbeat-path "$heartbeat_path" \
    --ray-temp-dir "$ray_tmpdir" \
    --ray-namespace "standalone_rl700_wc4_${RUN_ID}_${lane_name}" \
    > "$lane_dir/logs/serve_ray_vllm.log" 2>&1 &
  local serve_pid=$!

  for _ in $(seq 1 180); do
    if [[ -f "$server_manifest" ]]; then
      break
    fi
    if ! kill -0 "$serve_pid" 2>/dev/null; then
      echo "Ray/vLLM server exited before manifest was written for $lane_name" >&2
      wait "$serve_pid" || true
      exit 1
    fi
    sleep 2
  done
  if [[ ! -f "$server_manifest" ]]; then
    echo "Timed out waiting for Ray/vLLM server manifest: $server_manifest" >&2
    exit 1
  fi

  local job
  for job in "$@"; do
    IFS=: read -r rescale trial <<< "$job"
    run_eval "$gpus" "$server_manifest" "$model_config" "$rescale" "$trial"
  done
  cleanup_lane
  trap - EXIT
}

echo "output_root=$OUTPUT_ROOT"
echo "agent_worker_processes=$AGENT_WORKER_PROCESSES worker_concurrency=$WORKER_CONCURRENCY judge_workers=$JUDGE_WORKERS"
echo "judge_overlap=1 judge_logs=<output_dir>/logs/judge.log api_logger_default_dir=~/.dumps/api_requests"

run_lane lane_a "${LANE_A_GPUS:-0,1,2,3}" \
  0.25:0 0.5:0 0.35:1 0.25:2 0.5:2 \
  > "$OUTPUT_ROOT/logs/lane_a.log" 2>&1 &
lane_a_pid=$!

run_lane lane_b "${LANE_B_GPUS:-4,5,6,7}" \
  0.35:0 0.25:1 0.5:1 0.35:2 \
  > "$OUTPUT_ROOT/logs/lane_b.log" 2>&1 &
lane_b_pid=$!

echo "$lane_a_pid" > "$OUTPUT_ROOT/lane_a.pid"
echo "$lane_b_pid" > "$OUTPUT_ROOT/lane_b.pid"
echo "lane_a_pid=$lane_a_pid"
echo "lane_b_pid=$lane_b_pid"

set +e
wait "$lane_a_pid"
status_a=$?
wait "$lane_b_pid"
status_b=$?
set -e

if [[ "$status_a" -ne 0 || "$status_b" -ne 0 ]]; then
  echo "one or more lanes failed: lane_a=$status_a lane_b=$status_b" >&2
  exit 1
fi

touch "$OUTPUT_ROOT/done"
echo "all standalone rl_ckpt700 broad sweep evals complete: $OUTPUT_ROOT"
