#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
cd "$REPO_ROOT"

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
unset RAY_ADDRESS RAY_NAMESPACE

PYTHON="${PYTHON:-/home/ywxzml3j/ywxzml3juser40/.conda/envs/vllm-latest/bin/python}"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"
export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/standalone_broad_rescale025_rl700_gpt5nano_20260603}"
mkdir -p "$OUTPUT_ROOT/logs"

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for reward judging and HTTPS generation}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://globalai.vip/v1}"

RL_MODEL_PATH="${RL_MODEL_PATH:-/home/ywxzml3j/ywxzml3juser40/mms1_rl/ckpts/insight_doc/insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams/global_step_700__actor_merged_hf}"
AGENT_CONFIG="${AGENT_CONFIG:-recipe/vsearch/config/agent_insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml}"

BROAD_VAL_FILES='["/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet"]'
BROAD_VAL_FILES_NO_TOOL_NO_SYSTEM='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/dude200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/longdocurl200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlite200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlongbench200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/mpdocvqa200-insight_qwen_agent_no_tool_no_system.parquet"]'

COMMON_ARGS=(
  --agent-config "$AGENT_CONFIG"
  --agent-worker-processes 4
  --worker-concurrency 8
)

run_rl700_ray() {
  local out_dir="$OUTPUT_ROOT/rl_ckpt700_actor_merged_hf_broad_rescale025_ray_vllm"
  local log_file="$OUTPUT_ROOT/logs/rl_ckpt700_ray_vllm.log"
  local model_config="$out_dir/model_config.yaml"
  local server_manifest="$out_dir/ray_vllm_server_manifest.json"
  local heartbeat_path="$out_dir/ray_vllm_server.heartbeat"
  mkdir -p "$out_dir"
  cat > "$model_config" <<EOF
model: $RL_MODEL_PATH
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
  CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3}" "$PYTHON" -u scripts/serve_ray_vllm.py \
    --model-config "$model_config" \
    --server-manifest "$server_manifest" \
    --heartbeat-path "$heartbeat_path" \
    --ray-temp-dir "${RAY_TMPDIR:-/tmp/standalone_rl700_ray_20260603}" \
    --ray-namespace "${RAY_NAMESPACE_RL:-standalone_rl700_ray_20260603}" \
    > "$OUTPUT_ROOT/logs/rl_ckpt700_ray_vllm_serve.log" 2>&1 &
  local serve_pid=$!
  for _ in $(seq 1 180); do
    [[ -f "$server_manifest" ]] && break
    if ! kill -0 "$serve_pid" 2>/dev/null; then
      wait "$serve_pid" || true
      return 1
    fi
    sleep 2
  done
  [[ -f "$server_manifest" ]] || return 1
  CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3}" "$PYTHON" -u standalone_eval/rollout.py \
    --model-config "$model_config" \
    --val-files "$BROAD_VAL_FILES" \
    --output-dir "$out_dir" \
    --ray-server-manifest "$server_manifest" \
    "${COMMON_ARGS[@]}" \
    >"$log_file" 2>&1
  local rollout_status=$?
  "$PYTHON" scripts/stop_ray_vllm.py --server-manifest "$server_manifest" >> "$OUTPUT_ROOT/logs/rl_ckpt700_ray_vllm_stop.log" 2>&1 || true
  [[ "$rollout_status" -eq 0 ]] || return "$rollout_status"
  "$PYTHON" -u standalone_eval/judge.py \
    --rollout-dir "$out_dir" \
    --judge-model gpt-5-nano \
    --judge-workers 32 \
    >>"$log_file" 2>&1
}

run_gpt5nano_https() {
  local out_dir="$OUTPUT_ROOT/gpt5nano_no_tool_no_system_broad_rescale025_https"
  local log_file="$OUTPUT_ROOT/logs/gpt5nano_https.log"
  local model_config="$out_dir/model_config.yaml"
  mkdir -p "$out_dir"
  cat > "$model_config" <<EOF
model: ${HTTPS_MODEL:-gpt-5-nano}
backend: https_openai_chat

generation:
  max_tokens_after_initial_prompt: 16384

https_openai_chat:
  base_url: $OPENAI_BASE_URL
  api_key_env: OPENAI_API_KEY
  timeout: 180
  max_retries: 1
  image_format: png
  image_detail: high
  reasoning_effort: high
EOF
  CUDA_VISIBLE_DEVICES="" "$PYTHON" -u standalone_eval/rollout.py \
    --model-config "$model_config" \
    --val-files "$BROAD_VAL_FILES_NO_TOOL_NO_SYSTEM" \
    --output-dir "$out_dir" \
    "${COMMON_ARGS[@]}" \
    >"$log_file" 2>&1
  "$PYTHON" -u standalone_eval/judge.py \
    --rollout-dir "$out_dir" \
    --judge-model gpt-5-nano \
    --judge-workers 32 \
    >>"$log_file" 2>&1
}

run_rl700_ray &
pid_rl=$!
run_gpt5nano_https &
pid_https=$!

echo "$pid_rl" > "$OUTPUT_ROOT/rl_ckpt700_ray_vllm.pid"
echo "$pid_https" > "$OUTPUT_ROOT/gpt5nano_https.pid"
echo "launched rl_ckpt700_ray_vllm pid=$pid_rl"
echo "launched gpt5nano_https pid=$pid_https"
echo "output_root=$OUTPUT_ROOT"

status=0
wait "$pid_rl" || status=1
wait "$pid_https" || status=1
exit "$status"
