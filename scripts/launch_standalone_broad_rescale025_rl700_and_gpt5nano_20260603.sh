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
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
AGENT_CONFIG="${AGENT_CONFIG:-recipe/vsearch/config/agent_insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"

BROAD_VAL_FILES='["/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet"]'
BROAD_VAL_FILES_NO_TOOL_NO_SYSTEM='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/dude200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/longdocurl200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlite200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlongbench200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/mpdocvqa200-insight_qwen_agent_no_tool_no_system.parquet"]'

COMMON_ARGS=(
  --agent-config "$AGENT_CONFIG"
  --agent-config-name insight_qwen_agent_core
  --prompt-length 262144
  --response-length 15360
  --max-model-len 262144
  --max-pixels 12845056
  --image-patch-size 16
  --max-user-turns 10
  --max-assistant-turns 11
  --max-parallel-calls 1
  --concurrency 32
  --agent-worker-processes 4
  --worker-concurrency 8
  --processor-concurrency 8
  --judge-model gpt-5-nano
  --fallback-judge-model "$FALLBACK_JUDGE_MODEL"
  --judge-workers 32
  --judge-max-retries 10
  --judge-retry-interval 30
  --global-step 700
  --split val
  --run-name standalone_broad_rescale025_20260603
  --validation-image-token-reorder
  --validation-reorder-num-workers 8
  --validation-reorder-batch-size 32
  --validation-reorder-default-agent-loop insight_qwen_agent
  --progress-every 25
)

run_rl700_ray() {
  local out_dir="$OUTPUT_ROOT/rl_ckpt700_actor_merged_hf_broad_rescale025_ray_vllm"
  local log_file="$OUTPUT_ROOT/logs/rl_ckpt700_ray_vllm.log"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="${RL_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  "$PYTHON" evaluate.py \
    --generation-backend ray_vllm \
    --val-files "$BROAD_VAL_FILES" \
    --model-path "$RL_MODEL_PATH" \
    --output-dir "$out_dir" \
    --cache-dir "$out_dir/cache/rlhf" \
    --ray-num-replicas 4 \
    --ray-gpus-per-replica 1 \
    --ray-max-num-seqs 1024 \
    --ray-max-num-batched-tokens 32768 \
    --ray-gpu-memory-utilization 0.9 \
    --ray-load-format auto \
    --ray-temp-dir "${RAY_TMPDIR:-/tmp/standalone_rl700_ray_20260603}" \
    --ray-namespace "${RAY_NAMESPACE_RL:-standalone_rl700_ray_20260603}" \
    --ray-enable-prefix-caching \
    --ray-enable-chunked-prefill \
    --ray-enable-sleep-mode \
    --ray-enforce-eager \
    --ray-scheduling-policy fcfs \
    --no-ray-trust-remote-code \
    --trial-name rl_ckpt700_ray_vllm_rescale025 \
    "${COMMON_ARGS[@]}" \
    >"$log_file" 2>&1
}

run_gpt5nano_https() {
  local out_dir="$OUTPUT_ROOT/gpt5nano_no_tool_no_system_broad_rescale025_https"
  local log_file="$OUTPUT_ROOT/logs/gpt5nano_https.log"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="" \
  "$PYTHON" evaluate.py \
    --generation-backend https_openai_chat \
    --https-base-url "$OPENAI_BASE_URL" \
    --https-model "${HTTPS_MODEL:-gpt-5-nano}" \
    --https-api-key-env OPENAI_API_KEY \
    --https-timeout 600 \
    --https-max-retries 2 \
    --https-image-format PNG \
    --val-files "$BROAD_VAL_FILES_NO_TOOL_NO_SYSTEM" \
    --model-path "$TOKENIZER_MODEL_PATH" \
    --output-dir "$out_dir" \
    --cache-dir "$out_dir/cache/rlhf" \
    --no-tool-schema \
    --trial-name gpt5nano_no_tool_no_system_https_rescale025 \
    "${COMMON_ARGS[@]}" \
    >"$log_file" 2>&1
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
