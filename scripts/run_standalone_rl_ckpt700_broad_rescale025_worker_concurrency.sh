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
AGENT_CONFIG="${AGENT_CONFIG:-recipe/vsearch/config/agent_insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml}"

GPUS="${GPUS:-0,1,2,3}"
AGENT_WORKER_PROCESSES="${AGENT_WORKER_PROCESSES:-8}"
WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-1}"
CONCURRENCY="${CONCURRENCY:-$((AGENT_WORKER_PROCESSES * WORKER_CONCURRENCY))}"
PROCESSOR_CONCURRENCY="${PROCESSOR_CONCURRENCY:-4}"
JUDGE_WORKERS="${JUDGE_WORKERS:-32}"

echo "output_dir=$OUTPUT_DIR"
echo "gpus=$GPUS concurrency=$CONCURRENCY agent_worker_processes=$AGENT_WORKER_PROCESSES worker_concurrency=$WORKER_CONCURRENCY"

set +e
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -u evaluate.py \
  --val-files "$VAL_FILES" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --agent-config "$AGENT_CONFIG" \
  --agent-config-name insight_qwen_agent_core \
  --prompt-length 262144 \
  --response-length 15360 \
  --max-model-len 262144 \
  --max-pixels 12845056 \
  --image-patch-size 16 \
  --max-user-turns 10 \
  --max-assistant-turns 11 \
  --max-parallel-calls 1 \
  --qwen-tool-list "[image_zoom_in_tool_qwen3vl]" \
  --tool-parser hermes \
  --reward-agent-name insight_qwen_agent \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --presence-penalty 1.5 \
  --repetition-penalty 1.0 \
  --fallback-judge-model gemini-3.1-flash-lite-preview \
  --cache-dir "$OUTPUT_DIR/cache/rlhf" \
  --judge-workers "$JUDGE_WORKERS" \
  --judge-task-timeout 60 \
  --judge-min-success-rate 0.99 \
  --judge-max-retries 10 \
  --judge-retry-interval 30 \
  --concurrency "$CONCURRENCY" \
  --agent-worker-processes "$AGENT_WORKER_PROCESSES" \
  --worker-concurrency "$WORKER_CONCURRENCY" \
  --processor-concurrency "$PROCESSOR_CONCURRENCY" \
  --validation-image-token-reorder \
  --validation-reorder-num-workers 8 \
  --validation-reorder-batch-size 32 \
  --validation-reorder-default-agent-loop insight_qwen_agent \
  --ray-num-replicas 4 \
  --ray-gpus-per-replica 1 \
  --ray-temp-dir "/tmp/ray_worker_concurrency_${WORKER_CONCURRENCY}_${RUN_ID}" \
  --ray-namespace "standalone_worker_concurrency_${WORKER_CONCURRENCY}_${RUN_ID}" \
  --ray-dtype bfloat16 \
  --ray-load-format auto \
  --ray-max-num-seqs 1024 \
  --ray-max-num-batched-tokens 32768 \
  --ray-gpu-memory-utilization 0.9 \
  --ray-enable-prefix-caching \
  --ray-enable-chunked-prefill \
  --ray-enable-sleep-mode \
  --ray-enforce-eager \
  --ray-scheduling-policy fcfs \
  --no-ray-trust-remote-code \
  --progress-every 25
status=$?
set -e
echo "evaluate.py exited with status=$status"
exit "$status"
