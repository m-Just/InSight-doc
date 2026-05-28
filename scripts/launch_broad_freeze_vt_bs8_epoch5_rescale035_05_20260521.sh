#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
cd "$REPO_ROOT"

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" ]] && [[ -f /home/ywxzml3j/ywxzml3juser40/.bashrc ]]; then
  OPENAI_EXPORTS="$(grep -E '^export OPENAI_(API_KEY|BASE_URL)=' /home/ywxzml3j/ywxzml3juser40/.bashrc || true)"
  if [[ -n "$OPENAI_EXPORTS" ]]; then
    eval "$OPENAI_EXPORTS"
  fi
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set}"

MODEL_PATH="${MODEL_PATH:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_lr2e-6_cosine_minlr2e-7_len32768_bs8_full_clean_data_freeze_vt/sft_checkpoints/global_step_10530/huggingface}"
MODEL_LABEL="${MODEL_LABEL:-freeze_vt_bs8_epoch5_broad_rescale035_05_20260521}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/broad_freeze_vt_bs8_epoch5_rescale035_05_20260521}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"
TELEGRAM_NOTIFY_ON_FINISH="${TELEGRAM_NOTIFY_ON_FINISH:-1}"

VAL_FILES='["/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/o3bench0502-insight_qwen_agent.parquet"]'

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
MASTER_LOG="$OUTPUT_ROOT/queue.log"
printf "time\tmodel\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "$STATUS_TSV"

run_one() {
  local scale_id="$1"
  local scale_value="$2"
  local gpus="$3"
  local agent_cfg="$4"
  local run_name="${MODEL_LABEL}_broad_eval_0502_256k_zoom2_area3500_rescale${scale_id}"
  local work_dir="$OUTPUT_ROOT/$run_name"
  local ray_tmp="/tmp/rb${scale_id}fvt8"
  local status="success"
  local exit_code=0

  mkdir -p "$work_dir" "$ray_tmp"
  {
    echo "[broad:${run_name}] start=$(date -Is) gpus=${gpus} model=${MODEL_PATH} fallback=${FALLBACK_JUDGE_MODEL}"
    unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
    unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

    set +e
    EVAL_CUDA_VISIBLE_DEVICES="$gpus" \
    MODEL_PATH="$MODEL_PATH" \
    LOAD_FORMAT="safetensors" \
    WORK_DIR="$work_dir" \
    EXP_NAME="$run_name" \
    WANDB_NAME="$run_name" \
    VAL_FILES="$VAL_FILES" \
    AGENT_LOOP_CONFIG_PATH="$agent_cfg" \
    LOGGER="['console']" \
    MAX_RESPONSE_LENGTH="15360" \
    VAL_BATCH_SIZE="32" \
    TOOL_MAX_USER_TURNS="10" \
    TOOL_MAX_ASSISTANT_TURNS="11" \
    DATA_MAX_PROMPT_LENGTH="262144" \
    DATA_VALIDATION_MAX_PROMPT_LENGTH="262144" \
    ROLLOUT_MAX_MODEL_LEN="262144" \
    VLLM_MAX_MODEL_LEN="262144" \
    VLLM_GPU_MEMORY_UTILIZATION="0.9" \
    RAY_TMPDIR="$ray_tmp" \
    FALLBACK_JUDGE_MODEL="$FALLBACK_JUDGE_MODEL" \
    TELEGRAM_NOTIFY_ON_FINISH="$TELEGRAM_NOTIFY_ON_FINISH" \
    TELEGRAM_NOTIFY_LABEL="$run_name" \
    bash scripts/run_iq_base_eval_default_sampling_insight_rl15360.sh \
      trainer.val_only_hf_model_rollout=true \
      trainer.resume_mode=disable
    exit_code=$?
    set -e

    if (( exit_code != 0 )); then
      status="failed"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "$MODEL_LABEL" "$scale_id" "$scale_value" "$status" "$exit_code" "$work_dir" >> "$STATUS_TSV"
    echo "[broad:${run_name}] finished=$(date -Is) status=${status} exit_code=${exit_code}"
    return "$exit_code"
  } 2>&1 | tee -a "$MASTER_LOG"
}

echo "[broad] queue started=$(date -Is) model=${MODEL_PATH}" | tee -a "$MASTER_LOG"
status=0
if [[ "${SERIAL_QUEUE:-0}" == "1" ]]; then
  run_one 035 0.35 "${GPUS_035:-0,1,2,3}" recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml || status=1
  if (( status == 0 )); then
    run_one 05 0.5 "${GPUS_05:-${GPUS_035:-0,1,2,3}}" recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml || status=1
  fi
else
  run_one 035 0.35 "${GPUS_035:-0,1,2,3}" recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml &
  pid035=$!
  run_one 05 0.5 "${GPUS_05:-4,5,6,7}" recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml &
  pid05=$!

  wait "$pid035" || status=1
  wait "$pid05" || status=1
fi
echo "[broad] queue finished=$(date -Is) status=${status}" | tee -a "$MASTER_LOG"
exit "$status"
