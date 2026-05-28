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

MODEL_PATH="${MODEL_PATH:-/scratch/ywxzml3j/likaican/temp/full_sft_both_higher_dpi_unans02503505_fp32_sp4_epoch2_scratch_20260519/full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_freeze_vt_epoch2_fp32_scratch/sft_checkpoints/global_step_1118/huggingface}"
MODEL_LABEL="${MODEL_LABEL:-full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_epoch2_fp32_scratch}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"
TELEGRAM_NOTIFY_ON_FINISH="${TELEGRAM_NOTIFY_ON_FINISH:-0}"

run_highpage() {
  local output_root="${EVAL_OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_full_sft_both_higher_dpi_unans02503505_fp32_sp4_epoch2_scratch_20260520_fallback_restart}"
  EVAL_OUTPUT_ROOT="$output_root" \
  EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES:-0,1,2,3}" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_LABEL="$MODEL_LABEL" \
  FALLBACK_JUDGE_MODEL="$FALLBACK_JUDGE_MODEL" \
  TELEGRAM_NOTIFY_ON_FINISH="$TELEGRAM_NOTIFY_ON_FINISH" \
  bash "$REPO_ROOT/scripts/retry_highpage_full_sft_epoch1_fp32_sp4_eval_20260519.sh"
}

run_broad() {
  local gpus="${EVAL_CUDA_DEVICES:-4,5,6,7}"
  local output_root="${EVAL_OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/broad_full_sft_both_higher_dpi_unans02503505_fp32_sp4_epoch2_20260520_fallback_restart}"
  local status_tsv="$output_root/status.tsv"
  local master_log="$output_root/broad_queue.log"
  local val_files='["/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/o3bench0502-insight_qwen_agent.parquet"]'

  mkdir -p "$output_root"
  exec > >(tee -a "$master_log") 2>&1
  printf "time\tmodel\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "$status_tsv"

  run_broad_one() {
    local scale_id="$1"
    local scale_value="$2"
    local agent_cfg="$3"
    local run_name="${MODEL_LABEL}_broad_eval_0502_256k_zoom2_area3500_rescale${scale_id}"
    local work_dir="$output_root/$run_name"
    local status="success"
    local exit_code=0

    echo "[broad:${run_name}] start=$(date -Is) gpus=${gpus} model=${MODEL_PATH} fallback=${FALLBACK_JUDGE_MODEL}"
    mkdir -p "$work_dir"
    unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
    unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

    set +e
    EVAL_CUDA_VISIBLE_DEVICES="$gpus" \
    MODEL_PATH="$MODEL_PATH" \
    LOAD_FORMAT="safetensors" \
    WORK_DIR="$work_dir" \
    EXP_NAME="$run_name" \
    WANDB_NAME="$run_name" \
    VAL_FILES="$val_files" \
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
    RAY_TMPDIR="/tmp/rbroad_epoch2_fallback_restart_${scale_id}" \
    FALLBACK_JUDGE_MODEL="$FALLBACK_JUDGE_MODEL" \
    TELEGRAM_NOTIFY_ON_FINISH="$TELEGRAM_NOTIFY_ON_FINISH" \
    bash scripts/run_iq_base_eval_default_sampling_insight_rl15360.sh \
      trainer.val_only_hf_model_rollout=true \
      trainer.resume_mode=disable
    exit_code=$?
    set -e

    if (( exit_code != 0 )); then
      status="failed"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "$MODEL_LABEL" "$scale_id" "$scale_value" "$status" "$exit_code" "$work_dir" >> "$status_tsv"
    echo "[broad:${run_name}] finished=$(date -Is) status=${status} exit_code=${exit_code}"
  }

  echo "[broad] queue started=$(date -Is) gpus=${gpus} model=${MODEL_PATH}"
  run_broad_one 025 0.25 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml
  run_broad_one 035 0.35 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml
  run_broad_one 05 0.5 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml
  echo "[broad] queue finished=$(date -Is)"
}

case "${1:-}" in
  highpage) run_highpage ;;
  broad) run_broad ;;
  *)
    echo "usage: $0 {highpage|broad}" >&2
    exit 2
    ;;
esac
