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

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/broad_freeze_vt_bs32_epoch2_rl_ckpt425_rescale025_fast_20260524}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"
TELEGRAM_NOTIFY_ON_FINISH="${TELEGRAM_NOTIFY_ON_FINISH:-1}"

FREEZE_VT_BS32_EPOCH2_MODEL="/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_lr5e-6_cosine_minlr5e-7_len32768_bs32_full_clean_data_freeze_vt/sft_checkpoints/global_step_1052/huggingface"
RL_CKPT425_MODEL="/home/ywxzml3j/ywxzml3juser40/mms1_rl/ckpts/insight_doc/insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams/global_step_425__actor_merged_hf"

VAL_FILES='["/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/o3bench0502-insight_qwen_agent.parquet"]'
AGENT_CFG="recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml"

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
MASTER_LOG="$OUTPUT_ROOT/queue.log"
if [[ ! -s "$STATUS_TSV" ]]; then
  printf "time\tmodel\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "$STATUS_TSV"
fi

for required_path in "$FREEZE_VT_BS32_EPOCH2_MODEL" "$RL_CKPT425_MODEL" "$AGENT_CFG"; do
  if [[ ! -e "$required_path" ]]; then
    echo "missing required path: $required_path" >&2
    exit 2
  fi
done

run_one() {
  local model_label="$1"
  local model_path="$2"
  local load_format="$3"
  local gpus="$4"
  local ray_tag="$5"
  local run_name="${model_label}_broad_eval_0502_256k_zoom2_area3500_rescale025_fast"
  local work_dir="$OUTPUT_ROOT/$run_name"
  local ray_tmp="/tmp/${ray_tag}_br25"
  local status="success"
  local exit_code=0
  local launch_log="$work_dir/${run_name}.launch.log"

  mkdir -p "$work_dir" "$ray_tmp"
  {
    echo "[broad:${run_name}] start=$(date -Is) gpus=${gpus} model=${model_path} load_format=${load_format}"
    unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
    unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

    set +e
    EVAL_CUDA_VISIBLE_DEVICES="$gpus" \
    MODEL_PATH="$model_path" \
    LOAD_FORMAT="$load_format" \
    WORK_DIR="$work_dir" \
    EXP_NAME="$run_name" \
    WANDB_NAME="$run_name" \
    VAL_FILES="$VAL_FILES" \
    AGENT_LOOP_CONFIG_PATH="$AGENT_CFG" \
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

    if [[ -f "$launch_log" ]]; then
      if grep -q "'val_only_hf_model_rollout': True\\|val_only_hf_model_rollout: True" "$launch_log"; then
        echo "[broad:${run_name}] verified fast path in launch log"
      else
        echo "[broad:${run_name}] WARNING: fast path flag not found in launch log"
      fi
    else
      echo "[broad:${run_name}] WARNING: launch log not found at $launch_log"
    fi

    if (( exit_code != 0 )); then
      status="failed"
    fi
    printf "%s\t%s\t025\t0.25\t%s\t%s\t%s\n" "$(date -Is)" "$model_label" "$status" "$exit_code" "$work_dir" >> "$STATUS_TSV"
    echo "[broad:${run_name}] finished=$(date -Is) status=${status} exit_code=${exit_code}"
    return "$exit_code"
  } 2>&1 | tee -a "$MASTER_LOG"
}

echo "[broad] queue started=$(date -Is) output_root=${OUTPUT_ROOT}" | tee -a "$MASTER_LOG"
status=0

pid_freeze=""
pid_rl=""

if [[ "${RUN_FREEZE:-1}" == "1" ]]; then
  run_one "freeze_vt_bs32_epoch2" "$FREEZE_VT_BS32_EPOCH2_MODEL" "safetensors" "${FREEZE_GPUS:-0,1,2,3}" "fvt32" &
  pid_freeze=$!
fi

if [[ "${RUN_RL:-1}" == "1" ]]; then
  run_one "rl_ckpt425_actor_merged_hf" "$RL_CKPT425_MODEL" "auto" "${RL_GPUS:-4,5,6,7}" "rl425" &
  pid_rl=$!
fi

if [[ -n "$pid_freeze" ]]; then
  wait "$pid_freeze" || status=1
fi
if [[ -n "$pid_rl" ]]; then
  wait "$pid_rl" || status=1
fi

echo "[broad] queue finished=$(date -Is) status=${status}" | tee -a "$MASTER_LOG"
exit "$status"
