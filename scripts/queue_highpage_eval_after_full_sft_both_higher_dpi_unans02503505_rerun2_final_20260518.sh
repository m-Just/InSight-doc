#!/usr/bin/env bash
set -euo pipefail

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
SFT_ROOT="${SFT_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_both_higher_dpi_unans02503505_64k_sp2_bs32_8gpu_20260518_wandb_save2_rerun2/full_sft_both_higher_dpi_unans02503505_len65536_sp2_bs32_freeze_vt_epoch1_wandb_save2_rerun2/sft_checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_full_sft_both_higher_dpi_unans02503505_64k_sp2_bs32_8gpu_20260518_wandb_save2_rerun2}"
MODEL_LABEL="${MODEL_LABEL:-full_sft_both_higher_dpi_unans02503505_len65536_sp2_bs32_epoch1_wandb_save2_rerun2}"
TARGET_STEP="${TARGET_STEP:-559}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STATUS_TSV="${STATUS_TSV:-${OUTPUT_ROOT}/status.tsv}"
MASTER_LOG="${MASTER_LOG:-${OUTPUT_ROOT}/eval_queue.log}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"

VAL_FILES='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/longdocurl_highpage_0507-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/mmlongbench_highpage_0507-insight_qwen_agent.parquet"]'

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

cd "${REPO_ROOT}"

if [[ -f /home/ywxzml3j/ywxzml3juser40/.bashrc ]]; then
  OPENAI_EXPORTS="$(grep -E '^export OPENAI_(API_KEY|BASE_URL)=' /home/ywxzml3j/ywxzml3juser40/.bashrc || true)"
  if [[ -n "${OPENAI_EXPORTS}" ]]; then
    eval "${OPENAI_EXPORTS}"
  fi
fi

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set}"

printf "stage\tmodel\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "${STATUS_TSV}"

log() {
  echo "[$(date '+%F %T')] $*" >&2
}

target_hf_model() {
  local hf_dir="${SFT_ROOT}/global_step_${TARGET_STEP}/huggingface"
  if [[ -f "${hf_dir}/model.safetensors.index.json" ]]; then
    printf '%s\n' "${hf_dir}"
    return 0
  fi
  return 1
}

wait_for_hf_model() {
  local model_path=""
  while true; do
    if model_path="$(target_hf_model)"; then
      printf '%s\n' "${model_path}"
      return 0
    fi
    log "waiting for final HF checkpoint global_step_${TARGET_STEP} under ${SFT_ROOT}; next check in ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

record_status() {
  local stage="$1"
  local model="$2"
  local scale_id="$3"
  local scale_value="$4"
  local status="$5"
  local exit_code="$6"
  local path="$7"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${stage}" "${model}" "${scale_id}" "${scale_value}" "${status}" "${exit_code}" "${path}" >> "${STATUS_TSV}"
}

run_eval_one() {
  local model_path="$1"
  local scale_id="$2"
  local scale_value="$3"
  local agent_cfg="$4"
  local run_name="${MODEL_LABEL}_highpage_0507_rescale${scale_id}"
  local work_dir="${OUTPUT_ROOT}/${run_name}"
  local status="success"
  local exit_code=0

  log "starting ${run_name} initial_rescale=${scale_value}"

  unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
  unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

  set +e
  EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
  MODEL_PATH="${model_path}" \
  LOAD_FORMAT="safetensors" \
  WORK_DIR="${work_dir}" \
  EXP_NAME="${run_name}" \
  WANDB_NAME="${run_name}" \
  VAL_FILES="${VAL_FILES}" \
  AGENT_LOOP_CONFIG_PATH="${agent_cfg}" \
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
  FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL}" \
  TELEGRAM_NOTIFY_LABEL="${run_name}" \
  bash scripts/run_iq_ft_eval_default_sampling_rl15360.sh \
    trainer.val_only_hf_model_rollout=true \
    trainer.resume_mode=disable
  exit_code=$?
  set -e

  if (( exit_code != 0 )); then
    status="failed"
  fi
  record_status eval "${MODEL_LABEL}" "${scale_id}" "${scale_value}" "${status}" "${exit_code}" "${work_dir}"
  log "finished ${run_name} status=${status} exit_code=${exit_code}"
}

main() {
  log "eval watcher started"
  log "sft_root=${SFT_ROOT}"
  log "target_step=${TARGET_STEP}"
  log "output_root=${OUTPUT_ROOT}"
  log "eval_gpus=${EVAL_CUDA_VISIBLE_DEVICES}"

  local model_path
  model_path="$(wait_for_hf_model)"
  record_status checkpoint "${MODEL_LABEL}" "-" "-" success 0 "${model_path}"
  log "using final HF checkpoint ${model_path}"

  run_eval_one "${model_path}" 025 0.25 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml
  run_eval_one "${model_path}" 035 0.35 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml
  run_eval_one "${model_path}" 05 0.5 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml
  log "all evals finished"
}

main "$@"
