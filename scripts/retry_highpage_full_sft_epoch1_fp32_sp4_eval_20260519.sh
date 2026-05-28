#!/usr/bin/env bash
set -euo pipefail

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
MODEL_PATH="${MODEL_PATH:-/scratch/ywxzml3j/likaican/temp/full_sft_both_higher_dpi_unans02503505_fp32_sp4_scratch_20260519/full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_freeze_vt_epoch1_fp32_scratch/sft_checkpoints/global_step_559/huggingface}"
MODEL_LABEL="${MODEL_LABEL:-full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_epoch1_fp32_scratch}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_full_sft_both_higher_dpi_unans02503505_fp32_sp4_scratch_20260519_retry}"
EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES:-0,1,2,3}"
STATUS_TSV="${STATUS_TSV:-${EVAL_OUTPUT_ROOT}/status.tsv}"
MASTER_LOG="${MASTER_LOG:-${EVAL_OUTPUT_ROOT}/eval_queue.log}"

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set}"

VAL_FILES_HIGH_PAGE='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/longdocurl_highpage_0507-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/mmlongbench_highpage_0507-insight_qwen_agent.parquet"]'

mkdir -p "${EVAL_OUTPUT_ROOT}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

printf "time\tstage\tmodel\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "${STATUS_TSV}"

run_eval_one() {
  local scale_id="$1"
  local scale_value="$2"
  local agent_cfg="$3"
  local run_name="${MODEL_LABEL}_highpage_0507_rescale${scale_id}"
  local work_dir="${EVAL_OUTPUT_ROOT}/${run_name}"
  local status="success"
  local exit_code=0

  echo "[eval:${run_name}] start=$(date -Is) model_path=${MODEL_PATH}"

  unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
  unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

  set +e
  EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICES}" \
  MODEL_PATH="${MODEL_PATH}" \
  LOAD_FORMAT="safetensors" \
  WORK_DIR="${work_dir}" \
  EXP_NAME="${run_name}" \
  WANDB_NAME="${run_name}" \
  VAL_FILES="${VAL_FILES_HIGH_PAGE}" \
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
  TELEGRAM_NOTIFY_LABEL="${run_name}" \
  bash "${REPO_ROOT}/scripts/run_iq_ft_eval_default_sampling_rl15360.sh" \
    trainer.val_only_hf_model_rollout=true \
    trainer.resume_mode=disable
  exit_code=$?
  set -e

  if (( exit_code != 0 )); then
    status="failed"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" eval "${MODEL_LABEL}" "${scale_id}" "${scale_value}" "${status}" "${exit_code}" "${work_dir}" >> "${STATUS_TSV}"
  echo "[eval:${run_name}] finished=$(date -Is) status=${status} exit_code=${exit_code}"
}

cd "${REPO_ROOT}"
echo "[queue] started=$(date -Is) gpus=${EVAL_CUDA_DEVICES} model_path=${MODEL_PATH}"
run_eval_one 025 0.25 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml
run_eval_one 035 0.35 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml
run_eval_one 05 0.5 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml
echo "[queue] finished=$(date -Is)"
