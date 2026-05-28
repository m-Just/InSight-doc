#!/usr/bin/env bash
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"
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

export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MODEL_PATH="/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_both_higher_dpi_easy200_unans02503505_fixed_sources_20260518/lora_both_w_higher_dpi_easy200_unans02503505_len65536_sp2_bs32_rank32_alpha64_freeze_vt_fixed_sources_epoch1/sft_checkpoints/global_step_572/huggingface_merged_lora"
export LOAD_FORMAT="safetensors"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_epoch1_plus_easy200_unans02503505_fixed_sources_rescale035_retry_20260519}"
RUN_NAME="lora_both_higher_dpi_epoch1_plus_easy200_unans02503505_fixed_sources_highpage_0507_rescale035_retry"

mkdir -p "${OUTPUT_ROOT}"

export WORK_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
export EXP_NAME="${RUN_NAME}"
export WANDB_NAME="${RUN_NAME}"
export VAL_FILES='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/longdocurl_highpage_0507-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/mmlongbench_highpage_0507-insight_qwen_agent.parquet"]'
export AGENT_LOOP_CONFIG_PATH="recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml"
export LOGGER="['console']"
export MAX_RESPONSE_LENGTH="15360"
export VAL_BATCH_SIZE="32"
export TOOL_MAX_USER_TURNS="10"
export TOOL_MAX_ASSISTANT_TURNS="11"
export DATA_MAX_PROMPT_LENGTH="262144"
export DATA_VALIDATION_MAX_PROMPT_LENGTH="262144"
export ROLLOUT_MAX_MODEL_LEN="262144"
export VLLM_MAX_MODEL_LEN="262144"
export VLLM_GPU_MEMORY_UTILIZATION="0.9"
export FALLBACK_JUDGE_MODEL="gemini-3.1-flash-lite-preview"
export TELEGRAM_NOTIFY_LABEL="${RUN_NAME}"

bash scripts/run_iq_ft_eval_default_sampling_rl15360.sh \
  trainer.val_only_hf_model_rollout=true \
  trainer.resume_mode=disable
