#!/usr/bin/env bash
set -euo pipefail

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
cd "${REPO_ROOT}"

export RESET_STATUS="${RESET_STATUS:-1}"
export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_unanswerable02503505_epoch1_rerun_20260521}"
export MODEL_LABEL="${MODEL_LABEL:-lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch1_rerun_20260521}"
export BOTH_ROOT="${BOTH_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_unanswerable_medium_lane0_20260517/lora_both_higher_dpi_unanswerable02503505_len65536_sp2_bs32_rank32_alpha64_freeze_vt_medium_only_epoch1/sft_checkpoints}"
export FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"

exec scripts/run_with_telegram_notification.sh \
  --notify-start \
  --label "${MODEL_LABEL}" \
  -- bash scripts/launch_highpage_lora_both_higher_dpi_20260515.sh
