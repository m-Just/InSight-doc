#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script}"

export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH:-recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500.yaml}"
export TOOL_MAX_USER_TURNS="${TOOL_MAX_USER_TURNS:-10}"
export TOOL_MAX_ASSISTANT_TURNS="${TOOL_MAX_ASSISTANT_TURNS:-11}"

QUEUE=(
  "insight_qwen_agent_full_sft_all_convos_0426_exp1_lr3e-6_cosine_minlr3e-7_len32768_bs32_clean_data|/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_exp1_lr3e-6_cosine_minlr3e-7_len32768_bs32_clean_data/sft_checkpoints/global_step_1086/huggingface"
  "insight_qwen_agent_full_sft_all_convos_0426_exp2_lr5e-6_cosine_minlr5e-7_len32768_bs32|/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_exp2_lr5e-6_cosine_minlr5e-7_len32768_bs32/sft_checkpoints/global_step_1086/huggingface"
  "insight_qwen_agent_full_sft_all_convos_0426_exp3_lr3e-6_cosine_minlr3e-7_len32768_bs8|/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_exp3_lr3e-6_cosine_minlr3e-7_len32768_bs8/sft_checkpoints/global_step_4347/huggingface"
  "insight_qwen_agent_full_sft_all_convos_0426_exp4_lr2e-6_cosine_minlr2e-7_len32768_bs8|/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_exp4_lr2e-6_cosine_minlr2e-7_len32768_bs8/sft_checkpoints/global_step_4347/huggingface"
  "insight_qwen_agent_full_sft_all_convos_0426_exp5_lr1e-6_cosine_minlr1e-7_len32768_bs32|/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_exp5_lr1e-6_cosine_minlr1e-7_len32768_bs32/sft_checkpoints/global_step_1086/huggingface"
  "insight_qwen_agent_full_sft_all_convos_0426_exp1_lr5e-6_cosine_minlr5e-7_len32768_bs16|/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_exp1_lr5e-6_cosine_minlr5e-7_len32768_bs16/sft_checkpoints/global_step_2172/huggingface"
  "insight_qwen_agent_full_sft_all_convos_0426_exp2_lr3e-6_cosine_minlr3e-7_len32768_bs16|/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_exp2_lr3e-6_cosine_minlr3e-7_len32768_bs16/sft_checkpoints/global_step_2172/huggingface"
  "insight_qwen_agent_full_sft_all_convos_0426_exp3_lr8e-6_cosine_minlr8e-7_len32768_bs16|/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_exp3_lr8e-6_cosine_minlr8e-7_len32768_bs16/sft_checkpoints/global_step_2172/huggingface"
)

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs}"
VAL_FILES_DEFAULT='[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.test.parquet]'
export VAL_FILES="${VAL_FILES:-$VAL_FILES_DEFAULT}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"

for entry in "${QUEUE[@]}"; do
  exp_name="${entry%%|*}"
  model_path="${entry#*|}"
  work_dir="${OUTPUT_ROOT}/${exp_name}_eval_final_zoom2_area3500"
  log_path="${OUTPUT_ROOT}/${exp_name}_eval_final_zoom2_area3500.launch.log"
  mkdir -p "$work_dir"

  echo "[$(date -Is)] start ${exp_name}"
  MODEL_PATH="$model_path" \
  WORK_DIR="$work_dir" \
  EXP_NAME="${exp_name}_eval_final_zoom2_area3500" \
  EVAL_NAME="heldout" \
  CONVERSATION_EXPORT_DIR="$work_dir/exported_conversations" \
  bash "$REPO_ROOT/scripts/run_iq_ft_eval_default_sampling_rl15360.sh" \
    2>&1 | tee -a "$log_path"
  status=${PIPESTATUS[0]}
  echo "[$(date -Is)] end ${exp_name} status=${status}"
  if [[ $status -ne 0 ]]; then
    exit "$status"
  fi
done
