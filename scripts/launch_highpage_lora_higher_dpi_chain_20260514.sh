#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"
cd "$REPO_ROOT"

if [[ -f /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh ]]; then
  # Use the same env as the Qwen3-VL eval path; the merge env lacks qwen3_vl support.
  source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-vllm-latest}"
fi
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"

GPUS="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_higher_dpi_chain_20260514}"
MERGE_DTYPE="${MERGE_DTYPE:-bfloat16}"
POLL_SECONDS="${POLL_SECONDS:-600}"
O3_EXPECTED_STEP="${O3_EXPECTED_STEP:-1110}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"
START_STAGE="${START_STAGE:-basic}"

VAL_FILES='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/longdocurl_highpage_0507-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/mmlongbench_highpage_0507-insight_qwen_agent.parquet"]'

CURRENT_MODEL="/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_medium_data_ablation_40k_4gpu_lanes_wandb_noactoff_allowover_continue_20260513_210801/lora_basic_lr2e-4_cosine_minlr2e-5_len40960_bs32_rank32_alpha64_freeze_vt_medium_only/sft_checkpoints/global_step_792/huggingface_merged_lora"
ARXIV_ROOT="/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_medium_data_ablation_40k_4gpu_lanes_wandb_noactoff_allowover_continue_20260513_210801/lora_arxiv_w_higher_dpi_lr2e-4_cosine_minlr2e-5_len40960_bs32_rank32_alpha64_freeze_vt_medium_only/sft_checkpoints"
O3_ROOT="/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_medium_data_ablation_64k_sp2_failed_rerun_aspect_drop_20260514_110216/lora_O3_w_higher_dpi_lr2e-4_cosine_minlr2e-5_len65536_bs32_rank32_alpha64_freeze_vt_medium_only/sft_checkpoints"

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
if [[ "${RESET_STATUS:-0}" == "1" || ! -s "$STATUS_TSV" ]]; then
  printf "stage\tmodel\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "$STATUS_TSV"
fi

log() {
  echo "[$(date '+%F %T')] $*" >&2
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
    "$stage" "$model" "$scale_id" "$scale_value" "$status" "$exit_code" "$path" >> "$STATUS_TSV"
}

has_clean_hf_weights() {
  local model_dir="$1"
  python - "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
index_path = model_dir / "model.safetensors.index.json"
if not index_path.exists():
    raise SystemExit(1)
with index_path.open() as f:
    weight_map = json.load(f).get("weight_map", {})
bad = [k for k in weight_map if "lora_" in k or "base_layer" in k]
raise SystemExit(1 if bad else 0)
PY
}

patch_adapter_config() {
  local adapter_dir="$1"
  python - "$adapter_dir" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "adapter_config.json"
if not path.exists():
    raise SystemExit(f"missing adapter_config.json: {path}")
with path.open() as f:
    cfg = json.load(f)
cfg["r"] = 32
cfg["lora_alpha"] = 64
with path.open("w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
}

ensure_merged_lora() {
  local ckpt_dir="$1"
  local model_label="$2"
  local export_dir="$ckpt_dir/huggingface_base_lora_export"
  local adapter_dir="$export_dir/lora_adapter"
  local merged_dir="$ckpt_dir/huggingface_merged_lora"

  if has_clean_hf_weights "$merged_dir"; then
    log "${model_label}: using existing merged HF model: ${merged_dir}"
    echo "$merged_dir"
    return 0
  fi

  log "${model_label}: merging FSDP checkpoint to base+adapter HF export"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ckpt_dir" \
    --target_dir "$export_dir" >&2

  patch_adapter_config "$adapter_dir"

  log "${model_label}: merging PEFT adapter into standalone HF model"
  python scripts/merge_peft_lora_to_hf.py \
    --base-model "$export_dir" \
    --adapter "$adapter_dir" \
    --output-dir "$merged_dir" \
    --dtype "$MERGE_DTYPE" >&2

  if ! has_clean_hf_weights "$merged_dir"; then
    echo "ERROR: merged model still contains LoRA/base_layer keys: $merged_dir" >&2
    return 1
  fi

  echo "$merged_dir"
}

latest_step() {
  local root="$1"
  local marker="$root/latest_checkpointed_iteration.txt"
  if [[ -s "$marker" ]]; then
    tr -dc '0-9' < "$marker"
  else
    echo 0
  fi
}

wait_for_checkpoint() {
  local root="$1"
  local expected_step="$2"
  local model_label="$3"

  while true; do
    local step
    step="$(latest_step "$root")"
    local ckpt_dir="$root/global_step_${step}"
    if [[ "$step" -ge "$expected_step" && -f "$ckpt_dir/huggingface/model.safetensors.index.json" ]]; then
      log "${model_label}: found completed checkpoint global_step_${step}"
      echo "$ckpt_dir"
      return 0
    fi
    log "${model_label}: waiting for checkpoint >= ${expected_step}; latest=${step}"
    sleep "$POLL_SECONDS"
  done
}

ckpt_from_latest() {
  local root="$1"
  local model_label="$2"
  local step
  step="$(latest_step "$root")"
  local ckpt_dir="$root/global_step_${step}"
  if [[ "$step" -le 0 || ! -f "$ckpt_dir/huggingface/model.safetensors.index.json" ]]; then
    echo "ERROR: no usable latest checkpoint for ${model_label}: root=${root} latest=${step}" >&2
    return 1
  fi
  echo "$ckpt_dir"
}

run_eval_one() {
  local model_path="$1"
  local model_label="$2"
  local scale_id="$3"
  local scale_value="$4"
  local agent_cfg="$5"
  local run_name="${model_label}_highpage_0507_rescale${scale_id}"
  local work_dir="$OUTPUT_ROOT/$run_name"
  local status="success"
  local exit_code=0

  log "starting ${run_name} initial_rescale=${scale_value}"

  set +e
  EVAL_CUDA_VISIBLE_DEVICES="$GPUS" \
  MODEL_PATH="$model_path" \
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
  FALLBACK_JUDGE_MODEL="$FALLBACK_JUDGE_MODEL" \
  TELEGRAM_NOTIFY_LABEL="$run_name" \
  bash scripts/run_iq_ft_eval_default_sampling_rl15360.sh \
    trainer.val_only_hf_model_rollout=true \
    trainer.resume_mode=disable
  exit_code=$?
  set -e

  if (( exit_code != 0 )); then
    status="failed"
  fi
  record_status eval "$model_label" "$scale_id" "$scale_value" "$status" "$exit_code" "$work_dir"
  log "finished ${run_name} status=${status} exit_code=${exit_code}"
}

run_eval_sweep() {
  local model_path="$1"
  local model_label="$2"
  run_eval_one "$model_path" "$model_label" 025 0.25 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml
  run_eval_one "$model_path" "$model_label" 035 0.35 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml
  run_eval_one "$model_path" "$model_label" 05 0.5 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml
}

log "output_root=${OUTPUT_ROOT}"
log "GPUS=${GPUS}"
log "START_STAGE=${START_STAGE}"

if [[ "$START_STAGE" == "basic" ]]; then
  run_eval_one "$CURRENT_MODEL" "lora_basic_40k_freeze_vt_medium_only_retry" 035 0.35 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml
fi

if [[ "$START_STAGE" == "basic" || "$START_STAGE" == "arxiv" ]]; then
  ARXIV_CKPT="$(ckpt_from_latest "$ARXIV_ROOT" "lora_arxiv_w_higher_dpi")"
  ARXIV_MODEL="$(ensure_merged_lora "$ARXIV_CKPT" "lora_arxiv_w_higher_dpi")"
  record_status merge "lora_arxiv_w_higher_dpi" "-" "-" "success" 0 "$ARXIV_MODEL"
  run_eval_sweep "$ARXIV_MODEL" "lora_arxiv_w_higher_dpi"
fi

O3_CKPT="$(wait_for_checkpoint "$O3_ROOT" "$O3_EXPECTED_STEP" "lora_O3_w_higher_dpi")"
O3_MODEL="$(ensure_merged_lora "$O3_CKPT" "lora_O3_w_higher_dpi")"
record_status merge "lora_O3_w_higher_dpi" "-" "-" "success" 0 "$O3_MODEL"
run_eval_sweep "$O3_MODEL" "lora_O3_w_higher_dpi"

log "all chained highpage LoRA eval jobs finished"
