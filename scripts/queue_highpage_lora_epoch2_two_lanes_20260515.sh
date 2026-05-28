#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"
cd "$REPO_ROOT"

if [[ -f /home/ywxzml3j/ywxzml3juser40/.bashrc ]]; then
  OPENAI_EXPORTS="$(grep -E '^export OPENAI_(API_KEY|BASE_URL)=' /home/ywxzml3j/ywxzml3juser40/.bashrc || true)"
  if [[ -n "$OPENAI_EXPORTS" ]]; then
    eval "$OPENAI_EXPORTS"
  fi
fi

if [[ -f /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-vllm-latest}"
fi

export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

if [[ "${DISABLE_PROXY:-0}" == "1" ]]; then
  unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY
  unset https_proxy http_proxy all_proxy no_proxy
  log_proxy_state="disabled"
else
  log_proxy_state="inherited"
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"

LANE_NAME="${LANE_NAME:?Set LANE_NAME}"
LANE_GPUS="${LANE_GPUS:?Set LANE_GPUS}"
TASKS_FILE="${TASKS_FILE:?Set TASKS_FILE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_epoch2_two_lanes_20260515}"
MERGE_DTYPE="${MERGE_DTYPE:-bfloat16}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IDLE_STABLE_SECONDS="${IDLE_STABLE_SECONDS:-300}"
GPU_IDLE_MEMORY_THRESHOLD_MB="${GPU_IDLE_MEMORY_THRESHOLD_MB:-1024}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"

VAL_FILES='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/longdocurl_highpage_0507-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/mmlongbench_highpage_0507-insight_qwen_agent.parquet"]'

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status_${LANE_NAME}.tsv"
if [[ ! -s "$STATUS_TSV" ]]; then
  printf "lane\tstage\tmodel\tstep\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "$STATUS_TSV"
fi

log() {
  echo "[$(date '+%F %T')] [$LANE_NAME] $*" >&2
}

record_status() {
  local stage="$1"
  local model="$2"
  local step="$3"
  local scale_id="$4"
  local scale_value="$5"
  local status="$6"
  local exit_code="$7"
  local path="$8"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$LANE_NAME" "$stage" "$model" "$step" "$scale_id" "$scale_value" "$status" "$exit_code" "$path" >> "$STATUS_TSV"
}

gpu_has_memory_allocations() {
  local query
  if ! query="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
    log "nvidia-smi query failed; treating GPUs as busy"
    return 0
  fi
  awk -F', *' -v gpus="$LANE_GPUS" -v threshold="$GPU_IDLE_MEMORY_THRESHOLD_MB" '
    BEGIN {
      split(gpus, arr, ",")
      for (i in arr) wanted[arr[i]] = 1
    }
    ($1 in wanted) && ($2 + 0 > threshold) { busy = 1 }
    END { exit busy ? 0 : 1 }
  ' <<< "$query"
}

wait_for_idle_gpus() {
  local idle_started=""
  log "waiting for GPUs ${LANE_GPUS} to be memory-idle for ${IDLE_STABLE_SECONDS}s"
  while true; do
    if gpu_has_memory_allocations; then
      idle_started=""
      log "GPUs ${LANE_GPUS} busy; polling again in ${POLL_SECONDS}s"
      sleep "$POLL_SECONDS"
      continue
    fi

    if [[ -z "$idle_started" ]]; then
      idle_started="$(date +%s)"
      log "GPUs ${LANE_GPUS} below memory threshold; starting idle timer"
    fi

    local now elapsed
    now="$(date +%s)"
    elapsed=$((now - idle_started))
    if (( elapsed >= IDLE_STABLE_SECONDS )); then
      log "GPU idle window satisfied"
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
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

run_eval_one() {
  local model_path="$1"
  local model_label="$2"
  local step="$3"
  local scale_id="$4"
  local scale_value="$5"
  local agent_cfg="$6"
  local run_name="${model_label}_step${step}_highpage_0507_rescale${scale_id}"
  local work_dir="$OUTPUT_ROOT/$run_name"
  local status="success"
  local exit_code=0

  log "starting ${run_name} initial_rescale=${scale_value}"

  set +e
  EVAL_CUDA_VISIBLE_DEVICES="$LANE_GPUS" \
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
  record_status eval "$model_label" "$step" "$scale_id" "$scale_value" "$status" "$exit_code" "$work_dir"
  log "finished ${run_name} status=${status} exit_code=${exit_code}"
}

run_eval_sweep() {
  local model_path="$1"
  local model_label="$2"
  local step="$3"
  local scale_ids="${4:-025,035,05}"
  IFS=',' read -ra scales <<< "$scale_ids"
  for scale_id in "${scales[@]}"; do
    case "$scale_id" in
      025)
        run_eval_one "$model_path" "$model_label" "$step" 025 0.25 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml
        ;;
      035)
        run_eval_one "$model_path" "$model_label" "$step" 035 0.35 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml
        ;;
      05)
        run_eval_one "$model_path" "$model_label" "$step" 05 0.5 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml
        ;;
      *)
        log "ERROR: unknown scale_id=${scale_id} for ${model_label} step=${step}"
        record_status eval "$model_label" "$step" "$scale_id" "unknown" "failed_bad_scale" 1 "-"
        ;;
    esac
  done
}

log "output_root=${OUTPUT_ROOT}"
log "lane_gpus=${LANE_GPUS}"
log "tasks_file=${TASKS_FILE}"
log "proxy_state=${log_proxy_state}"
wait_for_idle_gpus

while IFS=$'\t' read -r model_label step ckpt_dir scale_ids; do
  [[ -z "${model_label:-}" || "${model_label:0:1}" == "#" ]] && continue
  scale_ids="${scale_ids:-025,035,05}"
  if [[ ! -f "$ckpt_dir/huggingface/model.safetensors.index.json" ]]; then
    log "ERROR: missing checkpoint HF index for ${model_label} step=${step}: ${ckpt_dir}"
    record_status merge "$model_label" "$step" "-" "-" "failed_missing_checkpoint" 1 "$ckpt_dir"
    continue
  fi
  merged_model="$(ensure_merged_lora "$ckpt_dir" "${model_label}_step${step}")"
  record_status merge "$model_label" "$step" "-" "-" "success" 0 "$merged_model"
  run_eval_sweep "$merged_model" "$model_label" "$step" "$scale_ids"
done < "$TASKS_FILE"

log "lane queue finished"
