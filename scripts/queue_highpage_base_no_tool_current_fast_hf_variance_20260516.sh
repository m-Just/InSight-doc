#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"
cd "$REPO_ROOT"

# The judge endpoint should be reached directly for this eval and future reruns
# launched through this wrapper.
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY
unset https_proxy http_proxy all_proxy no_proxy

if [[ -f /home/ywxzml3j/ywxzml3juser40/.bashrc ]]; then
  OPENAI_EXPORTS="$(grep -E '^export OPENAI_(API_KEY|BASE_URL)=' /home/ywxzml3j/ywxzml3juser40/.bashrc || true)"
  if [[ -n "$OPENAI_EXPORTS" ]]; then
    eval "$OPENAI_EXPORTS"
  fi
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_current_fast_hf_variance_20260516}"
VERIFY_TIMEOUT_SECONDS="${VERIFY_TIMEOUT_SECONDS:-1800}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"

VAL_FILES='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets_no_tool_no_system/longdocurl_highpage_0507-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets_no_tool_no_system/mmlongbench_highpage_0507-insight_qwen_agent_no_tool_no_system.parquet"]'

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
if [[ "${RESET_STATUS:-0}" == "1" || ! -s "$STATUS_TSV" ]]; then
  printf "lane\trepeat_id\tmodel\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "$STATUS_TSV"
fi

log() {
  echo "[$(date '+%F %T')] $*"
}

verify_fast_path_or_kill() {
  local launch_log="$1"
  local pid="$2"
  local waited=0

  while kill -0 "$pid" 2>/dev/null && (( waited < VERIFY_TIMEOUT_SECONDS )); do
    if [[ -f "$launch_log" ]]; then
      if grep -q "'val_only_hf_model_rollout': True" "$launch_log"; then
        log "verified fast path in ${launch_log}"
        return 0
      fi
      if grep -q "'val_only_hf_model_rollout': False" "$launch_log"; then
        log "ERROR: fast path override did not take effect in ${launch_log}; killing pid=${pid}"
        kill "$pid" 2>/dev/null || true
        return 1
      fi
    fi
    sleep 5
    waited=$((waited + 5))
  done

  if kill -0 "$pid" 2>/dev/null; then
    log "ERROR: timed out waiting for effective fast-path config in ${launch_log}; killing pid=${pid}"
    kill "$pid" 2>/dev/null || true
  fi
  return 1
}

run_eval_one() {
  local lane="$1"
  local repeat_id="$2"
  local gpus="$3"
  local scale_id="$4"
  local scale_value="$5"
  local agent_cfg="$6"
  local model_label="base_no_tool_no_system_current_lora_setting_fast_hf_val_only"
  local run_name="${model_label}_repeat${repeat_id}_highpage_0507_rescale${scale_id}"
  local work_dir="$OUTPUT_ROOT/$run_name"
  local launch_log="$work_dir/${run_name}.launch.log"
  local exit_code=0
  local status="success"

  log "[${lane}] starting ${run_name} gpus=${gpus} initial_rescale=${scale_value}"
  mkdir -p "$work_dir"

  set +e
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u NO_PROXY \
      -u https_proxy -u http_proxy -u all_proxy -u no_proxy \
  EVAL_CUDA_VISIBLE_DEVICES="$gpus" \
  MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct" \
  LOAD_FORMAT="auto" \
  WORK_DIR="$work_dir" \
  EXP_NAME="$run_name" \
  WANDB_NAME="$run_name" \
  VAL_FILES="$VAL_FILES" \
  AGENT_LOOP_CONFIG_PATH="$agent_cfg" \
  QWEN_TOOL_LIST="[]" \
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
  bash scripts/run_iq_base_eval_default_sampling_insight_rl15360.sh \
    trainer.val_only_hf_model_rollout=true \
    trainer.resume_mode=disable &
  local pid=$!

  verify_fast_path_or_kill "$launch_log" "$pid"
  local verify_status=$?
  wait "$pid"
  exit_code=$?
  set -e

  if (( verify_status != 0 || exit_code != 0 )); then
    status="failed"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$lane" "$repeat_id" "$model_label" "$scale_id" "$scale_value" "$status" "$exit_code" "$work_dir" >> "$STATUS_TSV"
  log "[${lane}] finished ${run_name} status=${status} exit_code=${exit_code}"
}

run_lane() {
  local lane="$1"
  local repeat_id="$2"
  local gpus="$3"
  run_eval_one "$lane" "$repeat_id" "$gpus" 025 0.25 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml
  run_eval_one "$lane" "$repeat_id" "$gpus" 035 0.35 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml
  run_eval_one "$lane" "$repeat_id" "$gpus" 05 0.5 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml
  log "[${lane}] repeat ${repeat_id} finished"
}

log "output_root=${OUTPUT_ROOT}"
log "proxy_state=disabled"
run_lane lane_a 1 "${LANE_A_GPUS:-0,1,2,3}" &
pid_a=$!
run_lane lane_b 2 "${LANE_B_GPUS:-4,5,6,7}" &
pid_b=$!

wait "$pid_a"
status_a=$?
wait "$pid_b"
status_b=$?

if (( status_a != 0 || status_b != 0 )); then
  log "one or more lanes failed: lane_a=${status_a} lane_b=${status_b}"
  exit 1
fi

log "all variance repeat lanes finished"
