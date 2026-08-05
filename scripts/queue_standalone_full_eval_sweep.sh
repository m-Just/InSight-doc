#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"


cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
PYTHONPATH_ENTRIES="$REPO_ROOT"
if [[ -n "${INSIGHT_O3_ROOT:-}" ]]; then
  PYTHONPATH_ENTRIES="$PYTHONPATH_ENTRIES:$INSIGHT_O3_ROOT"
fi
if [[ -n "${QWEN_AGENT_ROOT:-}" ]]; then
  PYTHONPATH_ENTRIES="$PYTHONPATH_ENTRIES:$QWEN_AGENT_ROOT"
fi
export PYTHONPATH="$PYTHONPATH_ENTRIES${PYTHONPATH:+:$PYTHONPATH}"
export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export ENSURE_API_LOGGER="${ENSURE_API_LOGGER:-1}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/workspace/standalone_full_eval_sweep_${RUN_ID}}"
AGENT_CONFIG="${AGENT_CONFIG:-standalone_eval/agent_configs/insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml}"
RESCALES="${RESCALES:-0.25 0.35 0.5}"
NUM_TRIALS="${NUM_TRIALS:-3}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
GROUP_VAL_FILES="${GROUP_VAL_FILES:-1}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
OVERLAP_JUDGE="${OVERLAP_JUDGE:-1}"

AGENT_WORKER_PROCESSES="${AGENT_WORKER_PROCESSES:-8}"
WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-4}"
HTTPS_WORKER_CONCURRENCY="${HTTPS_WORKER_CONCURRENCY:-32}"
CONTEXT_OVERFLOW_MAX_HALVING_TRIALS="${CONTEXT_OVERFLOW_MAX_HALVING_TRIALS:-4}"

JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"
JUDGE_WORKERS="${JUDGE_WORKERS:-32}"
INSIGHT_QWEN_JUDGE_MODE="${INSIGHT_QWEN_JUDGE_MODE:-legacy}"
if [[ -z "${JUDGE_TASK_TIMEOUT_SECONDS:-}" && ( "$INSIGHT_QWEN_JUDGE_MODE" == "single_call_v1" || "$INSIGHT_QWEN_JUDGE_MODE" == "single_call_v2" ) ]]; then
  JUDGE_TASK_TIMEOUT_SECONDS=600
fi
export JUDGE_TASK_TIMEOUT_SECONDS
SCORES_SUBDIR="${SCORES_SUBDIR:-scores}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-}"
JUDGE_RESCORE_EXISTING="${JUDGE_RESCORE_EXISTING:-0}"

RAY_IDLE_TIMEOUT_SECONDS="${RAY_IDLE_TIMEOUT_SECONDS:-1800}"
RAY_START_TIMEOUT_SECONDS="${RAY_START_TIMEOUT_SECONDS:-1200}"
RAY_CPUS_PER_SERVER="${RAY_CPUS_PER_SERVER:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-}"
RAY_ADDRESS="${RAY_ADDRESS:-}"
RAY_NAMESPACE="${RAY_NAMESPACE:-}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-}"

HTTPS_TIMEOUT_OVERRIDE="${HTTPS_TIMEOUT_OVERRIDE:-}"
HTTPS_MAX_RETRIES_OVERRIDE="${HTTPS_MAX_RETRIES_OVERRIDE:-}"

DEFAULT_MODEL_CONFIGS=(
  "standalone_eval/model_configs/release_ray_vllm.yaml"
)

DEFAULT_VAL_FILES=()
DEFAULT_VAL_FILES_NO_TOOL_NO_SYSTEM=()

split_env_list() {
  printf '%s\n' "$1" | tr ',\n' '  '
}

if [[ -n "${MODEL_CONFIGS:-}" ]]; then
  read -r -a MODEL_CONFIG_ARRAY <<< "$(split_env_list "$MODEL_CONFIGS")"
else
  MODEL_CONFIG_ARRAY=("${DEFAULT_MODEL_CONFIGS[@]}")
fi

USER_PROVIDED_VAL_FILES=0
if [[ -n "${VAL_FILES:-}" ]]; then
  USER_PROVIDED_VAL_FILES=1
  read -r -a VAL_FILE_ARRAY <<< "$(split_env_list "$VAL_FILES")"
else
  VAL_FILE_ARRAY=("${DEFAULT_VAL_FILES[@]}")
fi

if [[ -n "${VAL_FILES_NO_TOOL_NO_SYSTEM:-}" ]]; then
  read -r -a VAL_FILE_NO_TOOL_ARRAY <<< "$(split_env_list "$VAL_FILES_NO_TOOL_NO_SYSTEM")"
else
  VAL_FILE_NO_TOOL_ARRAY=("${DEFAULT_VAL_FILES_NO_TOOL_NO_SYSTEM[@]}")
fi

read -r -a RESCALE_ARRAY <<< "$(split_env_list "$RESCALES")"

CURRENT_RAY_MANIFEST=""
CURRENT_RAY_PID=""
CURRENT_RAY_KEY=""
ASYNC_JUDGE_PIDS=()
ASYNC_JUDGE_OUTPUT_DIRS=()
ASYNC_JUDGE_LABELS=()

slugify() {
  printf '%s' "$1" | tr '/ .:-' '_____' | tr -cs 'A-Za-z0-9_' '_' | sed 's/^_//;s/_$//'
}

rescale_slug() {
  printf '%s' "$1" | tr -d '.'
}

join_by_comma() {
  local IFS=,
  printf '%s' "$*"
}

realpath_csv() {
  local value="$1"
  local -a files resolved
  IFS=',' read -r -a files <<< "$value"
  resolved=()
  for file in "${files[@]}"; do
    resolved+=("$(realpath "$file")")
  done
  join_by_comma "${resolved[@]}"
}

print_command() {
  printf 'DRY_RUN:'
  printf ' %q' "$@"
  printf '\n'
}

with_api_env() {
  local -a env_args=(
    ENSURE_API_LOGGER="${ENSURE_API_LOGGER:-1}"
  )
  if [[ -n "${JUDGE_TASK_TIMEOUT_SECONDS:-}" ]]; then
    env_args+=(JUDGE_TASK_TIMEOUT_SECONDS="$JUDGE_TASK_TIMEOUT_SECONDS")
  fi
  if [[ -n "${JUDGE_BATCH_SIZE:-}" ]]; then
    env_args+=(JUDGE_BATCH_SIZE="$JUDGE_BATCH_SIZE")
  fi
  if [[ -n "${API_HTTP_PROXY:-}" ]]; then
    env_args+=(HTTP_PROXY="$API_HTTP_PROXY" http_proxy="$API_HTTP_PROXY")
  fi
  if [[ -n "${API_HTTPS_PROXY:-}" ]]; then
    env_args+=(HTTPS_PROXY="$API_HTTPS_PROXY" https_proxy="$API_HTTPS_PROXY")
  fi
  env "${env_args[@]}" "$@"
}

validate_api_env() {
  if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" ]]; then
    cat >&2 <<'EOF'
OPENAI_API_KEY and OPENAI_BASE_URL must be set before running this sweep.
The launcher intentionally does not set or change API credentials/endpoints.
EOF
    return 1
  fi
}

with_cuda_env() {
  if [[ -n "${EVAL_CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" "$@"
  else
    "$@"
  fi
}

read_model_config_field() {
  local model_config="$1"
  local field="$2"
  "$PYTHON_BIN" - "$model_config" "$field" <<'PY'
import sys
from standalone_eval.config.model import load_model_config

cfg = load_model_config(sys.argv[1])
value = cfg
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

model_config_file_sha() {
  local model_config="$1"
  "$PYTHON_BIN" - "$model_config" <<'PY'
import sys
from standalone_eval.config.model import sha256_file

print(sha256_file(sys.argv[1]))
PY
}

cleanup_current_ray_server() {
  if [[ -n "$CURRENT_RAY_MANIFEST" && -f "$CURRENT_RAY_MANIFEST" && "$DRY_RUN" != "1" ]]; then
    "$PYTHON_BIN" scripts/stop_ray_vllm.py --server-manifest "$CURRENT_RAY_MANIFEST" || true
  fi
  if [[ -n "$CURRENT_RAY_PID" && "$DRY_RUN" != "1" ]]; then
    wait "$CURRENT_RAY_PID" || true
  fi
  CURRENT_RAY_MANIFEST=""
  CURRENT_RAY_PID=""
  CURRENT_RAY_KEY=""
}

trap cleanup_current_ray_server EXIT

wait_for_ray_server() {
  local manifest="$1"
  local log_path="$2"
  local deadline=$((SECONDS + RAY_START_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if [[ -f "$manifest" ]] && "$PYTHON_BIN" - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get("status") == "running" else 1)
PY
    then
      return 0
    fi
    if [[ -n "$CURRENT_RAY_PID" ]] && ! kill -0 "$CURRENT_RAY_PID" 2>/dev/null; then
      echo "Ray server process exited before manifest became ready. Log: $log_path" >&2
      tail -n 120 "$log_path" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "Timed out waiting for Ray server manifest: $manifest. Log: $log_path" >&2
  tail -n 120 "$log_path" >&2 || true
  return 1
}

launch_ray_server_once() {
  local model_config="$1"
  local model_slug="$2"
  local server_root="$OUTPUT_ROOT/_ray_servers/$model_slug"
  local log_path="$server_root/serve.log"
  local heartbeat_path="$server_root/server_manifest.heartbeat"
  CURRENT_RAY_MANIFEST="$server_root/server_manifest.json"
  CURRENT_RAY_PID=""
  mkdir -p "$server_root"

  local -a serve_args=(
    scripts/serve_ray_vllm.py
    --model-config "$model_config"
    --server-manifest "$CURRENT_RAY_MANIFEST"
    --heartbeat-path "$heartbeat_path"
    --idle-timeout-seconds "$RAY_IDLE_TIMEOUT_SECONDS"
    --ray-cpus-per-server "$RAY_CPUS_PER_SERVER"
  )
  if [[ -n "$RAY_ADDRESS" ]]; then
    serve_args+=(--ray-address "$RAY_ADDRESS")
  fi
  if [[ -n "$RAY_NAMESPACE" ]]; then
    serve_args+=(--ray-namespace "$RAY_NAMESPACE")
  fi
  if [[ -n "$RAY_TEMP_DIR" ]]; then
    serve_args+=(--ray-temp-dir "$RAY_TEMP_DIR")
  fi
  if [[ -n "$RAY_NUM_CPUS" ]]; then
    serve_args+=(--ray-num-cpus "$RAY_NUM_CPUS")
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -u "${serve_args[@]}"
    return 0
  fi

  rm -f "$CURRENT_RAY_MANIFEST" "$heartbeat_path"
  with_cuda_env "$PYTHON_BIN" -u "${serve_args[@]}" > "$log_path" 2>&1 &
  CURRENT_RAY_PID="$!"
  wait_for_ray_server "$CURRENT_RAY_MANIFEST" "$log_path"
}

model_uses_no_tool_no_system_inputs() {
  local model_config="$1"
  local backend="$2"
  local name
  name="$(basename "$model_config" .yaml)"
  [[ "$backend" == "https_openai_chat" || "$name" == *"no_tool_no_system"* ]]
}

write_run_env() {
  local output_dir="$1"
  shift
  mkdir -p "$output_dir/logs"
  {
    echo "started_utc=$(date -u +'%F %T')"
    for item in "$@"; do
      echo "$item"
    done
  } > "$output_dir/run.env"
}

run_rollout() {
  local backend="$1"
  local model_config="$2"
  local val_file="$3"
  local rescale="$4"
  local output_dir="$5"
  local no_tool_override="$6"
  local log_path="$output_dir/logs/rollout.log"
  local -a rollout_args=(
    standalone_eval/rollout.py
    --model-config "$model_config"
    --val-files "$val_file"
    --output-dir "$output_dir"
    --agent-config "$AGENT_CONFIG"
    --agent-config-override "images.initial_rescale=$rescale"
    --max-samples "$MAX_SAMPLES"
    --num-trials "$NUM_TRIALS"
    --shuffle-rows
    --shuffle-seed "$SHUFFLE_SEED"
    --context-overflow-max-halving-trials "$CONTEXT_OVERFLOW_MAX_HALVING_TRIALS"
  )
  if [[ "$no_tool_override" == "1" ]]; then
    rollout_args+=(--agent-config-override "tools.qwen_tool_list=[]")
  fi
  if [[ "$backend" == "ray_vllm" ]]; then
    rollout_args+=(
      --ray-server-manifest "$CURRENT_RAY_MANIFEST"
      --agent-worker-processes "$AGENT_WORKER_PROCESSES"
      --worker-concurrency "$WORKER_CONCURRENCY"
    )
  else
    rollout_args+=(
      --agent-worker-processes 1
      --worker-concurrency "$HTTPS_WORKER_CONCURRENCY"
    )
    if [[ -n "$HTTPS_TIMEOUT_OVERRIDE" ]]; then
      rollout_args+=(--https-timeout-override "$HTTPS_TIMEOUT_OVERRIDE")
    fi
    if [[ -n "$HTTPS_MAX_RETRIES_OVERRIDE" ]]; then
      rollout_args+=(--https-max-retries-override "$HTTPS_MAX_RETRIES_OVERRIDE")
    fi
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -u "${rollout_args[@]}"
    return 0
  fi

  if [[ "$backend" == "https_openai_chat" ]]; then
    with_api_env "$PYTHON_BIN" -u "${rollout_args[@]}" > "$log_path" 2>&1
  else
    with_cuda_env "$PYTHON_BIN" -u "${rollout_args[@]}" > "$log_path" 2>&1
  fi
}

run_judge() {
  local output_dir="$1"
  local log_path="$output_dir/logs/judge.log"
  local -a judge_args=(
    standalone_eval/judge.py
    --rollout-dir "$output_dir"
    --scores-subdir "$SCORES_SUBDIR"
    --judge-model "$JUDGE_MODEL"
    --judge-workers "$JUDGE_WORKERS"
    --insight-qwen-judge-mode "$INSIGHT_QWEN_JUDGE_MODE"
  )
  if [[ "$JUDGE_RESCORE_EXISTING" == "1" ]]; then
    judge_args+=(--rescore-existing)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" -u "${judge_args[@]}"
    return 0
  fi
  with_api_env "$PYTHON_BIN" -u "${judge_args[@]}" > "$log_path" 2>&1
}

wait_for_async_judges() {
  local failures=0
  local i pid output_dir label judge_status
  if [[ "${#ASYNC_JUDGE_PIDS[@]}" == "0" ]]; then
    return 0
  fi

  echo "[$(date -u +'%F %T')] waiting for ${#ASYNC_JUDGE_PIDS[@]} overlap judge(s)"
  for i in "${!ASYNC_JUDGE_PIDS[@]}"; do
    pid="${ASYNC_JUDGE_PIDS[$i]}"
    output_dir="${ASYNC_JUDGE_OUTPUT_DIRS[$i]}"
    label="${ASYNC_JUDGE_LABELS[$i]}"
    judge_status=0
    if wait "$pid"; then
      judge_status=0
    else
      judge_status=$?
    fi
    echo "judge_exit_code=$judge_status" >> "$output_dir/run.env"
    if [[ "$judge_status" != "0" ]]; then
      failures=$((failures + 1))
      echo "[$(date -u +'%F %T')] overlap judge failed $label status=$judge_status"
    else
      echo "[$(date -u +'%F %T')] overlap judge done $label"
    fi
  done

  ASYNC_JUDGE_PIDS=()
  ASYNC_JUDGE_OUTPUT_DIRS=()
  ASYNC_JUDGE_LABELS=()
  return "$failures"
}

run_one_eval() {
  local backend="$1"
  local model_config="$2"
  local model_name="$3"
  local model_slug="$4"
  local val_files_arg="$5"
  local val_slug="$6"
  local rescale="$7"
  local no_tool_override="$8"
  local scale_slug output_dir rollout_status judge_status
  scale_slug="$(rescale_slug "$rescale")"
  output_dir="$OUTPUT_ROOT/$model_slug/$val_slug/rescale${scale_slug}"
  mkdir -p "$output_dir/logs"

  write_run_env "$output_dir" \
    "output_dir=$output_dir" \
    "model=$model_name" \
    "model_slug=$model_slug" \
    "model_config=$(realpath "$model_config")" \
    "backend=$backend" \
    "val_files=$(realpath_csv "$val_files_arg")" \
    "val_slug=$val_slug" \
    "rescale=$rescale" \
    "num_trials=$NUM_TRIALS" \
    "max_samples=$MAX_SAMPLES" \
    "shuffle_rows=1" \
    "shuffle_seed=$SHUFFLE_SEED" \
    "agent_config=$(realpath "$AGENT_CONFIG")" \
    "no_tool_no_system_inputs=$no_tool_override" \
    "agent_worker_processes=$AGENT_WORKER_PROCESSES" \
    "worker_concurrency=$WORKER_CONCURRENCY" \
    "https_worker_concurrency=$HTTPS_WORKER_CONCURRENCY" \
    "ray_server_manifest=$CURRENT_RAY_MANIFEST" \
    "judge_model=$JUDGE_MODEL" \
    "judge_workers=$JUDGE_WORKERS" \
    "insight_qwen_judge_mode=$INSIGHT_QWEN_JUDGE_MODE" \
    "scores_subdir=$SCORES_SUBDIR" \
    "judge_rescore_existing=$JUDGE_RESCORE_EXISTING" \
    "overlap_judge=$OVERLAP_JUDGE"

  echo "[$(date -u +'%F %T')] rollout model=$model_slug val=$val_slug rescale=$rescale output=$output_dir"
  rollout_status=0
  judge_status=0
  judge_pid=""
  if [[ "$OVERLAP_JUDGE" == "1" ]]; then
    echo "[$(date -u +'%F %T')] judge overlap start model=$model_slug val=$val_slug rescale=$rescale"
    run_judge "$output_dir" &
    judge_pid="$!"
    echo "judge_pid=$judge_pid" >> "$output_dir/run.env"
  fi

  run_rollout "$backend" "$model_config" "$val_files_arg" "$rescale" "$output_dir" "$no_tool_override" || rollout_status=$?
  echo "rollout_exit_code=$rollout_status" >> "$output_dir/run.env"

  if [[ "$OVERLAP_JUDGE" == "1" ]]; then
    if [[ "$rollout_status" != "0" ]]; then
      if [[ -n "$judge_pid" ]] && kill -0 "$judge_pid" 2>/dev/null; then
        kill "$judge_pid" 2>/dev/null || true
        wait "$judge_pid" 2>/dev/null || true
      fi
      judge_status=1
      echo "judge_exit_code=$judge_status" >> "$output_dir/run.env"
    elif [[ -n "$judge_pid" ]]; then
      ASYNC_JUDGE_PIDS+=("$judge_pid")
      ASYNC_JUDGE_OUTPUT_DIRS+=("$output_dir")
      ASYNC_JUDGE_LABELS+=("model=$model_slug val=$val_slug rescale=$rescale")
      echo "judge_exit_code=pending" >> "$output_dir/run.env"
    fi
  elif [[ "$DRY_RUN" == "1" || -f "$output_dir/samples.jsonl" || -d "$output_dir/checkpoints" ]]; then
    echo "[$(date -u +'%F %T')] judge model=$model_slug val=$val_slug rescale=$rescale"
    run_judge "$output_dir" || judge_status=$?
  else
    echo "[$(date -u +'%F %T')] skipping judge; rollout produced no samples: $output_dir"
    judge_status=1
  fi
  if [[ "$OVERLAP_JUDGE" != "1" ]]; then
    echo "judge_exit_code=$judge_status" >> "$output_dir/run.env"
  fi

  if [[ "$rollout_status" != "0" || "$judge_status" != "0" ]]; then
    echo "[$(date -u +'%F %T')] eval failed model=$model_slug val=$val_slug rescale=$rescale rollout=$rollout_status judge=$judge_status"
    return 1
  fi
  if [[ "$OVERLAP_JUDGE" == "1" ]]; then
    echo "[$(date -u +'%F %T')] rollout done model=$model_slug val=$val_slug rescale=$rescale judge=pending"
  else
    echo "[$(date -u +'%F %T')] eval done model=$model_slug val=$val_slug rescale=$rescale"
  fi
}

write_sweep_summary() {
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "$SCORES_SUBDIR" "$SUMMARY_PREFIX" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
scores_subdir = sys.argv[2]
summary_prefix = sys.argv[3]
filename_prefix = f"{summary_prefix}_" if summary_prefix else ""

def read_env(path: pathlib.Path) -> dict[str, str]:
    data = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            data[key] = value
    return data

def collect(kind: str) -> list[dict[str, str]]:
    rows = []
    for summary_path in sorted(root.glob(f"*/*/rescale*/{scores_subdir}/{kind}.tsv")):
        eval_dir = summary_path.parents[1]
        run_env = read_env(eval_dir / "run.env")
        with summary_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append({
                    "model": run_env.get("model", ""),
                    "model_slug": run_env.get("model_slug", ""),
                    "backend": run_env.get("backend", ""),
                    "model_config": run_env.get("model_config", ""),
                    "val_files": run_env.get("val_files", ""),
                    "val_slug": run_env.get("val_slug", ""),
                    "rescale": run_env.get("rescale", ""),
                    "num_trials": run_env.get("num_trials", ""),
                    "shuffle_seed": run_env.get("shuffle_seed", ""),
                    "no_tool_no_system_inputs": run_env.get("no_tool_no_system_inputs", ""),
                    "eval_output_dir": str(eval_dir),
                    "scores_subdir": scores_subdir,
                    **row,
                })
    return rows

def write(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

summary_rows = collect("eval_summary")
trial_rows = collect("eval_summary_by_trial")
failure_rows = collect("eval_failure_summary")
write(root / f"{filename_prefix}sweep_summary.tsv", summary_rows)
write(root / f"{filename_prefix}sweep_summary_by_trial.tsv", trial_rows)
write(root / f"{filename_prefix}sweep_failure_summary.tsv", failure_rows)
print(f"wrote {root / f'{filename_prefix}sweep_summary.tsv'} rows={len(summary_rows)}")
print(f"wrote {root / f'{filename_prefix}sweep_summary_by_trial.tsv'} rows={len(trial_rows)}")
print(f"wrote {root / f'{filename_prefix}sweep_failure_summary.tsv'} rows={len(failure_rows)}")
PY
}

mkdir -p "$OUTPUT_ROOT"

if [[ "${#VAL_FILE_ARRAY[@]}" == "0" ]]; then
  echo "VAL_FILES must be set to one or more eval parquet paths." >&2
  echo "Example: VAL_FILES=/path/a.parquet,/path/b.parquet MODEL_CONFIGS=standalone_eval/model_configs/release_ray_vllm.yaml scripts/queue_standalone_full_eval_sweep.sh" >&2
  exit 1
fi

echo "output_root=$OUTPUT_ROOT"
echo "model_configs=${MODEL_CONFIG_ARRAY[*]}"
echo "default_tool_val_files=${VAL_FILE_ARRAY[*]}"
echo "default_no_tool_no_system_val_files=${VAL_FILE_NO_TOOL_ARRAY[*]}"
echo "rescales=${RESCALE_ARRAY[*]}"
echo "num_trials=$NUM_TRIALS"
echo "shuffle_rows=1 shuffle_seed=$SHUFFLE_SEED group_val_files=$GROUP_VAL_FILES"
echo "scores_subdir=$SCORES_SUBDIR summary_prefix=$SUMMARY_PREFIX insight_qwen_judge_mode=$INSIGHT_QWEN_JUDGE_MODE"
echo "api_logger_dir=~/.dumps/api_requests"

validate_api_env

FAILURES=0
for model_config in "${MODEL_CONFIG_ARRAY[@]}"; do
  backend="$(read_model_config_field "$model_config" backend)"
  model_name="$(read_model_config_field "$model_config" model)"
  model_slug="$(slugify "$(basename "$model_config" .yaml)")"
  no_tool_override=0
  if model_uses_no_tool_no_system_inputs "$model_config" "$backend"; then
    no_tool_override=1
  fi

  if [[ "$USER_PROVIDED_VAL_FILES" == "1" ]]; then
    active_val_files=("${VAL_FILE_ARRAY[@]}")
  elif [[ "$no_tool_override" == "1" ]]; then
    active_val_files=("${VAL_FILE_NO_TOOL_ARRAY[@]}")
  else
    active_val_files=("${VAL_FILE_ARRAY[@]}")
  fi
  if [[ "$GROUP_VAL_FILES" == "1" ]]; then
    active_val_args=("$(join_by_comma "${active_val_files[@]}")")
    if [[ "$no_tool_override" == "1" ]]; then
      active_val_slugs=("full5_no_tool_no_system")
    else
      active_val_slugs=("full5_tool")
    fi
  else
    active_val_args=("${active_val_files[@]}")
    active_val_slugs=()
    for val_file in "${active_val_files[@]}"; do
      active_val_slugs+=("$(slugify "$(basename "$val_file" .parquet)")")
    done
  fi

  echo "[$(date -u +'%F %T')] model=$model_slug backend=$backend no_tool_no_system_inputs=$no_tool_override"

  if [[ "$backend" == "ray_vllm" ]]; then
    ray_key="$(model_config_file_sha "$model_config")"
    if [[ -n "$CURRENT_RAY_MANIFEST" && "$CURRENT_RAY_KEY" == "$ray_key" ]]; then
      echo "[$(date -u +'%F %T')] reusing Ray/vLLM server for $model_slug manifest=$CURRENT_RAY_MANIFEST"
    else
      cleanup_current_ray_server
      if ! launch_ray_server_once "$model_config" "$model_slug"; then
        echo "[$(date -u +'%F %T')] failed to launch Ray/vLLM server for $model_slug" >&2
        FAILURES=$((FAILURES + 1))
        if [[ "$STOP_ON_FAILURE" == "1" ]]; then
          exit 1
        fi
        continue
      fi
      CURRENT_RAY_KEY="$ray_key"
    fi
  else
    cleanup_current_ray_server
  fi

  for val_i in "${!active_val_args[@]}"; do
    val_files_arg="${active_val_args[$val_i]}"
    val_slug="${active_val_slugs[$val_i]}"
    for rescale in "${RESCALE_ARRAY[@]}"; do
      if ! run_one_eval "$backend" "$model_config" "$model_name" "$model_slug" "$val_files_arg" "$val_slug" "$rescale" "$no_tool_override"; then
        FAILURES=$((FAILURES + 1))
        if [[ "$STOP_ON_FAILURE" == "1" ]]; then
          exit 1
        fi
      fi
      write_sweep_summary || true
    done
  done
done

if ! wait_for_async_judges; then
  FAILURES=$((FAILURES + 1))
fi

write_sweep_summary

if [[ "$FAILURES" != "0" ]]; then
  echo "[$(date -u +'%F %T')] sweep completed with failures=$FAILURES output_root=$OUTPUT_ROOT"
  exit 1
fi

echo "[$(date -u +'%F %T')] sweep complete output_root=$OUTPUT_ROOT"
