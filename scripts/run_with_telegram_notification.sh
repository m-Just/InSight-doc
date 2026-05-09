#!/usr/bin/env bash

set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_with_telegram_notification.sh [options] -- <command> [args...]

Options:
  --label TEXT          Human-readable workload label. Default: basename of command.
  --chat-id ID          Telegram chat id. Default: 5907075712
  --token-file PATH     File containing the bot token.
                        Default: /scratch/ywxzml3j/likaican/secrets/telegram_bot_token
  --notify-start        Also send a start notification.
  -h, --help            Show this help.

Examples:
  scripts/run_with_telegram_notification.sh \
    --label uncapped-sweep \
    -- bash scripts/launch_longdocurl_mmlongbench_uncapped_initial_rescale_sweep.sh

  scripts/run_with_telegram_notification.sh \
    --chat-id 123456 \
    -- python -u scripts/some_workload.py
EOF
}

CHAT_ID="5907075712"
TOKEN_FILE="/scratch/ywxzml3j/likaican/secrets/telegram_bot_token"
LABEL=""
NOTIFY_START=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      LABEL="${2:?missing value for --label}"
      shift 2
      ;;
    --chat-id)
      CHAT_ID="${2:?missing value for --chat-id}"
      shift 2
      ;;
    --token-file)
      TOKEN_FILE="${2:?missing value for --token-file}"
      shift 2
      ;;
    --notify-start)
      NOTIFY_START=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Missing workload command." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "Telegram token file not found: $TOKEN_FILE" >&2
  exit 2
fi

if [[ -z "$LABEL" ]]; then
  LABEL="$(basename "$1")"
fi

WORKLOAD_CMD=("$@")
HOSTNAME_SHORT="$(hostname)"
START_TS="$(date +%s)"
START_ISO="$(date '+%F %T')"

send_telegram() {
  local raw_message="$1"
  local token
  local encoded_message

  token="$(cat "$TOKEN_FILE")"
  encoded_message="$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$raw_message")"
  curl --connect-timeout 5 -s \
    "https://api.telegram.org/bot${token}/sendMessage?chat_id=${CHAT_ID}&text=${encoded_message}" \
    >/dev/null || {
      echo "Warning: failed to send Telegram notification." >&2
      return 1
    }
}

if [[ "$NOTIFY_START" -eq 1 ]]; then
  start_message="START ${LABEL} host=${HOSTNAME_SHORT} time=${START_ISO}"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    start_message="${start_message} slurm_job_id=${SLURM_JOB_ID}"
  fi
  send_telegram "$start_message" || true
fi

"${WORKLOAD_CMD[@]}"
EXIT_CODE=$?

END_TS="$(date +%s)"
END_ISO="$(date '+%F %T')"
ELAPSED_SEC=$((END_TS - START_TS))

if [[ "$EXIT_CODE" -eq 0 ]]; then
  STATUS="SUCCESS"
else
  STATUS="FAILED"
fi

finish_message="${STATUS} ${LABEL} host=${HOSTNAME_SHORT} exit_code=${EXIT_CODE} elapsed_sec=${ELAPSED_SEC} end=${END_ISO}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  finish_message="${finish_message} slurm_job_id=${SLURM_JOB_ID}"
fi

send_telegram "$finish_message" || true
exit "$EXIT_CODE"
