#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

export REPO_ROOT
export RUN_ID
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/workspace/rl_ckpt700_highpage_standalone_rescale025_035_05_3trials_wc4_overlap_${RUN_ID}}"
export VAL_FILES="${VAL_FILES:-[\"/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/longdocurl_highpage_0507-insight_qwen_agent.parquet\",\"/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/mmlongbench_highpage_0507-insight_qwen_agent.parquet\"]}"

exec "$REPO_ROOT/scripts/queue_standalone_rl_ckpt700_broad_rescale_sweep_3trials_wc4_overlap_20260608.sh"
