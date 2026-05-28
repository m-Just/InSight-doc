#!/bin/bash
set -euo pipefail

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "OPENAI_BASE_URL is not set" >&2
  exit 1
fi

cd /scratch/ywxzml3j/likaican/src/verl-qwen3-vl

SAMPLE_ROOT="${SAMPLE_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/artifacts/synthetic_unanswerable_pipeline/balanced_sample5k_seed42}"
RUN_ROOT="${RUN_ROOT:-${SAMPLE_ROOT}_run1}"

GEN_MODEL="${GEN_MODEL:-gpt-5-nano}"
VERIFY_MODEL="${VERIFY_MODEL:-gemini-3.1-flash-lite-preview}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-8}"
VERIFY_CONCURRENCY="${VERIFY_CONCURRENCY:-8}"
NUM_CANDIDATES="${NUM_CANDIDATES:-1}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

INSIGHT_DOC_ROOT="${INSIGHT_DOC_ROOT:-/scratch/ywxzml3j/likaican/src/InSight-doc}"
O3_FINAL_OUTPUT_JSON="${O3_FINAL_OUTPUT_JSON:-$INSIGHT_DOC_ROOT/data/final_output_o3_data_mixed.json}"

RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_VERIFY="${RUN_VERIFY:-1}"

MERGED_CANDIDATES_JSONL="$RUN_ROOT/merged_candidates.jsonl"
VERIFY_DIR="$RUN_ROOT/verify_all"

SOURCE_DIRS=(
  "o3_part1"
  "o3_part2a"
  "o3_part2b"
  "o3_part2c"
  "arxiv_spanning"
  "arxiv_base_main"
  "arxiv_base_additional"
)

mkdir -p "$RUN_ROOT"

echo "SAMPLE_ROOT: $SAMPLE_ROOT"
echo "RUN_ROOT: $RUN_ROOT"
echo "GEN_MODEL: $GEN_MODEL"
echo "VERIFY_MODEL: $VERIFY_MODEL"
echo "NUM_CANDIDATES: $NUM_CANDIDATES"
echo "SAMPLE_SEED: $SAMPLE_SEED"

if [[ "$RUN_GENERATION" == "1" ]]; then
  for source_name in "${SOURCE_DIRS[@]}"; do
    manifest_path="$SAMPLE_ROOT/$source_name/manifest.jsonl"
    output_dir="$RUN_ROOT/${source_name}_gen"
    if [[ ! -f "$manifest_path" ]]; then
      echo "missing manifest: $manifest_path" >&2
      exit 1
    fi
    echo "== Generation: $source_name =="
    python ./scripts/generate_unanswerable_question_candidates_with_api.py \
      --manifest "$manifest_path" \
      --output-dir "$output_dir" \
      --model "$GEN_MODEL" \
      --concurrency "$GEN_CONCURRENCY" \
      --num-candidates "$NUM_CANDIDATES" \
      --sample-seed "$SAMPLE_SEED" \
      --insight-doc-root "$INSIGHT_DOC_ROOT" \
      --o3-final-output-json "$O3_FINAL_OUTPUT_JSON"
  done

  python - "$RUN_ROOT" "$MERGED_CANDIDATES_JSONL" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
output_path = Path(sys.argv[2])

source_dirs = [
    "o3_part1_gen",
    "o3_part2a_gen",
    "o3_part2b_gen",
    "o3_part2c_gen",
    "arxiv_spanning_gen",
    "arxiv_base_main_gen",
    "arxiv_base_additional_gen",
]

rows = []
for source_dir in source_dirs:
    path = run_root / source_dir / "candidates.jsonl"
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"merged_candidates={len(rows)}")
PY
fi

if [[ "$RUN_VERIFY" == "1" ]]; then
  if [[ ! -f "$MERGED_CANDIDATES_JSONL" ]]; then
    echo "missing merged candidates: $MERGED_CANDIDATES_JSONL" >&2
    exit 1
  fi
  echo "== Verification =="
  python ./scripts/verify_unanswerable_question_candidates_with_api.py \
    --candidates-jsonl "$MERGED_CANDIDATES_JSONL" \
    --output-dir "$VERIFY_DIR" \
    --model "$VERIFY_MODEL" \
    --concurrency "$VERIFY_CONCURRENCY" \
    --sample-seed "$SAMPLE_SEED" \
    --insight-doc-root "$INSIGHT_DOC_ROOT"
fi

echo "Done."
