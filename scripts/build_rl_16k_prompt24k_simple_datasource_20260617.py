#!/usr/bin/env python3
"""Build a 16k RL parquet with a 24k prompt cap recovery policy.

Input is the already-deduplicated 16k manifest from the 27k recovery build. For
rows whose original prompt estimate exceeds 24k and whose current/original
rescale is 0.5, this rewrites initial_rescale to 0.35. Data sources are
simplified to {fine_category}_{answerability}, for example
arxiv_veqa_answerable.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from rewrite_rl_manifest_category_datasource_20260616 import answerability, fine_category


REPO_ROOT = Path(__file__).resolve().parents[1]
INSIGHT_DOC_VERL_ROOT = Path("/scratch/ywxzml3j/likaican/src/InSight-doc/verl")
CREATE_PARQUET = INSIGHT_DOC_VERL_ROOT / "recipe/vsearch/create_parquet_dataset.py"

INPUT_MANIFEST = (
    REPO_ROOT
    / "notes/generated/rl_old_new_sft_dedup_prompt27k_recover_over27_r035_category_datasource_20260616/manifest.jsonl"
)
PROMPT_ESTIMATE_CSV = REPO_ROOT / "notes/generated/prompt_cap_11000_dropped_rows_20260601/dropped_rows.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "notes/generated/rl_16k_prompt24k_r05_to_r035_simple_datasource_20260617"
DEFAULT_OUTPUT_PARQUET = DEFAULT_OUTPUT_ROOT / (
    "insight_doc_rl_16k_prompt24k_r05_to_r035_simple_datasource-insight_qwen_agent.parquet"
)

PROMPT_CAP = 24000.0
RESCALE_OVER_CAP_FROM = 0.5
RESCALE_OVER_CAP_TO = 0.35
RESCALE_OVER_CAP_DPI = 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--prompt-estimate-csv", type=Path, default=PROMPT_ESTIMATE_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--validate-images", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rescale_tag(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "runknown"
    if abs(numeric - 0.25) < 1e-6:
        return "r025"
    if abs(numeric - 0.35) < 1e-6:
        return "r035"
    if abs(numeric - 0.5) < 1e-6:
        return "r05"
    return f"r{numeric:g}".replace(".", "")


def load_prompt_estimates(csv_path: Path) -> dict[tuple[str, str], dict[str, float]]:
    estimates: dict[tuple[str, str], dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            key = (str(record["question_id"]), rescale_tag(record["initial_rescale"]))
            estimates[key] = {
                "initial_rescale": float(record["initial_rescale"]),
                "estimated_prompt_tokens": float(record["estimated_prompt_tokens"]),
                "estimated_text_tokens": float(record["estimated_text_tokens"]),
                "estimated_image_tokens": float(record["estimated_image_tokens"]),
            }
    return estimates


def load_over_cap_keys(csv_path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for key, estimate in load_prompt_estimates(csv_path).items():
        if estimate["estimated_prompt_tokens"] <= PROMPT_CAP:
            continue
        if abs(estimate["initial_rescale"] - RESCALE_OVER_CAP_FROM) > 1e-6:
            continue
        keys.add(key)
    return keys


def row_original_rescale(row: dict[str, Any]) -> float:
    value = row.get("original_initial_rescale")
    if value is None:
        value = row.get("initial_rescale_before_prompt_cap_recovery")
    if value is None:
        value = row.get("initial_rescale")
    return float(value)


def row_original_rescale_tag(row: dict[str, Any]) -> str:
    return rescale_tag(row_original_rescale(row))


def simplify_and_recover_row(row: dict[str, Any], over_cap_keys: set[tuple[str, str]]) -> dict[str, Any]:
    out = dict(row)
    original_rescale = row_original_rescale(out)
    original_source = out.get("original_initial_rescale_source")
    if original_source is None:
        original_source = out.get("initial_rescale_source_before_prompt_cap_recovery")
    if original_source is None:
        original_source = out.get("initial_rescale_source")
    original_dpi = out.get("original_initial_rescale_dpi")
    if original_dpi is None:
        original_dpi = out.get("initial_rescale_dpi_before_prompt_cap_recovery")
    if original_dpi is None:
        original_dpi = out.get("initial_rescale_dpi")

    out["original_initial_rescale"] = original_rescale
    out["original_initial_rescale_source"] = original_source
    out["original_initial_rescale_dpi"] = original_dpi

    key = (str(out.get("question_id")), row_original_rescale_tag(out))
    if key in over_cap_keys and abs(original_rescale - RESCALE_OVER_CAP_FROM) < 1e-6:
        if abs(float(out.get("initial_rescale", original_rescale)) - RESCALE_OVER_CAP_TO) > 1e-6:
            out["initial_rescale_before_prompt_cap_recovery"] = out.get("initial_rescale")
            out["initial_rescale_source_before_prompt_cap_recovery"] = out.get("initial_rescale_source")
            out["initial_rescale_dpi_before_prompt_cap_recovery"] = out.get("initial_rescale_dpi")
        out["initial_rescale"] = RESCALE_OVER_CAP_TO
        out["initial_rescale_dpi"] = RESCALE_OVER_CAP_DPI
        out["initial_rescale_source"] = (
            f"{original_source}_promptcap{int(PROMPT_CAP)}_rescaled_to_{rescale_tag(RESCALE_OVER_CAP_TO)}"
        )

    category = fine_category(out)
    ans = answerability(out)
    out["category_data_source"] = category
    out["data_source"] = f"{category}_{ans}"
    return out


def estimate_prompt_tokens_after_recovery(
    row: dict[str, Any], prompt_estimates: dict[tuple[str, str], dict[str, float]]
) -> float | None:
    key = (str(row.get("question_id")), row_original_rescale_tag(row))
    estimate = prompt_estimates.get(key)
    if estimate is None:
        return None
    original_rescale = estimate["initial_rescale"]
    current_rescale = float(row.get("initial_rescale", original_rescale))
    if original_rescale <= 0:
        return estimate["estimated_prompt_tokens"]
    image_scale = (current_rescale / original_rescale) ** 2
    return estimate["estimated_text_tokens"] + estimate["estimated_image_tokens"] * image_scale


def run_create_parquet(data_root: Path, output_parquet: Path, validate_images: bool) -> None:
    cmd = [
        sys.executable,
        str(CREATE_PARQUET),
        "--dataset",
        "InSightDocRL",
        "--data_root",
        str(data_root),
        "--split",
        "all",
        "--prompt",
        "insight_qwen_agent",
        "--output_path",
        str(output_parquet),
        "--agent_name",
        "insight_qwen_agent",
        "--num_workers",
        "1",
        "--extra_options",
        json.dumps({"manifest_file": "manifest.jsonl"}),
    ]
    if validate_images:
        cmd.append("--validate_images")
    subprocess.run(cmd, cwd=INSIGHT_DOC_VERL_ROOT, check=True)


def main() -> int:
    args = parse_args()
    input_manifest = args.input_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_parquet = args.output_parquet.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    prompt_estimates = load_prompt_estimates(args.prompt_estimate_csv.expanduser().resolve())
    over_cap_keys = {
        key
        for key, estimate in prompt_estimates.items()
        if estimate["estimated_prompt_tokens"] > PROMPT_CAP
        and abs(estimate["initial_rescale"] - RESCALE_OVER_CAP_FROM) < 1e-6
    }
    input_rows = read_jsonl(input_manifest)
    recovered_rows = [simplify_and_recover_row(row, over_cap_keys) for row in input_rows]
    residual_over_cap_rows = [
        row
        for row in recovered_rows
        if (estimate_prompt_tokens_after_recovery(row, prompt_estimates) or 0.0) > PROMPT_CAP
    ]
    residual_over_cap_keys = {
        (str(row.get("question_id")), row_original_rescale_tag(row)) for row in residual_over_cap_rows
    }
    output_rows = [
        row
        for row in recovered_rows
        if (str(row.get("question_id")), row_original_rescale_tag(row)) not in residual_over_cap_keys
    ]

    pdf_image = output_root / "pdf_image"
    if not pdf_image.exists():
        pdf_image.symlink_to(Path("/"))
    elif not pdf_image.is_symlink() or pdf_image.resolve() != Path("/"):
        raise FileExistsError(f"refusing to use existing non-root pdf_image path: {pdf_image}")

    output_manifest = output_root / "manifest.jsonl"
    write_jsonl(output_manifest, output_rows)

    changed_rows = [
        row
        for row in output_rows
        if abs(float(row.get("original_initial_rescale", row.get("initial_rescale"))) - RESCALE_OVER_CAP_FROM) < 1e-6
        and abs(float(row.get("initial_rescale")) - RESCALE_OVER_CAP_TO) < 1e-6
    ]
    summary = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "output_parquet": str(output_parquet),
        "prompt_cap": PROMPT_CAP,
        "rescale_over_cap_from": RESCALE_OVER_CAP_FROM,
        "rescale_over_cap_to": RESCALE_OVER_CAP_TO,
        "rows": len(output_rows),
        "input_rows": len(input_rows),
        "over_cap_key_count": len(over_cap_keys),
        "dropped_residual_over_cap_after_recovery": len(residual_over_cap_rows),
        "dropped_residual_over_cap_question_ids": [str(row.get("question_id")) for row in residual_over_cap_rows],
        "rows_with_original_r05_current_r035": len(changed_rows),
        "data_source_counts": dict(Counter(str(row.get("data_source")) for row in output_rows)),
        "category_counts": dict(Counter(str(row.get("category_data_source")) for row in output_rows)),
        "answerability_counts": dict(Counter(answerability(row) for row in output_rows)),
        "current_rescale_counts": dict(Counter(rescale_tag(row.get("initial_rescale")) for row in output_rows)),
        "original_rescale_counts": dict(Counter(row_original_rescale_tag(row) for row in output_rows)),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    run_create_parquet(output_root, output_parquet, args.validate_images)

    table = pq.read_table(output_parquet, columns=["data_source"])
    verify = {
        "parquet_rows": table.num_rows,
        "parquet_data_source_counts": dict(Counter(table.column("data_source").to_pylist())),
    }
    with (output_root / "verify.json").open("w", encoding="utf-8") as handle:
        json.dump(verify, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"summary": summary, "verify": verify}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
