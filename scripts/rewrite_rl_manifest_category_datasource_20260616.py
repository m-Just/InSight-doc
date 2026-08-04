#!/usr/bin/env python3
"""Rewrite RL manifest data_source to category_answerability_rescale.

This is a fast post-processing step for manifests already produced by
build_rl_old_new_sft_dedup_prompt27k_parquet_20260616.py. It avoids redoing the
old/new/SFT dedup scan and only rewrites data_source before parquet conversion.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
INSIGHT_DOC_VERL_ROOT = Path("/scratch/ywxzml3j/likaican/src/InSight-doc/verl")
CREATE_PARQUET = INSIGHT_DOC_VERL_ROOT / "recipe/vsearch/create_parquet_dataset.py"

DEFAULT_INPUT_MANIFEST = (
    REPO_ROOT
    / "notes/generated/rl_old_new_sft_dedup_prompt27k_recover_over27_r035_20260616/manifest.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "notes/generated/rl_old_new_sft_dedup_prompt27k_recover_over27_r035_category_datasource_20260616"
)
DEFAULT_OUTPUT_PARQUET = DEFAULT_OUTPUT_ROOT / (
    "insight_doc_rl_old_new_sft_dedup_prompt27k_recover_over27_r035_category_datasource"
    "-insight_qwen_agent.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
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
    return "r" + re.sub(r"[^0-9]+", "", f"{numeric:.6g}")


def answerability(row: dict[str, Any]) -> str:
    if row.get("is_unanswerable"):
        return "unanswerable"
    data_source = str(row.get("data_source") or "").lower()
    answer = str(row.get("answer") or "").lower()
    question_type = json.dumps(row.get("question_type"), ensure_ascii=False).lower()
    if "unanswerable" in data_source or "unanswerable" in answer or "unanswerable" in question_type:
        return "unanswerable"
    return "answerable"


def coarse_bucket(row: dict[str, Any]) -> str:
    bucket = str(row.get("source_bucket") or "").lower()
    if bucket:
        return bucket
    text = " ".join(
        [
            str(row.get("subset") or ""),
            str(row.get("document_id") or ""),
            str(row.get("question_id") or ""),
        ]
    ).lower()
    if "arxiv" in text or row.get("subset") in {"veqa", "mveqa"}:
        return "arxiv"
    if "map" in text or "metromap" in text or "travelmap" in text:
        return "map"
    if "poster" in text:
        return "poster"
    if "info" in text:
        return "info"
    if "dude" in text:
        return "dude"
    if "docvqa" in text:
        return "docvqa"
    if "mpdocvqa" in text:
        return "mpdocvqa"
    return "unknown"


def fine_category(row: dict[str, Any]) -> str:
    bucket = coarse_bucket(row)
    subset = str(row.get("subset") or "").lower()
    document_id = str(row.get("document_id") or "").lower()
    question_id = str(row.get("question_id") or "").lower()
    text = " ".join([subset, document_id, question_id])

    if bucket == "arxiv":
        if subset in {"veqa", "mveqa"}:
            return f"arxiv_{subset}"
        if "mveqa" in text or "_mvqa" in text:
            return "arxiv_mveqa"
        if "veqa" in text or "_qa" in text:
            return "arxiv_veqa"
        return "arxiv_unknown"

    if bucket == "map":
        if "metromap" in text or "metro" in text:
            return "map_metro"
        if "travelmap" in text or "travel" in text:
            return "map_travel"
        return "map_unknown"

    return bucket


def set_category_data_source(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    category = fine_category(out)
    ans = answerability(out)
    tag = rescale_tag(out.get("initial_rescale"))
    out["category_data_source"] = category
    out["data_source"] = f"{category}_{ans}_{tag}"
    return out


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

    rows = [set_category_data_source(row) for row in read_jsonl(input_manifest)]

    pdf_image = output_root / "pdf_image"
    if not pdf_image.exists():
        pdf_image.symlink_to(Path("/"))
    elif not pdf_image.is_symlink() or pdf_image.resolve() != Path("/"):
        raise FileExistsError(f"refusing to use existing non-root pdf_image path: {pdf_image}")

    output_manifest = output_root / "manifest.jsonl"
    write_jsonl(output_manifest, rows)

    summary = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest),
        "output_parquet": str(output_parquet),
        "rows": len(rows),
        "data_source_counts": dict(Counter(str(row.get("data_source")) for row in rows)),
        "category_counts": dict(Counter(str(row.get("category_data_source")) for row in rows)),
        "answerability_counts": dict(Counter(answerability(row) for row in rows)),
        "rescale_counts": dict(Counter(rescale_tag(row.get("initial_rescale")) for row in rows)),
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
