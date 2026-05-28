#!/usr/bin/env python3
"""Build one RL parquet from filtered answerable + synthetic-unanswerable rows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ANSWERABLE_ROOT = REPO_ROOT / "notes/generated/answerable_rl_afterfiltering_manifest_mapping_20260519"
UNANSWERABLE_ROOT = REPO_ROOT / "notes/generated/synthetic_unanswerable_vr2_wrong_rl_20260519"
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/ywxzml3j/likaican/data/insight_doc/"
    "answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale"
)
DEFAULT_OUTPUT_PARQUET = Path(
    "/scratch/ywxzml3j/likaican/data/insight_doc/"
    "answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale-insight_qwen_agent.parquet"
)

ANSWERABLE_0518_INITIAL_RESCALE = 0.25
ANSWERABLE_0519_RESCALING_BY_LOGICAL_PART = {
    "arxiv/train_part4": 0.35,
    "O3_data_0424/train_part3b": 0.35,
    "O3_data_0424/train_part3d": 0.35,
    "arxiv/train_part5": 0.5,
    "O3_data_0424/train_part3a": 0.5,
    "O3_data_0424/train_part3c": 0.5,
}
SYNTHETIC_UNANSWERABLE_INITIAL_RESCALE_BY_SPLIT = {
    "rescale025": 0.25,
    "rescale035": 0.35,
    "rescale05": 0.5,
}
SYNTHETIC_UNANSWERABLE_DPI_BY_SPLIT = {
    "rescale025": 50,
    "rescale035": 70,
    "rescale05": 100,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    duplicate_group = parser.add_mutually_exclusive_group()
    duplicate_group.add_argument("--keep-answerable-duplicates", dest="keep_answerable_duplicates", action="store_true")
    duplicate_group.add_argument("--drop-answerable-duplicates", dest="keep_answerable_duplicates", action="store_false")
    parser.set_defaults(keep_answerable_duplicates=True)
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


def pdf_image_root_from_manifest_path(manifest_path: str | Path) -> Path:
    return Path(manifest_path).expanduser().resolve().parent / "pdf_image"


def normalize_manifest_row(row: dict[str, Any], *, abs_image_root: Path, data_source: str) -> dict[str, Any]:
    out = dict(row)
    out["data_source"] = data_source
    images = out.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"row missing images: {out.get('question_id')}")
    abs_images: list[str] = []
    for image in images:
        image_path = Path(str(image))
        if not image_path.is_absolute():
            image_path = abs_image_root / image_path
        image_path = image_path.resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"image for {out.get('question_id')} does not exist: {image_path}")
        abs_images.append(str(image_path))
    out["images"] = abs_images
    return out


def set_initial_rescale(row: dict[str, Any], *, initial_rescale: float, source: str, dpi: int | None = None) -> None:
    row["initial_rescale"] = float(initial_rescale)
    row["initial_rescale_source"] = source
    if dpi is not None:
        row["initial_rescale_dpi"] = int(dpi)


def infer_answerable_initial_rescale(record: dict[str, Any], path: Path) -> tuple[float, str, int | None]:
    selected = record.get("selected_match") or {}
    batch = selected.get("batch")
    logical_part = selected.get("logical_part")
    if batch == "0518" or path.name == "answerable_RL_afterfiltering0518_mapped.jsonl":
        return ANSWERABLE_0518_INITIAL_RESCALE, "answerable_0518", 50
    if batch == "0519" or path.name == "answerable_RL_afterfiltering0519_mapped.jsonl":
        if logical_part not in ANSWERABLE_0519_RESCALING_BY_LOGICAL_PART:
            raise ValueError(f"cannot infer 0519 initial_rescale for logical_part={logical_part!r}")
        initial_rescale = ANSWERABLE_0519_RESCALING_BY_LOGICAL_PART[logical_part]
        dpi = 70 if initial_rescale == 0.35 else 100
        return initial_rescale, f"answerable_0519_{logical_part}", dpi
    raise ValueError(f"cannot infer answerable initial_rescale for {record.get('question_id')}")


def load_answerable_rows(keep_duplicates: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = [
        ANSWERABLE_ROOT / "answerable_RL_afterfiltering0518_mapped.jsonl",
        ANSWERABLE_ROOT / "answerable_RL_afterfiltering0519_mapped.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_question_ids: list[str] = []

    for path in paths:
        for record in read_jsonl(path):
            qid = str(record["question_id"])
            if qid in seen and not keep_duplicates:
                duplicate_question_ids.append(qid)
                continue
            seen.add(qid)
            selected = record.get("selected_match") or {}
            manifest_path = selected.get("manifest_path")
            if not isinstance(manifest_path, str):
                raise ValueError(f"missing selected manifest path for answerable {qid}")
            row = normalize_manifest_row(
                record["row"],
                abs_image_root=pdf_image_root_from_manifest_path(manifest_path),
                data_source="insight_doc_rl_answerable",
            )
            initial_rescale, source, dpi = infer_answerable_initial_rescale(record, path)
            set_initial_rescale(row, initial_rescale=initial_rescale, source=source, dpi=dpi)
            rows.append(row)

    return rows, {
        "input_files": [str(path) for path in paths],
        "rows": len(rows),
        "duplicate_question_ids_dropped": len(duplicate_question_ids),
        "duplicate_question_ids_dropped_file": None,
    }


def load_synthetic_unanswerable_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = UNANSWERABLE_ROOT / "combined/mapped.jsonl"
    rows: list[dict[str, Any]] = []
    for record in read_jsonl(path):
        rescale_split = str(record.get("rescale"))
        if rescale_split not in SYNTHETIC_UNANSWERABLE_INITIAL_RESCALE_BY_SPLIT:
            raise ValueError(f"cannot infer synthetic-unanswerable initial_rescale for {rescale_split!r}")
        row = normalize_manifest_row(
            record["row"],
            abs_image_root=pdf_image_root_from_manifest_path(record["source_manifest_path"]),
            data_source="insight_doc_rl_synthetic_unanswerable_vr2_wrong",
        )
        set_initial_rescale(
            row,
            initial_rescale=SYNTHETIC_UNANSWERABLE_INITIAL_RESCALE_BY_SPLIT[rescale_split],
            source=f"synthetic_unanswerable_{rescale_split}",
            dpi=SYNTHETIC_UNANSWERABLE_DPI_BY_SPLIT[rescale_split],
        )
        rows.append(row)
    return rows, {"input_file": str(path), "rows": len(rows)}


def run_create_parquet(data_root: Path, output_parquet: Path, validate_images: bool) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "recipe/vsearch/create_parquet_dataset.py"),
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
        "8",
        "--extra_options",
        json.dumps({"manifest_file": "manifest.jsonl"}),
    ]
    if validate_images:
        cmd.append("--validate_images")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_parquet = args.output_parquet.expanduser().resolve()

    answerable_rows, answerable_summary = load_answerable_rows(args.keep_answerable_duplicates)
    unanswerable_rows, unanswerable_summary = load_synthetic_unanswerable_rows()
    all_rows = answerable_rows + unanswerable_rows
    qid_counts = Counter(str(row.get("question_id")) for row in all_rows)
    duplicate_after_join = {qid: count for qid, count in qid_counts.items() if count > 1}

    output_root.mkdir(parents=True, exist_ok=True)
    # Absolute image paths are intentional here because the joined rows span
    # both /home and /scratch source trees.
    pdf_image = output_root / "pdf_image"
    if pdf_image.exists() or pdf_image.is_symlink():
        if not pdf_image.is_symlink() or pdf_image.resolve() != Path("/"):
            raise FileExistsError(f"refusing to replace existing pdf_image: {pdf_image}")
    else:
        pdf_image.symlink_to(Path("/"))

    manifest_path = output_root / "manifest.jsonl"
    write_jsonl(manifest_path, all_rows)
    run_create_parquet(output_root, output_parquet, args.validate_images)

    summary = {
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "pdf_image_symlink": str(pdf_image),
        "output_parquet": str(output_parquet),
        "prompt": "insight_qwen_agent",
        "agent_name": "insight_qwen_agent",
        "answerable": answerable_summary,
        "synthetic_unanswerable_vr2_wrong": unanswerable_summary,
        "total_manifest_rows": len(all_rows),
        "unique_question_ids": len(qid_counts),
        "duplicate_question_ids_after_join": duplicate_after_join,
        "data_source_counts": dict(Counter(str(row.get("data_source")) for row in all_rows)),
        "initial_rescale_counts": dict(Counter(str(row.get("initial_rescale")) for row in all_rows)),
        "initial_rescale_source_counts": dict(Counter(str(row.get("initial_rescale_source")) for row in all_rows)),
        "initial_rescale_dpi_counts": dict(Counter(str(row.get("initial_rescale_dpi")) for row in all_rows)),
        "subset_counts": dict(Counter(str(row.get("subset")) for row in all_rows)),
        "question_type_counts": dict(Counter(str(row.get("question_type")) for row in all_rows)),
        "keep_answerable_duplicates": args.keep_answerable_duplicates,
    }
    if not args.keep_answerable_duplicates and answerable_summary["duplicate_question_ids_dropped"]:
        dropped_path = output_root / "answerable_duplicate_question_ids_dropped.txt"
        # Recompute only for writing the exact ids in original order.
        seen: set[str] = set()
        dropped: list[str] = []
        for path in [
            ANSWERABLE_ROOT / "answerable_RL_afterfiltering0518_mapped.jsonl",
            ANSWERABLE_ROOT / "answerable_RL_afterfiltering0519_mapped.jsonl",
        ]:
            for record in read_jsonl(path):
                qid = str(record["question_id"])
                if qid in seen:
                    dropped.append(qid)
                seen.add(qid)
        dropped_path.write_text("".join(f"{qid}\n" for qid in dropped), encoding="utf-8")
        summary["answerable"]["duplicate_question_ids_dropped_file"] = str(dropped_path)

    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
