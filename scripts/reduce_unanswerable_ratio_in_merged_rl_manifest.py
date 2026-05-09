#!/usr/bin/env python3
"""Reduce unanswerable rows in a merged RL manifest on a per-subset basis.

Policy:
- keep all answerable rows
- for each subset, keep at most floor(answerable_count / denominator) unanswerable rows
  so the resulting unanswerable fraction is <= 1 / (denominator + 1)

This preserves the InSightDocBase layout:

    <output_root>/
      ├── manifest.jsonl
      ├── meta.json
      └── pdf_image -> same symlink target as input
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INPUT_ROOT_DEFAULT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/"
    "_rl_wrong_question_manifests_medium_only_balanced_half_unanswerable_dude_reduced_merged_for_parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT_DEFAULT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=INPUT_ROOT_DEFAULT.parent / f"{INPUT_ROOT_DEFAULT.name}_u25",
    )
    parser.add_argument(
        "--unanswerable-denominator",
        type=int,
        default=3,
        help=(
            "Keep at most floor(answerable_count / denominator) unanswerable rows per subset. "
            "For denominator=3 this yields <=25%% unanswerable."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_unanswerable(row: dict[str, Any]) -> bool:
    return "not-answerable" in json.dumps(row.get("question_type"), ensure_ascii=False).lower()


def tie_break_key(question_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()


def load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_is_unanswerable"] = is_unanswerable(row)
            row["_tie_break_key"] = tie_break_key(str(row["question_id"]), 0)
            rows.append(row)
    return rows


def select_rows(rows: list[dict[str, Any]], denominator: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["_tie_break_key"] = tie_break_key(str(row["question_id"]), seed)
        by_subset[str(row.get("subset", "<missing>"))].append(row)

    selected: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "unanswerable_denominator": denominator,
        "subset_selection": {},
    }

    for subset in sorted(by_subset):
        subset_rows = by_subset[subset]
        answerable_rows = [row for row in subset_rows if not row["_is_unanswerable"]]
        unanswerable_rows = sorted(
            (row for row in subset_rows if row["_is_unanswerable"]),
            key=lambda row: row["_tie_break_key"],
        )
        max_unanswerable = len(answerable_rows) // denominator if denominator > 0 else len(unanswerable_rows)
        chosen_unanswerable = unanswerable_rows[:max_unanswerable]
        chosen = sorted(answerable_rows + chosen_unanswerable, key=lambda row: row["_tie_break_key"])
        selected.extend(chosen)
        summary["subset_selection"][subset] = {
            "before_total": len(subset_rows),
            "before_answerable": len(answerable_rows),
            "before_unanswerable": len(unanswerable_rows),
            "selected_total": len(chosen),
            "selected_answerable": len(answerable_rows),
            "selected_unanswerable": len(chosen_unanswerable),
        }

    selected.sort(key=lambda row: row["_tie_break_key"])
    return selected, summary


def build_summary(selected_rows: list[dict[str, Any]], summary: dict[str, Any], input_root: Path, output_root: Path) -> dict[str, Any]:
    subset_counts = Counter(str(row.get("subset", "<missing>")) for row in selected_rows)
    subset_unanswerable_counts = Counter(
        str(row.get("subset", "<missing>")) for row in selected_rows if row["_is_unanswerable"]
    )
    return {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "rows": len(selected_rows),
        "selected_subset_counts": dict(sorted(subset_counts.items())),
        "selected_unanswerable_subset_counts": dict(sorted(subset_unanswerable_counts.items())),
        "selected_unanswerable_total": sum(subset_unanswerable_counts.values()),
        "policy": summary,
    }


def write_output(input_root: Path, output_root: Path, selected_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            clean_row = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(clean_row, ensure_ascii=False) + "\n")

    input_symlink = input_root / "pdf_image"
    if not input_symlink.is_symlink():
        raise ValueError(f"expected pdf_image symlink at {input_symlink}")
    target = input_symlink.resolve()
    output_symlink = output_root / "pdf_image"
    if output_symlink.exists() or output_symlink.is_symlink():
        if output_symlink.is_symlink():
            output_symlink.unlink()
        else:
            raise FileExistsError(f"refusing to replace real directory at {output_symlink}")
    output_symlink.symlink_to(target)

    meta = build_summary(selected_rows, summary, input_root=input_root, output_root=output_root)
    meta["common_pdf_image_root"] = str(target)
    meta["manifest_path"] = str(manifest_path)
    meta["pdf_image_symlink"] = str(output_symlink)
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.unanswerable_denominator <= 0:
        raise ValueError("--unanswerable-denominator must be positive")

    manifest_path = input_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    rows = load_rows(manifest_path)
    selected_rows, selection_summary = select_rows(
        rows=rows,
        denominator=args.unanswerable_denominator,
        seed=args.seed,
    )
    summary = build_summary(selected_rows, selection_summary, input_root=input_root, output_root=output_root)

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    write_output(input_root=input_root, output_root=output_root, selected_rows=selected_rows, summary=selection_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
