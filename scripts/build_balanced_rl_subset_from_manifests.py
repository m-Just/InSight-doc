#!/usr/bin/env python3
"""Build a more balanced RL subset from grouped manifest files.

Default policy:
- keep all rows from protected minority subsets such as bigpage_* and docvqa
- for every other subset, cap selected rows at the largest protected subset size
- within each capped subset, prioritize not-answerable rows first, then fill
  remaining slots deterministically

Outputs a new manifest tree grouped by original source manifest directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INPUT_ROOT_DEFAULT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_rl_wrong_question_manifests_medium_only"
)
OUTPUT_ROOT_DEFAULT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_rl_wrong_question_manifests_medium_only_balanced"
)


@dataclass(frozen=True)
class RowRef:
    source_manifest: Path
    row_index: int
    question_id: str
    subset: str
    is_unanswerable: bool
    tie_break_key: str
    row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument(
        "--protected-subset-pattern",
        action="append",
        default=[r"^bigpage_.*$", r"^docvqa$"],
        help="Regex for subsets that should be kept in full. May be passed multiple times.",
    )
    parser.add_argument(
        "--cap-mode",
        choices=("max_protected_count", "manual"),
        default="max_protected_count",
        help="How to set the cap for non-protected subsets.",
    )
    parser.add_argument(
        "--cap-count",
        type=int,
        default=None,
        help="Manual cap per non-protected subset when --cap-mode=manual.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the selection summary without writing outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic tie-breaking among equally prioritized rows.",
    )
    parser.add_argument(
        "--balanced-half-subset",
        action="append",
        default=[],
        help=(
            "Subset that should use an exact half answerable / half unanswerable split, "
            "subject to the non-protected cap. May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--subset-count-override",
        action="append",
        default=[],
        metavar="SUBSET=COUNT",
        help=(
            "Override the selected row count for a specific subset. Useful when one "
            "subset should be smaller than the shared non-protected cap."
        ),
    )
    return parser.parse_args()


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns]


def is_protected_subset(subset: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(subset) for pattern in patterns)


def question_type_contains_not_answerable(question_type: Any) -> bool:
    return "not-answerable" in json.dumps(question_type, ensure_ascii=False).lower()


def stable_tie_break(question_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()


def parse_subset_count_overrides(items: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --subset-count-override value: {item!r}")
        subset, count_str = item.split("=", 1)
        subset = subset.strip()
        if not subset:
            raise ValueError(f"invalid --subset-count-override subset: {item!r}")
        try:
            count = int(count_str)
        except ValueError as exc:
            raise ValueError(f"invalid --subset-count-override count: {item!r}") from exc
        if count <= 0:
            raise ValueError(f"--subset-count-override must be positive: {item!r}")
        overrides[subset] = count
    return overrides


def load_rows(input_root: Path, seed: int) -> tuple[list[RowRef], dict[Path, list[RowRef]]]:
    all_rows: list[RowRef] = []
    by_manifest: dict[Path, list[RowRef]] = {}
    for manifest in sorted(input_root.glob("*/manifest.jsonl")):
        manifest_rows: list[RowRef] = []
        with manifest.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                question_id = str(row["question_id"])
                subset = str(row.get("subset", "<missing>"))
                row_ref = RowRef(
                    source_manifest=manifest,
                    row_index=row_index,
                    question_id=question_id,
                    subset=subset,
                    is_unanswerable=question_type_contains_not_answerable(row.get("question_type")),
                    tie_break_key=stable_tie_break(question_id, seed),
                    row=row,
                )
                manifest_rows.append(row_ref)
                all_rows.append(row_ref)
        by_manifest[manifest] = manifest_rows
    return all_rows, by_manifest


def select_rows(
    rows: list[RowRef],
    protected_patterns: list[re.Pattern[str]],
    cap_mode: str,
    cap_count: int | None,
    balanced_half_subsets: set[str],
    subset_count_overrides: dict[str, int],
) -> tuple[list[RowRef], dict[str, Any]]:
    by_subset: dict[str, list[RowRef]] = defaultdict(list)
    subset_counts = Counter()
    subset_unanswerable_counts = Counter()
    for row in rows:
        by_subset[row.subset].append(row)
        subset_counts[row.subset] += 1
        if row.is_unanswerable:
            subset_unanswerable_counts[row.subset] += 1

    protected_subsets = sorted(subset for subset in by_subset if is_protected_subset(subset, protected_patterns))
    if not protected_subsets:
        raise ValueError("no protected subsets matched; refine --protected-subset-pattern")

    if cap_mode == "max_protected_count":
        non_protected_cap = max(subset_counts[subset] for subset in protected_subsets)
    else:
        if cap_count is None or cap_count <= 0:
            raise ValueError("--cap-count must be positive when --cap-mode=manual")
        non_protected_cap = cap_count

    selected: list[RowRef] = []
    selection_summary: dict[str, Any] = {
        "protected_subsets": protected_subsets,
        "balanced_half_subsets": sorted(balanced_half_subsets),
        "subset_count_overrides": dict(sorted(subset_count_overrides.items())),
        "subset_counts_before": dict(sorted(subset_counts.items())),
        "subset_unanswerable_counts_before": dict(sorted(subset_unanswerable_counts.items())),
        "non_protected_cap": non_protected_cap,
        "subset_selection": {},
    }

    for subset in sorted(by_subset):
        subset_rows = by_subset[subset]
        protected = subset in protected_subsets
        subset_target = subset_count_overrides.get(subset, non_protected_cap)
        if protected:
            chosen = sorted(subset_rows, key=lambda row: row.tie_break_key)
        elif subset in balanced_half_subsets:
            if subset_target % 2 != 0:
                raise ValueError(
                    f"subset target must be even for balanced-half subsets, got {subset_target} for {subset}"
                )
            target_each = subset_target // 2
            unanswerable_rows = sorted(
                (row for row in subset_rows if row.is_unanswerable),
                key=lambda row: row.tie_break_key,
            )
            answerable_rows = sorted(
                (row for row in subset_rows if not row.is_unanswerable),
                key=lambda row: row.tie_break_key,
            )
            if len(unanswerable_rows) < target_each:
                raise ValueError(
                    f"subset {subset} has only {len(unanswerable_rows)} unanswerable rows, needs {target_each}"
                )
            if len(answerable_rows) < target_each:
                raise ValueError(
                    f"subset {subset} has only {len(answerable_rows)} answerable rows, needs {target_each}"
                )
            chosen = sorted(
                unanswerable_rows[:target_each] + answerable_rows[:target_each],
                key=lambda row: row.tie_break_key,
            )
        else:
            ranked = sorted(
                subset_rows,
                key=lambda row: (
                    0 if row.is_unanswerable else 1,
                    row.tie_break_key,
                ),
            )
            chosen = ranked[:subset_target]
        selected.extend(chosen)
        selection_summary["subset_selection"][subset] = {
            "protected": protected,
            "balanced_half": subset in balanced_half_subsets,
            "target": len(subset_rows) if protected else subset_target,
            "before": len(subset_rows),
            "before_unanswerable": sum(1 for row in subset_rows if row.is_unanswerable),
            "selected": len(chosen),
            "selected_unanswerable": sum(1 for row in chosen if row.is_unanswerable),
        }

    selected.sort(key=lambda row: (str(row.source_manifest), row.row_index))
    return selected, selection_summary


def write_outputs(
    input_root: Path, output_root: Path, selected_rows: list[RowRef], selection_summary: dict[str, Any]
) -> dict[str, Any]:
    by_manifest: dict[Path, list[RowRef]] = defaultdict(list)
    for row in selected_rows:
        by_manifest[row.source_manifest].append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_outputs: list[dict[str, Any]] = []
    overall_subset_counts = Counter()
    overall_unanswerable_subset_counts = Counter()

    for input_manifest, rows in sorted(by_manifest.items(), key=lambda item: str(item[0])):
        group_dir = output_root / input_manifest.parent.name
        group_dir.mkdir(parents=True, exist_ok=True)
        output_manifest = group_dir / "manifest.jsonl"
        with output_manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.row, ensure_ascii=False) + "\n")
                overall_subset_counts[row.subset] += 1
                if row.is_unanswerable:
                    overall_unanswerable_subset_counts[row.subset] += 1

        meta = {
            "input_manifest": str(input_manifest),
            "output_manifest": str(output_manifest),
            "rows": len(rows),
            "subset_counts": dict(sorted(Counter(row.subset for row in rows).items())),
            "subset_unanswerable_counts": dict(
                sorted(Counter(row.subset for row in rows if row.is_unanswerable).items())
            ),
        }
        (group_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_outputs.append(meta)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "policy": selection_summary,
        "manifests": manifest_outputs,
        "selected_rows_total": len(selected_rows),
        "selected_subset_counts": dict(sorted(overall_subset_counts.items())),
        "selected_unanswerable_subset_counts": dict(sorted(overall_unanswerable_subset_counts.items())),
        "selected_unanswerable_total": sum(overall_unanswerable_subset_counts.values()),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_summary_preview(
    input_root: Path,
    selected_rows: list[RowRef],
    selection_summary: dict[str, Any],
) -> dict[str, Any]:
    selected_subset_counts = Counter(row.subset for row in selected_rows)
    selected_unanswerable_subset_counts = Counter(row.subset for row in selected_rows if row.is_unanswerable)
    return {
        "input_root": str(input_root),
        "policy": selection_summary,
        "selected_rows_total": len(selected_rows),
        "selected_subset_counts": dict(sorted(selected_subset_counts.items())),
        "selected_unanswerable_subset_counts": dict(sorted(selected_unanswerable_subset_counts.items())),
        "selected_unanswerable_total": sum(selected_unanswerable_subset_counts.values()),
    }


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    patterns = compile_patterns(args.protected_subset_pattern)
    subset_count_overrides = parse_subset_count_overrides(args.subset_count_override)

    rows, _ = load_rows(input_root, seed=args.seed)
    selected_rows, selection_summary = select_rows(
        rows=rows,
        protected_patterns=patterns,
        cap_mode=args.cap_mode,
        cap_count=args.cap_count,
        balanced_half_subsets=set(args.balanced_half_subset),
        subset_count_overrides=subset_count_overrides,
    )

    if args.dry_run:
        print(json.dumps(build_summary_preview(input_root, selected_rows, selection_summary), ensure_ascii=False, indent=2))
        return 0

    summary = write_outputs(input_root, output_root, selected_rows, selection_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
