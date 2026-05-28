#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/scratch/ywxzml3j/likaican/src/verl-qwen3-vl")
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "synthetic_unanswerable_pipeline"
INSIGHT_DOC_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc")
O3_ROOT = INSIGHT_DOC_ROOT / "O3_data_0424"
ARXIV_POSTPROCESS_ROOT = INSIGHT_DOC_ROOT / "arxiv_0307_sample" / "qa_gen" / "postprocess"


@dataclass(frozen=True)
class SourceSpec:
    sample_group: str
    source_name: str
    manifest_path: Path
    pdf_image_root: Path
    family: str
    allow_multipart: bool


SOURCE_SPECS: list[SourceSpec] = [
    SourceSpec(
        sample_group="o3",
        source_name="o3_part1",
        manifest_path=O3_ROOT / "0426_selected_train_part1" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
        pdf_image_root=O3_ROOT / "0426_selected_train_part1" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        family="o3",
        allow_multipart=True,
    ),
    SourceSpec(
        sample_group="o3",
        source_name="o3_part2a",
        manifest_path=O3_ROOT / "0426_selected_train_part2a" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
        pdf_image_root=O3_ROOT / "0426_selected_train_part2a" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        family="o3",
        allow_multipart=True,
    ),
    SourceSpec(
        sample_group="o3",
        source_name="o3_part2b",
        manifest_path=O3_ROOT / "0426_selected_train_part2b" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
        pdf_image_root=O3_ROOT / "0426_selected_train_part2b" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        family="o3",
        allow_multipart=True,
    ),
    SourceSpec(
        sample_group="o3",
        source_name="o3_part2c",
        manifest_path=O3_ROOT / "0426_selected_train_part2c" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
        pdf_image_root=O3_ROOT / "0426_selected_train_part2c" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        family="o3",
        allow_multipart=True,
    ),
    SourceSpec(
        sample_group="arxiv_spanning",
        source_name="arxiv_spanning",
        manifest_path=ARXIV_POSTPROCESS_ROOT
        / "spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning"
        / "dpi200_aug_noaug_maxp40"
        / "manifest.jsonl",
        pdf_image_root=ARXIV_POSTPROCESS_ROOT
        / "spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning"
        / "dpi200_aug_noaug_maxp40"
        / "pdf_image",
        family="arxiv_spanning",
        allow_multipart=True,
    ),
    SourceSpec(
        sample_group="arxiv_base",
        source_name="arxiv_base_main",
        manifest_path=ARXIV_POSTPROCESS_ROOT
        / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
        / "dpi200_aug_noaug_maxp40"
        / "manifest.jsonl",
        pdf_image_root=ARXIV_POSTPROCESS_ROOT
        / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
        / "dpi200_aug_noaug_maxp40"
        / "pdf_image",
        family="arxiv_base",
        allow_multipart=False,
    ),
    SourceSpec(
        sample_group="arxiv_base",
        source_name="arxiv_base_additional",
        manifest_path=ARXIV_POSTPROCESS_ROOT
        / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
        / "dpi200_aug_noaug_maxp40_jitter_seed0"
        / "manifest.jsonl",
        pdf_image_root=ARXIV_POSTPROCESS_ROOT
        / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
        / "dpi200_aug_noaug_maxp40_jitter_seed0"
        / "pdf_image",
        family="arxiv_base",
        allow_multipart=False,
    ),
]


def normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def is_not_answerable(row: dict[str, Any]) -> bool:
    qtype = row.get("question_type")
    if qtype == "not-answerable":
        return True
    if isinstance(qtype, (list, tuple, set)):
        return "not-answerable" in qtype
    return False


def looks_like_explicit_multipart(question: str) -> bool:
    question = question.lower()
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    return sum(1 for label in labels if label in question) >= 2


def row_is_eligible(row: dict[str, Any], spec: SourceSpec) -> bool:
    if is_not_answerable(row):
        return False
    images = row.get("images")
    if not isinstance(images, list) or not images:
        return False
    if not spec.allow_multipart and looks_like_explicit_multipart(normalize_text(row.get("question"))):
        return False
    return True


def load_rows(spec: SourceSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with spec.manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row_is_eligible(row, spec):
                rows.append(row)
    return rows


def load_excluded_pairs(path: Path) -> set[tuple[str, str]]:
    excluded: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_name = normalize_text(row.get("source_name"))
            question_id = normalize_text(row.get("question_id"))
            if not source_name or not question_id:
                continue
            excluded.add((source_name, question_id))
    return excluded


def proportional_counts(total_target: int, buckets: dict[str, int]) -> dict[str, int]:
    total_available = sum(buckets.values())
    if total_available < total_target:
        raise ValueError(f"Requested {total_target} rows but only {total_available} are available")
    raw = {key: total_target * value / total_available for key, value in buckets.items()}
    counts = {key: math.floor(value) for key, value in raw.items()}
    remaining = total_target - sum(counts.values())
    remainders = sorted(
        ((raw[key] - counts[key], key) for key in buckets),
        key=lambda item: (-item[0], item[1]),
    )
    for _, key in remainders[:remaining]:
        counts[key] += 1
    return counts


def stable_sample(rows: list[dict[str, Any]], n: int, seed: int, sample_key: str) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{sample_key}")
    indices = list(range(len(rows)))
    selected_indices = sorted(rng.sample(indices, n))
    return [rows[i] for i in selected_indices]


def symlink_document_dirs(sample_dir: Path, spec: SourceSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pdf_root = sample_dir / "pdf_image"
    pdf_root.mkdir(parents=True, exist_ok=True)
    linked_dirs: set[str] = set()
    for row in rows:
        for rel_image in row.get("images") or []:
            rel_dir = str(Path(rel_image).parent)
            if rel_dir in linked_dirs:
                continue
            src_dir = spec.pdf_image_root / rel_dir
            dst_dir = pdf_root / rel_dir
            if not src_dir.exists():
                raise FileNotFoundError(f"Missing source image dir: {src_dir}")
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            if dst_dir.exists() or dst_dir.is_symlink():
                linked_dirs.add(rel_dir)
                continue
            os.symlink(src_dir, dst_dir, target_is_directory=True)
            linked_dirs.add(rel_dir)
    return {
        "pdf_image_root": str(pdf_root),
        "linked_document_dirs": len(linked_dirs),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_question_id_list(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{normalize_text(row.get('question_id'))}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced synthetic-unanswerable seed sample.")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--o3-ratio", type=float, default=0.5)
    parser.add_argument("--arxiv-spanning-frac", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--exclude-original-question-ids-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional original_question_ids.jsonl from a prior sample. "
            "Rows with the same (source_name, question_id) will be excluded before sampling."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ARTIFACTS_ROOT / "balanced_sample5k_seed42",
        help="Output directory for sampled manifests.",
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    excluded_pairs: set[tuple[str, str]] = set()
    if args.exclude_original_question_ids_jsonl is not None:
        excluded_pairs = load_excluded_pairs(args.exclude_original_question_ids_jsonl.expanduser().resolve())

    loaded: dict[str, list[dict[str, Any]]] = {}
    for spec in SOURCE_SPECS:
        rows = load_rows(spec)
        if excluded_pairs:
            rows = [
                row for row in rows
                if (spec.source_name, normalize_text(row.get("question_id"))) not in excluded_pairs
            ]
        loaded[spec.source_name] = rows

    total = args.sample_size
    o3_target = round(total * args.o3_ratio)
    arxiv_target = total - o3_target
    requested_arxiv_spanning_target = round(arxiv_target * args.arxiv_spanning_frac)

    o3_specs = [spec for spec in SOURCE_SPECS if spec.family == "o3"]
    arxiv_base_specs = [spec for spec in SOURCE_SPECS if spec.family == "arxiv_base"]
    arxiv_spanning_spec = next(spec for spec in SOURCE_SPECS if spec.family == "arxiv_spanning")
    available_arxiv_spanning = len(loaded[arxiv_spanning_spec.source_name])
    arxiv_spanning_target = min(requested_arxiv_spanning_target, available_arxiv_spanning)
    arxiv_base_target = arxiv_target - arxiv_spanning_target

    o3_counts = proportional_counts(
        o3_target,
        {spec.source_name: len(loaded[spec.source_name]) for spec in o3_specs},
    )
    arxiv_base_counts = proportional_counts(
        arxiv_base_target,
        {spec.source_name: len(loaded[spec.source_name]) for spec in arxiv_base_specs},
    )
    sampled_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    sampled_rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for spec in o3_specs:
        rows = stable_sample(loaded[spec.source_name], o3_counts[spec.source_name], args.sample_seed, spec.source_name)
        sampled_rows_by_source[spec.source_name] = rows
        sampled_rows_by_group[spec.sample_group].extend(rows)

    for spec in arxiv_base_specs:
        rows = stable_sample(
            loaded[spec.source_name],
            arxiv_base_counts[spec.source_name],
            args.sample_seed,
            spec.source_name,
        )
        sampled_rows_by_source[spec.source_name] = rows
        sampled_rows_by_group[spec.sample_group].extend(rows)

    spanning_rows = stable_sample(
        loaded[arxiv_spanning_spec.source_name],
        arxiv_spanning_target,
        args.sample_seed,
        arxiv_spanning_spec.source_name,
    )
    sampled_rows_by_source[arxiv_spanning_spec.source_name] = spanning_rows
    sampled_rows_by_group[arxiv_spanning_spec.sample_group].extend(spanning_rows)

    overall_qids: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "sample_seed": args.sample_seed,
        "sample_size": total,
        "o3_target": o3_target,
        "arxiv_target": arxiv_target,
        "requested_arxiv_spanning_target": requested_arxiv_spanning_target,
        "arxiv_spanning_target": arxiv_spanning_target,
        "arxiv_base_target": arxiv_base_target,
        "excluded_prior_rows": len(excluded_pairs),
        "sources": {},
    }

    for spec in SOURCE_SPECS:
        rows = sampled_rows_by_source.get(spec.source_name, [])
        sample_dir = output_root / spec.source_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(sample_dir / "manifest.jsonl", rows)
        write_question_id_list(sample_dir / "original_question_ids.txt", rows)
        pdf_info = symlink_document_dirs(sample_dir, spec, rows)
        source_payload = {
            "sample_group": spec.sample_group,
            "source_name": spec.source_name,
            "source_manifest_path": str(spec.manifest_path),
            "source_pdf_image_root": str(spec.pdf_image_root),
            "sample_count": len(rows),
            "sample_seed": args.sample_seed,
            "allow_multipart": spec.allow_multipart,
            **pdf_info,
        }
        write_json(sample_dir / "sources.json", source_payload)
        summary["sources"][spec.source_name] = source_payload
        for row in rows:
            overall_qids.append(
                {
                    "question_id": normalize_text(row.get("question_id")),
                    "sample_group": spec.sample_group,
                    "source_name": spec.source_name,
                    "source_manifest_path": str(spec.manifest_path),
                    "subset": normalize_text(row.get("subset")),
                }
            )

    overall_qids.sort(key=lambda item: (item["sample_group"], item["source_name"], item["question_id"]))
    write_jsonl(output_root / "original_question_ids.jsonl", overall_qids)
    with (output_root / "original_question_ids.txt").open("w", encoding="utf-8") as handle:
        for item in overall_qids:
            handle.write(f"{item['question_id']}\n")

    summary["group_counts"] = {
        key: len(value) for key, value in sorted(sampled_rows_by_group.items())
    }
    summary["total_sampled"] = sum(summary["group_counts"].values())
    summary["question_id_policy"] = (
        "Augmented unanswerable rows will keep the current verified-manifest scheme: "
        "<original_question_id>__mut_unanswerable_<mutation_type>_<candidate_id8>."
    )
    write_json(output_root / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
