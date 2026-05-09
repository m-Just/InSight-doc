#!/usr/bin/env python3
"""Build one merged manifest from generated/*/medium/processed/wrong_question_ids.txt.

The output layout matches create_parquet_dataset.py expectations:

    <output_root>/
      ├── manifest.jsonl
      ├── meta.json
      └── pdf_image -> <deepest common source image root>

Each merged row is retrieved from the original source manifest and its ``images``
paths are rewritten relative to the shared ``pdf_image`` symlink target.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GENERATED_ROOT_DEFAULT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated")
ARXIV_POSTPROCESS_ROOT_DEFAULT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess"
)
O3_ROOT_DEFAULT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424")
OUTPUT_ROOT_DEFAULT = GENERATED_ROOT_DEFAULT / "_medium_processed_wrong_question_manifests_merged_for_parquet"


@dataclass(frozen=True)
class DatasetManifestSpec:
    manifest_path: Path
    pdf_image_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=Path, default=GENERATED_ROOT_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--arxiv-postprocess-root", type=Path, default=ARXIV_POSTPROCESS_ROOT_DEFAULT)
    parser.add_argument("--o3-root", type=Path, default=O3_ROOT_DEFAULT)
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium"),
        default="medium",
        help="Which generated/*/<difficulty>/processed/wrong_question_ids.txt trees to scan.",
    )
    parser.add_argument(
        "--manifest-subpath",
        default="processed/wrong_question_ids.txt",
        help="Relative path under each dataset difficulty root to scan.",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Skip source processed roots whose generated-relative path contains this substring.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dataset_specs(arxiv_postprocess_root: Path, o3_root: Path) -> dict[tuple[str, str], DatasetManifestSpec]:
    return {
        ("O3_data_0424", "train_part1"): DatasetManifestSpec(
            manifest_path=o3_root / "0426_selected_train_part1" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            pdf_image_root=o3_root / "0426_selected_train_part1" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        ),
        ("O3_data_0424", "train_part2a"): DatasetManifestSpec(
            manifest_path=o3_root / "0426_selected_train_part2a" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            pdf_image_root=o3_root / "0426_selected_train_part2a" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        ),
        ("O3_data_0424", "train_part2b"): DatasetManifestSpec(
            manifest_path=o3_root / "0426_selected_train_part2b" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            pdf_image_root=o3_root / "0426_selected_train_part2b" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        ),
        ("O3_data_0424", "train_part2c"): DatasetManifestSpec(
            manifest_path=o3_root / "0426_selected_train_part2c" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            pdf_image_root=o3_root / "0426_selected_train_part2c" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        ),
        ("O3_data_0424", "dude_poster_unanswerable"): DatasetManifestSpec(
            manifest_path=o3_root / "dude_poster_unanswerable" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            pdf_image_root=o3_root / "dude_poster_unanswerable" / "dpi200_aug_noaug_maxp40" / "pdf_image",
        ),
        ("arxiv", "train_part1"): DatasetManifestSpec(
            manifest_path=arxiv_postprocess_root
            / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
            / "dpi200_aug_noaug_maxp40"
            / "manifest.jsonl",
            pdf_image_root=arxiv_postprocess_root
            / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
            / "dpi200_aug_noaug_maxp40"
            / "pdf_image",
        ),
        ("arxiv", "train_part2"): DatasetManifestSpec(
            manifest_path=arxiv_postprocess_root
            / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
            / "dpi200_aug_noaug_maxp40"
            / "manifest.jsonl",
            pdf_image_root=arxiv_postprocess_root
            / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
            / "dpi200_aug_noaug_maxp40"
            / "pdf_image",
        ),
        ("arxiv", "train_part3"): DatasetManifestSpec(
            manifest_path=arxiv_postprocess_root
            / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
            / "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
            / "manifest.jsonl",
            pdf_image_root=arxiv_postprocess_root
            / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
            / "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
            / "pdf_image",
        ),
        ("arxiv", "spanning_train_part1"): DatasetManifestSpec(
            manifest_path=arxiv_postprocess_root
            / "spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning"
            / "dpi200_aug_noaug_maxp40"
            / "manifest.jsonl",
            pdf_image_root=arxiv_postprocess_root
            / "spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning"
            / "dpi200_aug_noaug_maxp40"
            / "pdf_image",
        ),
    }


def iter_wrong_id_files(generated_root: Path, difficulty: str, subpath: str, exclude_patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    pattern = f"**/{difficulty}/{subpath}"
    for path in sorted(generated_root.glob(pattern)):
        rel = str(path.relative_to(generated_root))
        if any(pattern and pattern in rel for pattern in exclude_patterns):
            continue
        paths.append(path)
    return paths


def load_wrong_ids(path: Path) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                out.append(value)
    return out


def question_id_from_manifest_row(row: dict[str, Any]) -> str:
    qid = row.get("question_id")
    if isinstance(qid, str) and qid:
        return qid
    raise ValueError(f"manifest row missing string question_id: {row}")


def deepest_common_root(paths: list[Path]) -> Path:
    if not paths:
        raise ValueError("cannot compute common root for empty image path list")
    return Path(os.path.commonpath([str(path) for path in paths]))


def write_output(
    output_root: Path,
    rows: list[dict[str, Any]],
    common_root: Path,
    summary: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    symlink_path = output_root / "pdf_image"
    if symlink_path.exists() or symlink_path.is_symlink():
        if symlink_path.is_symlink() and symlink_path.resolve() == common_root.resolve():
            pass
        else:
            if symlink_path.is_dir() and not symlink_path.is_symlink():
                raise FileExistsError(f"refusing to replace real directory at {symlink_path}")
            symlink_path.unlink()
            symlink_path.symlink_to(common_root)
    else:
        symlink_path.symlink_to(common_root)

    meta = dict(summary)
    meta["manifest_path"] = str(manifest_path)
    meta["pdf_image_symlink"] = str(symlink_path)
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    generated_root = args.generated_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    specs = dataset_specs(
        arxiv_postprocess_root=args.arxiv_postprocess_root.expanduser().resolve(),
        o3_root=args.o3_root.expanduser().resolve(),
    )

    wrong_id_files = iter_wrong_id_files(
        generated_root=generated_root,
        difficulty=args.difficulty,
        subpath=args.manifest_subpath,
        exclude_patterns=args.exclude_pattern,
    )
    if not wrong_id_files:
        raise SystemExit(f"no wrong_question_ids.txt files found under {generated_root} for difficulty={args.difficulty}")

    wanted_by_dataset: dict[tuple[str, str], set[str]] = {}
    source_counts: dict[str, int] = {}
    total_input_ids = 0
    for wrong_id_path in wrong_id_files:
        rel_parts = wrong_id_path.relative_to(generated_root).parts
        if len(rel_parts) < 4:
            raise SystemExit(f"unexpected wrong ids path shape: {wrong_id_path}")
        dataset_key = (rel_parts[0], rel_parts[1])
        if dataset_key not in specs:
            raise SystemExit(f"no manifest mapping for {wrong_id_path}")
        ids = load_wrong_ids(wrong_id_path)
        total_input_ids += len(ids)
        wanted_by_dataset.setdefault(dataset_key, set()).update(ids)
        source_counts[str(wrong_id_path)] = len(ids)

    merged_rows: list[dict[str, Any]] = []
    absolute_images: list[Path] = []
    matched_counts: dict[str, int] = {}
    missing_by_source: dict[str, list[str]] = {}
    global_seen_qids: set[str] = set()

    for dataset_key in sorted(wanted_by_dataset):
        spec = specs[dataset_key]
        wanted = wanted_by_dataset[dataset_key]
        matched = 0
        seen_local: set[str] = set()
        with spec.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = question_id_from_manifest_row(row)
                if qid not in wanted or qid in global_seen_qids:
                    continue
                images = row.get("images")
                if not isinstance(images, list) or not images:
                    raise ValueError(f"row missing images for {qid} in {spec.manifest_path}")
                abs_paths: list[Path] = []
                for rel in images:
                    abs_path = (spec.pdf_image_root / rel).resolve()
                    if not abs_path.exists():
                        raise FileNotFoundError(
                            f"image file not found for {qid}: {abs_path} "
                            f"(dataset={dataset_key}, rel={rel})"
                        )
                    abs_paths.append(abs_path)
                    absolute_images.append(abs_path)
                out_row = dict(row)
                out_row["images"] = abs_paths
                merged_rows.append(out_row)
                matched += 1
                seen_local.add(qid)
                global_seen_qids.add(qid)
        matched_counts[f"{dataset_key[0]}/{dataset_key[1]}"] = matched
        missing = sorted(wanted - seen_local)
        if missing:
            missing_by_source[f"{dataset_key[0]}/{dataset_key[1]}"] = missing

    common_root = deepest_common_root(absolute_images)
    rewritten_rows: list[dict[str, Any]] = []
    for row in merged_rows:
        out = dict(row)
        out["images"] = [str(path.relative_to(common_root)) for path in row["images"]]
        rewritten_rows.append(out)

    summary = {
        "generated_root": str(generated_root),
        "output_root": str(output_root),
        "difficulty": args.difficulty,
        "wrong_id_files": [str(path) for path in wrong_id_files],
        "source_counts": source_counts,
        "matched_counts": matched_counts,
        "rows": len(rewritten_rows),
        "input_wrong_question_ids_total": total_input_ids,
        "unique_question_ids_total": len(global_seen_qids),
        "missing_question_ids_total": sum(len(v) for v in missing_by_source.values()),
        "missing_question_ids_by_dataset": {k: len(v) for k, v in missing_by_source.items()},
        "common_pdf_image_root": str(common_root),
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    write_output(output_root=output_root, rows=rewritten_rows, common_root=common_root, summary=summary)

    missing_path = output_root / "missing_question_ids.json"
    missing_path.write_text(json.dumps(missing_by_source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
