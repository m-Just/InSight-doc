#!/usr/bin/env python3
"""Merge grouped RL manifests into one manifest usable by create_parquet_dataset.py.

The output layout matches InSightDocBase expectations:

    <output_root>/
      ├── manifest.jsonl
      ├── meta.json
      └── pdf_image -> <deepest common source image root>

Each row's ``images`` field is rewritten to be relative to the symlink target.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


INPUT_ROOT_DEFAULT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/"
    "_rl_wrong_question_manifests_medium_only_balanced_half_unanswerable_dude_reduced"
)
ARXIV_POSTPROCESS_ROOT_DEFAULT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess"
)
O3_ROOT_DEFAULT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT_DEFAULT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=INPUT_ROOT_DEFAULT.parent / f"{INPUT_ROOT_DEFAULT.name}_merged_for_parquet",
    )
    parser.add_argument("--arxiv-postprocess-root", type=Path, default=ARXIV_POSTPROCESS_ROOT_DEFAULT)
    parser.add_argument("--o3-root", type=Path, default=O3_ROOT_DEFAULT)
    parser.add_argument(
        "--manifest-name",
        default="manifest.jsonl",
        help="Name of each grouped manifest file under the input root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def group_to_pdf_image_root(group_name: str, arxiv_root: Path, o3_root: Path) -> Path:
    mapping = {
        "O3_data_0424__0426_selected_train_part1__dpi200_aug_noaug_maxp40": (
            o3_root / "0426_selected_train_part1" / "dpi200_aug_noaug_maxp40" / "pdf_image"
        ),
        "O3_data_0424__0426_selected_train_part2a__dpi200_aug_noaug_maxp40": (
            o3_root / "0426_selected_train_part2a" / "dpi200_aug_noaug_maxp40" / "pdf_image"
        ),
        "O3_data_0424__0426_selected_train_part2b__dpi200_aug_noaug_maxp40": (
            o3_root / "0426_selected_train_part2b" / "dpi200_aug_noaug_maxp40" / "pdf_image"
        ),
        "O3_data_0424__0426_selected_train_part2c__dpi200_aug_noaug_maxp40": (
            o3_root / "0426_selected_train_part2c" / "dpi200_aug_noaug_maxp40" / "pdf_image"
        ),
        "O3_data_0424__dude_poster_unanswerable__dpi200_aug_noaug_maxp40": (
            o3_root / "dude_poster_unanswerable" / "dpi200_aug_noaug_maxp40" / "pdf_image"
        ),
        "arxiv__spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning__dpi200_aug_noaug_maxp40": (
            arxiv_root
            / "spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning"
            / "dpi200_aug_noaug_maxp40"
            / "pdf_image"
        ),
        "arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0": (
            arxiv_root
            / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
            / "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
            / "pdf_image"
        ),
        "arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__dpi200_aug_noaug_maxp40_jitter_seed0": (
            arxiv_root
            / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
            / "dpi200_aug_noaug_maxp40_jitter_seed0"
            / "pdf_image"
        ),
        "arxiv__veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train__dpi200_aug_noaug_maxp40": (
            arxiv_root
            / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
            / "dpi200_aug_noaug_maxp40"
            / "pdf_image"
        ),
    }
    try:
        return mapping[group_name]
    except KeyError as exc:
        raise KeyError(f"unsupported group name: {group_name}") from exc


def iter_group_manifests(input_root: Path, manifest_name: str) -> list[Path]:
    manifests = sorted(path for path in input_root.glob(f"*/{manifest_name}") if path.is_file())
    if not manifests:
        raise FileNotFoundError(f"no grouped manifests named {manifest_name!r} under {input_root}")
    return manifests


def load_merged_rows(
    input_root: Path,
    manifest_name: str,
    arxiv_root: Path,
    o3_root: Path,
) -> tuple[list[dict[str, Any]], list[Path], dict[str, int]]:
    merged_rows: list[dict[str, Any]] = []
    absolute_images: list[Path] = []
    group_counts: dict[str, int] = {}

    for manifest_path in iter_group_manifests(input_root, manifest_name):
        group_name = manifest_path.parent.name
        pdf_image_root = group_to_pdf_image_root(group_name, arxiv_root=arxiv_root, o3_root=o3_root)
        if not pdf_image_root.is_dir():
            raise FileNotFoundError(f"pdf_image root not found for {group_name}: {pdf_image_root}")

        count = 0
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                images = row.get("images")
                if not isinstance(images, list) or not images:
                    raise ValueError(f"row missing images list in {manifest_path}: {row.get('question_id')}")
                row = dict(row)
                row["_group_name"] = group_name
                row["_pdf_image_root"] = str(pdf_image_root)
                abs_paths = []
                for rel in images:
                    abs_path = (pdf_image_root / rel).resolve()
                    if not abs_path.exists():
                        raise FileNotFoundError(
                            f"image file not found for {row.get('question_id')}: {abs_path} "
                            f"(group={group_name}, rel={rel})"
                        )
                    abs_paths.append(abs_path)
                    absolute_images.append(abs_path)
                row["_abs_images"] = abs_paths
                merged_rows.append(row)
                count += 1
        group_counts[group_name] = count
    return merged_rows, absolute_images, group_counts


def deepest_common_root(paths: list[Path]) -> Path:
    if not paths:
        raise ValueError("cannot compute common root for empty path list")
    return Path(os.path.commonpath([str(path) for path in paths]))


def rewrite_rows(rows: list[dict[str, Any]], common_root: Path) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for row in rows:
        out = {key: value for key, value in row.items() if not key.startswith("_")}
        out["images"] = [str(path.relative_to(common_root)) for path in row["_abs_images"]]
        rewritten.append(out)
    return rewritten


def write_output(output_root: Path, rows: list[dict[str, Any]], common_root: Path, group_counts: dict[str, int]) -> None:
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

    meta = {
        "rows": len(rows),
        "common_pdf_image_root": str(common_root),
        "group_counts": dict(sorted(group_counts.items())),
        "manifest_path": str(manifest_path),
        "pdf_image_symlink": str(symlink_path),
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    arxiv_root = args.arxiv_postprocess_root.expanduser().resolve()
    o3_root = args.o3_root.expanduser().resolve()

    rows, absolute_images, group_counts = load_merged_rows(
        input_root=input_root,
        manifest_name=args.manifest_name,
        arxiv_root=arxiv_root,
        o3_root=o3_root,
    )
    common_root = deepest_common_root(absolute_images)
    rewritten_rows = rewrite_rows(rows, common_root=common_root)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "rows": len(rewritten_rows),
        "groups": len(group_counts),
        "common_pdf_image_root": str(common_root),
        "group_counts": dict(sorted(group_counts.items())),
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    write_output(output_root=output_root, rows=rewritten_rows, common_root=common_root, group_counts=group_counts)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
