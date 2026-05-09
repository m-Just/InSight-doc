#!/usr/bin/env python3
"""Split a merged manifest into 70-DPI and 100-DPI parts.

Policy:
- estimate which rows would exceed the 100-DPI vreasoner_v2 prompt budget
  using the repo's image-token proxy at ``initial_rescale=0.5`` plus a
  conservative text upper bound and fixed slack
- force all such rows into the low-DPI (70-DPI) part
- fill the rest of the low-DPI part up to a target ratio (default 60%)
  using a deterministic hash of question_id
- assign the remainder to the high-DPI (100-DPI) part

The output layout preserves the create_parquet_dataset.py expectations:

    <output_root_low>/
      ├── manifest.jsonl
      ├── meta.json
      └── pdf_image -> same symlink target as input

    <output_root_high>/
      ├── manifest.jsonl
      ├── meta.json
      └── pdf_image -> same symlink target as input
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


INPUT_ROOT_DEFAULT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_medium_processed_wrong_question_manifests_merged_for_parquet"
)
INSIGHT_DOC_CREATE_PARQUET_PATH = Path(
    "/scratch/ywxzml3j/likaican/src/InSight-doc/verl/recipe/vsearch/create_parquet_dataset.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT_DEFAULT)
    parser.add_argument(
        "--output-root-low",
        type=Path,
        default=INPUT_ROOT_DEFAULT.parent / f"{INPUT_ROOT_DEFAULT.name}_70dpi_part60",
    )
    parser.add_argument(
        "--output-root-high",
        type=Path,
        default=INPUT_ROOT_DEFAULT.parent / f"{INPUT_ROOT_DEFAULT.name}_100dpi_part40",
    )
    parser.add_argument("--manifest-file", default="manifest.jsonl")
    parser.add_argument("--target-low-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--high-limit", type=int, default=32768)
    parser.add_argument("--high-initial-rescale", type=float, default=0.5)
    parser.add_argument("--slack-tokens", type=int, default=512)
    parser.add_argument("--gpt-image-max-area", type=int, default=1638400)
    parser.add_argument("--prompt-style", default="vreasoner")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_prompt_templates() -> dict[str, Any]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("create_parquet_dataset_mod", INSIGHT_DOC_CREATE_PARQUET_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PROMPTS


def load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def cap_size_by_area(size: tuple[int, int], max_area: int) -> tuple[int, int]:
    width, height = size
    area = width * height
    if max_area <= 0 or area <= max_area:
        return width, height
    scale = (max_area / float(area)) ** 0.5
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def estimate_image_token_cost(
    image_paths: list[str],
    pdf_root: Path,
    initial_rescale: float,
    gpt_image_max_area: int,
    size_cache: dict[str, tuple[int, int]],
) -> int:
    total = 0
    for rel_path in image_paths:
        abs_path = str((pdf_root / rel_path).resolve())
        if abs_path in size_cache:
            width, height = size_cache[abs_path]
        else:
            with Image.open(abs_path) as image:
                width, height = image.size
            size_cache[abs_path] = (width, height)
        width = max(1, int(round(width * initial_rescale)))
        height = max(1, int(round(height * initial_rescale)))
        width, height = cap_size_by_area((width, height), gpt_image_max_area)
        total += max(1, math.ceil(width / 32.0) * math.ceil(height / 32.0))
    return total


def prompt_char_upper_bound(prompts: dict[str, Any], prompt_style: str, question: str, num_images: int) -> int:
    prompt_cfg = prompts[prompt_style]
    user_prompt = prompt_cfg["user_template"].format(question=question)
    if user_prompt.count("<image>") == 1 and num_images > 1:
        user_prompt = user_prompt.replace("<image>", "<image>" * num_images, 1)
    total = len(user_prompt)
    system_prompt = prompt_cfg.get("system")
    if system_prompt is not None:
        total += len(system_prompt)
    return total


def tie_break_key(question_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subset_counts = Counter(str(row.get("subset", "<missing>")) for row in rows)
    return {
        "rows": len(rows),
        "subset_counts": dict(sorted(subset_counts.items())),
    }


def write_partition(
    output_root: Path,
    rows: list[dict[str, Any]],
    pdf_target: Path,
    meta: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    symlink_path = output_root / "pdf_image"
    if symlink_path.exists() or symlink_path.is_symlink():
        if symlink_path.is_symlink():
            symlink_path.unlink()
        else:
            raise FileExistsError(f"refusing to replace real directory at {symlink_path}")
    symlink_path.symlink_to(pdf_target)

    meta = dict(meta)
    meta["manifest_path"] = str(manifest_path)
    meta["pdf_image_symlink"] = str(symlink_path)
    meta["common_pdf_image_root"] = str(pdf_target)
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    manifest_path = input_root / args.manifest_file
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    pdf_symlink = input_root / "pdf_image"
    if not pdf_symlink.exists():
        raise FileNotFoundError(f"pdf_image not found: {pdf_symlink}")
    pdf_target = pdf_symlink.resolve()
    if not pdf_target.exists():
        raise FileNotFoundError(f"pdf_image symlink target missing: {pdf_target}")
    if not (0.0 < args.target_low_ratio < 1.0):
        raise ValueError("--target-low-ratio must be between 0 and 1")

    prompts = load_prompt_templates()
    rows = load_rows(manifest_path)
    total_rows = len(rows)
    low_target = int(round(total_rows * args.target_low_ratio))
    high_target = total_rows - low_target

    size_cache: dict[str, tuple[int, int]] = {}
    forced_low: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    forced_examples: list[dict[str, Any]] = []

    for row in rows:
        image_cost = estimate_image_token_cost(
            image_paths=row["images"],
            pdf_root=pdf_target,
            initial_rescale=args.high_initial_rescale,
            gpt_image_max_area=args.gpt_image_max_area,
            size_cache=size_cache,
        )
        prompt_bound = prompt_char_upper_bound(
            prompts=prompts,
            prompt_style=args.prompt_style,
            question=row["question"],
            num_images=len(row["images"]),
        )
        estimated_tokens = image_cost + prompt_bound + args.slack_tokens
        row["_estimated_100dpi_tokens"] = estimated_tokens
        row["_estimated_100dpi_image_cost"] = image_cost
        row["_tie_break_key"] = tie_break_key(str(row["question_id"]), args.seed)
        if estimated_tokens > args.high_limit:
            forced_low.append(row)
            if len(forced_examples) < 20:
                forced_examples.append(
                    {
                        "question_id": row["question_id"],
                        "subset": row.get("subset"),
                        "estimated_100dpi_tokens": estimated_tokens,
                        "image_cost": image_cost,
                        "num_images": len(row["images"]),
                    }
                )
        else:
            candidate_rows.append(row)

    if len(forced_low) > low_target:
        raise ValueError(
            f"forced-low rows ({len(forced_low)}) exceed target low partition size ({low_target}); "
            "cannot satisfy requested split"
        )

    candidate_rows.sort(key=lambda row: row["_tie_break_key"])
    extra_low_needed = low_target - len(forced_low)
    selected_low_extra = candidate_rows[:extra_low_needed]
    selected_high = candidate_rows[extra_low_needed:]

    low_rows = sorted(forced_low + selected_low_extra, key=lambda row: row["_tie_break_key"])
    high_rows = sorted(selected_high, key=lambda row: row["_tie_break_key"])

    def clean(rows_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows_in]

    low_rows_clean = clean(low_rows)
    high_rows_clean = clean(high_rows)

    summary = {
        "input_root": str(input_root),
        "target_low_ratio": args.target_low_ratio,
        "target_high_ratio": 1.0 - args.target_low_ratio,
        "target_low_rows": low_target,
        "target_high_rows": high_target,
        "rows_total": total_rows,
        "forced_low_rows": len(forced_low),
        "forced_low_examples": forced_examples,
        "high_limit": args.high_limit,
        "high_initial_rescale": args.high_initial_rescale,
        "slack_tokens": args.slack_tokens,
        "prompt_style": args.prompt_style,
        "low_partition": build_summary(low_rows_clean),
        "high_partition": build_summary(high_rows_clean),
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    write_partition(args.output_root_low.expanduser().resolve(), low_rows_clean, pdf_target, {**summary, "partition": "low"})
    write_partition(
        args.output_root_high.expanduser().resolve(), high_rows_clean, pdf_target, {**summary, "partition": "high"}
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
