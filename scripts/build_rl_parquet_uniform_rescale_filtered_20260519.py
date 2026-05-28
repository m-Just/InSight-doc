#!/usr/bin/env python3
"""Build a filtered RL parquet for uniform-rescale no-tool baselines.

The source parquet may contain per-sample initial_rescale overrides in extra_info.
This script estimates prompt tokens under a uniform rescale, drops rows over a
threshold, and removes initial_rescale provenance keys so the training config's
global agent setting controls image presentation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd


DEFAULT_SOURCE_PARQUET = Path(
    "/scratch/ywxzml3j/likaican/data/insight_doc/"
    "answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale-insight_qwen_agent.parquet"
)
DEFAULT_OUTPUT_PARQUET = Path(
    "/scratch/ywxzml3j/likaican/data/insight_doc/"
    "answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_uniform_rescale05_"
    "maxarea3500x3500_le11000_no_initial_rescale-insight_qwen_agent.parquet"
)
INITIAL_RESCALE_EXTRA_INFO_KEYS = (
    "initial_rescale",
    "initial_rescale_source",
    "initial_rescale_dpi",
)
WORD_RE = re.compile(r"\b\S+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, default=DEFAULT_SOURCE_PARQUET)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--uniform-initial-rescale", type=float, default=0.5)
    parser.add_argument("--gpt-image-max-area", type=int, default=3500 * 3500)
    parser.add_argument("--max-prompt-tokens", type=float, default=11000)
    return parser.parse_args()


def image_path_from_value(value: Any) -> str:
    if isinstance(value, dict):
        if "image" in value:
            value = value["image"]
        elif "image_url" in value:
            value = value["image_url"]
    if isinstance(value, dict):
        value = value.get("url") or value.get("image") or ""
    image_path = str(value)
    if image_path.startswith("file://"):
        return unquote(urlparse(image_path).path)
    return image_path


def fast_image_size(path: str, size_cache: dict[str, tuple[int, int]]) -> tuple[int, int]:
    path = image_path_from_value(path)
    if path in size_cache:
        return size_cache[path]
    with open(path, "rb") as handle:
        header = handle.read(32)
    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        size = struct.unpack(">II", header[16:24])
    else:
        from PIL import Image

        with Image.open(path) as image:
            size = image.size
    size_cache[path] = size
    return size


def cap_dims_by_area(width: int, height: int, max_area: int) -> tuple[int, int]:
    if max_area <= 0 or width * height <= max_area:
        return width, height
    scale = math.sqrt(max_area / float(width * height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def image_tokens(size: tuple[int, int], *, initial_rescale: float, max_area: int) -> int:
    width = max(1, int(round(size[0] * initial_rescale)))
    height = max(1, int(round(size[1] * initial_rescale)))
    width, height = cap_dims_by_area(width, height, max_area)
    return math.ceil(width / 32.0) * math.ceil(height / 32.0)


def iter_text_parts(obj: Any):
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        if obj.get("type") in {"image", "image_url"} or "image" in obj or "image_url" in obj:
            return
        for key in ("text", "content"):
            if key in obj:
                yield from iter_text_parts(obj[key])
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from iter_text_parts(item)


def estimate_text_tokens(prompt: Any) -> float:
    words = 0
    for message in prompt:
        content = message.get("content") if isinstance(message, dict) else message
        for text in iter_text_parts(content):
            words += len(WORD_RE.findall(text))
    return words / 0.75


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return []


def remove_initial_rescale_extra_info(extra_info: Any) -> dict[str, Any]:
    cleaned = dict(extra_info or {})
    for key in INITIAL_RESCALE_EXTRA_INFO_KEYS:
        cleaned.pop(key, None)
    return cleaned


def main() -> int:
    args = parse_args()
    source_parquet = args.source_parquet.expanduser().resolve()
    output_parquet = args.output_parquet.expanduser().resolve()
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(source_parquet)
    size_cache: dict[str, tuple[int, int]] = {}
    estimated_tokens: list[float] = []
    keep_mask: list[bool] = []

    for record in df.to_dict("records"):
        total = estimate_text_tokens(record["prompt"])
        for image in as_list(record.get("images")):
            total += image_tokens(
                fast_image_size(image_path_from_value(image), size_cache),
                initial_rescale=args.uniform_initial_rescale,
                max_area=args.gpt_image_max_area,
            )
        estimated_tokens.append(total)
        keep_mask.append(total <= args.max_prompt_tokens)

    filtered = df.loc[keep_mask].copy()
    filtered["extra_info"] = filtered["extra_info"].map(remove_initial_rescale_extra_info)
    filtered.to_parquet(output_parquet, index=False)

    dropped = df.loc[[not keep for keep in keep_mask]].copy()
    dropped_qids = [
        str((extra_info or {}).get("question_id"))
        for extra_info in dropped["extra_info"].tolist()
        if isinstance(extra_info, dict)
    ]
    dropped_qids_path = output_parquet.with_suffix(".dropped_question_ids.txt")
    dropped_qids_path.write_text("".join(f"{qid}\n" for qid in dropped_qids), encoding="utf-8")

    kept_estimates = [value for value, keep in zip(estimated_tokens, keep_mask, strict=True) if keep]
    dropped_estimates = [value for value, keep in zip(estimated_tokens, keep_mask, strict=True) if not keep]
    summary = {
        "source_parquet": str(source_parquet),
        "output_parquet": str(output_parquet),
        "dropped_question_ids": str(dropped_qids_path),
        "rows_source": int(len(df)),
        "rows_kept": int(len(filtered)),
        "rows_dropped": int(len(dropped)),
        "uniform_initial_rescale": args.uniform_initial_rescale,
        "gpt_image_max_area": args.gpt_image_max_area,
        "max_prompt_tokens": args.max_prompt_tokens,
        "removed_extra_info_keys": list(INITIAL_RESCALE_EXTRA_INFO_KEYS),
        "source_initial_rescale_counts": dict(
            Counter(str((extra_info or {}).get("initial_rescale")) for extra_info in df["extra_info"].tolist())
        ),
        "kept_estimated_prompt_tokens_max": max(kept_estimates) if kept_estimates else None,
        "dropped_estimated_prompt_tokens_min": min(dropped_estimates) if dropped_estimates else None,
    }
    summary_path = output_parquet.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
