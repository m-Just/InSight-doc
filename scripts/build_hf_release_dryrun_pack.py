#!/usr/bin/env python3
"""Build a small, self-contained dry-run pack for the planned HF release.

The pack intentionally preserves the current verl-compatible columns while
copying image assets next to the sampled parquets.  It emits both absolute
file:// parquets for immediate local verl smoke tests and relative-path
parquets for transfer/HF-style inspection.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd


DEFAULT_SFT_LOG = Path(
    "/scratch/ywxzml3j/likaican/temp/"
    "full_sft_both_higher_dpi_unans02503505_fp32_sp4_epoch2_scratch_20260519/"
    "full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_freeze_vt_epoch2_fp32_scratch/"
    "train.log"
)
DEFAULT_RL_PARQUET = Path(
    "notes/generated/"
    "rl_16k_prompt24k_r05_to_r035_simple_datasource_plus_mc_false_e05_plus_arxiv_struct1k_llm_20260722/"
    "insight_doc_rl_16k_prompt24k_r05_to_r035_simple_datasource_plus_mc_false_e_arxiv_struct1k_llm-"
    "insight_qwen_agent.parquet"
)
DEFAULT_OUT_DIR = Path("notes/generated/hf_release_dryrun_100_20260727")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-log", type=Path, default=DEFAULT_SFT_LOG)
    parser.add_argument("--rl-parquet", type=Path, default=DEFAULT_RL_PARQUET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sft-rows", type=int, default=50)
    parser.add_argument("--rl-rows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args()


def normalize_obj(value: Any) -> Any:
    """Convert pandas/numpy containers into JSON/Arrow-friendly Python values."""
    if isinstance(value, np.ndarray):
        return [normalize_obj(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): normalize_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_obj(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def extract_first_python_dict(text: str, needle: str = "{'model':") -> dict[str, Any]:
    start = text.find(needle)
    if start < 0:
        raise ValueError(f"Could not find config dict starting with {needle!r}")

    depth = 0
    end = None
    for idx, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        raise ValueError("Could not find end of config dict")
    return ast.literal_eval(text[start:end])


def sft_train_files_from_log(log_path: Path) -> list[Path]:
    cfg = extract_first_python_dict(log_path.read_text(errors="replace"))
    train_files = cfg["data"]["train_files"]
    return [Path(path) for path in train_files]


def local_path_from_image_uri(uri: str) -> Path:
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        return Path(unquote(parsed.path))
    return Path(uri)


def image_uri_from_local_path(path: Path) -> str:
    return "file://" + str(path.resolve())


def safe_fragment(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:max_len] or "image"


class ImagePacker:
    def __init__(self, pack_root: Path):
        self.pack_root = pack_root.resolve()
        self.image_root = self.pack_root / "images"
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.src_to_rel: dict[str, str] = {}
        self.manifest: list[dict[str, Any]] = []

    def pack_one(self, image_value: str) -> tuple[str, str]:
        src = local_path_from_image_uri(image_value)
        if not src.exists():
            raise FileNotFoundError(f"Missing image file: {src}")

        src_key = str(src.resolve())
        if src_key not in self.src_to_rel:
            digest = hashlib.sha1(src_key.encode("utf-8")).hexdigest()
            ext = src.suffix.lower() or ".png"
            rel = Path("images") / digest[:2] / f"{digest[:20]}_{safe_fragment(src.stem)}{ext}"
            dst = self.pack_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            self.src_to_rel[src_key] = rel.as_posix()
            self.manifest.append(
                {
                    "relative_path": rel.as_posix(),
                    "source_path": src_key,
                    "size_bytes": dst.stat().st_size,
                    "sha1_source_path": digest,
                }
            )

        rel_path = self.src_to_rel[src_key]
        abs_uri = image_uri_from_local_path(self.pack_root / rel_path)
        return rel_path, abs_uri

    def rewrite_images(self, images: Any, *, mode: str) -> Any:
        rewritten = []
        for image in normalize_obj(images):
            if isinstance(image, dict):
                item = dict(image)
                image_value = item.get("image")
                if isinstance(image_value, str):
                    rel_path, abs_uri = self.pack_one(image_value)
                    item["image"] = rel_path if mode == "relative" else abs_uri
                rewritten.append(item)
            elif isinstance(image, str):
                rel_path, abs_uri = self.pack_one(image)
                rewritten.append({"image": rel_path if mode == "relative" else abs_uri})
            else:
                raise TypeError(f"Unsupported image entry type: {type(image)}")
        return rewritten


def load_and_sample_sft(files: list[Path], n_rows: int, seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path, columns=["messages", "images", "tools", "message_loss_mask"])
        df["_source_parquet"] = str(path)
        df["_source_row_index"] = np.arange(len(df), dtype=np.int64)
        frames.append(df)

    all_rows = pd.concat(frames, ignore_index=True)
    sample = all_rows.sample(n=min(n_rows, len(all_rows)), random_state=seed).reset_index(drop=True)
    records = []
    provenance = []
    for idx, row in sample.iterrows():
        records.append(
            {
                "id": f"sft_dryrun_{idx:05d}",
                "messages": normalize_obj(row["messages"]),
                "images": normalize_obj(row["images"]),
                "tools": normalize_obj(row["tools"]),
                "message_loss_mask": normalize_obj(row["message_loss_mask"]),
            }
        )
        provenance.append(
            {
                "split": "sft",
                "id": f"sft_dryrun_{idx:05d}",
                "source_parquet": str(row["_source_parquet"]),
                "source_row_index": int(row["_source_row_index"]),
            }
        )
    return pd.DataFrame(records), provenance


def load_and_sample_rl(path: Path, n_rows: int, seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(
        path,
        columns=["prompt", "images", "reward_model", "data_source", "extra_info", "agent_name"],
    )
    sample = df.sample(n=min(n_rows, len(df)), random_state=seed).reset_index(drop=True)
    records = []
    provenance = []
    for idx, row in sample.iterrows():
        records.append(
            {
                "id": f"rl_dryrun_{idx:05d}",
                "prompt": normalize_obj(row["prompt"]),
                "images": normalize_obj(row["images"]),
                "reward_model": normalize_obj(row["reward_model"]),
                "data_source": normalize_obj(row["data_source"]),
                "extra_info": normalize_obj(row["extra_info"]),
                "agent_name": normalize_obj(row["agent_name"]),
            }
        )
        extra_info = normalize_obj(row["extra_info"])
        provenance.append(
            {
                "split": "rl",
                "id": f"rl_dryrun_{idx:05d}",
                "source_parquet": str(path),
                "source_row_index": int(extra_info.get("index", idx)) if isinstance(extra_info, dict) else idx,
                "data_source": normalize_obj(row["data_source"]),
                "question_id": extra_info.get("question_id") if isinstance(extra_info, dict) else None,
            }
        )
    return pd.DataFrame(records), provenance


def rewrite_dataframe_images(df: pd.DataFrame, packer: ImagePacker, *, mode: str) -> pd.DataFrame:
    out = df.copy(deep=True)
    out["images"] = out["images"].map(lambda images: packer.rewrite_images(images, mode=mode))
    return out


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(normalize_obj(record), ensure_ascii=False) + "\n")


def count_image_placeholders(messages: Any) -> int:
    count = 0
    for message in normalize_obj(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            count += content.count("<image>")
    return count


def split_stats(df: pd.DataFrame, message_col: str) -> dict[str, Any]:
    image_counts = df["images"].map(len)
    placeholder_counts = df[message_col].map(count_image_placeholders)
    stats = {
        "rows": int(len(df)),
        "image_refs": int(image_counts.sum()),
        "mean_images_per_row": float(image_counts.mean()) if len(df) else 0.0,
        "placeholder_mismatches": int((image_counts != placeholder_counts).sum()),
    }
    if "data_source" in df.columns:
        stats["data_source_counts"] = df["data_source"].value_counts(dropna=False).to_dict()
    return stats


def write_readme(out_dir: Path, summary: dict[str, Any]) -> None:
    readme = f"""# InSight-doc HF Release Dry-Run Pack

This is a 100-row dry-run bundle: {summary["sft"]["rows"]} SFT rows and {summary["rl"]["rows"]} RL rows.

## Files

- `sft/sft_dryrun_relative.parquet`: relocatable SFT parquet with image paths relative to this directory.
- `sft/sft_dryrun_verl_abs.parquet`: SFT parquet with absolute `file://` image paths for immediate local verl smoke tests.
- `rl/rl_dryrun_relative.parquet`: relocatable RL parquet with image paths relative to this directory.
- `rl/rl_dryrun_verl_abs.parquet`: RL parquet with absolute `file://` image paths for immediate local verl smoke tests.
- `images/`: copied image assets referenced by the sampled rows.
- `sample_manifest.jsonl`: source row mapping for audit only; not part of the training schema.
- `image_manifest.jsonl`: copied image mapping for audit only.
- `summary.json`: pack statistics.

## Schemas

SFT columns: `id`, `messages`, `images`, `tools`, `message_loss_mask`.

RL columns: `id`, `prompt`, `images`, `reward_model`, `data_source`, `extra_info`, `agent_name`.

The `*_verl_abs.parquet` files preserve current verl compatibility.  The `*_relative.parquet`
files are intended for transfer/HF-style packaging; for current verl, either run from this
directory or rewrite relative image paths to absolute `file://` paths after download.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def make_archive(out_dir: Path) -> Path:
    archive_path = out_dir.with_suffix(".tar.gz")
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    return archive_path


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    sft_dir = out_dir / "sft"
    rl_dir = out_dir / "rl"
    sft_dir.mkdir(parents=True)
    rl_dir.mkdir(parents=True)

    sft_files = sft_train_files_from_log(args.sft_log)
    sft_df, sft_prov = load_and_sample_sft(sft_files, args.sft_rows, args.seed)
    rl_df, rl_prov = load_and_sample_rl(args.rl_parquet, args.rl_rows, args.seed + 1)

    packer = ImagePacker(out_dir)

    sft_rel = rewrite_dataframe_images(sft_df, packer, mode="relative")
    sft_abs = rewrite_dataframe_images(sft_df, packer, mode="absolute")
    rl_rel = rewrite_dataframe_images(rl_df, packer, mode="relative")
    rl_abs = rewrite_dataframe_images(rl_df, packer, mode="absolute")

    sft_rel.to_parquet(sft_dir / "sft_dryrun_relative.parquet", index=False)
    sft_abs.to_parquet(sft_dir / "sft_dryrun_verl_abs.parquet", index=False)
    rl_rel.to_parquet(rl_dir / "rl_dryrun_relative.parquet", index=False)
    rl_abs.to_parquet(rl_dir / "rl_dryrun_verl_abs.parquet", index=False)

    write_jsonl(sft_rel, sft_dir / "sft_dryrun_relative.jsonl")
    write_jsonl(rl_rel, rl_dir / "rl_dryrun_relative.jsonl")

    sample_manifest = sft_prov + rl_prov
    with (out_dir / "sample_manifest.jsonl").open("w", encoding="utf-8") as f:
        for record in sample_manifest:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with (out_dir / "image_manifest.jsonl").open("w", encoding="utf-8") as f:
        for record in sorted(packer.manifest, key=lambda x: x["relative_path"]):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "pack_root": str(out_dir),
        "seed": args.seed,
        "sft_source_log": str(args.sft_log),
        "sft_source_files": [str(path) for path in sft_files],
        "rl_source_file": str(args.rl_parquet),
        "sft": split_stats(sft_rel, "messages"),
        "rl": split_stats(rl_rel, "prompt"),
        "unique_images": len(packer.manifest),
        "copied_image_bytes": int(sum(item["size_bytes"] for item in packer.manifest)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out_dir, summary)

    archive = None
    if not args.no_archive:
        archive = make_archive(out_dir)
        summary["archive"] = str(archive)
        summary["archive_size_bytes"] = archive.stat().st_size
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
