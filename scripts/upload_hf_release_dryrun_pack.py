#!/usr/bin/env python3
"""Upload the dry-run release pack to Hugging Face with embedded images.

The upload can target either one repo with two configs, or two separate repos:

- SFT preserves the release-facing SFT columns (`messages`, `images`, `tools`)
  plus lightweight viewer/helper fields.
- RL preserves the current RL columns (`prompt`, `images`, `reward_model`,
  `data_source`, `extra_info`, `agent_name`) plus lightweight viewer helper
  fields.

The `images` column is a Hugging Face `Sequence(Image)` column with image bytes
embedded into the parquet shards.  No separate image files are uploaded unless
`--upload-raw-pack` is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, Image, Sequence
from huggingface_hub import HfApi


DEFAULT_PACK_ROOT = Path("notes/generated/hf_release_dryrun_100_20260727")

RL_EXTRA_INFO_TOP_LEVEL_KEYS = ("document_id", "question_id", "initial_rescale")
RL_EXTRA_INFO_PROVENANCE_KEY_MAP = {
    "question": "original_question",
    "index": "index",
    "initial_rescale_dpi": "initial_rescale_dpi",
    "question_type": "question_type",
    "subset": "subset",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="Single HF dataset repo id; uploads SFT/RL as configs")
    parser.add_argument("--sft-repo-id", help="HF dataset repo id for SFT-only upload")
    parser.add_argument("--rl-repo-id", help="HF dataset repo id for RL-only upload")
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--private", action="store_true", help="Create/upload as a private dataset repo")
    parser.add_argument("--upload-raw-pack", action="store_true", help="Also upload the raw pack tar.gz sidecar")
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument("--sft-num-shards", type=int, default=None)
    parser.add_argument("--rl-num-shards", type=int, default=None)
    parser.add_argument("--commit-message", default="Upload InSight-doc dry-run release pack")
    parser.add_argument(
        "--write-local-embedded-parquets",
        action="store_true",
        help="Also write schema-preserving embedded parquets under the pack root before upload.",
    )
    return parser.parse_args()


def normalize(value: Any) -> Any:
    if isinstance(value, bytes):
        return value
    if hasattr(value, "tolist"):
        return normalize(value.tolist())
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if not isinstance(value, (list, tuple, dict)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def image_path_from_item(pack_root: Path, item: dict[str, Any]) -> Path:
    value = item["image"]
    if value.startswith("file://"):
        return Path(value[7:])
    return (pack_root / value).resolve()


def embed_images(pack_root: Path, images: Any) -> list[dict[str, Any]]:
    embedded = []
    for item in normalize(images):
        path = image_path_from_item(pack_root, item)
        embedded.append({"bytes": path.read_bytes(), "path": None})
    return embedded


def load_sample_manifest(pack_root: Path) -> dict[str, dict[str, Any]]:
    path = pack_root / "sample_manifest.jsonl"
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            row_id = record.get("id")
            if row_id:
                out[str(row_id)] = record
    return out


def load_image_source_map(pack_root: Path) -> dict[str, str]:
    path = pack_root / "image_manifest.jsonl"
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            rel = record.get("relative_path")
            src = record.get("source_path")
            if rel and src:
                out[str(rel)] = str(src)
    return out


def normalize_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def data_source_from_question_id(question_id: str | None) -> str:
    qid = str(question_id or "")
    answerability = "unanswerable" if "__mut_unanswerable" in qid or "unanswerable" in qid else "answerable"
    base_qid = qid.split("__mut_unanswerable", maxsplit=1)[0]

    if "_mvqa" in base_qid:
        category = "arxiv_mveqa"
    elif re.search(r"_qa\d*$", base_qid) or "_qa" in base_qid:
        category = "arxiv_veqa"
    elif base_qid.startswith("docvqa") or "_docvqa" in base_qid:
        category = "docvqa"
    elif base_qid.startswith("dude") or "_dude" in base_qid:
        category = "dude"
    elif "travelmap" in base_qid:
        category = "map_travel"
    elif "metromap" in base_qid:
        category = "map_metro"
    elif "p2p" in base_qid or "poster" in base_qid:
        category = "poster"
    elif "info" in base_qid:
        category = "info"
    else:
        return "unknown"
    return f"{category}_{answerability}"


def paired_export_dir_from_image_source(image_source: str | None) -> Path | None:
    if not image_source:
        return None
    path = Path(image_source)
    run_dir = path.parent.parent if path.parent.name == "images" else path.parent
    parts = list(run_dir.parts)
    if "converted_sft" not in parts:
        return None
    idx = parts.index("converted_sft")
    export_dir = Path(*parts[:idx], "exported_conversations", *parts[idx + 1 :])
    if export_dir.exists():
        return export_dir.resolve()
    return None


def raw_export_dirs_for_sft_row(row: pd.Series, source_record: dict[str, Any] | None, image_source_by_rel: dict[str, str]) -> list[Path]:
    dirs: list[Path] = []
    if source_record:
        source_parquet = Path(str(source_record.get("source_parquet") or ""))
        for candidate in (source_parquet.parent.parent / "raw", source_parquet.parent.parent / "raw_gpt5_nano_rewrite"):
            if candidate.exists():
                dirs.append(candidate.resolve())

    images = normalize(row.get("images", []))
    if images:
        first_image = images[0]
        if isinstance(first_image, dict):
            image_source = image_source_by_rel.get(str(first_image.get("image") or ""))
            paired_dir = paired_export_dir_from_image_source(image_source)
            if paired_dir:
                dirs.append(paired_dir)

    unique_dirs = []
    for directory in dirs:
        if directory not in unique_dirs:
            unique_dirs.append(directory)
    return unique_dirs


def question_and_id_from_export(path: Path) -> tuple[str | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None, None
    extra_info = data.get("extra_info") if isinstance(data, dict) else None
    extra_info = extra_info if isinstance(extra_info, dict) else {}
    question = extra_info.get("question")
    if not question:
        for message in data.get("conversation") or []:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, dict) and content.get("question"):
                question = content["question"]
                break
    return question, extra_info.get("question_id")


@lru_cache(maxsize=None)
def question_id_map_for_export_dir(export_dir: str) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for path in Path(export_dir).glob("*.json"):
        question, question_id = question_and_id_from_export(path)
        if question and question_id:
            mapping.setdefault(normalize_question_text(question), set()).add(str(question_id))
    return mapping


def sft_data_source_from_row(row: pd.Series, manifest_by_id: dict[str, dict[str, Any]], image_source_by_rel: dict[str, str]) -> str:
    source_record = manifest_by_id.get(str(row.get("id")))
    question = normalize_question_text(first_user_text(row.get("messages", [])))
    data_sources = set()
    for export_dir in raw_export_dirs_for_sft_row(row, source_record, image_source_by_rel):
        question_ids = question_id_map_for_export_dir(str(export_dir)).get(question, set())
        data_sources.update(data_source_from_question_id(question_id) for question_id in question_ids)
    data_sources.discard("unknown")
    if len(data_sources) == 1:
        return next(iter(data_sources))
    return "unknown"


def clean_extra_info(extra_info: Any) -> dict[str, Any]:
    extra_info = normalize(extra_info)
    if not isinstance(extra_info, dict):
        return {}
    cleaned = {
        key: extra_info[key]
        for key in RL_EXTRA_INFO_TOP_LEVEL_KEYS
        if key in extra_info and extra_info[key] is not None
    }
    provenance = {
        out_key: extra_info[in_key]
        for in_key, out_key in RL_EXTRA_INFO_PROVENANCE_KEY_MAP.items()
        if in_key in extra_info and extra_info[in_key] is not None
    }
    if provenance:
        cleaned["_provenance"] = provenance
    return cleaned


def count_sft_tool_calls(messages: Any) -> int:
    total = 0
    for message in normalize(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = normalize(message.get("tool_calls"))
        if isinstance(tool_calls, list):
            total += len([call for call in tool_calls if call is not None])
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += content.count("<tool_call>")
    return total


def first_user_text(messages: Any) -> str:
    for message in normalize(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped == "---":
                continue
            if stripped.startswith("Image ") and "<image>" in stripped:
                continue
            if stripped == "<image>":
                continue
            lines.append(line)
        text = "\n".join(lines).strip()
        if text:
            return text[:4000]
    return ""


def question_from_rl_row(row: pd.Series) -> str:
    extra_info = normalize(row.get("extra_info"))
    if isinstance(extra_info, dict):
        question = extra_info.get("question") or extra_info.get("original_question")
        if question:
            return str(question)[:4000]
    return first_user_text(row.get("prompt", []))


def answer_from_rl_row(row: pd.Series) -> str:
    reward_model = normalize(row.get("reward_model"))
    if isinstance(reward_model, dict):
        return str(reward_model.get("ground_truth") or "")[:4000]
    return ""


def build_dataset_from_df(df: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(df, preserve_index=False)
    return dataset.cast_column("images", Sequence(Image()))


def build_sft_dataset(pack_root: Path) -> Dataset:
    df = pd.read_parquet(pack_root / "sft" / "sft_dryrun_relative.parquet")
    df = df.copy()
    manifest_by_id = load_sample_manifest(pack_root)
    image_source_by_rel = load_image_source_map(pack_root)
    if "message_loss_mask" in df.columns:
        df = df.drop(columns=["message_loss_mask"])
    df["question"] = df["messages"].map(first_user_text)
    df["data_source"] = df.apply(lambda row: sft_data_source_from_row(row, manifest_by_id, image_source_by_rel), axis=1)
    df["num_tool_calls"] = df["messages"].map(count_sft_tool_calls).astype("int32")
    df["images"] = df["images"].map(lambda images: embed_images(pack_root, images))
    df = df[["id", "images", "question", "messages", "num_tool_calls", "data_source", "tools"]]
    return build_dataset_from_df(df)


def build_rl_dataset(pack_root: Path) -> Dataset:
    df = pd.read_parquet(pack_root / "rl" / "rl_dryrun_relative.parquet")
    df = df.copy()
    df["question"] = df.apply(question_from_rl_row, axis=1)
    df["answer"] = df.apply(answer_from_rl_row, axis=1)
    df["extra_info"] = df["extra_info"].map(clean_extra_info)
    df["images"] = df["images"].map(lambda images: embed_images(pack_root, images))
    df = df[["id", "images", "question", "answer", "prompt", "reward_model", "data_source", "extra_info", "agent_name"]]
    return build_dataset_from_df(df)


def write_local_parquet(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    dataset.to_parquet(path)


def upload_file_if_present(pack_root: Path, repo_id: str, local_name: str, repo_name: str, commit_message: str) -> None:
    path = pack_root / local_name
    if not path.exists():
        return
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=repo_name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )


def upload_readme(pack_root: Path, repo_id: str, commit_message: str) -> None:
    readme = pack_root / "README.md"
    if not readme.exists():
        return
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README_dryrun_pack.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"{commit_message}: dry-run README",
    )


def upload_raw_archive(pack_root: Path, repo_id: str, commit_message: str) -> None:
    archive = pack_root.with_suffix(".tar.gz")
    if not archive.exists():
        raise FileNotFoundError(archive)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(archive),
        path_in_repo=archive.name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"{commit_message}: raw pack archive",
    )


def write_dataset_card(pack_root: Path, *, task: str, repo_id: str) -> Path:
    card_path = pack_root / f"README_{task}_hf.md"
    if task == "sft":
        native_columns = "`messages`, `images`, `tools`"
        description = "100-row dry-run SFT release pack with embedded image bytes."
    elif task == "rl":
        native_columns = "`prompt`, `images`, `reward_model`, `data_source`, `extra_info`, `agent_name`"
        description = "100-row dry-run RL release pack with embedded image bytes."
    else:
        raise ValueError(task)

    card_path.write_text(
        f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# {repo_id}

{description}

The `images` column is a Hugging Face `Sequence(Image)` column with image bytes
embedded directly in the parquet shards.  No separate image files are required
for the Dataset Viewer.

Native verl-compatible columns are preserved: {native_columns}.

Additional viewer/helper columns:

- `id`
- `question`
- `answer` (RL only)
- `data_source` (SFT: recovered from raw-export question ids; RL: native row-level source)
- `num_tool_calls` (SFT only)
""",
        encoding="utf-8",
    )
    return card_path


def upload_dataset(
    dataset: Dataset,
    *,
    repo_id: str,
    task: str,
    pack_root: Path,
    private: bool,
    max_shard_size: str,
    num_shards: int | None,
    commit_message: str,
    config_name: str = "default",
) -> None:
    api = HfApi()
    try:
        existing_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        existing_files = []
    stale_train_parquets = [
        path for path in existing_files if path.startswith("data/train-") and path.endswith(".parquet")
    ]
    for path in stale_train_parquets:
        api.delete_file(
            path_in_repo=path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"{commit_message}: remove stale {task} shard {path}",
        )

    push_kwargs: dict[str, Any] = {}
    if num_shards is not None:
        push_kwargs["num_shards"] = num_shards
    else:
        push_kwargs["max_shard_size"] = max_shard_size

    dataset.push_to_hub(
        repo_id,
        config_name=config_name,
        split="train",
        private=private,
        commit_message=f"{commit_message}: {task}",
        embed_external_files=True,
        **push_kwargs,
    )

    card = write_dataset_card(pack_root, task=task, repo_id=repo_id)
    HfApi().upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"{commit_message}: dataset card",
    )


def main() -> None:
    args = parse_args()
    pack_root = args.pack_root.resolve()

    # Keep HF cache in /tmp by default in this environment; the repo parent may be read-only.
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache_upload")

    sft = build_sft_dataset(pack_root)
    rl = build_rl_dataset(pack_root)

    if args.write_local_embedded_parquets:
        write_local_parquet(sft, pack_root / "sft" / "sft_dryrun_embedded_images.parquet")
        write_local_parquet(rl, pack_root / "rl" / "rl_dryrun_embedded_images.parquet")

    if args.sft_repo_id or args.rl_repo_id:
        if not args.sft_repo_id or not args.rl_repo_id:
            raise ValueError("--sft-repo-id and --rl-repo-id must be provided together")
        upload_dataset(
            sft,
            repo_id=args.sft_repo_id,
            task="sft",
            pack_root=pack_root,
            private=args.private,
            max_shard_size=args.max_shard_size,
            num_shards=args.sft_num_shards,
            commit_message=args.commit_message,
        )
        upload_dataset(
            rl,
            repo_id=args.rl_repo_id,
            task="rl",
            pack_root=pack_root,
            private=args.private,
            max_shard_size=args.max_shard_size,
            num_shards=args.rl_num_shards,
            commit_message=args.commit_message,
        )

        if args.upload_raw_pack:
            upload_raw_archive(pack_root, args.sft_repo_id, args.commit_message)
            upload_raw_archive(pack_root, args.rl_repo_id, args.commit_message)

        print(f"Uploaded SFT embedded-image dry-run dataset to https://huggingface.co/datasets/{args.sft_repo_id}")
        print(f"Uploaded RL embedded-image dry-run dataset to https://huggingface.co/datasets/{args.rl_repo_id}")
        print("No separate image files were uploaded unless --upload-raw-pack was used.")
        return

    if not args.repo_id:
        raise ValueError("Provide either --repo-id or both --sft-repo-id and --rl-repo-id")

    upload_dataset(
        sft,
        repo_id=args.repo_id,
        task="sft",
        pack_root=pack_root,
        private=args.private,
        max_shard_size=args.max_shard_size,
        num_shards=args.sft_num_shards,
        commit_message=args.commit_message,
        config_name="sft",
    )
    upload_dataset(
        rl,
        repo_id=args.repo_id,
        task="rl",
        pack_root=pack_root,
        private=args.private,
        max_shard_size=args.max_shard_size,
        num_shards=args.rl_num_shards,
        commit_message=args.commit_message,
        config_name="rl",
    )

    upload_readme(pack_root, args.repo_id, args.commit_message)
    if args.upload_raw_pack:
        upload_raw_archive(pack_root, args.repo_id, args.commit_message)

    print(f"Uploaded embedded-image dry-run dataset to https://huggingface.co/datasets/{args.repo_id}")
    print("Configs: sft, rl")
    print("No separate image files were uploaded unless --upload-raw-pack was used.")


if __name__ == "__main__":
    main()
