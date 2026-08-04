#!/usr/bin/env python3
"""Upload the full InSight-doc SFT/RL release datasets with embedded images.

This script is intentionally shard-oriented: it never materializes the full
embedded-image dataset locally or in memory.  It writes a small batch of parquet
shards under a work directory, commits that batch to Hugging Face, and deletes
the local shard files after the commit succeeds.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import Dataset, Image, Sequence
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from build_hf_release_dryrun_pack import (
    DEFAULT_RL_PARQUET,
    DEFAULT_SFT_LOG,
    local_path_from_image_uri,
    normalize_obj,
    sft_train_files_from_log,
)
from upload_hf_release_dryrun_pack import (
    answer_from_rl_row,
    clean_extra_info,
    count_sft_tool_calls,
    data_source_from_question_id,
    first_user_text,
    normalize_question_text,
    paired_export_dir_from_image_source,
    question_from_rl_row,
    question_id_map_for_export_dir,
)


DEFAULT_WORK_DIR = Path("/tmp/insight_doc_hf_full_upload")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-repo-id", required=True)
    parser.add_argument("--rl-repo-id", required=True)
    parser.add_argument("--sft-log", type=Path, default=DEFAULT_SFT_LOG)
    parser.add_argument("--rl-parquet", type=Path, default=DEFAULT_RL_PARQUET)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--private", action="store_true", help="Create/upload private dataset repos.")
    parser.add_argument("--resume", action="store_true", help="Skip remote shards that already exist.")
    parser.add_argument(
        "--delete-existing-data-files",
        action="store_true",
        help="Delete existing remote data/* files for the selected repo before uploading shards.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build local shards but do not upload.")
    parser.add_argument("--splits", choices=["both", "sft", "rl"], default="both")
    parser.add_argument("--max-sft-rows", type=int, default=None)
    parser.add_argument("--max-rl-rows", type=int, default=None)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Deterministically shuffle rows before row limiting and sharding.",
    )
    parser.add_argument("--target-shard-mb", type=float, default=1024.0)
    parser.add_argument("--commit-batch-shards", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=5)
    parser.add_argument("--parquet-batch-size", type=int, default=1, help="Rows per parquet row group.")
    parser.add_argument("--commit-prefix", default="Upload InSight-doc full release")
    parser.add_argument(
        "--inline-sft-tool-calls",
        action="store_true",
        help="Inline structured assistant tool_calls into message content for viewer-friendly SFT release messages.",
    )
    return parser.parse_args()


def image_paths_from_images(images: Any) -> list[Path]:
    paths = []
    for item in normalize_obj(images):
        value = item.get("image") if isinstance(item, dict) else item
        if isinstance(value, str):
            paths.append(local_path_from_image_uri(value))
    return paths


def estimate_image_bytes(images: Any) -> tuple[int, int]:
    total = 0
    missing = 0
    for path in image_paths_from_images(images):
        try:
            total += path.stat().st_size
        except OSError:
            missing += 1
    return total, missing


def embed_original_images(images: Any) -> list[dict[str, Any]]:
    embedded = []
    for path in image_paths_from_images(images):
        embedded.append({"bytes": path.read_bytes(), "path": None})
    return embedded


def raw_export_dirs_for_sft(source_parquet: Path, images: Any) -> list[Path]:
    dirs = []
    for candidate in (source_parquet.parent.parent / "raw", source_parquet.parent.parent / "raw_gpt5_nano_rewrite"):
        if candidate.exists():
            dirs.append(candidate.resolve())

    image_paths = image_paths_from_images(images)
    if image_paths:
        paired_dir = paired_export_dir_from_image_source(str(image_paths[0]))
        if paired_dir:
            dirs.append(paired_dir)

    unique_dirs = []
    for directory in dirs:
        if directory not in unique_dirs:
            unique_dirs.append(directory)
    return unique_dirs


def recover_sft_data_source(question: str, source_parquet: Path, images: Any) -> str:
    data_sources = set()
    normalized_question = normalize_question_text(question)
    for export_dir in raw_export_dirs_for_sft(source_parquet, images):
        question_ids = question_id_map_for_export_dir(str(export_dir)).get(normalized_question, set())
        data_sources.update(data_source_from_question_id(question_id) for question_id in question_ids)
    data_sources.discard("unknown")
    if len(data_sources) == 1:
        return next(iter(data_sources))
    return "unknown"


def assistant_text_content(content: Any) -> str:
    """Render assistant text content the same way as the Qwen3-VL chat template."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and "text" in item
        )
    return str(content)


def inline_sft_tool_calls(messages: Any) -> list[dict[str, Any]]:
    """Move assistant tool calls into content using Qwen3-VL chat-template syntax."""
    inlined = []
    for message in normalize_obj(messages):
        if not isinstance(message, dict):
            inlined.append(message)
            continue

        tool_calls = message.get("tool_calls")
        out_message = {key: value for key, value in message.items() if key != "tool_calls"}
        if message.get("role") != "assistant" or not tool_calls:
            inlined.append(out_message)
            continue

        content = assistant_text_content(message.get("content"))
        blocks = []
        for tool_call in tool_calls:
            if not tool_call:
                continue
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            call = function or tool_call
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            arguments = call.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            blocks.append(f'<tool_call>\n{{"name": "{name}", "arguments": {arguments}}}\n</tool_call>')

        if blocks:
            out_message["content"] = content + ("\n" if content else "") + "\n".join(blocks)
        inlined.append(out_message)
    return inlined


def build_dataset_from_records(records: list[dict[str, Any]]) -> Dataset:
    dataset = Dataset.from_pandas(pd.DataFrame(records), preserve_index=False)
    return dataset.cast_column("images", Sequence(Image()))


def rl_tool_schema_sidecar() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "agent_name": "insight_qwen_agent",
        "tool_schema_format": "qwen_agent_unwrapped_function",
        "qwen_tool_list": ["image_zoom_in_tool_qwen3vl"],
        "tool_name_aliases": {"image_zoom_in_tool_qwen3vl": "image_zoom_in_tool"},
        "source_definition": {
            "registry_tool": "image_zoom_in_tool_qwen3vl",
            "runtime_file": "verl/experimental/agent_loop/qwen_agent_tools/image_zoom_in_qwen3vl.py",
            "runtime_class": "ImageZoomInToolQwen3VL",
            "schema_loaded_by": "verl/experimental/agent_loop/qwen_agent_loop.py",
        },
        "tools": [
            {
                "name": "image_zoom_in_tool",
                "description": (
                    "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) "
                    "and an optional object label"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": (
                                "The bounding box of the region to zoom in, as [x1, y1, x2, y2], "
                                "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner"
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": "The name or label of the object in the specified bounding box",
                        },
                        "img_idx": {
                            "type": "number",
                            "description": "The index of the zoomed-in image (starting from 0)",
                        },
                    },
                    "required": ["bbox_2d", "label", "img_idx"],
                },
            }
        ],
    }


def write_rl_tool_schema_sidecar(path: Path) -> None:
    path.write_text(json.dumps(rl_tool_schema_sidecar(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def chat_template_readme_snippet(repo_id: str, *, split_name: str) -> str:
    if split_name == "sft":
        row_source = 'row = load_dataset("' + repo_id + '", split="train[:1]")[0]\nmessages = attach_images(row["messages"], row["images"])\ntools = row["tools"]\nadd_generation_prompt = False'
        note = (
            "For SFT rows, `tools` is stored as a column.  The Qwen3-VL chat template "
            "renders that schema into the system prompt during tokenization."
        )
        extra_imports = ""
    elif split_name == "rl":
        row_source = (
            'row = load_dataset("' + repo_id + '", split="train[:1]")[0]\n'
            'tool_schema_path = hf_hub_download("' + repo_id + '", "tool_schemas/insight_qwen_agent_tools.json", repo_type="dataset")\n'
            'with open(tool_schema_path, encoding="utf-8") as f:\n'
            '    tools = json.load(f)["tools"]\n'
            'messages = attach_images(row["prompt"], row["images"])\n'
            'add_generation_prompt = True'
        )
        note = (
            "For RL rows, this release stores the raw prompt messages but not a row-level "
            "`tools` column.  The training/eval agent supplies the same tool schema from "
            "its config at rollout time; the matching model-facing schema is provided as "
            "`tool_schemas/insight_qwen_agent_tools.json`."
        )
        extra_imports = "import json\n\nfrom huggingface_hub import hf_hub_download"
    else:
        raise ValueError(split_name)

    return f"""
## Reconstructing The Model Prompt

{note}

```python
import copy
import re

from datasets import load_dataset
from transformers import AutoProcessor
{extra_imports}

def attach_images(messages, images):
    \"\"\"Convert '<image>' placeholders to Qwen-VL image content items.\"\"\"
    messages = copy.deepcopy(messages)
    image_idx = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or "<image>" not in content:
            continue
        content_items = []
        for part in re.split(r"(<image>)", content):
            if not part:
                continue
            if part == "<image>":
                content_items.append({{"type": "image", "image": images[image_idx]}})
                image_idx += 1
            else:
                content_items.append({{"type": "text", "text": part}})
        message["content"] = content_items
    assert image_idx == len(images), (image_idx, len(images))
    return messages


processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
{row_source}

prompt_text = processor.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=add_generation_prompt,
)
model_inputs = processor(text=[prompt_text], images=row["images"], return_tensors="pt")
```
"""


def write_dataset_card(path: Path, *, repo_id: str, split_name: str) -> None:
    if split_name == "sft":
        columns = "`id`, `images`, `question`, `messages`, `num_tool_calls`, `data_source`, `tools`"
        description = "Full InSight-doc SFT release dataset with image bytes embedded in parquet shards."
    elif split_name == "rl":
        columns = "`id`, `images`, `question`, `answer`, `prompt`, `reward_model`, `data_source`, `extra_info`, `agent_name`"
        description = "Full InSight-doc RL release dataset with image bytes embedded in parquet shards."
    else:
        raise ValueError(split_name)

    path.write_text(
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
embedded directly in the parquet shards.  Parquet shards are written with one
row per row group to keep Dataset Viewer random-access scans bounded.

Columns: {columns}.

{chat_template_readme_snippet(repo_id, split_name=split_name)}
""",
        encoding="utf-8",
    )


class BatchedShardUploader:
    def __init__(
        self,
        *,
        repo_id: str,
        split_name: str,
        work_dir: Path,
        private: bool,
        resume: bool,
        delete_existing_data_files: bool,
        dry_run: bool,
        commit_batch_shards: int,
        num_threads: int,
        parquet_batch_size: int,
        commit_prefix: str,
    ):
        self.repo_id = repo_id
        self.split_name = split_name
        self.work_dir = work_dir / split_name
        self.private = private
        self.resume = resume
        self.delete_existing_data_files = delete_existing_data_files
        self.dry_run = dry_run
        self.commit_batch_shards = commit_batch_shards
        self.num_threads = num_threads
        self.parquet_batch_size = parquet_batch_size
        self.commit_prefix = commit_prefix
        self.api = HfApi()
        self.pending: list[Path] = []
        self.shard_idx = 0
        self.uploaded_shards = 0
        self.skipped_shards = 0
        self.total_rows = 0
        self.total_image_bytes = 0
        self.max_row_image_bytes = 0
        self.missing_images = 0
        self.existing_files: set[str] = set()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def setup_repo(self) -> None:
        if self.dry_run:
            return
        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="dataset",
            private=self.private,
            exist_ok=True,
        )
        if self.resume:
            self.existing_files = set(self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset"))
        if self.delete_existing_data_files:
            existing_files = self.existing_files or set(
                self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            )
            data_files = sorted(path for path in existing_files if path.startswith("data/"))
            if data_files:
                print(f"[{self.split_name}] deleting {len(data_files)} existing remote data files", flush=True)
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=[CommitOperationDelete(path_in_repo=path) for path in data_files],
                    commit_message=f"{self.commit_prefix}: clear existing {self.split_name} data files",
                )
                self.existing_files.difference_update(data_files)
        readme = self.work_dir / "README.md"
        write_dataset_card(readme, repo_id=self.repo_id, split_name=self.split_name)
        self.api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"{self.commit_prefix}: {self.split_name} dataset card",
        )
        if self.split_name == "rl":
            sidecar = self.work_dir / "insight_qwen_agent_tools.json"
            write_rl_tool_schema_sidecar(sidecar)
            self.api.upload_file(
                path_or_fileobj=str(sidecar),
                path_in_repo="tool_schemas/insight_qwen_agent_tools.json",
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"{self.commit_prefix}: RL tool schema sidecar",
            )

    def write_shard(self, records: list[dict[str, Any]], estimated_bytes: int) -> None:
        if not records:
            return
        shard_name = f"train-{self.shard_idx:06d}.parquet"
        path_in_repo = f"data/{shard_name}"
        self.shard_idx += 1
        self.total_rows += len(records)
        self.total_image_bytes += estimated_bytes
        if self.resume and path_in_repo in self.existing_files:
            self.skipped_shards += 1
            print(f"[{self.split_name}] skip existing {path_in_repo}", flush=True)
            return

        local_path = self.work_dir / shard_name
        embedded_records = []
        for record in records:
            out = dict(record)
            out["images"] = embed_original_images(record["images"])
            embedded_records.append(out)

        dataset = build_dataset_from_records(embedded_records)
        dataset.to_parquet(local_path, batch_size=self.parquet_batch_size)
        local_size = local_path.stat().st_size
        print(
            f"[{self.split_name}] wrote {shard_name}: rows={len(records)} "
            f"image_bytes={estimated_bytes / 1e6:.1f}MB parquet={local_size / 1e6:.1f}MB",
            flush=True,
        )
        del dataset, embedded_records
        gc.collect()

        if self.dry_run:
            return
        self.pending.append(local_path)
        if len(self.pending) >= self.commit_batch_shards:
            self.commit_pending()

    def commit_pending(self) -> None:
        if not self.pending:
            return
        ops = [
            CommitOperationAdd(path_in_repo=f"data/{path.name}", path_or_fileobj=str(path))
            for path in self.pending
        ]
        first = self.pending[0].stem
        last = self.pending[-1].stem
        print(f"[{self.split_name}] uploading {len(self.pending)} shards {first}..{last}", flush=True)
        self.api.create_commit(
            repo_id=self.repo_id,
            repo_type="dataset",
            operations=ops,
            commit_message=f"{self.commit_prefix}: {self.split_name} {first}..{last}",
            num_threads=self.num_threads,
        )
        self.uploaded_shards += len(self.pending)
        for path in self.pending:
            path.unlink(missing_ok=True)
        self.pending = []

    def finish(self) -> None:
        if not self.dry_run:
            self.commit_pending()
        state = {
            "repo_id": self.repo_id,
            "split_name": self.split_name,
            "uploaded_shards": self.uploaded_shards,
            "skipped_shards": self.skipped_shards,
            "total_shards_seen": self.shard_idx,
            "total_rows": self.total_rows,
            "total_image_bytes": self.total_image_bytes,
            "max_row_image_bytes": self.max_row_image_bytes,
            "missing_images": self.missing_images,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        }
        (self.work_dir / "upload_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[{self.split_name}] finished: {json.dumps(state, sort_keys=True)}", flush=True)


def flush_if_needed(
    uploader: BatchedShardUploader,
    buffer: list[dict[str, Any]],
    buffer_bytes: int,
    row_estimate: int,
    target_bytes: int,
) -> tuple[list[dict[str, Any]], int]:
    if buffer and buffer_bytes + row_estimate > target_bytes:
        uploader.write_shard(buffer, buffer_bytes)
        return [], 0
    return buffer, buffer_bytes


def upload_sft(args: argparse.Namespace) -> None:
    uploader = BatchedShardUploader(
        repo_id=args.sft_repo_id,
        split_name="sft",
        work_dir=args.work_dir,
        private=args.private,
        resume=args.resume,
        delete_existing_data_files=args.delete_existing_data_files,
        dry_run=args.dry_run,
        commit_batch_shards=args.commit_batch_shards,
        num_threads=args.num_threads,
        parquet_batch_size=args.parquet_batch_size,
        commit_prefix=args.commit_prefix,
    )
    uploader.setup_repo()
    target_bytes = int(args.target_shard_mb * 1_000_000)
    row_limit = args.max_sft_rows
    rows_seen = 0
    buffer: list[dict[str, Any]] = []
    buffer_bytes = 0
    data_source_counts: dict[str, int] = {}

    sft_frames = []
    for source_parquet in sft_train_files_from_log(args.sft_log):
        df = pd.read_parquet(source_parquet, columns=["messages", "images", "tools"])
        df["_source_parquet"] = str(source_parquet)
        df["_source_row_index"] = range(len(df))
        print(f"[sft] loaded {source_parquet} rows={len(df)}", flush=True)
        sft_frames.append(df)
    all_sft = pd.concat(sft_frames, ignore_index=True)
    if args.shuffle_seed is not None:
        all_sft = all_sft.sample(frac=1, random_state=args.shuffle_seed)
    if row_limit is not None:
        all_sft = all_sft.head(row_limit)
    print(f"[sft] selected rows={len(all_sft)} shuffle_seed={args.shuffle_seed}", flush=True)

    for _, row in all_sft.iterrows():
        question = first_user_text(row["messages"])
        image_bytes, missing = estimate_image_bytes(row["images"])
        uploader.max_row_image_bytes = max(uploader.max_row_image_bytes, image_bytes)
        uploader.missing_images += missing
        buffer, buffer_bytes = flush_if_needed(uploader, buffer, buffer_bytes, image_bytes, target_bytes)
        source_parquet = Path(row["_source_parquet"])
        data_source = recover_sft_data_source(question, source_parquet, row["images"])
        data_source_counts[data_source] = data_source_counts.get(data_source, 0) + 1
        buffer.append(
            {
                "id": f"sft_{rows_seen:06d}",
                "images": normalize_obj(row["images"]),
                "question": question,
                "messages": inline_sft_tool_calls(row["messages"])
                if args.inline_sft_tool_calls
                else normalize_obj(row["messages"]),
                "num_tool_calls": int(count_sft_tool_calls(row["messages"])),
                "data_source": data_source,
                "tools": normalize_obj(row["tools"]),
            }
        )
        buffer_bytes += image_bytes
        rows_seen += 1

    uploader.write_shard(buffer, buffer_bytes)
    uploader.finish()
    (uploader.work_dir / "data_source_counts.json").write_text(
        json.dumps(data_source_counts, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def upload_rl(args: argparse.Namespace) -> None:
    uploader = BatchedShardUploader(
        repo_id=args.rl_repo_id,
        split_name="rl",
        work_dir=args.work_dir,
        private=args.private,
        resume=args.resume,
        delete_existing_data_files=args.delete_existing_data_files,
        dry_run=args.dry_run,
        commit_batch_shards=args.commit_batch_shards,
        num_threads=args.num_threads,
        parquet_batch_size=args.parquet_batch_size,
        commit_prefix=args.commit_prefix,
    )
    uploader.setup_repo()
    target_bytes = int(args.target_shard_mb * 1_000_000)
    row_limit = args.max_rl_rows
    df = pd.read_parquet(
        args.rl_parquet,
        columns=["prompt", "images", "reward_model", "data_source", "extra_info", "agent_name"],
    )
    if args.shuffle_seed is not None:
        df = df.sample(frac=1, random_state=args.shuffle_seed)
    if row_limit is not None:
        df = df.head(row_limit)
    print(f"[rl] loaded {args.rl_parquet} rows={len(df)}", flush=True)
    buffer: list[dict[str, Any]] = []
    buffer_bytes = 0
    data_source_counts: dict[str, int] = {}

    for row_idx, (_, row) in enumerate(df.iterrows()):
        image_bytes, missing = estimate_image_bytes(row["images"])
        uploader.max_row_image_bytes = max(uploader.max_row_image_bytes, image_bytes)
        uploader.missing_images += missing
        buffer, buffer_bytes = flush_if_needed(uploader, buffer, buffer_bytes, image_bytes, target_bytes)
        data_source = normalize_obj(row["data_source"])
        data_source_counts[str(data_source)] = data_source_counts.get(str(data_source), 0) + 1
        buffer.append(
            {
                "id": f"rl_{row_idx:06d}",
                "images": normalize_obj(row["images"]),
                "question": question_from_rl_row(row),
                "answer": answer_from_rl_row(row),
                "prompt": normalize_obj(row["prompt"]),
                "reward_model": normalize_obj(row["reward_model"]),
                "data_source": data_source,
                "extra_info": clean_extra_info(row["extra_info"]),
                "agent_name": normalize_obj(row["agent_name"]),
            }
        )
        buffer_bytes += image_bytes

    uploader.write_shard(buffer, buffer_bytes)
    uploader.finish()
    (uploader.work_dir / "data_source_counts.json").write_text(
        json.dumps(data_source_counts, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache_upload")
    if args.splits in {"both", "sft"}:
        upload_sft(args)
    if args.splits in {"both", "rl"}:
        upload_rl(args)


if __name__ == "__main__":
    main()
