#!/usr/bin/env python3
"""Run a small end-to-end smoke test for the Qwen3-VL zoom-in SFT pipeline.

This script performs three steps:
1. Convert exported vReasoner/VSearcher conversations into SFT parquet files.
2. Run a tiny engine SFT job that saves an HF-exported checkpoint.
3. Load that checkpoint into AgentLoopManager and run one QwenAgentLoop inference sample.

The inference sample is built from one converted parquet row by:
- selecting messages up to the first assistant turn
- reconstructing multimodal user content from "<image>" placeholders plus the row's flat images list

The goal is not quality evaluation. It is a plumbing check that:
- conversion produces usable rows
- SFT can train and export an HF checkpoint
- QwenAgentLoop can load that checkpoint and execute a tool-using inference path
"""

from __future__ import annotations

import argparse
import math
import io
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

CONDA_PREFIX = os.environ.get("CONDA_PREFIX")
if CONDA_PREFIX:
    conda_lib = str(Path(CONDA_PREFIX) / "lib")
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in current_ld_library_path.split(":") if part]
    if conda_lib not in ld_parts and os.environ.get("_QWEN3_VL_SMOKE_REEXEC") != "1":
        new_env = os.environ.copy()
        new_env["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld_library_path}" if current_ld_library_path else conda_lib
        new_env["_QWEN3_VL_SMOKE_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], new_env)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from hydra import compose, initialize_config_dir
from PIL import Image

# Ensure local companion repos are importable in this workspace.
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_SRC = REPO_ROOT.parent
EXTRA_REPOS = (WORKSPACE_SRC / "InSight-o3", WORKSPACE_SRC / "Qwen-Agent")
for extra_repo in EXTRA_REPOS:
    extra_repo_str = str(extra_repo)
    if extra_repo.exists() and extra_repo_str not in sys.path:
        sys.path.insert(0, extra_repo_str)
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            f"{extra_repo_str}:{current_pythonpath}" if current_pythonpath else extra_repo_str
        )
os.environ.setdefault("VERL_PROJ_DIR", str(REPO_ROOT))

from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True, help="Directory containing exported conversation JSON files.")
    parser.add_argument("--base-model", required=True, help="Base Qwen3-VL model path for SFT initialization.")
    parser.add_argument("--work-dir", required=True, help="Working directory for converted data and checkpoints.")
    parser.add_argument("--rollout-backend", default="vllm", choices=["vllm", "sglang"], help="Inference backend.")
    parser.add_argument(
        "--system-prompt-mode",
        default="exported",
        choices=["exported", "vsearcher_qwen3_vl"],
        help="Which system prompt to use during conversion, and therefore for both training and inference.",
    )
    parser.add_argument(
        "--assistant-format-mode",
        default="tagged",
        choices=["tagged", "plain"],
        help="Whether assistant targets keep <think>/<answer> tags or are written as plain text.",
    )
    parser.add_argument("--train-max-samples", type=int, default=2, help="Maximum train samples for smoke SFT.")
    parser.add_argument("--val-max-samples", type=int, default=1, help="Maximum val samples for smoke SFT.")
    parser.add_argument("--total-training-steps", type=int, default=1, help="Tiny SFT step budget.")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun local process count.")
    parser.add_argument("--row-index", type=int, default=None, help="Explicit converted train row to use for inference.")
    parser.add_argument(
        "--agent-loop-name",
        default="qwen_agent",
        help="Agent loop name to use for inference smoke run.",
    )
    parser.add_argument(
        "--agent-loop-config-path",
        default="",
        help="Optional external agent loop config path for inference smoke run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_model_ref(model_ref: str) -> str:
    model_path = Path(model_ref).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return model_ref


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
    print(f"\n[run] {printable}\n")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def find_free_port() -> int:
    return random.randint(20000, 45000)


def load_image(item: Any) -> Image.Image:
    if not isinstance(item, dict) or "bytes" not in item:
        raise ValueError(f"Expected image item with raw bytes, got {type(item)}")
    image = Image.open(io.BytesIO(item["bytes"]))
    image.load()
    return image.convert("RGB")


def split_text_with_images(text: str, images: list[Image.Image], image_cursor: int) -> tuple[str | list[dict[str, Any]], int]:
    if "<image>" not in text:
        return text, image_cursor

    parts: list[dict[str, Any]] = []
    chunks = text.split("<image>")
    for idx, chunk in enumerate(chunks):
        if chunk:
            parts.append({"type": "text", "text": chunk})
        if idx < len(chunks) - 1:
            if image_cursor >= len(images):
                raise ValueError("Not enough images to satisfy <image> placeholders in converted row")
            parts.append({"type": "image", "image": images[image_cursor]})
            image_cursor += 1
    return parts, image_cursor


def build_inference_prefix(row: pd.Series) -> list[dict[str, Any]]:
    messages = row["messages"]
    flat_images = [load_image(item) for item in row.get("images", [])]
    prefix: list[dict[str, Any]] = []
    image_cursor = 0

    for message in messages:
        if message.get("role") == "assistant":
            break
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"Expected converted message content to be str, got {type(content)}")
        rebuilt_content, image_cursor = split_text_with_images(content, flat_images, image_cursor)
        prefix.append({"role": message["role"], "content": rebuilt_content})

    if not prefix:
        raise ValueError("Failed to build inference prefix from converted row")
    return prefix


def select_row(df: pd.DataFrame, explicit_row_index: int | None) -> tuple[int, pd.Series]:
    if explicit_row_index is not None:
        return explicit_row_index, df.iloc[explicit_row_index]

    for idx, row in df.iterrows():
        for message in row["messages"]:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                return int(idx), row
    return 0, df.iloc[0]


def has_tool_call(row: pd.Series) -> bool:
    for message in row["messages"]:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return True
    return False


def row_cost(row: pd.Series) -> tuple[int, int, int]:
    return (
        len(row.get("images", [])),
        len(row.get("messages", [])),
        sum(len(str(message.get("content", ""))) for message in row.get("messages", [])),
    )


def write_subset(df: pd.DataFrame, output_path: Path, max_rows: int) -> Path | None:
    candidate_indices = [idx for idx, row in df.iterrows() if has_tool_call(row)]
    if not candidate_indices:
        return None

    ranked = sorted(candidate_indices, key=lambda idx: row_cost(df.iloc[idx]))
    subset = df.iloc[ranked[: max_rows if max_rows > 0 else len(ranked)]].reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(subset, preserve_index=False), output_path)
    return output_path


def latest_hf_checkpoint(ckpt_root: Path) -> Path:
    candidates = []
    for step_dir in ckpt_root.glob("global_step_*"):
        try:
            step = int(step_dir.name.split("_")[-1])
        except ValueError:
            continue
        hf_dir = step_dir / "huggingface"
        if hf_dir.is_dir():
            candidates.append((step, hf_dir))
    if not candidates:
        raise FileNotFoundError(f"No HF checkpoint found under {ckpt_root}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def run_inference_smoke(
    model_path: Path,
    train_parquet: Path,
    row_index: int | None,
    rollout_backend: str,
    agent_loop_name: str,
    agent_loop_config_path: str,
) -> None:
    df = pd.read_parquet(train_parquet)
    selected_index, row = select_row(df, row_index)
    raw_prompt = build_inference_prefix(row)

    print(f"\n[inference] using train row {selected_index}")
    print("[inference] prompt prefix:")
    for message in raw_prompt:
        print(message)

    env_vars = {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
    }
    if rollout_backend == "vllm":
        env_vars["VLLM_LOGGING_LEVEL"] = "INFO"
        env_vars["VLLM_USE_V1"] = "1"

    ray.shutdown()
    ray.init(runtime_env={"env_vars": env_vars}, ignore_reinit_error=True)

    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=os.path.abspath(repo_root / "verl" / "trainer" / "config")):
        overrides = [
            f"actor_rollout_ref.model.path={model_path}",
            f"actor_rollout_ref.rollout.name={rollout_backend}",
            "actor_rollout_ref.rollout.mode=async",
            "actor_rollout_ref.rollout.multi_turn.qwen_tool_list=[image_zoom_in_tool_qwen3vl]",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.data_parallel_size=1",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
            "actor_rollout_ref.rollout.enforce_eager=True",
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=16384",
            "actor_rollout_ref.rollout.prompt_length=8192",
            "actor_rollout_ref.rollout.response_length=2048",
            "actor_rollout_ref.rollout.n=1",
            "actor_rollout_ref.rollout.agent.num_workers=1",
            "actor_rollout_ref.rollout.skip_tokenizer_init=True",
            "trainer.nnodes=1",
            "trainer.n_gpus_per_node=1",
            "reward_model.enable=False",
            "reward_model.use_reward_loop=False",
            f"actor_rollout_ref.rollout.agent.default_agent_loop={agent_loop_name}",
        ]
        if agent_loop_config_path:
            overrides.append(f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_loop_config_path}")
        config = compose(
            config_name="ppo_trainer",
            overrides=overrides,
        )

    agent_loop_manager = AgentLoopManager(config)
    tokenizer = hf_tokenizer(str(model_path))

    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([raw_prompt], dtype=object),
            "agent_name": np.array([agent_loop_name], dtype=object),
            "insight_images_are_presented": np.array([agent_loop_name == "insight_qwen_agent"], dtype=object),
            "data_source": np.array(["qwen3_vl_zoom_smoke"], dtype=object),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}], dtype=object),
        }
    )
    result = agent_loop_manager.generate_sequences(prompts=batch)

    num_turns = int(result.non_tensor_batch["__num_turns__"][0])
    responses = result.batch["responses"]
    response_mask = result.batch["response_mask"]
    attention_mask = result.batch["attention_mask"]
    response_length = response_mask.size(1)

    valid_tokens = responses[0][attention_mask[0][-response_length:].bool()]
    response_with_obs = tokenizer.decode(valid_tokens)

    valid_tokens = responses[0][response_mask[0].bool()]
    response_without_obs = tokenizer.decode(valid_tokens)

    print(f"\n[inference] num_turns = {num_turns}")
    print("\n[inference] decoded response including tool observations:\n")
    print(response_with_obs)
    print("\n[inference] decoded assistant-only response tokens:\n")
    print(response_without_obs)

    ray.shutdown()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    export_dir = Path(args.export_dir).expanduser().resolve()
    base_model = resolve_model_ref(args.base_model)
    work_dir = Path(args.work_dir).expanduser().resolve()
    data_dir = work_dir / "data"
    ckpt_dir = work_dir / "checkpoints"
    smoke_data_dir = work_dir / "smoke_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    smoke_data_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    env.setdefault("PYTHONUNBUFFERED", "1")

    run_command(
        [
            "python",
            "scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py",
            "--input-dir",
            str(export_dir),
            "--output-dir",
            str(data_dir),
            "--val-ratio",
            "0.2",
            "--seed",
            str(args.seed),
            "--system-prompt-mode",
            args.system_prompt_mode,
            "--assistant-format-mode",
            args.assistant_format_mode,
            "--stitch-runtime-hints",
        ],
        cwd=repo_root,
        env=env,
    )

    train_parquet = data_dir / "train.parquet"
    val_parquet = data_dir / "val.parquet"
    train_df = pd.read_parquet(train_parquet)
    val_df = pd.read_parquet(val_parquet) if val_parquet.exists() else pd.DataFrame()

    smoke_train_parquet = smoke_data_dir / "train.parquet"
    smoke_val_parquet = smoke_data_dir / "val.parquet"
    min_train_rows = max(1, args.nproc_per_node * 2)
    target_train_rows = max(min_train_rows, args.train_max_samples)
    selected_train = write_subset(train_df, smoke_train_parquet, max_rows=target_train_rows)
    selected_val = write_subset(val_df, smoke_val_parquet, max_rows=max(1, args.val_max_samples)) if len(val_df) > 0 else None

    if selected_train is None:
        raise RuntimeError("No converted train rows contain assistant tool calls; cannot run smoke test.")

    train_parquet = selected_train
    train_df = pd.read_parquet(train_parquet)
    has_val = selected_val is not None and selected_val.exists() and len(pd.read_parquet(selected_val)) > 0
    if has_val:
        val_parquet = selected_val

    train_batch_size = max(args.nproc_per_node * 2, 2)
    steps_per_epoch = max(1, math.ceil(len(train_df) / train_batch_size))
    total_epochs = max(1, math.ceil(args.total_training_steps / steps_per_epoch))

    rdzv_port = find_free_port()

    sft_cmd = [
        "torchrun",
        "--nnodes=1",
        "--node_rank=0",
        f"--nproc-per-node={args.nproc_per_node}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=127.0.0.1:{rdzv_port}",
        "-m",
        "verl.trainer.sft_trainer",
        f"data.train_files={train_parquet}",
        f"data.train_max_samples={len(train_df)}",
        f"data.val_max_samples={args.val_max_samples}",
        "data.messages_key=messages",
        "data.tools_key=tools",
        f"data.train_batch_size={train_batch_size}",
        "data.micro_batch_size_per_gpu=1",
        "data.use_dynamic_bsz=False",
        "data.max_token_len_per_gpu=32768",
        "data.max_length=16384",
        "data.pad_mode=no_padding",
        "data.truncation=error",
        f"model.path={base_model}",
        "model.use_remove_padding=True",
        "engine=fsdp",
        "optim=fsdp",
        "optim.lr=2e-5",
        "trainer.project_name=qwen3_vl_zoom_smoke",
        "trainer.experiment_name=e2e_smoke",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.total_training_steps={args.total_training_steps}",
        f"trainer.save_freq={args.total_training_steps}",
        "trainer.logger=['console']",
        f"trainer.default_local_dir={ckpt_dir}",
        "trainer.resume_mode=disable",
        "checkpoint.save_contents=[model,optimizer,extra,hf_model]",
    ]
    sft_cmd.append("trainer.test_freq=-1")

    run_command(sft_cmd, cwd=repo_root, env=env)

    hf_checkpoint = latest_hf_checkpoint(ckpt_dir)
    print(f"\n[checkpoint] using HF checkpoint: {hf_checkpoint}")

    run_inference_smoke(
        model_path=hf_checkpoint,
        train_parquet=train_parquet,
        row_index=args.row_index,
        rollout_backend=args.rollout_backend,
        agent_loop_name=args.agent_loop_name,
        agent_loop_config_path=args.agent_loop_config_path,
    )


if __name__ == "__main__":
    main()
