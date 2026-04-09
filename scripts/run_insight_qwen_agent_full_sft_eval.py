#!/usr/bin/env python3
"""Run aligned Qwen3-VL zoom-in SFT and held-out evaluation.

This script waits for separate exported-conversation directories for train and
validation, converts them into SFT parquet files, launches a full-rank SFT run,
and then evaluates the resulting HF checkpoint with the aligned
`InSightQwenAgentLoop` on a held-out evaluation parquet.

Required environment variables:
- WANDB_API_KEY
- WANDB_PROJECT
- WANDB_ENTITY
- OPENAI_API_KEY
- OPENAI_BASE_URL

Recommended environment:
- conda env: vllm-latest
- PYTHONPATH includes local InSight-o3 and Qwen-Agent repos
- VERL_PROJ_DIR points at this repo root
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_SRC = REPO_ROOT.parent
EXTRA_REPOS = (WORKSPACE_SRC / "InSight-o3", WORKSPACE_SRC / "Qwen-Agent")

CONDA_PREFIX = os.environ.get("CONDA_PREFIX")
if CONDA_PREFIX:
    conda_lib = str(Path(CONDA_PREFIX) / "lib")
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in current_ld_library_path.split(":") if part]
    if conda_lib not in ld_parts and os.environ.get("_INSIGHT_QWEN_SFT_REEXEC") != "1":
        new_env = os.environ.copy()
        new_env["LD_LIBRARY_PATH"] = f"{conda_lib}:{current_ld_library_path}" if current_ld_library_path else conda_lib
        new_env["_INSIGHT_QWEN_SFT_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], new_env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-export-dir",
        default=(
            "/scratch/ywxzml3j/likaican/mms1_rl/multi_agent_vsearch/"
            "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/"
            "veqa_batch_0350_mveqa_batch_0352_sample_408_maxp40_simple_prompt_answer_tag_export"
        ),
        help="Train exported-conversation directory.",
    )
    parser.add_argument(
        "--val-export-dir",
        default=(
            "/scratch/ywxzml3j/likaican/mms1_rl/multi_agent_vsearch/"
            "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/"
            "veqa_batch_0350_mveqa_batch_0352_sample_102_maxp40_simple_prompt_answer_tag_export"
        ),
        help="Validation exported-conversation directory.",
    )
    parser.add_argument(
        "--eval-parquet",
        default=(
            "/scratch/ywxzml3j/likaican/temp/"
            "arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.test.parquet"
        ),
        help="Held-out evaluation parquet for judge-based validation.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base model path or HF id for SFT initialization.",
    )
    parser.add_argument(
        "--work-dir",
        default="/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_eval",
        help="Working directory for converted data, checkpoints, and eval outputs.",
    )
    parser.add_argument(
        "--run-name",
        default="arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent",
        help="Base run name for SFT and eval artifacts.",
    )
    parser.add_argument(
        "--system-prompt-mode",
        default="vsearcher_qwen3_vl",
        choices=["exported", "vsearcher_qwen3_vl"],
        help="System prompt mode for SFT conversion.",
    )
    parser.add_argument(
        "--assistant-format-mode",
        default="plain",
        choices=["tagged", "plain"],
        help="Assistant target format for SFT conversion.",
    )
    parser.add_argument(
        "--image-storage-mode",
        default="path",
        choices=["bytes", "path"],
        help="How converted SFT rows store images.",
    )
    parser.add_argument(
        "--convert-workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 2),
        help="Worker processes for exported-conversation conversion.",
    )
    parser.add_argument("--min-train-json-count", type=int, default=408, help="Minimum train export JSON count.")
    parser.add_argument("--min-val-json-count", type=int, default=102, help="Minimum val export JSON count.")
    parser.add_argument("--nproc-per-node", type=int, default=8, help="torchrun local process count for SFT.")
    parser.add_argument("--eval-gpus", type=int, default=4, help="GPU count for held-out evaluation.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Global train batch size for SFT.")
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=1, help="Per-GPU micro batch size for SFT.")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="SFT learning rate.")
    parser.add_argument("--total-epochs", type=int, default=4, help="SFT epochs.")
    parser.add_argument("--max-token-len-per-gpu", type=int, default=65536, help="Token budget per GPU.")
    parser.add_argument("--max-length", type=int, default=65536, help="Dataset max length.")
    parser.add_argument("--eval-max-model-len", type=int, default=32768, help="Max model length for eval rollout.")
    parser.add_argument("--judge-model", default="gpt-5-nano", help="Judge model for held-out evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval while waiting for exports.")
    parser.add_argument(
        "--resume-mode",
        default="auto",
        choices=["auto", "disable", "resume_path"],
        help="SFT resume mode passed to the trainer.",
    )
    parser.add_argument(
        "--resume-from-path",
        default="",
        help="Optional explicit checkpoint path when resume-mode is auto or resume_path.",
    )
    parser.add_argument(
        "--train-cuda-visible-devices",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES override for SFT. Empty keeps current visibility.",
    )
    parser.add_argument(
        "--eval-cuda-visible-devices",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES override for eval. Empty keeps current visibility.",
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Fail immediately if export dirs are not ready instead of waiting.",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def ensure_runtime_env() -> None:
    for repo in EXTRA_REPOS:
        repo_str = str(repo)
        if repo.exists() and repo_str not in sys.path:
            sys.path.insert(0, repo_str)
            current_pythonpath = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = f"{repo_str}:{current_pythonpath}" if current_pythonpath else repo_str
    os.environ.setdefault("VERL_PROJ_DIR", str(REPO_ROOT))
    require_env("WANDB_API_KEY")
    require_env("WANDB_PROJECT")
    require_env("WANDB_ENTITY")
    require_env("OPENAI_API_KEY")
    require_env("OPENAI_BASE_URL")


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
    print(f"\n[run] {printable}\n", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def export_dir_json_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob("*.json"))


def wait_for_exports(
    train_dir: Path,
    val_dir: Path,
    train_min_count: int,
    val_min_count: int,
    poll_seconds: int,
    skip_wait: bool,
) -> None:
    while True:
        train_count = export_dir_json_count(train_dir)
        val_count = export_dir_json_count(val_dir)
        train_ready = train_count >= train_min_count
        val_ready = val_count >= val_min_count
        if train_ready and val_ready:
            print(f"[ready] train exports: {train_count} json files")
            print(f"[ready] val exports: {val_count} json files")
            return
        if skip_wait:
            raise RuntimeError(
                f"Export dirs are not ready: train_ready={train_ready}, val_ready={val_ready}, "
                f"train_count={train_count}, val_count={val_count}, "
                f"train_dir={train_dir}, val_dir={val_dir}"
            )
        print(
            f"[wait] train_count={train_count}/{train_min_count} val_count={val_count}/{val_min_count}; "
            f"sleeping {poll_seconds}s before retry",
            flush=True,
        )
        time.sleep(poll_seconds)


def export_dir_stats(path: Path) -> tuple[int, int]:
    if not path.is_dir():
        return 0, 0
    json_paths = list(path.glob("*.json"))
    if not json_paths:
        return 0, 0
    latest_mtime_ns = max(json_path.stat().st_mtime_ns for json_path in json_paths)
    return len(json_paths), latest_mtime_ns


def parquet_num_rows(path: Path) -> int:
    return pq.read_metadata(path).num_rows


def conversion_meta_path(output_dir: Path) -> Path:
    return output_dir / "conversion_meta.json"


def expected_conversion_meta(input_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    json_count, latest_mtime_ns = export_dir_stats(input_dir)
    return {
        "input_dir": str(input_dir),
        "json_count": json_count,
        "latest_mtime_ns": latest_mtime_ns,
        "system_prompt_mode": args.system_prompt_mode,
        "assistant_format_mode": args.assistant_format_mode,
        "image_storage_mode": args.image_storage_mode,
        "stitch_runtime_hints": True,
        "seed": args.seed,
    }


def can_reuse_converted(output_dir: Path, input_dir: Path, args: argparse.Namespace) -> bool:
    train_parquet = output_dir / "train.parquet"
    meta_path = conversion_meta_path(output_dir)
    if not train_parquet.exists() or not meta_path.exists():
        return False
    try:
        actual_meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    expected_meta = expected_conversion_meta(input_dir, args)
    return actual_meta == expected_meta


def convert_exports(input_dir: Path, output_dir: Path, args: argparse.Namespace, env: dict[str, str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_parquet = output_dir / "train.parquet"
    if can_reuse_converted(output_dir, input_dir, args):
        print(f"[reuse] using cached converted parquet: {train_parquet}", flush=True)
        return train_parquet

    cmd = [
        sys.executable,
        "scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--val-ratio",
        "0.0",
        "--seed",
        str(args.seed),
        "--system-prompt-mode",
        args.system_prompt_mode,
        "--assistant-format-mode",
        args.assistant_format_mode,
        "--image-storage-mode",
        args.image_storage_mode,
        "--num-workers",
        str(args.convert_workers),
        "--stitch-runtime-hints",
    ]
    run_command(cmd, cwd=REPO_ROOT, env=env)
    if not train_parquet.exists():
        raise FileNotFoundError(f"Converter did not create {train_parquet}")
    conversion_meta_path(output_dir).write_text(json.dumps(expected_conversion_meta(input_dir, args), indent=2) + "\n")
    return train_parquet


def latest_hf_checkpoint(ckpt_root: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
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


def read_latest_completed_step(ckpt_root: Path) -> int | None:
    tracker = ckpt_root / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return None
    try:
        return int(tracker.read_text().strip())
    except (OSError, ValueError):
        return None


def prune_incomplete_checkpoints(ckpt_root: Path) -> None:
    latest_completed = read_latest_completed_step(ckpt_root)
    if latest_completed is None:
        return
    for step_dir in ckpt_root.glob("global_step_*"):
        try:
            step = int(step_dir.name.split("_")[-1])
        except ValueError:
            continue
        if step > latest_completed and step_dir.is_dir():
            print(f"[cleanup] removing incomplete checkpoint directory: {step_dir}", flush=True)
            shutil.rmtree(step_dir, ignore_errors=True)


def find_free_port() -> int:
    return random.randint(20000, 45000)


def run_sft(
    args: argparse.Namespace,
    train_parquet: Path,
    val_parquet: Path,
    train_rows: int,
    ckpt_dir: Path,
    env: dict[str, str],
) -> Path:
    if args.train_batch_size % args.nproc_per_node != 0:
        raise ValueError(
            f"train_batch_size={args.train_batch_size} must be divisible by nproc_per_node={args.nproc_per_node}"
        )
    steps_per_epoch = max(1, train_rows // args.train_batch_size)
    rdzv_port = find_free_port()
    sft_run_name = args.run_name
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
        f"data.val_files={val_parquet}",
        f"data.train_max_samples={train_rows}",
        f"data.val_max_samples={parquet_num_rows(val_parquet)}",
        "data.messages_key=messages",
        "data.tools_key=tools",
        "+data.message_loss_mask_key=message_loss_mask",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        "data.use_dynamic_bsz=False",
        f"data.max_token_len_per_gpu={args.max_token_len_per_gpu}",
        f"data.max_length={args.max_length}",
        "data.pad_mode=no_padding",
        "data.truncation=error",
        f"model.path={args.base_model}",
        "model.use_remove_padding=True",
        f"optim.lr={args.learning_rate}",
        "engine=fsdp",
        "optim=fsdp",
        "trainer.logger=['console','wandb']",
        "trainer.project_name=insight_doc",
        f"trainer.experiment_name={sft_run_name}",
        f"trainer.total_epochs={args.total_epochs}",
        f"trainer.test_freq={steps_per_epoch}",
        f"trainer.save_freq={steps_per_epoch}",
        f"trainer.default_local_dir={ckpt_dir}",
        f"trainer.resume_mode={args.resume_mode}",
        "checkpoint.save_contents=[model,optimizer,extra,hf_model]",
    ]
    if args.resume_from_path:
        sft_cmd.append(f"trainer.resume_from_path={args.resume_from_path}")
    run_command(sft_cmd, cwd=REPO_ROOT, env=env)
    return latest_hf_checkpoint(ckpt_dir)


def run_eval(args: argparse.Namespace, model_path: Path, env: dict[str, str]) -> None:
    eval_env = env.copy()
    eval_env.setdefault("OPENAI_CLIENT_TIMEOUT", "60")
    eval_env["MODEL_PATH"] = str(model_path)
    eval_env["WORK_DIR"] = str(Path(args.work_dir).resolve() / "eval")
    eval_tmp_suffix = hashlib.sha1(f"eval:{Path(args.work_dir).resolve()}".encode("utf-8")).hexdigest()[:10]
    eval_tmp_dir = Path("/dev/shm") / f"iqae-eval-{eval_tmp_suffix}"
    eval_tmp_dir.mkdir(parents=True, exist_ok=True)
    eval_env["TMPDIR"] = str(eval_tmp_dir)
    eval_env["TMP"] = str(eval_tmp_dir)
    eval_env["TEMP"] = str(eval_tmp_dir)
    eval_env["RAY_TMPDIR"] = str(eval_tmp_dir / "ray")
    eval_env["PROJECT_NAME"] = "insight_doc"
    eval_env["EXP_NAME"] = f"{args.run_name}.eval_ft"
    eval_env["EVAL_NAME"] = "heldout"
    eval_env["VAL_ONLY"] = "True"
    eval_env["TRAIN_FILES"] = f"[{args.eval_parquet}]"
    eval_env["VAL_FILES"] = f"[{args.eval_parquet}]"
    eval_env["NUM_VAL_TRIALS"] = "1"
    eval_env["JUDGE_MODEL"] = args.judge_model
    eval_env["MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE"] = "200"
    eval_env["LOGGER"] = "['console','wandb']"
    eval_env["WANDB_NAME"] = f"{args.run_name}.eval_ft"
    if args.eval_cuda_visible_devices:
        eval_env["CUDA_VISIBLE_DEVICES"] = args.eval_cuda_visible_devices

    run_experiment_cmd = [
        "/bin/bash",
        "-lc",
        (
            "source recipe/vsearch/_base.sh && "
            "run_experiment "
            f"trainer.n_gpus_per_node={args.eval_gpus} "
            "data.max_prompt_length=17408 "
            "data.validation_max_prompt_length=17408 "
            "actor_rollout_ref.model.custom_chat_template=null "
            "actor_rollout_ref.rollout.n=1 "
            "actor_rollout_ref.rollout.agent.default_agent_loop=insight_qwen_agent "
            "actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/vsearch/config/agent_insight_qwen_agent.yaml "
            "actor_rollout_ref.rollout.multi_turn.qwen_tool_list=[image_zoom_in_tool_qwen3vl] "
            "actor_rollout_ref.rollout.val_kwargs.temperature=0.0 "
            "actor_rollout_ref.rollout.val_kwargs.top_p=1.0 "
            "actor_rollout_ref.rollout.val_kwargs.top_k=-1 "
            f"+actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len={args.eval_max_model_len}"
        ),
    ]
    run_command(run_experiment_cmd, cwd=REPO_ROOT, env=eval_env)


def main() -> None:
    args = parse_args()
    ensure_runtime_env()

    train_export_dir = Path(args.train_export_dir).expanduser().resolve()
    val_export_dir = Path(args.val_export_dir).expanduser().resolve()
    args.eval_parquet = str(Path(args.eval_parquet).expanduser().resolve())
    work_dir = Path(args.work_dir).expanduser().resolve()
    converted_train_dir = work_dir / "converted_train"
    converted_val_dir = work_dir / "converted_val"
    ckpt_dir = work_dir / "sft_checkpoints"
    tmp_suffix = hashlib.sha1(str(work_dir).encode("utf-8")).hexdigest()[:10]
    runtime_tmp_dir = Path("/dev/shm") / f"iqae-{tmp_suffix}"
    wandb_dir = work_dir / "wandb"

    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runtime_tmp_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    prune_incomplete_checkpoints(ckpt_dir)

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["TMPDIR"] = str(runtime_tmp_dir)
    env["TMP"] = str(runtime_tmp_dir)
    env["TEMP"] = str(runtime_tmp_dir)
    env["WANDB_DIR"] = str(wandb_dir)
    env["WANDB_PROJECT"] = require_env("WANDB_PROJECT")
    env["WANDB_ENTITY"] = require_env("WANDB_ENTITY")
    env["WANDB_NAME"] = args.run_name
    if args.train_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.train_cuda_visible_devices

    wait_for_exports(
        train_export_dir,
        val_export_dir,
        args.min_train_json_count,
        args.min_val_json_count,
        args.poll_seconds,
        args.skip_wait,
    )

    train_converted = convert_exports(train_export_dir, converted_train_dir, args, env)
    val_converted = convert_exports(val_export_dir, converted_val_dir, args, env)

    train_rows = parquet_num_rows(train_converted)
    val_rows = parquet_num_rows(val_converted)
    if train_rows == 0:
        raise RuntimeError("Converted train split is empty")
    if val_rows == 0:
        raise RuntimeError("Converted validation split is empty")
    print(f"[data] train rows: {train_rows}")
    print(f"[data] val rows: {val_rows}")

    hf_checkpoint = run_sft(
        args=args,
        train_parquet=train_converted,
        val_parquet=val_converted,
        train_rows=train_rows,
        ckpt_dir=ckpt_dir,
        env=env,
    )
    print(f"[checkpoint] latest hf checkpoint: {hf_checkpoint}")

    run_eval(args=args, model_path=hf_checkpoint, env=env)


if __name__ == "__main__":
    main()
