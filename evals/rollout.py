#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

VERL_ROOT = Path(os.environ.get("VERL_ROOT", REPO_ROOT / "verl")).resolve()
extra_paths = [REPO_ROOT, VERL_ROOT]

for extra_path in extra_paths:
    if extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from evals.config.agent import apply_agent_settings_to_args, load_agent_settings
from evals.config.model import (
    apply_model_config,
    load_model_config,
    normalize_https_max_retries,
    normalize_https_timeout,
    semantic_model_config_sha256,
    sha256_file,
)
from evals.core.orchestrator import run_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InSight-Doc rollout-only eval. Generates and exports conversations without scoring.",
    )
    parser.add_argument("--model-config", dest="model_config", required=True, help="Standalone model/inference config YAML.")
    parser.add_argument("--val-files", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--agent-config",
        default="evals/agent_configs/insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml",
    )
    parser.add_argument(
        "--agent-config-override",
        action="append",
        default=[],
        help="Dotlist override applied to --agent-config; repeatable, e.g. images.initial_rescale=0.35.",
    )
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--shuffle-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle loaded val rows before trial expansion. Enabled by default for cross-benchmark load balancing.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--agent-worker-processes", type=int, default=1)
    parser.add_argument(
        "--worker-concurrency",
        type=int,
        required=True,
        help=(
            "Ray/vLLM: per-agent-worker concurrency, so total concurrency is "
            "agent_worker_processes * worker_concurrency. HTTPS: total async request concurrency."
        ),
    )
    parser.add_argument("--ray-server-manifest", help="Manifest written by scripts/serve_ray_vllm.py.")
    parser.add_argument("--context-overflow-max-halving-trials", type=int, default=4)
    parser.add_argument(
        "--https-timeout-override",
        help=(
            "Override https_openai_chat.timeout from --model-config without changing resume identity. "
            "Use only for transport retries against the same model/data/config."
        ),
    )
    parser.add_argument(
        "--https-max-retries-override",
        help=(
            "Override https_openai_chat.max_retries from --model-config without changing resume identity. "
            "Use only for transport retries against the same model/data/config."
        ),
    )
    args = parser.parse_args()

    model_config_data = load_model_config(args.model_config)
    model_config_sha256 = sha256_file(args.model_config)
    model_config_semantic_sha256 = semantic_model_config_sha256(model_config_data)
    apply_model_config(args, model_config_data)
    args._model_config_data = model_config_data
    args._model_config_sha256 = model_config_sha256
    args._model_config_semantic_sha256 = model_config_semantic_sha256
    args.custom_chat_template_file = None
    args.logprobs = False

    if args.generation_backend == "https_openai_chat":
        if args.https_timeout_override is not None:
            args.https_timeout = normalize_https_timeout(args.https_timeout_override)
        if args.https_max_retries_override is not None:
            args.https_max_retries = normalize_https_max_retries(args.https_max_retries_override)

    if args.max_samples == 0:
        parser.error("--max-samples=0 is not useful for eval; use -1 for uncapped or a positive cap")
    if args.generation_backend == "ray_vllm" and not args.ray_server_manifest:
        parser.error("--ray-server-manifest is required for backend=ray_vllm")
    if args.generation_backend == "https_openai_chat" and not args.https_base_url:
        parser.error("--model-config https_openai_chat.base_url or OPENAI_BASE_URL is required")
    if args.agent_worker_processes < 1:
        parser.error("--agent-worker-processes must be >= 1")
    if args.worker_concurrency < 1:
        parser.error("--worker-concurrency must be >= 1")
    return args


async def async_main(args: argparse.Namespace) -> None:
    agent_settings, agent_name = load_agent_settings(args.agent_config, args.agent_config_override)
    apply_agent_settings_to_args(args, agent_settings)
    export_dir = Path(args.output_dir) / "exported_conversations"
    if args.generation_backend == "ray_vllm":
        from evals.backends.ray_vllm import RayVLLMBackend

        backend = RayVLLMBackend(args, agent_settings, export_dir)
    elif args.generation_backend == "https_openai_chat":
        from evals.backends.https_openai_chat import HTTPSOpenAIChatBackend

        backend = HTTPSOpenAIChatBackend(args, agent_settings, export_dir)
    else:
        raise ValueError(f"unsupported generation backend: {args.generation_backend}")
    await run_rollout(args, backend=backend, agent_settings=agent_settings, agent_name=agent_name)


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
