#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = Path(os.environ.get("VERL_ROOT", REPO_ROOT / "verl")).resolve()
for extra_path in (REPO_ROOT, VERL_ROOT):
    if extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from evals.backends.ray_vllm_servers import cleanup_ray_vllm_servers, launch_ray_vllm_servers
from evals.config.model import apply_model_config, load_model_config, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a standalone ray/vLLM model pool for evals/rollout.py.")
    parser.add_argument("--model-config", dest="model_config", required=True)
    parser.add_argument("--server-manifest", required=True)
    parser.add_argument("--heartbeat-path")
    parser.add_argument("--idle-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--ray-address")
    parser.add_argument("--ray-namespace")
    parser.add_argument("--ray-temp-dir")
    parser.add_argument("--ray-num-cpus", type=int)
    parser.add_argument("--ray-cpus-per-server", type=float, default=4.0)
    return parser.parse_args()


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


async def main_async() -> None:
    args = parse_args()
    model_config_data = load_model_config(args.model_config)
    model_config_sha256 = sha256_file(args.model_config)
    apply_model_config(args, model_config_data)
    if args.generation_backend != "ray_vllm":
        raise ValueError("scripts/serve_ray_vllm.py only supports backend=ray_vllm")

    manifest_path = Path(args.server_manifest).resolve()
    heartbeat_path = Path(args.heartbeat_path).resolve() if args.heartbeat_path else manifest_path.with_suffix(".heartbeat")
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path.touch()

    shutdown_event = threading.Event()

    def handle_signal(signum, _frame) -> None:
        print(f"received signal {signum}; shutting down Ray vLLM servers", flush=True)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    handles = []
    try:
        handles, actor_names, server_metadata, namespace = await launch_ray_vllm_servers(args)
        manifest = {
            "backend": "ray_vllm",
            "status": "running",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "serve_pid": os.getpid(),
            "model": args.model,
            "model_config": {
                "path": str(Path(args.model_config).resolve()),
                "sha256": model_config_sha256,
                "data": model_config_data,
            },
            "ray": {
                "address": getattr(args, "_ray_address", None),
                "namespace": namespace,
                "actor_names": actor_names,
            },
            "server_metadata": server_metadata,
            "heartbeat_path": str(heartbeat_path),
            "idle_timeout_seconds": args.idle_timeout_seconds,
        }
        write_manifest(manifest_path, manifest)
        print(f"Ray vLLM server manifest written to {manifest_path}", flush=True)

        while not shutdown_event.wait(30.0):
            if args.idle_timeout_seconds <= 0:
                continue
            try:
                last_used = heartbeat_path.stat().st_mtime
            except FileNotFoundError:
                last_used = time.time()
                heartbeat_path.touch()
            idle_seconds = time.time() - last_used
            if idle_seconds >= args.idle_timeout_seconds:
                print(
                    f"Ray vLLM server idle for {idle_seconds:.1f}s "
                    f"(timeout={args.idle_timeout_seconds:.1f}s); shutting down",
                    flush=True,
                )
                break
    finally:
        cleanup_ray_vllm_servers(handles)
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["status"] = "stopped"
                manifest["stopped_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                write_manifest(manifest_path, manifest)
            except Exception:
                pass
        print("Ray vLLM servers stopped", flush=True)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
