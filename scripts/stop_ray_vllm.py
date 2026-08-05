#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = Path(os.environ.get("VERL_ROOT", REPO_ROOT / "verl")).resolve()
for extra_path in (REPO_ROOT, VERL_ROOT):
    if extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop Ray/vLLM servers from a standalone server manifest.")
    parser.add_argument("--server-manifest", required=True)
    parser.add_argument("--no-kill-serve-process", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.server_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ray_info = manifest.get("ray") or {}
    address = ray_info.get("address") or manifest.get("ray_address") or "auto"
    namespace = ray_info.get("namespace") or manifest.get("ray_namespace")
    actor_names = list(ray_info.get("actor_names") or manifest.get("actor_names") or [])

    try:
        import ray

        ray.init(address=address, namespace=namespace, ignore_reinit_error=True)
        for actor_name in actor_names:
            try:
                actor = ray.get_actor(actor_name, namespace=namespace)
                ray.kill(actor, no_restart=True)
                print(f"killed Ray actor {actor_name}", flush=True)
            except Exception as exc:
                print(f"warning: failed to kill Ray actor {actor_name}: {exc}", flush=True)
        ray.shutdown()
    except Exception as exc:
        print(f"warning: failed to connect to Ray for cleanup: {exc}", flush=True)

    serve_pid = manifest.get("serve_pid")
    if serve_pid and not args.no_kill_serve_process and int(serve_pid) != os.getpid():
        try:
            os.kill(int(serve_pid), signal.SIGTERM)
            print(f"sent SIGTERM to serve process pid={serve_pid}", flush=True)
        except ProcessLookupError:
            print(f"serve process pid={serve_pid} is not running", flush=True)
        except Exception as exc:
            print(f"warning: failed to terminate serve process pid={serve_pid}: {exc}", flush=True)


if __name__ == "__main__":
    main()
