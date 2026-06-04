#!/usr/bin/env python3
"""Dump a resolved verl Hydra config without launching training.

Use this by reusing the overrides from an existing verl launch script:

  python scripts/dump_verl_resolved_config.py \
    --output /tmp/resolved_verl_config.yaml \
    -- actor_rollout_ref.rollout.prompt_length=262144 ...

The ``--`` separator is intentional: everything after it is passed to Hydra as
an override. This script registers verl's ``eval`` resolver before composition,
which avoids resolver-order issues with Hydra's generic ``--cfg job --resolve``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="verl/trainer/config")
    parser.add_argument("--config-name", default="ppo_trainer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.overrides and args.overrides[0] == "--":
        args.overrides = args.overrides[1:]
    return args


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_path).expanduser().resolve()
    if not any(item.startswith("project_root=") or item.startswith("+project_root=") for item in args.overrides):
        args.overrides.insert(0, f"project_root={Path.cwd().resolve()}")
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
