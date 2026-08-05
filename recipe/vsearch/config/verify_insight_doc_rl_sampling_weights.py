#!/usr/bin/env python3
"""Verify release InSight-Doc RL sampling weights."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


DEFAULT_WEIGHTS_FILE = Path(__file__).with_name("insight_doc_rl_sampling_weights_release.yaml")


def load_weights(path: Path) -> dict[str, float]:
    config = OmegaConf.load(path)
    values = config.get("weights", config)
    if isinstance(values, DictConfig):
        values = OmegaConf.to_container(values, resolve=True)
    if not isinstance(values, dict):
        raise TypeError(f"weights file must contain a mapping or a top-level weights mapping: {path}")
    weights = {str(key): float(value) for key, value in values.items()}
    if not weights:
        raise ValueError(f"weights file is empty: {path}")
    bad = {key: value for key, value in weights.items() if value < 0.0}
    if bad:
        raise ValueError(f"weights must be non-negative: {bad}")
    return weights


def answerability_of(source: str) -> str:
    if "_unanswerable" in source:
        return "unanswerable"
    if "_answerable" in source:
        return "answerable"
    return "unknown"


def category_of(source: str) -> str:
    for suffix in ("_answerable_mc_false_e", "_answerable", "_unanswerable"):
        marker = source.find(suffix)
        if marker != -1:
            return source[:marker]
    return source


def assert_close(name: str, actual: float, expected: float, tol: float) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: expected {expected:.12f}, got {actual:.12f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-file", type=Path, default=DEFAULT_WEIGHTS_FILE)
    parser.add_argument("--expected-answerable", type=float, default=0.86)
    parser.add_argument("--expected-unanswerable", type=float, default=0.14)
    parser.add_argument("--expected-mc-false-e", type=float, default=0.05)
    parser.add_argument("--expected-arxiv-struct", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = load_weights(args.weights_file)

    total = sum(weights.values())
    answerability = defaultdict(float)
    category = defaultdict(float)
    mc_false_e = 0.0
    arxiv_struct = 0.0

    for source, weight in weights.items():
        answerability[answerability_of(source)] += weight
        category[category_of(source)] += weight
        if "mc_false_e" in source:
            mc_false_e += weight
        if source.startswith("arxiv_struct_"):
            arxiv_struct += weight

    assert_close("total", total, 1.0, args.tolerance)
    assert_close("answerable", answerability["answerable"], args.expected_answerable, args.tolerance)
    assert_close("unanswerable", answerability["unanswerable"], args.expected_unanswerable, args.tolerance)
    assert_close("mc_false_e", mc_false_e, args.expected_mc_false_e, args.tolerance)
    assert_close("arxiv_struct", arxiv_struct, args.expected_arxiv_struct, args.tolerance)

    print(f"OK: {len(weights)} source weights in {args.weights_file}")
    print(f"total={total:.12f}")
    print("answerability:")
    for key in sorted(answerability):
        print(f"  {key}: {answerability[key]:.12f}")
    print("category:")
    for key in sorted(category):
        print(f"  {key}: {category[key]:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
