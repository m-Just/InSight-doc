#!/usr/bin/env python3
"""Verify insight_doc RL category/answerability/rescale sampling weights."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


WEIGHTS_FILE = Path(__file__).with_name("insight_doc_rl_category_answerability_rescale_sampling_weights.yaml")

CATEGORY_TARGETS = {
    "arxiv_veqa": 0.175,
    "arxiv_mveqa": 0.175,
    "map_metro": 0.15,
    "map_travel": 0.05,
    "docvqa": 0.10,
    "dude": 0.20,
    "poster": 0.10,
    "info": 0.05,
}
ANSWERABILITY_TARGETS = {
    "answerable": 0.85,
    "unanswerable": 0.15,
}
RESCALE_TARGETS = {
    "r025": 0.7,
    "r035": 0.2,
    "r05": 0.1,
}


def parse_data_source(name: str) -> tuple[str, str, str]:
    parts = name.split("_")
    rescale = parts[-1]
    answerability = parts[-2]
    category = "_".join(parts[:-2])
    return category, answerability, rescale


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: expected {expected:.12f}, got {actual:.12f}")


def main() -> int:
    config = OmegaConf.load(WEIGHTS_FILE)
    weights = {str(key): float(value) for key, value in config.weights.items()}

    expected_keys = {
        f"{category}_{answerability}_{rescale}"
        for category in CATEGORY_TARGETS
        for answerability in ANSWERABILITY_TARGETS
        for rescale in RESCALE_TARGETS
    }
    actual_keys = set(weights)
    if actual_keys != expected_keys:
        raise AssertionError(
            f"weight keys mismatch: missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    assert_close("total weight", sum(weights.values()), 1.0)

    category_marginals = {key: 0.0 for key in CATEGORY_TARGETS}
    answerability_marginals = {key: 0.0 for key in ANSWERABILITY_TARGETS}
    rescale_marginals = {key: 0.0 for key in RESCALE_TARGETS}

    for key, weight in weights.items():
        category, answerability, rescale = parse_data_source(key)
        expected = CATEGORY_TARGETS[category] * ANSWERABILITY_TARGETS[answerability] * RESCALE_TARGETS[rescale]
        assert_close(key, weight, expected)
        category_marginals[category] += weight
        answerability_marginals[answerability] += weight
        rescale_marginals[rescale] += weight

    for key, expected in CATEGORY_TARGETS.items():
        assert_close(f"category marginal {key}", category_marginals[key], expected)
    for key, expected in ANSWERABILITY_TARGETS.items():
        assert_close(f"answerability marginal {key}", answerability_marginals[key], expected)
    for key, expected in RESCALE_TARGETS.items():
        assert_close(f"rescale marginal {key}", rescale_marginals[key], expected)

    print(f"OK: {len(weights)} weights in {WEIGHTS_FILE} match target product distribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
