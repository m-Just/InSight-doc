#!/usr/bin/env python3
"""Create a no-tool/no-system parquet variant.

The transformed parquet has system messages removed from ``prompt``. If a
``tools`` column exists, it is dropped as well.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def _to_python_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return list(value)


def strip_system_messages(prompt: Any) -> list[dict[str, Any]]:
    messages = _to_python_list(prompt)
    out = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            continue
        out.append(message)
    if not out:
        raise ValueError("prompt became empty after removing system messages")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")

    df = pd.read_parquet(args.input)
    if "prompt" not in df.columns:
        raise ValueError(f"missing prompt column in {args.input}")

    before_rows = len(df)
    before_system_rows = sum(
        any(isinstance(msg, dict) and msg.get("role") == "system" for msg in _to_python_list(prompt))
        for prompt in df["prompt"]
    )
    df = df.copy()
    df["prompt"] = df["prompt"].map(strip_system_messages)
    dropped_tools = False
    if "tools" in df.columns:
        df = df.drop(columns=["tools"])
        dropped_tools = True
    after_system_rows = sum(
        any(isinstance(msg, dict) and msg.get("role") == "system" for msg in _to_python_list(prompt))
        for prompt in df["prompt"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"rows={before_rows}")
    print(f"rows_with_system_before={before_system_rows}")
    print(f"rows_with_system_after={after_system_rows}")
    print(f"dropped_tools_column={dropped_tools}")


if __name__ == "__main__":
    main()
