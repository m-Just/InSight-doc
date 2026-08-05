#!/usr/bin/env python
"""Rewrite SFT parquet tool-call argument JSON into base-model key order."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def rewrite_arguments(arguments: str) -> str:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    if not isinstance(parsed, dict):
        return arguments

    ordered = OrderedDict()
    for key in ("label", "bbox_2d", "img_idx"):
        if key in parsed:
            ordered[key] = parsed[key]
    for key, value in parsed.items():
        if key not in ordered:
            ordered[key] = value
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ": "))


def rewrite_row(row: dict) -> dict:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return row
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                function["arguments"] = rewrite_arguments(arguments)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = pq.read_table(args.input)
    rows = [rewrite_row(row) for row in table.to_pylist()]
    output = pa.Table.from_pylist(rows, schema=table.schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    pq.write_table(output, tmp)
    tmp.replace(args.output)
    print(f"wrote {args.output} rows={output.num_rows}")


if __name__ == "__main__":
    main()
