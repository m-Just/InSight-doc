#!/usr/bin/env python3
"""Build deterministic 500-row easy subsets from high-DPI easy sources.

The output is two parquets, one labelled for rescale=0.35 and one for
rescale=0.5.  The rows are sampled from the combined O3 part3a-d and arxiv
part4-5 easy pool, with no overlap between the two labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


GENERATED_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated")

EASY_PARQUETS = [
    GENERATED_ROOT / "O3_data_0424/train_part3a/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "O3_data_0424/train_part3b/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "O3_data_0424/train_part3c/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "O3_data_0424/train_part3d/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "arxiv/train_part4/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "arxiv/train_part5/easy/processed_drop_degenerate/sft_data.parquet",
]

VSEARCHER_SYSTEM_PROMPT = """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
- **Next, find information:** Use a tool to research the things you need to find out.
- **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

Continue this loop until your research is complete.

To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question."""


def normalize_system(row: dict) -> dict:
    """Ensure easy rows are trained under the same tool-capable system prompt."""
    messages = list(row["messages"])
    loss_mask = list(row["message_loss_mask"])
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": VSEARCHER_SYSTEM_PROMPT})
        loss_mask.insert(0, False)
    row = dict(row)
    row["messages"] = messages
    row["message_loss_mask"] = loss_mask
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("notes/generated/highpage_extra_easy500_rescale_samples_20260517"),
    )
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--rows-per-rescale", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in EASY_PARQUETS if not path.exists()]
    if missing:
        raise SystemExit("Missing source parquets:\n" + "\n".join(missing))

    tables = {str(path): pq.read_table(path) for path in EASY_PARQUETS}
    pool: list[tuple[str, int]] = []
    for source, table in tables.items():
        pool.extend((source, idx) for idx in range(table.num_rows))

    needed = args.rows_per_rescale * 2
    if len(pool) < needed:
        raise SystemExit(f"Need {needed} rows but only found {len(pool)}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pool))[:needed]
    scale_to_entries = {
        "035": [pool[int(i)] for i in perm[: args.rows_per_rescale]],
        "05": [pool[int(i)] for i in perm[args.rows_per_rescale :]],
    }

    manifest: dict[str, object] = {
        "seed": args.seed,
        "rows_per_rescale": args.rows_per_rescale,
        "sources": [str(path) for path in EASY_PARQUETS],
        "outputs": {},
    }

    for scale_id, entries in scale_to_entries.items():
        out_path = args.out_root / f"easy_random{args.rows_per_rescale}_rescale{scale_id}_with_vsearcher_system.parquet"
        if out_path.exists() and not args.overwrite:
            print(f"exists: {out_path}", flush=True)
            manifest["outputs"][scale_id] = str(out_path)  # type: ignore[index]
            continue

        rows = []
        source_counts: dict[str, int] = {}
        grouped: dict[str, list[int]] = {}
        for source, idx in entries:
            grouped.setdefault(source, []).append(idx)
            source_counts[source] = source_counts.get(source, 0) + 1

        for source, indices in grouped.items():
            table = tables[source].take(sorted(indices))
            rows.extend(normalize_system(row) for row in table.to_pylist())

        # Shuffle again so rows from the same source are not clustered after grouping.
        rng.shuffle(rows)
        pq.write_table(pa.Table.from_pylist(rows), out_path)
        manifest["outputs"][scale_id] = str(out_path)  # type: ignore[index]
        manifest[f"source_counts_rescale{scale_id}"] = source_counts
        print(f"rescale={scale_id} rows={len(rows)} -> {out_path}", flush=True)

    manifest_path = args.out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
