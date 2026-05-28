#!/usr/bin/env python3
"""Create deterministic random easy-data subsets for LoRA ablations."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import pyarrow.parquet as pq


EASY_PARQUETS = [
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part1/easy/processed_drop_degenerate/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2a/easy/processed_drop_degenerate/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2b/easy/processed_drop_degenerate/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2c/easy/processed_drop_degenerate/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/dude_poster_unanswerable/easy/processed_drop_degenerate/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part1/easy/processed_drop_degenerate/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part2/easy/processed_drop_degenerate/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part3/easy/processed_drop_degenerate/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part4/easy/processed_drop_degenerate/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part5/easy/processed_drop_degenerate/sft_data.parquet",
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/spanning_train_part1/easy/processed_drop_degenerate/sft_data.parquet",
]


def stable_indices(n_rows: int, fraction: float, seed: int, source: str) -> list[int]:
    n_keep = int(round(n_rows * fraction))
    n_keep = min(n_rows, max(1 if n_rows else 0, n_keep))
    key = f"{seed}:{fraction:.12g}:{source}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    rng_seed = int.from_bytes(digest[:8], "big", signed=False)

    # Use NumPy only if pyarrow is available in this env; its permutation is fast and stable.
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    return sorted(int(i) for i in rng.choice(n_rows, size=n_keep, replace=False))


def frac_label(fraction: float) -> str:
    return f"{int(math.floor(fraction * 1000 + 0.5)):03d}"


def rel_output_path(source: str, out_root: Path, fraction: float) -> Path:
    path = Path(source)
    try:
        rel = path.relative_to("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated")
    except ValueError:
        rel = Path(path.name)
    return out_root / f"frac_{frac_label(fraction)}" / rel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("notes/generated/easy_fraction_samples_20260516"),
    )
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--fraction", type=float, action="append", default=[0.5, 0.25])
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for fraction in args.fraction:
        if not (0 < fraction <= 1):
            raise SystemExit(f"invalid fraction: {fraction}")
        for source in EASY_PARQUETS:
            src = Path(source)
            if not src.exists():
                raise SystemExit(f"missing source parquet: {src}")
            table = pq.read_table(src)
            indices = stable_indices(table.num_rows, fraction, args.seed, source)
            sampled = table.take(indices)
            out_path = rel_output_path(source, args.out_root, fraction)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(sampled, out_path)
            manifest_rows.append(
                (
                    frac_label(fraction),
                    f"{fraction:.12g}",
                    source,
                    str(out_path),
                    str(table.num_rows),
                    str(sampled.num_rows),
                )
            )
            print(
                f"fraction={fraction:.4g} rows={sampled.num_rows}/{table.num_rows} "
                f"{src} -> {out_path}",
                flush=True,
            )

    manifest = args.out_root / "manifest.tsv"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("fraction_label\tfraction\tsource\toutput\tinput_rows\toutput_rows\n")
        for row in manifest_rows:
            f.write("\t".join(row) + "\n")
    print(f"manifest={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
