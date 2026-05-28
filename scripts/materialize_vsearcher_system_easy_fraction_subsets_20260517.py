#!/usr/bin/env python3
"""Materialize vsearcher-system variants for existing easy-fraction subsets."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import pyarrow.parquet as pq


GENERATED_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated")

BASIC_EASY_PARQUETS = [
    GENERATED_ROOT / "O3_data_0424/train_part1/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "O3_data_0424/train_part2a/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "O3_data_0424/train_part2b/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "O3_data_0424/train_part2c/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "O3_data_0424/dude_poster_unanswerable/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "arxiv/train_part1/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "arxiv/train_part2/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "arxiv/train_part3/easy/processed_drop_degenerate/sft_data.parquet",
    GENERATED_ROOT / "arxiv/spanning_train_part1/easy/processed_drop_degenerate/sft_data.parquet",
]


def stable_indices(n_rows: int, fraction: float, seed: int, source: str) -> list[int]:
    n_keep = int(round(n_rows * fraction))
    n_keep = min(n_rows, max(1 if n_rows else 0, n_keep))
    key = f"{seed}:{fraction:.12g}:{source}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    rng_seed = int.from_bytes(digest[:8], "big", signed=False)

    import numpy as np

    rng = np.random.default_rng(rng_seed)
    return sorted(int(i) for i in rng.choice(n_rows, size=n_keep, replace=False))


def frac_label(fraction: float) -> str:
    return f"{int(math.floor(fraction * 1000 + 0.5)):03d}"


def subset_output_path(source: Path, out_root: Path, fraction: float) -> Path:
    rel = source.relative_to(GENERATED_ROOT)
    return out_root / f"frac_{frac_label(fraction)}" / rel.parent / "sft_data_with_vsearcher_system.parquet"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("notes/generated/easy_fraction_samples_20260516"),
    )
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--fraction", type=float, action="append", default=[0.5, 0.25])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = []
    for fraction in args.fraction:
        if not (0 < fraction <= 1):
            raise SystemExit(f"invalid fraction: {fraction}")
        for source in BASIC_EASY_PARQUETS:
            vsearcher_source = source.with_name("sft_data_with_vsearcher_system.parquet")
            if not source.exists():
                raise SystemExit(f"missing source parquet: {source}")
            if not vsearcher_source.exists():
                raise SystemExit(f"missing vsearcher-system parquet: {vsearcher_source}")

            source_meta = pq.read_metadata(source)
            vsearcher_table = pq.read_table(vsearcher_source)
            if source_meta.num_rows != vsearcher_table.num_rows:
                raise SystemExit(
                    f"row-count mismatch: {source} has {source_meta.num_rows}, "
                    f"{vsearcher_source} has {vsearcher_table.num_rows}"
                )

            out_path = subset_output_path(source, args.out_root, fraction)
            if out_path.exists() and not args.overwrite:
                print(f"exists: {out_path}", flush=True)
                continue

            indices = stable_indices(source_meta.num_rows, fraction, args.seed, str(source))
            sampled = vsearcher_table.take(indices)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(sampled, out_path)
            rows.append(
                (
                    frac_label(fraction),
                    f"{fraction:.12g}",
                    str(vsearcher_source),
                    str(out_path),
                    str(vsearcher_table.num_rows),
                    str(sampled.num_rows),
                )
            )
            print(
                f"fraction={fraction:.4g} rows={sampled.num_rows}/{vsearcher_table.num_rows} "
                f"{vsearcher_source} -> {out_path}",
                flush=True,
            )

    manifest = args.out_root / "manifest_vsearcher_system.tsv"
    if rows or not manifest.exists():
        with manifest.open("w", encoding="utf-8") as f:
            f.write("fraction_label\tfraction\tsource\toutput\tinput_rows\toutput_rows\n")
            for row in rows:
                f.write("\t".join(row) + "\n")
        print(f"manifest={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
