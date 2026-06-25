#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_ld_pca_lle import evaluate_latents, write_metrics_csv


def read_sample_stats(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "observed_fraction": float(row["observed_fraction"]),
                    "country": row.get("country", ""),
                    "original_group_id": row.get("original_group_id", ""),
                    "date_mean_bp": float(row.get("date_mean_bp") or "nan"),
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate latent matrix against sample_stats.tsv")
    p.add_argument("--latents_npy", required=True)
    p.add_argument("--sample_stats_tsv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--primary_k", type=int, default=32)
    p.add_argument("--compat_k", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latents = np.load(args.latents_npy)
    rows = read_sample_stats(Path(args.sample_stats_tsv))
    if latents.shape[0] != len(rows):
        raise SystemExit(f"ERROR: latent/sample count mismatch: {latents.shape[0]} vs {len(rows)}")
    metrics = evaluate_latents(latents, rows, ks=[args.primary_k, args.compat_k], seed=args.seed)
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    write_metrics_csv(output_dir / "metrics.csv", metrics)
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
