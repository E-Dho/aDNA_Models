#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from overlap_embed.data import load_sample_stats_tsv, write_tsv
from overlap_embed.eval import read_latents_csv, write_latents_csv
from overlap_embed.oos_encoder import evaluate_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate out-of-sample encoder predictions against teacher CS-OGE latents.")
    parser.add_argument("--predicted_latents_csv", required=True)
    parser.add_argument("--teacher_latents_csv", required=True)
    parser.add_argument("--sample_stats_tsv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_ids, predicted = read_latents_csv(Path(args.predicted_latents_csv))
    teacher_ids, teacher = read_latents_csv(Path(args.teacher_latents_csv))
    teacher_by_id = {sample_id: idx for idx, sample_id in enumerate(teacher_ids)}
    stats_by_id = {str(row["sample_id"]): row for row in load_sample_stats_tsv(Path(args.sample_stats_tsv))}

    keep_pred_idx = []
    matched_ids = []
    target_rows = []
    stats_rows = []
    coverage = []
    for pred_idx, sample_id in enumerate(pred_ids):
        teacher_idx = teacher_by_id.get(sample_id)
        stats = stats_by_id.get(sample_id)
        if teacher_idx is None or stats is None:
            continue
        keep_pred_idx.append(pred_idx)
        matched_ids.append(sample_id)
        target_rows.append(teacher[teacher_idx])
        stats_rows.append(stats)
        coverage.append(float(stats["observed_fraction"]))

    if not matched_ids:
        raise SystemExit("No matched sample IDs across predicted latents, teacher latents, and sample stats.")

    pred_matched = predicted[np.asarray(keep_pred_idx, dtype=np.int64)]
    target = np.asarray(target_rows, dtype=np.float32)
    coverage_arr = np.asarray(coverage, dtype=np.float32)
    metrics = evaluate_predictions(
        predicted=pred_matched,
        target=target,
        sample_ids=matched_ids,
        sample_stats_tsv=Path(args.sample_stats_tsv),
        coverage=coverage_arr,
        k=args.k,
    )
    metrics["n_matched_samples"] = int(len(matched_ids))
    metrics["latent_dim"] = int(pred_matched.shape[1])

    write_latents_csv(output_dir / "matched_predicted_latents.csv", matched_ids, pred_matched)
    write_latents_csv(output_dir / "matched_teacher_latents.csv", matched_ids, target)
    write_tsv(
        output_dir / "matched_sample_stats.tsv",
        stats_rows,
        ["sample_id", "observed_fraction", "country", "original_group_id", "date_mean_bp"],
    )
    with (output_dir / "oos_evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
