#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from overlap_embed.outlier import maybe_write_plots, score_migrant_outliers, write_outlier_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect candidate migrant outliers from CS-OGE latent-space neighborhoods."
    )
    parser.add_argument("--latents_csv", required=True, help="final_latents.csv or compatible latent CSV")
    parser.add_argument("--sample_stats_tsv", required=True, help="sample_stats.tsv for the same samples")
    parser.add_argument("--anno", required=True, help=".anno file containing Lat. and Long. columns")
    parser.add_argument("--output_dir", required=True, help="Directory for outlier reports")
    parser.add_argument("--k", type=int, default=15, help="Number of latent nearest neighbors")
    parser.add_argument("--epsilon_km", type=float, default=None, help="Manual geographic displacement threshold")
    parser.add_argument(
        "--epsilon_quantile",
        type=float,
        default=0.95,
        help="Auto threshold quantile among high adaptive-purity samples",
    )
    parser.add_argument(
        "--adaptive_purity_min",
        type=float,
        default=0.80,
        help="Minimum per-sample adaptive original-group purity for epsilon calibration",
    )
    parser.add_argument(
        "--min_original_group_size",
        type=int,
        default=2,
        help="Minimum original-group size for original-group isolation flagging",
    )
    parser.add_argument(
        "--max_rank1_neighbor_latent_distance",
        type=float,
        default=0.6,
        help="Drop geographic candidates whose closest latent neighbor is farther than this distance",
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip optional PNG diagnostics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scores, calibration, summary = score_migrant_outliers(
        latents_csv=Path(args.latents_csv),
        sample_stats_tsv=Path(args.sample_stats_tsv),
        anno_path=Path(args.anno),
        k=args.k,
        epsilon_km=args.epsilon_km,
        epsilon_quantile=args.epsilon_quantile,
        adaptive_purity_min=args.adaptive_purity_min,
        min_original_group_size=args.min_original_group_size,
        max_rank1_neighbor_latent_distance=args.max_rank1_neighbor_latent_distance,
    )
    output_dir = Path(args.output_dir)
    write_outlier_outputs(scores=scores, calibration=calibration, summary=summary, output_dir=output_dir)
    if not args.no_plots:
        summary["plots"] = maybe_write_plots(scores, output_dir)
        write_outlier_outputs(scores=scores, calibration=calibration, summary=summary, output_dir=output_dir)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
