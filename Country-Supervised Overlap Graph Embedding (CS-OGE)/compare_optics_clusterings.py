#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from overlap_embed.cluster_compare import (
    AssignmentSpec,
    compare_assignments,
    load_assignments,
    parse_assignment_arg,
    write_comparison_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
DEFAULT_ASSIGNMENTS = [
    AssignmentSpec(
        "latent3",
        ANALYSIS_DIR
        / "csoge_ho_dropgt90_obs005_country000_origgroup020_globalog_latent3_17439177"
        / "optics_clusters_single_smoke"
        / "optics_cluster_assignments.tsv",
    ),
    AssignmentSpec(
        "latent6",
        ANALYSIS_DIR
        / "csoge_ho_dropgt90_obs005_country005_origgroup020_17125764"
        / "optics_clusters_default_sweep"
        / "optics_cluster_assignments.tsv",
    ),
    AssignmentSpec(
        "latent20",
        ANALYSIS_DIR
        / "csoge_ho_dropgt90_obs005_country000_origgroup020_globalog_latent20_17630186"
        / "optics_clusters_default_sweep"
        / "optics_cluster_assignments.tsv",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare OPTICS clustering assignments across CS-OGE latent dimensions.",
    )
    parser.add_argument(
        "--assignment",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Assignment TSV to compare. May be repeated. Defaults to latent3, latent6, latent20.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ANALYSIS_DIR / "optics_latent_dim_comparison_3d_6d_20d",
        help="Directory for metrics, overlap matrices, and heatmaps.",
    )
    parser.add_argument("--cluster_col", default="cluster_id")
    parser.add_argument("--sample_col", default="sample_id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    specs = [parse_assignment_arg(value) for value in args.assignment] if args.assignment else DEFAULT_ASSIGNMENTS
    assignments = load_assignments(
        specs,
        sample_col=args.sample_col,
        cluster_col=args.cluster_col,
    )
    results = compare_assignments(assignments)
    summary = write_comparison_outputs(
        output_dir=args.output_dir,
        specs=specs,
        assignments=assignments,
        results=results,
        sample_col=args.sample_col,
        cluster_col=args.cluster_col,
    )
    print(f"Wrote metrics: {summary['metrics_tsv']}")
    print(f"Wrote overlap matrices: {summary['overlap_matrices_dir']}")
    print(f"Wrote heatmaps: {summary['overlap_heatmaps_dir']}")
    print(f"Common samples across all runs: {summary['n_common_samples_all_runs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
