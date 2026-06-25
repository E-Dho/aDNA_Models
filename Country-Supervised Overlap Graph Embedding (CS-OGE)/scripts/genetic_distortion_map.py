#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from overlap_embed.geo import run_distortion_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a genetic distortion map by aligning latent 2D structure to geography."
    )
    parser.add_argument("--latents_csv", required=True, help="CSV with sample_id and latent columns")
    parser.add_argument("--metadata_tsv", required=True, help="TSV with sample IDs and coordinates")
    parser.add_argument("--lat_col", default="latitude")
    parser.add_argument("--lon_col", default="longitude")
    parser.add_argument("--id_col", default="sample_id")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--time_col", default=None)
    parser.add_argument("--reduce_method", choices=("pca", "none"), default="pca")
    parser.add_argument("--n_components", type=int, default=2)
    parser.add_argument(
        "--pc_indices",
        default=None,
        help="Optional comma-separated 1-based PCA components or raw dimensions for 2D, e.g. '1,3'",
    )
    parser.add_argument("--project_coords", type=int, choices=(0, 1), default=1)
    parser.add_argument("--crs", default=None, help="Optional projected CRS string, e.g. EPSG:3035")
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--plot_arrows", type=int, choices=(0, 1), default=1)
    parser.add_argument("--plot_labels", type=int, choices=(0, 1), default=0)
    parser.add_argument("--arrow_scale", type=float, default=None)
    parser.add_argument("--arrow_alpha_mode", choices=("density", "fixed"), default="density")
    parser.add_argument("--arrow_alpha_min", type=float, default=0.05)
    parser.add_argument("--arrow_alpha_max", type=float, default=0.65)
    parser.add_argument(
        "--aligned_color_mode",
        choices=("sample", "label", "time", "fixed"),
        default="label",
        help="How to assign matching colors in aligned_vs_true.png",
    )
    parser.add_argument("--knn_k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_distortion_pipeline(
            latents_csv=Path(args.latents_csv),
            metadata_tsv=Path(args.metadata_tsv),
            out_dir=Path(args.out_dir),
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            id_col=args.id_col,
            label_col=args.label_col,
            time_col=args.time_col,
            reduce_method=args.reduce_method,
            n_components=args.n_components,
            pc_indices=args.pc_indices,
            project_coords=args.project_coords,
            crs=args.crs,
            min_samples=args.min_samples,
            plot_arrows=args.plot_arrows,
            plot_labels=args.plot_labels,
            arrow_scale=args.arrow_scale,
            arrow_alpha_mode=args.arrow_alpha_mode,
            arrow_alpha_min=args.arrow_alpha_min,
            arrow_alpha_max=args.arrow_alpha_max,
            aligned_color_mode=args.aligned_color_mode,
            knn_k=args.knn_k,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    metrics = result["metrics"]
    summary = {
        "out_dir": result["out_dir"],
        "n_samples": result["n_samples"],
        "rms_alignment_error": metrics["rms_alignment_error"],
        "median_distortion_magnitude": metrics["median_distortion_magnitude"],
        "pairwise_latent_geo_pearson": metrics["pairwise_latent_geo_pearson"],
        "pairwise_latent_geo_spearman": metrics["pairwise_latent_geo_spearman"],
    }
    print(json.dumps(summary, indent=2))
    print("Artifacts:")
    for name in (
        "distortion_vectors.tsv",
        "distortion_metrics.json",
        "distortion_map.png",
        "aligned_vs_true.png",
        "distortion_histogram.png",
        "distance_correlation.png",
    ):
        print(f"  {Path(result['out_dir']) / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
