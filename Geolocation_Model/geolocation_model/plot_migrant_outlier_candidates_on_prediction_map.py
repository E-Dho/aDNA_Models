#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_prediction_coordinate_map import (  # noqa: E402
    add_map_background,
    as_float,
    compute_arrow_density_alpha,
    draw_quiver_by_alpha_bins,
    finite_filter,
    read_predictions,
)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"TSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite_candidate_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    required = ["sample_id", "lat", "lon", "knn_mean_lat", "knn_mean_lon", "knn_geo_distance_km"]
    out = []
    for row in rows:
        try:
            vals = [float(row[k]) for k in required[1:]]
        except (KeyError, ValueError):
            continue
        if all(np.isfinite(vals)):
            out.append(row)
    return out


def write_candidate_vectors(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fieldnames = [
        "sample_id",
        "country",
        "original_group_id",
        "date_mean_bp",
        "lat",
        "lon",
        "knn_mean_lat",
        "knn_mean_lon",
        "knn_geo_distance_km",
        "observed_fraction",
        "adaptive_original_group_purity",
        "neighbor_countries",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def render(args: argparse.Namespace) -> Dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    background_rows = finite_filter(read_predictions(Path(args.predictions_tsv)))
    candidates = finite_candidate_rows(read_tsv(Path(args.candidates_tsv)))
    if not candidates:
        raise ValueError("No finite candidate rows found")

    true_lat = as_float(background_rows, "true_lat")
    true_lon = as_float(background_rows, "true_lon")
    pred_lat = as_float(background_rows, "pred_lat")
    pred_lon = as_float(background_rows, "pred_lon")
    error = as_float(background_rows, "error_km")
    dx = pred_lon - true_lon
    dy = pred_lat - true_lat
    bg_alpha = compute_arrow_density_alpha(
        true_lon,
        true_lat,
        dx,
        dy,
        alpha_min=args.background_arrow_alpha_min,
        alpha_max=args.background_arrow_alpha_max,
        grid_size=args.background_density_grid_size,
    )

    cand_lat = np.asarray([float(r["lat"]) for r in candidates], dtype=np.float64)
    cand_lon = np.asarray([float(r["lon"]) for r in candidates], dtype=np.float64)
    cand_dst_lat = np.asarray([float(r["knn_mean_lat"]) for r in candidates], dtype=np.float64)
    cand_dst_lon = np.asarray([float(r["knn_mean_lon"]) for r in candidates], dtype=np.float64)
    cand_dx = cand_dst_lon - cand_lon
    cand_dy = cand_dst_lat - cand_lat

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / "migrant_outlier_candidate_vectors.tsv"
    write_candidate_vectors(vectors_path, candidates)

    try:
        import cartopy.crs as ccrs

        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
    except Exception:
        fig, ax = plt.subplots(figsize=(12, 10))

    map_geojson = Path(args.map_geojson).resolve() if args.map_geojson else None
    data_crs = add_map_background(ax, args.lon_min, args.lon_max, args.lat_min, args.lat_max, map_geojson=map_geojson)
    transform_kwargs = {"transform": data_crs} if data_crs is not None else {}

    ax.scatter(true_lon, true_lat, s=5, color="#2c7fb8", alpha=0.25, linewidths=0, zorder=3, label="background true", **transform_kwargs)
    ax.scatter(pred_lon, pred_lat, s=6, c=error, cmap="magma_r", alpha=0.24, linewidths=0, zorder=3, label="background predicted", **transform_kwargs)
    draw_quiver_by_alpha_bins(
        ax,
        true_lon,
        true_lat,
        dx,
        dy,
        bg_alpha,
        transform=data_crs,
        color=args.background_arrow_color,
        n_bins=args.background_alpha_bins,
    )

    ax.quiver(
        cand_lon,
        cand_lat,
        cand_dx,
        cand_dy,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=args.candidate_arrow_width,
        headwidth=5.5,
        headlength=7.0,
        headaxislength=6.0,
        color=args.candidate_color,
        alpha=args.candidate_arrow_alpha,
        zorder=40,
        **transform_kwargs,
    )
    ax.scatter(cand_lon, cand_lat, s=args.candidate_point_size, color=args.candidate_color, edgecolors="white", linewidths=1.0, marker="o", zorder=41, label="candidate true", **transform_kwargs)
    ax.scatter(cand_dst_lon, cand_dst_lat, s=args.candidate_point_size, color=args.candidate_color, edgecolors="black", linewidths=1.0, marker="X", zorder=42, label="candidate kNN mean", **transform_kwargs)

    if args.label_candidates:
        for row, x0, y0, x1, y1 in zip(candidates, cand_lon, cand_lat, cand_dst_lon, cand_dst_lat):
            sid = row.get("sample_id", "")
            ax.text(x0, y0, sid, color=args.label_color, alpha=args.label_alpha, fontsize=args.label_size, fontweight="bold", zorder=43, **transform_kwargs)
            if args.label_destination:
                ax.text(x1, y1, f"{sid} kNN", color=args.label_color, alpha=args.label_alpha, fontsize=args.label_size, fontweight="bold", zorder=43, **transform_kwargs)

    ax.legend(loc="lower left", frameon=True, fontsize=8)
    ax.set_title(args.title)
    fig.tight_layout()
    png_path = out_dir / args.output_png
    svg_path = out_dir / args.output_svg if args.output_svg else None
    fig.savefig(png_path, dpi=args.dpi)
    if svg_path:
        fig.savefig(svg_path)
    plt.close(fig)

    distances = np.asarray([float(r["knn_geo_distance_km"]) for r in candidates], dtype=np.float64)
    summary = {
        "predictions_tsv": str(Path(args.predictions_tsv).resolve()),
        "candidates_tsv": str(Path(args.candidates_tsv).resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_background_samples": int(len(background_rows)),
        "n_candidates": int(len(candidates)),
        "median_candidate_knn_geo_distance_km": float(np.median(distances)),
        "mean_candidate_knn_geo_distance_km": float(np.mean(distances)),
        "background_arrow_alpha_min": args.background_arrow_alpha_min,
        "background_arrow_alpha_max": args.background_arrow_alpha_max,
        "files": {
            "png": str(png_path.resolve()),
            "svg": str(svg_path.resolve()) if svg_path else None,
            "candidate_vectors_tsv": str(vectors_path.resolve()),
        },
    }
    (out_dir / "migrant_outlier_candidate_map_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay migrant outlier displacement candidates on a faint geolocation prediction map.")
    parser.add_argument("--predictions_tsv", required=True)
    parser.add_argument("--candidates_tsv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--map_geojson", default=None)
    parser.add_argument("--output_png", default="migrant_outlier_candidates_on_prediction_map.png")
    parser.add_argument("--output_svg", default="migrant_outlier_candidates_on_prediction_map.svg")
    parser.add_argument("--title", default="Migrant outlier candidates over geolocation displacement background")
    parser.add_argument("--lon_min", type=float, default=-12.0)
    parser.add_argument("--lon_max", type=float, default=45.0)
    parser.add_argument("--lat_min", type=float, default=30.0)
    parser.add_argument("--lat_max", type=float, default=69.0)
    parser.add_argument("--background_arrow_alpha_min", type=float, default=0.005)
    parser.add_argument("--background_arrow_alpha_max", type=float, default=0.22)
    parser.add_argument("--background_density_grid_size", type=int, default=120)
    parser.add_argument("--background_alpha_bins", type=int, default=10)
    parser.add_argument("--background_arrow_color", default="#111111")
    parser.add_argument("--candidate_color", default="#00a651")
    parser.add_argument("--candidate_arrow_width", type=float, default=0.0048)
    parser.add_argument("--candidate_arrow_alpha", type=float, default=0.62)
    parser.add_argument("--candidate_point_size", type=float, default=72.0)
    parser.add_argument("--label_candidates", type=int, choices=(0, 1), default=1)
    parser.add_argument("--label_destination", type=int, choices=(0, 1), default=0)
    parser.add_argument("--label_size", type=float, default=7.2)
    parser.add_argument("--label_color", default="#000000")
    parser.add_argument("--label_alpha", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        print(json.dumps(render(args), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
