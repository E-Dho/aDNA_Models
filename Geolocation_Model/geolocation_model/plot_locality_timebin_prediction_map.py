#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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

AGE_BIN_ORDER = ["0-1000 BP", "1000-2500 BP", "2500-4000 BP", "4000-6000 BP", ">6000 BP"]
AGE_BIN_COLORS = {
    "0-1000 BP": "#1b9e77",
    "1000-2500 BP": "#377eb8",
    "2500-4000 BP": "#984ea3",
    "4000-6000 BP": "#ff7f00",
    ">6000 BP": "#e41a1c",
}


def mean_float(rows: Sequence[Dict[str, str]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except ValueError:
            pass
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def aggregate_locality_timebins(rows: Sequence[Dict[str, str]], min_samples: int) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        locality = row.get("locality_id", "missing") or "missing"
        age_bin = row.get("age_bin", "missing") or "missing"
        groups[(locality, age_bin)].append(row)

    out = []
    for (locality, age_bin), rs in groups.items():
        if len(rs) < min_samples:
            continue
        true_lat = mean_float(rs, "true_lat")
        true_lon = mean_float(rs, "true_lon")
        pred_lat = mean_float(rs, "pred_lat")
        pred_lon = mean_float(rs, "pred_lon")
        error = mean_float(rs, "error_km")
        alpha_precision = mean_float(rs, "alpha_precision") if "alpha_precision" in rs[0] else float("nan")
        age = mean_float(rs, "age") if "age" in rs[0] else float("nan")
        if not np.isfinite([true_lat, true_lon, pred_lat, pred_lon]).all():
            continue
        out.append({
            "locality_id": locality,
            "age_bin": age_bin,
            "n_samples": len(rs),
            "true_lat_mean": true_lat,
            "true_lon_mean": true_lon,
            "pred_lat_mean": pred_lat,
            "pred_lon_mean": pred_lon,
            "mean_error_km": error,
            "mean_alpha_precision": alpha_precision,
            "mean_age_bp": age,
        })
    order = {label: i for i, label in enumerate(AGE_BIN_ORDER)}
    out.sort(key=lambda r: (str(r["locality_id"]), order.get(str(r["age_bin"]), 99)))
    return out


def write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "locality_id",
        "age_bin",
        "n_samples",
        "true_lat_mean",
        "true_lon_mean",
        "pred_lat_mean",
        "pred_lon_mean",
        "mean_error_km",
        "mean_alpha_precision",
        "mean_age_bp",
        "arrow_density_alpha",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{float(row[k]):.8f}" if isinstance(row.get(k), float) else row.get(k, "")) for k in fieldnames})


def render(args: argparse.Namespace) -> Dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = finite_filter(read_predictions(Path(args.predictions_tsv)))
    aggregated = aggregate_locality_timebins(rows, min_samples=args.min_samples_per_arrow)
    if not aggregated:
        raise ValueError("No locality/time-bin groups remain after aggregation")

    true_lon = np.asarray([r["true_lon_mean"] for r in aggregated], dtype=np.float64)
    true_lat = np.asarray([r["true_lat_mean"] for r in aggregated], dtype=np.float64)
    pred_lon = np.asarray([r["pred_lon_mean"] for r in aggregated], dtype=np.float64)
    pred_lat = np.asarray([r["pred_lat_mean"] for r in aggregated], dtype=np.float64)
    dx = pred_lon - true_lon
    dy = pred_lat - true_lat
    arrow_alpha = compute_arrow_density_alpha(
        true_lon,
        true_lat,
        dx,
        dy,
        alpha_min=args.arrow_alpha_min,
        alpha_max=args.arrow_alpha_max,
        grid_size=args.density_grid_size,
    )
    for row, alpha in zip(aggregated, arrow_alpha):
        row["arrow_density_alpha"] = float(alpha)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "locality_timebin_prediction_vectors.tsv"
    write_tsv(tsv_path, aggregated)

    lon_min = args.lon_min
    lon_max = args.lon_max
    lat_min = args.lat_min
    lat_max = args.lat_max

    try:
        import cartopy.crs as ccrs

        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
    except Exception:
        fig, ax = plt.subplots(figsize=(12, 10))

    map_geojson = Path(args.map_geojson).resolve() if args.map_geojson else None
    data_crs = add_map_background(ax, lon_min, lon_max, lat_min, lat_max, map_geojson=map_geojson)
    transform_kwargs = {"transform": data_crs} if data_crs is not None else {}

    ax.scatter(true_lon, true_lat, s=10, color="#222222", alpha=0.65, linewidths=0, zorder=4, label="locality centroid", **transform_kwargs)
    ax.scatter(pred_lon, pred_lat, s=np.clip(np.asarray([r["n_samples"] for r in aggregated], dtype=np.float64) * 6.0, 12, 90), color="#f6f0d0", edgecolors="#333333", linewidths=0.25, alpha=0.8, zorder=5, label="mean prediction", **transform_kwargs)

    for age_bin in AGE_BIN_ORDER:
        mask = np.asarray([r["age_bin"] == age_bin for r in aggregated], dtype=bool)
        if not mask.any():
            continue
        draw_quiver_by_alpha_bins(
            ax,
            true_lon[mask],
            true_lat[mask],
            dx[mask],
            dy[mask],
            arrow_alpha[mask],
            transform=data_crs,
            color=AGE_BIN_COLORS[age_bin],
            n_bins=args.alpha_bins,
        )

    legend_handles = [Line2D([0], [0], color=AGE_BIN_COLORS[label], lw=3, label=label) for label in AGE_BIN_ORDER]
    legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#222222", markersize=6, label="true locality centroid"))
    legend_handles.append(Line2D([0], [0], marker="o", color="#333333", markerfacecolor="#f6f0d0", markersize=7, label="mean prediction"))
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, fontsize=8)
    ax.set_title(args.title)
    fig.tight_layout()

    png_path = out_dir / args.output_png
    svg_path = out_dir / args.output_svg if args.output_svg else None
    fig.savefig(png_path, dpi=args.dpi)
    if svg_path:
        fig.savefig(svg_path)
    plt.close(fig)

    summary = {
        "predictions_tsv": str(Path(args.predictions_tsv).resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_input_samples": len(rows),
        "n_locality_timebin_arrows": len(aggregated),
        "min_samples_per_arrow": args.min_samples_per_arrow,
        "age_bins": AGE_BIN_ORDER,
        "map_geojson": str(map_geojson) if map_geojson else None,
        "files": {
            "png": str(png_path.resolve()),
            "svg": str(svg_path.resolve()) if svg_path else None,
            "vectors_tsv": str(tsv_path.resolve()),
        },
    }
    (out_dir / "locality_timebin_prediction_map_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map locality-level mean prediction arrows by age bin.")
    parser.add_argument("--predictions_tsv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--map_geojson", default=None)
    parser.add_argument("--output_png", default="locality_timebin_prediction_map.png")
    parser.add_argument("--output_svg", default="locality_timebin_prediction_map.svg")
    parser.add_argument("--title", default="Locality mean predicted coordinates by time bin")
    parser.add_argument("--min_samples_per_arrow", type=int, default=1)
    parser.add_argument("--lon_min", type=float, default=-12.0)
    parser.add_argument("--lon_max", type=float, default=45.0)
    parser.add_argument("--lat_min", type=float, default=30.0)
    parser.add_argument("--lat_max", type=float, default=69.0)
    parser.add_argument("--arrow_alpha_min", type=float, default=0.08)
    parser.add_argument("--arrow_alpha_max", type=float, default=0.78)
    parser.add_argument("--density_grid_size", type=int, default=90)
    parser.add_argument("--alpha_bins", type=int, default=8)
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
