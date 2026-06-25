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
    compute_arrow_density_alpha,
    draw_quiver_by_alpha_bins,
    finite_filter,
    read_predictions,
)

SEQ_COLORS = {
    "Shotgun": "#1b9e77",
    "1240k": "#377eb8",
}
SEQ_ORDER = ["Shotgun", "1240k"]


def selected_seq_order(value: str) -> list[str]:
    if value == "both":
        return list(SEQ_ORDER)
    return [value]


def mean_float(rows: Sequence[Dict[str, str]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except ValueError:
            pass
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(rows: Sequence[Dict[str, str]], min_samples: int, allowed_seq: set[str]) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        seq = row.get("sequencing_type", "")
        if seq not in allowed_seq:
            continue
        locality = row.get("locality_id", "missing") or "missing"
        groups[(locality, seq)].append(row)
    out = []
    for (locality, seq), rs in groups.items():
        if len(rs) < min_samples:
            continue
        out.append({
            "locality_id": locality,
            "sequencing_type": seq,
            "n_samples": len(rs),
            "true_lat_mean": mean_float(rs, "true_lat"),
            "true_lon_mean": mean_float(rs, "true_lon"),
            "pred_lat_mean": mean_float(rs, "pred_lat"),
            "pred_lon_mean": mean_float(rs, "pred_lon"),
            "mean_error_km": mean_float(rs, "error_km"),
            "mean_alpha_precision": mean_float(rs, "alpha_precision"),
        })
    out = [r for r in out if np.isfinite([r["true_lat_mean"], r["true_lon_mean"], r["pred_lat_mean"], r["pred_lon_mean"]]).all()]
    out.sort(key=lambda r: (str(r["sequencing_type"]), str(r["locality_id"])))
    return out


def write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "locality_id",
        "sequencing_type",
        "n_samples",
        "true_lat_mean",
        "true_lon_mean",
        "pred_lat_mean",
        "pred_lon_mean",
        "mean_error_km",
        "mean_alpha_precision",
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
    seq_order = selected_seq_order(args.sequencing_method)
    allowed_seq = set(seq_order)
    agg = aggregate(rows, min_samples=args.min_samples_per_arrow, allowed_seq=allowed_seq)
    if not agg:
        raise ValueError("No Shotgun/1240k locality groups remain after filtering")

    true_lon = np.asarray([r["true_lon_mean"] for r in agg], dtype=np.float64)
    true_lat = np.asarray([r["true_lat_mean"] for r in agg], dtype=np.float64)
    pred_lon = np.asarray([r["pred_lon_mean"] for r in agg], dtype=np.float64)
    pred_lat = np.asarray([r["pred_lat_mean"] for r in agg], dtype=np.float64)
    dx = pred_lon - true_lon
    dy = pred_lat - true_lat
    alpha = compute_arrow_density_alpha(true_lon, true_lat, dx, dy, alpha_min=args.arrow_alpha_min, alpha_max=args.arrow_alpha_max, grid_size=args.density_grid_size)
    for row, a in zip(agg, alpha):
        row["arrow_density_alpha"] = float(a)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vector_path = out_dir / "locality_sequencing_prediction_vectors.tsv"
    write_tsv(vector_path, agg)

    try:
        import cartopy.crs as ccrs

        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
    except Exception:
        fig, ax = plt.subplots(figsize=(12, 10))

    map_geojson = Path(args.map_geojson).resolve() if args.map_geojson else None
    data_crs = add_map_background(ax, args.lon_min, args.lon_max, args.lat_min, args.lat_max, map_geojson=map_geojson)
    transform_kwargs = {"transform": data_crs} if data_crs is not None else {}

    ax.scatter(true_lon, true_lat, s=10, color="#222222", alpha=0.55, linewidths=0, zorder=4, label="true locality centroid", **transform_kwargs)
    ax.scatter(
        pred_lon,
        pred_lat,
        s=np.clip(np.asarray([r["n_samples"] for r in agg], dtype=np.float64) * 7.0, 14, 95),
        color="#f6f0d0",
        edgecolors="#333333",
        linewidths=0.25,
        alpha=0.82,
        zorder=5,
        label="mean prediction",
        **transform_kwargs,
    )

    seq_arr = np.asarray([r["sequencing_type"] for r in agg], dtype=object)
    for seq in seq_order:
        mask = seq_arr == seq
        if not mask.any():
            continue
        draw_quiver_by_alpha_bins(ax, true_lon[mask], true_lat[mask], dx[mask], dy[mask], alpha[mask], transform=data_crs, color=SEQ_COLORS[seq], n_bins=args.alpha_bins)

    legend = [Line2D([0], [0], color=SEQ_COLORS[seq], lw=3, label=seq) for seq in seq_order]
    legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#222222", markersize=6, label="true locality centroid"))
    legend.append(Line2D([0], [0], marker="o", color="#333333", markerfacecolor="#f6f0d0", markersize=7, label="mean prediction"))
    ax.legend(handles=legend, loc="lower left", frameon=True, fontsize=8)
    ax.set_title(args.title)
    fig.tight_layout()

    png_path = out_dir / args.output_png
    svg_path = out_dir / args.output_svg if args.output_svg else None
    fig.savefig(png_path, dpi=args.dpi)
    if svg_path:
        fig.savefig(svg_path)
    plt.close(fig)

    counts = {seq: int(sum(1 for r in rows if r.get("sequencing_type") == seq)) for seq in seq_order}
    arrow_counts = {seq: int(sum(1 for r in agg if r["sequencing_type"] == seq)) for seq in seq_order}
    summary = {
        "predictions_tsv": str(Path(args.predictions_tsv).resolve()),
        "out_dir": str(out_dir.resolve()),
        "included_sample_counts": counts,
        "locality_arrow_counts": arrow_counts,
        "n_locality_sequencing_arrows": len(agg),
        "sequencing_method_filter": args.sequencing_method,
        "excluded_rule": "only selected exact sequencing_type values are visualized",
        "files": {"png": str(png_path.resolve()), "svg": str(svg_path.resolve()) if svg_path else None, "vectors_tsv": str(vector_path.resolve())},
    }
    (out_dir / "locality_sequencing_prediction_map_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map locality mean prediction arrows for Shotgun vs 1240k samples only.")
    parser.add_argument("--predictions_tsv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--map_geojson", default=None)
    parser.add_argument("--output_png", default="locality_sequencing_prediction_map.png")
    parser.add_argument("--output_svg", default="locality_sequencing_prediction_map.svg")
    parser.add_argument("--title", default="Locality mean prediction arrows by sequencing method")
    parser.add_argument("--sequencing_method", choices=["both", "Shotgun", "1240k"], default="both")
    parser.add_argument("--min_samples_per_arrow", type=int, default=1)
    parser.add_argument("--lon_min", type=float, default=-12.0)
    parser.add_argument("--lon_max", type=float, default=45.0)
    parser.add_argument("--lat_min", type=float, default=30.0)
    parser.add_argument("--lat_max", type=float, default=69.0)
    parser.add_argument("--arrow_alpha_min", type=float, default=0.20)
    parser.add_argument("--arrow_alpha_max", type=float, default=0.85)
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
