#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def read_predictions(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction TSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample_id", "true_lat", "true_lon", "pred_lat", "pred_lon", "error_km"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def as_float(rows: Sequence[Dict[str, str]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        raw = row.get(key, "")
        try:
            vals.append(float(raw))
        except ValueError:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float64)


def finite_filter(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        try:
            vals = [float(row[k]) for k in ("true_lat", "true_lon", "pred_lat", "pred_lon", "error_km")]
        except ValueError:
            continue
        if all(math.isfinite(v) for v in vals):
            out.append(row)
    return out


def compute_arrow_density_alpha(
    x: Sequence[float],
    y: Sequence[float],
    dx: Sequence[float],
    dy: Sequence[float],
    *,
    alpha_min: float = 0.03,
    alpha_max: float = 0.65,
    grid_size: int = 120,
) -> np.ndarray:
    """Return per-arrow transparency from midpoint density on a deterministic 2D grid."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    dx_arr = np.asarray(dx, dtype=np.float64)
    dy_arr = np.asarray(dy, dtype=np.float64)
    if not (x_arr.shape == y_arr.shape == dx_arr.shape == dy_arr.shape):
        raise ValueError("x, y, dx, and dy must have matching shapes")
    if x_arr.size == 0:
        return np.asarray([], dtype=np.float64)

    lo = float(min(alpha_min, alpha_max))
    hi = float(max(alpha_min, alpha_max))
    mid_x = x_arr + 0.5 * dx_arr
    mid_y = y_arr + 0.5 * dy_arr
    finite = np.isfinite(mid_x) & np.isfinite(mid_y)
    if not finite.any():
        return np.full(x_arr.shape, lo, dtype=np.float64)

    grid = max(int(grid_size), 2)
    x_edges = np.linspace(float(np.nanmin(mid_x[finite])), float(np.nanmax(mid_x[finite])), grid + 1)
    y_edges = np.linspace(float(np.nanmin(mid_y[finite])), float(np.nanmax(mid_y[finite])), grid + 1)
    if np.isclose(x_edges[0], x_edges[-1]):
        x_edges = np.asarray([x_edges[0] - 0.5, x_edges[0] + 0.5], dtype=np.float64)
    if np.isclose(y_edges[0], y_edges[-1]):
        y_edges = np.asarray([y_edges[0] - 0.5, y_edges[0] + 0.5], dtype=np.float64)

    hist, _, _ = np.histogram2d(mid_x[finite], mid_y[finite], bins=(x_edges, y_edges))
    x_bin = np.clip(np.searchsorted(x_edges, mid_x, side="right") - 1, 0, hist.shape[0] - 1)
    y_bin = np.clip(np.searchsorted(y_edges, mid_y, side="right") - 1, 0, hist.shape[1] - 1)
    density = hist[x_bin, y_bin].astype(np.float64)
    density[~finite] = 1.0

    low = float(np.percentile(density[finite], 5))
    high = float(np.percentile(density[finite], 95))
    if high <= low:
        return np.full(x_arr.shape, (lo + hi) / 2.0, dtype=np.float64)
    normalized = np.clip((density - low) / (high - low), 0.0, 1.0)
    return lo + normalized * (hi - lo)


def draw_geojson_boundaries(ax, geojson_path: Path, *, transform=None, linewidth: float = 0.45, color: str = "#666666") -> bool:
    try:
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    def iter_rings(geometry):
        if not geometry:
            return
        gtype = geometry.get("type")
        coords = geometry.get("coordinates")
        if gtype == "Polygon":
            for ring in coords or []:
                yield ring
        elif gtype == "MultiPolygon":
            for poly in coords or []:
                for ring in poly:
                    yield ring
        elif gtype == "GeometryCollection":
            for geom in geometry.get("geometries", []):
                yield from iter_rings(geom)

    features = payload.get("features", []) if payload.get("type") == "FeatureCollection" else [{"geometry": payload}]
    drawn = 0
    for feature in features:
        for ring in iter_rings(feature.get("geometry")) or []:
            arr = np.asarray(ring, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            kwargs = {"transform": transform} if transform is not None else {}
            ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=linewidth, alpha=0.85, zorder=1, **kwargs)
            drawn += 1
    return drawn > 0


def add_map_background(ax, lon_min: float, lon_max: float, lat_min: float, lat_max: float, map_geojson: Path | None = None):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.25, linewidth=0.5)
        drew_geojson = False
        if map_geojson is not None and map_geojson.is_file():
            drew_geojson = draw_geojson_boundaries(ax, map_geojson)
        ax.text(
            0.01,
            0.01,
            "Cartopy unavailable: " + ("using supplied GeoJSON boundaries" if drew_geojson else "drawing lon/lat frame without coastlines"),
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
            ha="left",
            va="bottom",
        )
        return None

    data_crs = ccrs.PlateCarree()
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1efe6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d8e6ef", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, edgecolor="#555555", zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="#888888", zorder=1)
    if map_geojson is not None and map_geojson.is_file():
        draw_geojson_boundaries(ax, map_geojson, transform=data_crs, linewidth=0.25, color="#444444")
    gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="#999999", alpha=0.45, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    return data_crs


def draw_quiver_by_alpha_bins(ax, x, y, dx, dy, alpha_values, *, transform=None, color="#1f1f1f", n_bins=10):
    if len(x) == 0:
        return
    lo = float(np.nanmin(alpha_values))
    hi = float(np.nanmax(alpha_values))
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        bins = np.asarray([lo - 0.5, hi + 0.5], dtype=np.float64)
    else:
        bins = np.linspace(lo, hi, max(int(n_bins), 1) + 1)
    for idx in range(len(bins) - 1):
        if idx == len(bins) - 2:
            mask = (alpha_values >= bins[idx]) & (alpha_values <= bins[idx + 1])
        else:
            mask = (alpha_values >= bins[idx]) & (alpha_values < bins[idx + 1])
        if not mask.any():
            continue
        alpha = float(np.nanmean(alpha_values[mask]))
        kwargs = {}
        if transform is not None:
            kwargs["transform"] = transform
        ax.quiver(
            x[mask],
            y[mask],
            dx[mask],
            dy[mask],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0022,
            headwidth=3.8,
            headlength=4.8,
            headaxislength=4.2,
            color=color,
            alpha=alpha,
            zorder=3,
            **kwargs,
        )


def write_augmented_vectors(path: Path, rows: Sequence[Dict[str, str]], arrow_alpha: np.ndarray) -> None:
    fieldnames = [
        "sample_id",
        "true_lat",
        "true_lon",
        "pred_lat",
        "pred_lon",
        "error_km",
        "alpha_precision",
        "arrow_density_alpha",
        "sequencing_type",
        "age",
        "age_bin",
        "locality_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row, alpha in zip(rows, arrow_alpha):
            writer.writerow({
                "sample_id": row.get("sample_id", ""),
                "true_lat": row.get("true_lat", ""),
                "true_lon": row.get("true_lon", ""),
                "pred_lat": row.get("pred_lat", ""),
                "pred_lon": row.get("pred_lon", ""),
                "error_km": row.get("error_km", ""),
                "alpha_precision": row.get("alpha_precision", ""),
                "arrow_density_alpha": f"{float(alpha):.8f}",
                "sequencing_type": row.get("sequencing_type", ""),
                "age": row.get("age", ""),
                "age_bin": row.get("age_bin", ""),
                "locality_id": row.get("locality_id", ""),
            })


def parse_highlight_ids(value: str | None) -> list[str]:
    if not value:
        return []
    raw = value
    maybe_path = Path(value)
    if maybe_path.is_file():
        raw = maybe_path.read_text(encoding="utf-8")
    ids = []
    for token in raw.replace("\n", ",").replace(";", ",").split(","):
        token = token.strip().strip('"').strip("'")
        if token and token not in ids:
            ids.append(token)
    return ids


def write_highlight_status(path: Path, requested: Sequence[str], rows: Sequence[Dict[str, str]]) -> list[Dict[str, str]]:
    by_id = {row.get("sample_id", ""): row for row in rows}
    status = []
    fieldnames = ["sample_id", "status", "true_lat", "true_lon", "pred_lat", "pred_lon", "error_km", "split", "age", "age_bin", "locality_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for sid in requested:
            row = by_id.get(sid)
            if row is None:
                out = {"sample_id": sid, "status": "missing_from_predictions"}
            else:
                out = {
                    "sample_id": sid,
                    "status": "found",
                    "true_lat": row.get("true_lat", ""),
                    "true_lon": row.get("true_lon", ""),
                    "pred_lat": row.get("pred_lat", ""),
                    "pred_lon": row.get("pred_lon", ""),
                    "error_km": row.get("error_km", ""),
                    "split": row.get("split", ""),
                    "age": row.get("age", ""),
                    "age_bin": row.get("age_bin", ""),
                    "locality_id": row.get("locality_id", ""),
                }
            for key in fieldnames:
                out.setdefault(key, "")
            writer.writerow(out)
            status.append(out)
    return status


def make_map(args: argparse.Namespace) -> Dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = finite_filter(read_predictions(Path(args.predictions_tsv)))
    highlight_ids = parse_highlight_ids(args.highlight_ids)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No finite rows available after filtering")

    true_lat = as_float(rows, "true_lat")
    true_lon = as_float(rows, "true_lon")
    pred_lat = as_float(rows, "pred_lat")
    pred_lon = as_float(rows, "pred_lon")
    error = as_float(rows, "error_km")
    alpha_precision = as_float(rows, "alpha_precision") if "alpha_precision" in rows[0] else np.full(len(rows), np.nan)

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_augmented_vectors(out_dir / "prediction_coordinate_vectors.tsv", rows, arrow_alpha)
    highlight_status = write_highlight_status(out_dir / "highlight_sample_status.tsv", highlight_ids, rows) if highlight_ids else []

    lon_min = args.lon_min if args.lon_min is not None else float(np.nanmin(np.r_[true_lon, pred_lon])) - 2.0
    lon_max = args.lon_max if args.lon_max is not None else float(np.nanmax(np.r_[true_lon, pred_lon])) + 2.0
    lat_min = args.lat_min if args.lat_min is not None else float(np.nanmin(np.r_[true_lat, pred_lat])) - 2.0
    lat_max = args.lat_max if args.lat_max is not None else float(np.nanmax(np.r_[true_lat, pred_lat])) + 2.0

    projection = None
    try:
        import cartopy.crs as ccrs

        projection = ccrs.PlateCarree()
        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=projection)
    except Exception:
        fig, ax = plt.subplots(figsize=(12, 10))

    map_geojson = Path(args.map_geojson).resolve() if args.map_geojson else None
    data_crs = add_map_background(ax, lon_min, lon_max, lat_min, lat_max, map_geojson=map_geojson)
    transform_kwargs = {"transform": data_crs} if data_crs is not None else {}

    true_color = "#1f77b4"
    pred_color = "#d62728"
    ax.scatter(true_lon, true_lat, s=args.true_point_size, color=true_color, alpha=0.65, linewidths=0, zorder=4, label="true", **transform_kwargs)
    pred_scatter = ax.scatter(
        pred_lon,
        pred_lat,
        s=args.pred_point_size,
        c=error,
        cmap="magma_r",
        alpha=0.78,
        linewidths=0,
        zorder=5,
        label="predicted",
        **transform_kwargs,
    )
    draw_quiver_by_alpha_bins(ax, true_lon, true_lat, dx, dy, arrow_alpha, transform=data_crs, color=args.arrow_color, n_bins=args.alpha_bins)

    if highlight_ids:
        by_id = {row.get("sample_id", ""): row for row in rows}
        for sid in highlight_ids:
            row = by_id.get(sid)
            if row is None:
                continue
            tlon = float(row["true_lon"])
            tlat = float(row["true_lat"])
            plon = float(row["pred_lon"])
            plat = float(row["pred_lat"])
            kwargs = {"transform": data_crs} if data_crs is not None else {}
            ax.quiver(
                [tlon],
                [tlat],
                [plon - tlon],
                [plat - tlat],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=args.highlight_arrow_width,
                headwidth=5.5,
                headlength=7.0,
                headaxislength=6.0,
                color=args.highlight_color,
                alpha=1.0,
                zorder=30,
                **kwargs,
            )
            ax.scatter([tlon], [tlat], s=args.highlight_point_size, color=args.highlight_color, edgecolors="white", linewidths=1.1, marker="o", zorder=31, **kwargs)
            ax.scatter([plon], [plat], s=args.highlight_point_size, color=args.highlight_color, edgecolors="black", linewidths=1.1, marker="X", zorder=32, **kwargs)
            ax.text(tlon, tlat, f"{sid} true", color=args.highlight_color, fontsize=args.highlight_label_size, fontweight="bold", zorder=33, **kwargs)
            ax.text(plon, plat, f"{sid} pred", color=args.highlight_color, fontsize=args.highlight_label_size, fontweight="bold", zorder=33, **kwargs)

    cbar = fig.colorbar(pred_scatter, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label("Prediction error (km)")
    ax.set_title(args.title)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    png_path = out_dir / args.output_png
    fig.savefig(png_path, dpi=args.dpi)
    if args.output_svg:
        fig.savefig(out_dir / args.output_svg)
    plt.close(fig)

    summary = {
        "predictions_tsv": str(Path(args.predictions_tsv).resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_samples": int(len(rows)),
        "median_error_km": float(np.nanmedian(error)),
        "mean_error_km": float(np.nanmean(error)),
        "p90_error_km": float(np.nanpercentile(error, 90)),
        "mean_alpha_precision": float(np.nanmean(alpha_precision)),
        "median_alpha_precision": float(np.nanmedian(alpha_precision)),
        "mean_arrow_density_alpha": float(np.nanmean(arrow_alpha)),
        "median_arrow_density_alpha": float(np.nanmedian(arrow_alpha)),
        "extent": {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max},
        "map_geojson": str(map_geojson) if map_geojson else None,
        "highlight_ids": highlight_ids,
        "highlight_status": highlight_status,
        "files": {
            "png": str(png_path.resolve()),
            "svg": str((out_dir / args.output_svg).resolve()) if args.output_svg else None,
            "vectors_tsv": str((out_dir / "prediction_coordinate_vectors.tsv").resolve()),
            "highlight_status_tsv": str((out_dir / "highlight_sample_status.tsv").resolve()) if highlight_ids else None,
        },
    }
    (out_dir / "prediction_coordinate_map_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map predicted sample coordinates against true coordinates with density-alpha arrows.")
    parser.add_argument("--predictions_tsv", required=True, help="Prediction TSV containing true_lat/true_lon/pred_lat/pred_lon/error_km")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--output_png", default="prediction_coordinate_map.png")
    parser.add_argument("--output_svg", default="prediction_coordinate_map.svg")
    parser.add_argument("--map_geojson", default=None, help="Optional GeoJSON with country/coastline polygons to draw without Cartopy")
    parser.add_argument("--title", default="Predicted vs true sample coordinates")
    parser.add_argument("--lon_min", type=float, default=-12.0)
    parser.add_argument("--lon_max", type=float, default=45.0)
    parser.add_argument("--lat_min", type=float, default=30.0)
    parser.add_argument("--lat_max", type=float, default=69.0)
    parser.add_argument("--arrow_alpha_min", type=float, default=0.025)
    parser.add_argument("--arrow_alpha_max", type=float, default=0.72)
    parser.add_argument("--density_grid_size", type=int, default=120)
    parser.add_argument("--alpha_bins", type=int, default=12)
    parser.add_argument("--arrow_color", default="#171717")
    parser.add_argument("--true_point_size", type=float, default=9.0)
    parser.add_argument("--pred_point_size", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional first-N sample limit for quick previews")
    parser.add_argument("--highlight_ids", default=None, help="Comma-separated sample IDs or file containing IDs to highlight")
    parser.add_argument("--highlight_color", default="#00a651")
    parser.add_argument("--highlight_point_size", type=float, default=95.0)
    parser.add_argument("--highlight_arrow_width", type=float, default=0.0048)
    parser.add_argument("--highlight_label_size", type=float, default=8.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = make_map(args)
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
