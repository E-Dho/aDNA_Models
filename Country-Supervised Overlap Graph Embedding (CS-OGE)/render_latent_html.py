#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from overlap_embed.eval import read_latents_csv


def _parse_axes(text: str) -> List[int]:
    axes = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(axes) not in (2, 3):
        raise argparse.ArgumentTypeError("--axes must contain exactly 2 or 3 comma-separated integers")
    if any(axis < 1 for axis in axes):
        raise argparse.ArgumentTypeError("--axes are 1-based and must be >= 1")
    return axes


def _read_latents(latents_csv: Path | None, latents_npy: Path | None, sample_stats: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    if (latents_csv is None) == (latents_npy is None):
        raise ValueError("Provide exactly one of --latents_csv or --latents_npy")
    if latents_csv is not None:
        sample_ids, latents = read_latents_csv(latents_csv)
        return sample_ids, latents.astype(np.float32)
    assert latents_npy is not None
    latents = np.load(latents_npy).astype(np.float32)
    sample_ids = sample_stats["sample_id"].astype(str).tolist()
    if latents.shape[0] != len(sample_ids):
        raise ValueError(
            f"latents_npy row count {latents.shape[0]} does not match sample_stats rows {len(sample_ids)}"
        )
    return sample_ids, latents


def _project_axes(latents: np.ndarray, axes: Sequence[int], projection: str) -> Tuple[np.ndarray, List[str], List[float] | None]:
    width = latents.shape[1]
    if projection == "direct":
        if max(axes) > width:
            raise ValueError(f"Requested axis {max(axes)} but latent width is only {width}")
        coords = np.column_stack([latents[:, axis - 1] for axis in axes]).astype(np.float32)
        labels = [f"z{axis}" for axis in axes]
        return coords, labels, None

    centered = latents.astype(np.float64) - latents.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    if max(axes) > vt.shape[0]:
        raise ValueError(f"Requested PCA component {max(axes)} but only {vt.shape[0]} components exist")
    coords = np.column_stack([centered @ vt[axis - 1].T for axis in axes]).astype(np.float32)
    variances = (s ** 2) / max(centered.shape[0] - 1, 1)
    variance_ratio = (variances / variances.sum()) * 100.0
    labels = [f"PC{axis} ({variance_ratio[axis - 1]:.2f}%)" for axis in axes]
    return coords, labels, [float(variance_ratio[axis - 1]) for axis in axes]


def _render_plot(
    frame: pd.DataFrame,
    axes_labels: Sequence[str],
    color_by: str,
    output_html: Path,
    title: str,
    include_plotlyjs: str | bool,
) -> None:
    try:
        import plotly.express as px
    except ImportError:
        _render_plot_fallback(frame, axes_labels, color_by, output_html, title)
        return

    hover_data = {
        "sample_id": False,
        "country": True,
        "original_group_id": True,
        "date_mean_bp": True,
        "observed_fraction": True,
        "axis1": ":.4f",
        "axis2": ":.4f",
    }
    if color_by in hover_data:
        hover_data[color_by] = False
    if "axis3" in frame.columns:
        hover_data["axis3"] = ":.4f"

    if "axis3" in frame.columns:
        fig = px.scatter_3d(
            frame,
            x="axis1",
            y="axis2",
            z="axis3",
            color=color_by,
            hover_name="sample_id",
            hover_data=hover_data,
            title=title,
        )
        fig.update_traces(marker={"size": 3, "opacity": 0.78})
        fig.update_layout(
            scene={
                "xaxis_title": axes_labels[0],
                "yaxis_title": axes_labels[1],
                "zaxis_title": axes_labels[2],
            },
            legend={"itemsizing": "constant"},
        )
    else:
        fig = px.scatter(
            frame,
            x="axis1",
            y="axis2",
            color=color_by,
            hover_name="sample_id",
            hover_data=hover_data,
            title=title,
        )
        fig.update_traces(marker={"size": 5, "opacity": 0.78})
        fig.update_layout(
            xaxis_title=axes_labels[0],
            yaxis_title=axes_labels[1],
            legend={"itemsizing": "constant"},
        )
    fig.write_html(output_html, include_plotlyjs=include_plotlyjs)


def _render_plot_fallback(
    frame: pd.DataFrame,
    axes_labels: Sequence[str],
    color_by: str,
    output_html: Path,
    title: str,
) -> None:
    palette = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    categories = sorted(frame[color_by].astype(str).unique())
    traces = []
    use_3d = "axis3" in frame.columns
    for idx, category in enumerate(categories):
        sub = frame[frame[color_by].astype(str) == category]
        customdata = (
            sub[["country", "original_group_id", "date_mean_bp", "observed_fraction"]]
            .astype(object)
            .where(pd.notnull(sub[["country", "original_group_id", "date_mean_bp", "observed_fraction"]]), "")
            .values
            .tolist()
        )
        if use_3d:
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers",
                    "name": category,
                    "legendgroup": category,
                    "showlegend": True,
                    "x": sub["axis1"].round(8).tolist(),
                    "y": sub["axis2"].round(8).tolist(),
                    "z": sub["axis3"].round(8).tolist(),
                    "text": sub["sample_id"].astype(str).tolist(),
                    "customdata": customdata,
                    "marker": {"size": 3, "opacity": 0.78, "color": palette[idx % len(palette)]},
                    "hovertemplate": (
                        "<b>%{text}</b><br><br>"
                        f"{html.escape(color_by)}={html.escape(category)}<br>"
                        "country=%{customdata[0]}<br>"
                        "original_group_id=%{customdata[1]}<br>"
                        "axis1=%{x:.4f}<br>"
                        "axis2=%{y:.4f}<br>"
                        "axis3=%{z:.4f}<br>"
                        "date_mean_bp=%{customdata[2]}<br>"
                        "observed_fraction=%{customdata[3]}<extra></extra>"
                    ),
                }
            )
        else:
            traces.append(
                {
                    "type": "scattergl",
                    "mode": "markers",
                    "name": category,
                    "legendgroup": category,
                    "showlegend": True,
                    "x": sub["axis1"].round(8).tolist(),
                    "y": sub["axis2"].round(8).tolist(),
                    "text": sub["sample_id"].astype(str).tolist(),
                    "customdata": customdata,
                    "marker": {"size": 5, "opacity": 0.78, "color": palette[idx % len(palette)]},
                    "hovertemplate": (
                        "<b>%{text}</b><br><br>"
                        f"{html.escape(color_by)}={html.escape(category)}<br>"
                        "country=%{customdata[0]}<br>"
                        "original_group_id=%{customdata[1]}<br>"
                        "axis1=%{x:.4f}<br>"
                        "axis2=%{y:.4f}<br>"
                        "date_mean_bp=%{customdata[2]}<br>"
                        "observed_fraction=%{customdata[3]}<extra></extra>"
                    ),
                }
            )

    if use_3d:
        layout = {
            "title": title,
            "scene": {
                "xaxis": {"title": axes_labels[0]},
                "yaxis": {"title": axes_labels[1]},
                "zaxis": {"title": axes_labels[2]},
            },
            "legend": {"title": {"text": color_by}, "itemsizing": "constant"},
            "margin": {"l": 0, "r": 0, "t": 70, "b": 0},
        }
    else:
        layout = {
            "title": title,
            "xaxis": {"title": axes_labels[0]},
            "yaxis": {"title": axes_labels[1]},
            "legend": {"title": {"text": color_by}, "itemsizing": "constant"},
            "margin": {"l": 60, "r": 20, "t": 70, "b": 60},
        }

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div id="plot" style="width:100%;height:100vh;"></div>
  <script>
    const data = {json.dumps(traces)};
    const layout = {json.dumps(layout)};
    Plotly.newPlot('plot', data, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    output_html.write_text(html_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render interactive 2D/3D latent-space HTML plots.")
    parser.add_argument("--latents_csv", default=None, help="Path to final_latents.csv or compatible latent CSV")
    parser.add_argument("--latents_npy", default=None, help="Path to spectral_init.npy or compatible latent matrix")
    parser.add_argument("--sample_stats_tsv", required=True, help="sample_stats.tsv for the same sample set")
    parser.add_argument("--output_html", required=True, help="Output HTML path")
    parser.add_argument("--color_by", default="country", help="Column from sample_stats.tsv used for color")
    parser.add_argument(
        "--axes",
        type=_parse_axes,
        default=[1, 2, 3],
        help="2 or 3 comma-separated 1-based axes/components, e.g. 1,2,3 or 1,3",
    )
    parser.add_argument(
        "--projection",
        choices=("direct", "pca"),
        default="direct",
        help="Use latent dimensions directly or PCA-project before plotting",
    )
    parser.add_argument("--title", default="", help="Optional plot title override")
    parser.add_argument(
        "--exclude_samples",
        default="",
        help="Optional comma-separated sample IDs to drop before rendering",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Optional merged coordinate CSV path. Defaults to <output_html stem>_coords.csv",
    )
    parser.add_argument(
        "--embed_plotlyjs",
        action="store_true",
        help="Embed Plotly JS into the HTML so it renders offline and works with local screenshot/video tools",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sample_stats_tsv = Path(args.sample_stats_tsv)
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    sample_stats = pd.read_csv(sample_stats_tsv, sep="\t")
    sample_stats["sample_id"] = sample_stats["sample_id"].astype(str)
    sample_ids, latents = _read_latents(
        Path(args.latents_csv) if args.latents_csv else None,
        Path(args.latents_npy) if args.latents_npy else None,
        sample_stats,
    )

    latents_frame = pd.DataFrame({"sample_id": sample_ids})
    merged = sample_stats.merge(latents_frame, on="sample_id", how="inner")
    if len(merged) != len(sample_stats):
        raise ValueError(
            f"sample_stats rows={len(sample_stats)} but only merged {len(merged)} rows with latent sample IDs"
        )

    reorder = pd.Index(sample_ids, name="sample_id")
    merged = merged.set_index("sample_id").loc[reorder].reset_index()
    coords, axis_labels, explained = _project_axes(latents, args.axes, args.projection)

    frame = merged.copy()
    frame["axis1"] = coords[:, 0]
    frame["axis2"] = coords[:, 1]
    if coords.shape[1] == 3:
        frame["axis3"] = coords[:, 2]

    if args.exclude_samples:
        excluded = {sample_id.strip() for sample_id in args.exclude_samples.split(",") if sample_id.strip()}
        if excluded:
            frame = frame[~frame["sample_id"].astype(str).isin(excluded)].copy()

    if args.color_by not in frame.columns:
        raise ValueError(f"color_by={args.color_by!r} is not a column in sample_stats.tsv")
    frame[args.color_by] = frame[args.color_by].fillna("(missing)").replace("", "(missing)")

    output_csv = Path(args.output_csv) if args.output_csv else output_html.with_name(f"{output_html.stem}_coords.csv")
    frame.to_csv(output_csv, index=False)

    title = args.title
    if not title:
        source_name = Path(args.latents_csv or args.latents_npy).name
        projection_label = "PCA" if args.projection == "pca" else "latent"
        title = f"{source_name}: {projection_label} axes {','.join(str(a) for a in args.axes)} colored by {args.color_by}"
    include_plotlyjs = True if args.embed_plotlyjs else "cdn"
    _render_plot(frame, axis_labels, args.color_by, output_html, title, include_plotlyjs)

    summary = {
        "output_html": str(output_html),
        "output_csv": str(output_csv),
        "n_samples": int(len(frame)),
        "color_by": args.color_by,
        "projection": args.projection,
        "axes": list(args.axes),
    }
    if explained is not None:
        summary["explained_variance_percent"] = explained
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
