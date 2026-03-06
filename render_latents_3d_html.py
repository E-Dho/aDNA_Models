#!/usr/bin/env python3
"""Render interactive 3D HTML plots from latent coordinate CSV exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render 3D latent coordinate HTML plots.")
    p.add_argument("--coords_dir", required=True, help="Directory containing *_3d_coords.csv files")
    p.add_argument("--output_dir", default=None, help="Output dir for HTML files (default: <coords_dir>/html)")
    p.add_argument("--model_label", default="model", help="Title label for plots")
    p.add_argument("--sample_key", default="sample_id", help="Sample ID column in coords CSV")
    p.add_argument("--metadata_tsv", default=None, help="Optional metadata TSV/ANNO to merge for coloring")
    p.add_argument("--metadata_key", default=None, help="Metadata key column (default: first metadata column)")
    p.add_argument(
        "--color_by",
        action="append",
        default=[],
        help="Metadata field to color by; can be repeated",
    )
    p.add_argument("--category_top_k", type=int, default=20, help="Top K categories to keep before grouping as Other")
    return p.parse_args()


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def prepare_color_column(df: pd.DataFrame, col: str, top_k: int) -> pd.Series:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return s
    filled = s.fillna("NA").astype(str)
    vc = filled.value_counts()
    keep = set(vc.index[:top_k])
    return filled.where(filled.isin(keep), other="Other")


def render_one(
    frame: pd.DataFrame,
    method: str,
    output_path: Path,
    title: str,
    color: str | None,
) -> None:
    kwargs = {
        "x": "dim1",
        "y": "dim2",
        "z": "dim3",
        "hover_name": "sample_id",
        "title": title,
        "opacity": 0.75,
    }
    if color is not None:
        kwargs["color"] = color
    fig = px.scatter_3d(frame, **kwargs)
    fig.update_traces(marker={"size": 2})
    fig.update_layout(scene=dict(xaxis_title="dim1", yaxis_title="dim2", zaxis_title="dim3"))
    fig.write_html(output_path, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()

    coords_dir = Path(args.coords_dir)
    out_dir = Path(args.output_dir) if args.output_dir else coords_dir / "html"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = None
    meta_key = None
    if args.metadata_tsv:
        meta = pd.read_csv(args.metadata_tsv, sep="\t", low_memory=False)
        meta_key = args.metadata_key or meta.columns[0]
        if meta_key not in meta.columns:
            raise ValueError(f"metadata_key '{meta_key}' not found")

    coord_files = sorted(coords_dir.glob("*_3d_coords.csv"))
    if not coord_files:
        raise ValueError(f"No *_3d_coords.csv files found in {coords_dir}")

    produced: list[str] = []

    for path in coord_files:
        method = path.stem.replace("_3d_coords", "")
        df = pd.read_csv(path)
        if args.sample_key not in df.columns:
            raise ValueError(f"{path} is missing sample key column '{args.sample_key}'")

        df = df.rename(columns={args.sample_key: "sample_id"})

        if meta is not None:
            frame = df.merge(meta, left_on="sample_id", right_on=meta_key, how="left")
        else:
            frame = df

        base_name = f"{method}_3d"
        no_color_path = out_dir / f"{base_name}.html"
        render_one(
            frame=frame,
            method=method,
            output_path=no_color_path,
            title=f"{method.upper()} 3D of {args.model_label} latents",
            color=None,
        )
        produced.append(str(no_color_path))

        for color_col in args.color_by:
            if color_col not in frame.columns:
                continue
            tmp_col = f"_color_{sanitize_name(color_col)}"
            frame[tmp_col] = prepare_color_column(frame, color_col, top_k=args.category_top_k)
            out_path = out_dir / f"{base_name}_{sanitize_name(color_col)}.html"
            render_one(
                frame=frame,
                method=method,
                output_path=out_path,
                title=f"{method.upper()} 3D of {args.model_label} latents colored by {color_col}",
                color=tmp_col,
            )
            produced.append(str(out_path))

    print("Generated HTML files:")
    for p in produced:
        print(p)


if __name__ == "__main__":
    main()
