#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from overlap_embed.optics import write_cluster_member_tables, write_optics_cluster_browser_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an interactive OPTICS cluster browser.")
    parser.add_argument("--assignments_tsv", required=True, help="optics_cluster_assignments.tsv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--latent_cols", default=None, help="Comma-separated 3D display columns; default first three z* columns")
    parser.add_argument("--group_col", default="country", help="Column used as group_id for sorting member lists")
    parser.add_argument("--include_noise", action="store_true", help="Include OPTICS noise as cluster -1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assignments = pd.read_csv(args.assignments_tsv, sep="\t")
    if args.latent_cols:
        latent_cols = [col.strip() for col in args.latent_cols.split(",") if col.strip()]
    else:
        latent_cols = [col for col in assignments.columns if col.startswith("z")][:3]
    if len(latent_cols) < 3:
        raise ValueError("Need at least three latent columns for the browser")
    output_dir = Path(args.output_dir)
    combined, overview, members_dir = write_cluster_member_tables(
        assignments,
        output_dir,
        group_col=args.group_col,
        include_noise=args.include_noise,
    )
    html = write_optics_cluster_browser_html(
        assignments,
        latent_cols,
        output_dir / "optics_cluster_browser_3d.html",
        group_col=args.group_col,
        include_noise=args.include_noise,
    )
    print(f"HTML\t{html}")
    print(f"COMBINED_MEMBERS\t{combined}")
    print(f"OVERVIEW\t{overview}")
    print(f"MEMBERS_DIR\t{members_dir}")


if __name__ == "__main__":
    main()
