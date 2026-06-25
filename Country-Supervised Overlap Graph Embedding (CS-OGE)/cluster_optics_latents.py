#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, TypeVar

from overlap_embed.optics import (
    OpticsConfig,
    assignments_frame,
    default_sweep_configs,
    load_latents_with_stats,
    maybe_write_optics_3d_html,
    maybe_write_reachability_plot,
    run_optics_sweep,
    summarize_clusters,
    sweep_summary,
    write_optics_outputs,
)


T = TypeVar("T")


def _csv_values(value: str | None, cast: Callable[[str], T]) -> list[T]:
    if value is None:
        return []
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def _min_cluster_size(value: str) -> int | float:
    parsed = float(value)
    return int(parsed) if parsed >= 1.0 and parsed.is_integer() else parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster CS-OGE latents with OPTICS.")
    parser.add_argument("--latents_csv", required=True)
    parser.add_argument("--sample_stats_tsv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--component_cols", default=None, help="Comma-separated latent columns; default all z*")
    parser.add_argument("--metric", default="minkowski")
    parser.add_argument("--min_samples", default=None, help="Comma-separated sweep values")
    parser.add_argument("--min_cluster_size", default=None, help="Comma-separated integer counts or fractions")
    parser.add_argument("--xi", default=None, help="Comma-separated sweep values")
    parser.add_argument("--single_config", action="store_true", help="Use one config instead of a sweep")
    parser.add_argument("--no_html", action="store_true", help="Skip optional Plotly HTML")
    parser.add_argument("--no_reachability_plot", action="store_true", help="Skip reachability PNG")
    return parser


def build_configs(args: argparse.Namespace) -> list[OpticsConfig]:
    min_samples = _csv_values(args.min_samples, int)
    min_cluster_sizes = _csv_values(args.min_cluster_size, _min_cluster_size)
    xis = _csv_values(args.xi, float)
    if not min_samples and not min_cluster_sizes and not xis and not args.single_config:
        return default_sweep_configs(args.metric)
    min_samples = min_samples or [15]
    min_cluster_sizes = min_cluster_sizes or [0.005]
    xis = xis or [0.03]
    if args.single_config:
        return [
            OpticsConfig(
                min_samples=min_samples[0],
                min_cluster_size=min_cluster_sizes[0],
                xi=xis[0],
                metric=args.metric,
            )
        ]
    return [
        OpticsConfig(min_samples=ms, min_cluster_size=mcs, xi=xi, metric=args.metric)
        for ms in min_samples
        for mcs in min_cluster_sizes
        for xi in xis
    ]


def main() -> None:
    args = build_parser().parse_args()
    component_cols = _csv_values(args.component_cols, str) or None
    frame, values, latent_cols = load_latents_with_stats(
        Path(args.latents_csv),
        Path(args.sample_stats_tsv),
        component_cols=component_cols,
    )
    configs = build_configs(args)
    chosen, runs = run_optics_sweep(values, configs)
    assignments = assignments_frame(frame, latent_cols, chosen)
    output_dir = Path(args.output_dir)
    summary = write_optics_outputs(
        output_dir=output_dir,
        assignments=assignments,
        cluster_summary=summarize_clusters(assignments),
        sweep=sweep_summary(runs, chosen),
        chosen=chosen,
        input_summary={
            "latents_csv": str(args.latents_csv),
            "sample_stats_tsv": str(args.sample_stats_tsv),
            "latent_columns": latent_cols,
            "n_configs": len(configs),
        },
    )
    if not args.no_reachability_plot:
        summary["reachability_plot"] = maybe_write_reachability_plot(chosen, output_dir)
    if not args.no_html:
        summary["html"] = maybe_write_optics_3d_html(assignments, latent_cols, output_dir)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
