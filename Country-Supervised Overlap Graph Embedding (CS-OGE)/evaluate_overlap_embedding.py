#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from overlap_embed.eval import export_evaluation_bundle, read_latents_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate overlap embedding latents against aDNA metadata.")
    parser.add_argument("--latents_csv", required=True, help="Path to latents CSV")
    parser.add_argument("--anno", default=None, help="Unused compatibility argument; sample_stats_tsv is the source of truth")
    parser.add_argument("--sample_stats_tsv", required=True, help="Path to sample_stats.tsv")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--label", default="country", choices=("country", "original_group_id"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sample_ids, latents = read_latents_csv(Path(args.latents_csv))
    summary = export_evaluation_bundle(
        latents=latents,
        sample_ids=sample_ids,
        sample_stats_tsv=Path(args.sample_stats_tsv),
        output_dir=Path(args.output_dir),
        prefix="evaluated",
        plot_label=args.label,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
