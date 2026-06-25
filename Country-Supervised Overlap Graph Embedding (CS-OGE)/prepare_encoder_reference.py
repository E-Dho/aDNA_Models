#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from overlap_embed.oos_encoder import prepare_encoder_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare SNP manifest and teacher references for OOS encoder training.")
    parser.add_argument("--data_prefix", required=True, help="Teacher data prefix with .snp file")
    parser.add_argument("--teacher_run_dir", required=True, help="CS-OGE teacher run directory")
    parser.add_argument("--graph_dir", required=True, help="Graph directory containing sample_stats.tsv")
    parser.add_argument("--out_dir", required=True, help="Output directory for encoder reference files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_encoder_reference(
        data_prefix=Path(args.data_prefix),
        teacher_run_dir=Path(args.teacher_run_dir),
        graph_dir=Path(args.graph_dir),
        out_dir=Path(args.out_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
