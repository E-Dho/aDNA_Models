#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from overlap_embed.data import build_memmap_from_eigenstrat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert EIGENSTRAT .geno to int8 memmap for overlap embedding.")
    parser.add_argument("--geno", required=True, help="Path to EIGENSTRAT .geno")
    parser.add_argument("--ind", default=None, help="Optional path to .ind for sample IDs")
    parser.add_argument("--out_prefix", required=True, help="Output prefix without extension")
    parser.add_argument(
        "--ploidy",
        choices=("diploid", "pseudohaploid"),
        default="pseudohaploid",
        help="Genotype category mode.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    meta = build_memmap_from_eigenstrat(
        geno_path=Path(args.geno),
        ind_path=Path(args.ind) if args.ind else None,
        out_prefix=Path(args.out_prefix),
        ploidy=args.ploidy,
    )
    print(json.dumps(meta.__dict__, indent=2))


if __name__ == "__main__":
    main()
