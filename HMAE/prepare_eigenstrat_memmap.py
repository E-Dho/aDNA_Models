#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hmae.data import build_memmap_from_eigenstrat


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert EIGENSTRAT .geno to int8 memmap for hierarchical masked "
            "autoencoder training."
        )
    )
    p.add_argument("--geno", required=True, help="Path to EIGENSTRAT .geno")
    p.add_argument("--ind", default=None, help="Optional path to .ind for sample IDs")
    p.add_argument(
        "--out_prefix",
        required=True,
        help="Output prefix (without extension), e.g. /path/to/aadr_unfiltered",
    )
    p.add_argument(
        "--ploidy",
        choices=("diploid", "pseudohaploid"),
        default="diploid",
        help="Genotype category mode.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    meta = build_memmap_from_eigenstrat(
        geno_path=Path(args.geno),
        out_prefix=Path(args.out_prefix),
        ploidy=args.ploidy,
        ind_path=Path(args.ind) if args.ind else None,
    )
    print(json.dumps(meta.__dict__, indent=2))


if __name__ == "__main__":
    main()

