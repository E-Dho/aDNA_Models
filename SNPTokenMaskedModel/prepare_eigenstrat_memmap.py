#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from token_snp.data import build_memmap_from_eigenstrat


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert EIGENSTRAT .geno to int8 memmap for tokenized SNP model."
    )
    p.add_argument("--geno", required=True, help="Path to EIGENSTRAT .geno file")
    p.add_argument("--out_prefix", required=True, help="Output prefix for generated files")
    p.add_argument(
        "--ploidy",
        choices=("diploid", "pseudohaploid"),
        required=True,
        help="Genotype mode used to map token classes",
    )
    p.add_argument("--ind", default=None, help="Optional .ind file for sample IDs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    meta = build_memmap_from_eigenstrat(
        geno_path=Path(args.geno),
        out_prefix=Path(args.out_prefix),
        ploidy=args.ploidy,
        ind_path=Path(args.ind) if args.ind is not None else None,
    )

    print("Wrote memmap metadata:")
    print(f"  memmap_path: {meta.memmap_path}")
    print(f"  n_samples:   {meta.n_samples}")
    print(f"  n_snps:      {meta.n_snps}")
    print(f"  n_classes:   {meta.n_classes}")
    print(f"  missing:     {meta.missing_value}")
    if meta.sample_ids_path is not None:
        print(f"  sample_ids:  {meta.sample_ids_path}")


if __name__ == "__main__":
    main()
