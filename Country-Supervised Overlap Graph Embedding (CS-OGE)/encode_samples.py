#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from overlap_embed.oos_encoder import align_eigenstrat_to_manifest, encode_aligned_memmap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode new samples into a trained CS-OGE latent space.")
    parser.add_argument("--encoder_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--geno", default="", help="EIGENSTRAT text .geno for new samples")
    parser.add_argument("--snp", default="", help=".snp matching --geno")
    parser.add_argument("--ind", default="", help=".ind matching --geno")
    parser.add_argument("--meta_json", default="", help="Already aligned memmap meta JSON")
    parser.add_argument("--ploidy", choices=("diploid", "pseudohaploid"), default="pseudohaploid")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--low_overlap_fraction", type=float, default=0.05)
    parser.add_argument("--teacher_sample_stats_tsv", default="")
    parser.add_argument("--teacher_latents_csv", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    encoder_dir = Path(args.encoder_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.meta_json:
        meta_json = Path(args.meta_json)
    else:
        for required_name, value in (("--geno", args.geno), ("--snp", args.snp), ("--ind", args.ind)):
            if not value:
                raise SystemExit(f"{required_name} is required when --meta_json is not provided")
        meta, report = align_eigenstrat_to_manifest(
            geno_path=Path(args.geno),
            snp_path=Path(args.snp),
            ind_path=Path(args.ind),
            manifest_path=encoder_dir / "snp_manifest.tsv",
            out_prefix=out_dir / "aligned_input",
            ploidy=args.ploidy,
            low_overlap_fraction=args.low_overlap_fraction,
        )
        meta_json = out_dir / "aligned_input.meta.json"
        with (out_dir / "alignment_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report.__dict__, handle, indent=2)
    summary = encode_aligned_memmap(
        encoder_dir=encoder_dir,
        meta_json=meta_json,
        out_dir=out_dir,
        batch_size=args.batch_size,
        teacher_latents_csv=Path(args.teacher_latents_csv) if args.teacher_latents_csv else None,
        teacher_sample_stats_tsv=Path(args.teacher_sample_stats_tsv) if args.teacher_sample_stats_tsv else None,
        low_overlap_fraction=args.low_overlap_fraction,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
