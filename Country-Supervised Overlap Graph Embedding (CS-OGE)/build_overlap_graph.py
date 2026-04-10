#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from overlap_embed.graph import build_overlap_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build overlap-aware similarity graph for aDNA embedding.")
    parser.add_argument("--geno", default=None, help="Path to EIGENSTRAT .geno")
    parser.add_argument("--ind", default=None, help="Path to EIGENSTRAT .ind")
    parser.add_argument("--anno", required=True, help="Path to .anno metadata")
    parser.add_argument("--meta_json", default=None, help="Existing memmap meta JSON")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--ploidy", choices=("diploid", "pseudohaploid"), default="pseudohaploid")
    parser.add_argument("--chunk_snps", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--min_overlap", type=int, default=50000)
    parser.add_argument("--overlap_shrinkage", type=float, default=100000.0)
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_snps", type=int, default=-1)
    parser.add_argument("--min_observed_fraction", type=float, default=0.0)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--compute_backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--compute_device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--resume_existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = build_overlap_graph(
        out_dir=Path(args.out_dir),
        anno_path=Path(args.anno),
        meta_json=Path(args.meta_json) if args.meta_json else None,
        geno_path=Path(args.geno) if args.geno else None,
        ind_path=Path(args.ind) if args.ind else None,
        ploidy=args.ploidy,
        chunk_snps=args.chunk_snps,
        min_overlap=args.min_overlap,
        top_k=args.top_k,
        overlap_shrinkage=args.overlap_shrinkage,
        latent_dim=args.latent_dim,
        max_samples=args.max_samples,
        max_snps=args.max_snps,
        min_observed_fraction=args.min_observed_fraction,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        compute_backend=args.compute_backend,
        compute_device=args.compute_device,
        reuse_existing=args.reuse_existing,
        resume_existing=args.resume_existing,
    )
    print(
        json.dumps(
            {
                "graph_npz": str(artifacts.graph_npz_path),
                "sample_stats_tsv": str(artifacts.sample_stats_tsv_path),
                "spectral_init": str(artifacts.spectral_init_path),
                "build_summary": str(artifacts.build_summary_path),
                "meta_json": str(artifacts.meta_json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
