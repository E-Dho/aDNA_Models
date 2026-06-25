#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from overlap_embed.train import TrainConfig, train_overlap_embedding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train overlap graph embedding for aDNA.")
    parser.add_argument("--graph_npz", required=True, help="Path to graph_edges.npz")
    parser.add_argument("--spectral_init", required=True, help="Path to spectral_init.npy")
    parser.add_argument("--sample_stats_tsv", required=True, help="Path to sample_stats.tsv")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--objective_mode", choices=("label_supervised", "snp_similarity"), default="label_supervised")
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--negatives_per_anchor", type=int, default=16)
    parser.add_argument("--lambda_cov", type=float, default=0.5)
    parser.add_argument("--lambda_var", type=float, default=1.0)
    parser.add_argument("--lambda_covariance", type=float, default=0.1)
    parser.add_argument("--lambda_graph_pos", type=float, default=1.0)
    parser.add_argument("--lambda_graph_neg", type=float, default=1.0)
    parser.add_argument("--lambda_graph_similarity", type=float, default=1.0)
    parser.add_argument("--graph_similarity_temperature", type=float, default=1.0)
    parser.add_argument("--lambda_country", type=float, default=0.0)
    parser.add_argument("--lambda_original_group", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--country_margin", type=float, default=1.5)
    parser.add_argument("--country_negatives_per_anchor", type=int, default=8)
    parser.add_argument("--original_group_negatives_per_anchor", type=int, default=4)
    parser.add_argument("--original_group_temperature", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", dest="wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.set_defaults(wandb=True)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--wandb_project", default="adna-overlap-embed")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_group", default="europe_drop_premerge_gt90")
    parser.add_argument("--wandb_name", default="overlap_embed")
    parser.add_argument("--wandb_tags", default="overlap-embed,europe")
    parser.add_argument("--wandb_log_every", type=int, default=10)
    parser.add_argument("--wandb_init_timeout", type=int, default=30)
    parser.add_argument("--wandb_watch", type=int, default=0)
    parser.add_argument("--wandb_online_fallback", dest="wandb_online_fallback", action="store_true")
    parser.add_argument("--no-wandb_online_fallback", dest="wandb_online_fallback", action="store_false")
    parser.set_defaults(wandb_online_fallback=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        objective_mode=args.objective_mode,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        negatives_per_anchor=args.negatives_per_anchor,
        lambda_cov=args.lambda_cov,
        lambda_var=args.lambda_var,
        lambda_covariance=args.lambda_covariance,
        lambda_graph_pos=args.lambda_graph_pos,
        lambda_graph_neg=args.lambda_graph_neg,
        lambda_graph_similarity=args.lambda_graph_similarity,
        graph_similarity_temperature=args.graph_similarity_temperature,
        lambda_country=args.lambda_country,
        lambda_original_group=args.lambda_original_group,
        margin=args.margin,
        country_margin=args.country_margin,
        country_negatives_per_anchor=args.country_negatives_per_anchor,
        original_group_negatives_per_anchor=args.original_group_negatives_per_anchor,
        original_group_temperature=args.original_group_temperature,
        val_ratio=args.val_ratio,
        patience=args.patience,
        seed=args.seed,
        wandb=args.wandb and args.wandb_mode != "disabled",
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_log_every=args.wandb_log_every,
        wandb_init_timeout=args.wandb_init_timeout,
        wandb_watch=args.wandb_watch,
        wandb_online_fallback=args.wandb_online_fallback,
    )
    summary = train_overlap_embedding(
        graph_npz=Path(args.graph_npz),
        spectral_init=Path(args.spectral_init),
        sample_stats_tsv=Path(args.sample_stats_tsv),
        output_dir=Path(args.output_dir),
        config=config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
