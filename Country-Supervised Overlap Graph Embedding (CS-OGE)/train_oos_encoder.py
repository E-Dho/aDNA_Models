#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from overlap_embed.oos_encoder import train_oos_encoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an out-of-sample encoder for a fixed CS-OGE latent space.")
    parser.add_argument("--meta_json", required=True)
    parser.add_argument("--snp_manifest", required=True)
    parser.add_argument("--teacher_latents_csv", required=True)
    parser.add_argument("--sample_stats_tsv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument(
        "--model_type",
        choices=("sparse_residual_encoder", "sparse_deep_encoder"),
        default="sparse_residual_encoder",
        help="Encoder architecture. sparse_deep_encoder adds sparse 584k->100k and dense 100k->1k->latent layers.",
    )
    parser.add_argument("--deep_hidden_dim", type=int, default=100000)
    parser.add_argument("--deep_bottleneck_dim", type=int, default=1024)
    parser.add_argument("--sparse_fanout", type=int, default=2)
    parser.add_argument("--sparse_init", choices=("legacy", "fan_in"), default="fan_in")
    parser.add_argument("--dst_enable", action="store_true")
    parser.add_argument("--dst_update_interval_steps", type=int, default=100)
    parser.add_argument("--dst_start_epoch", type=int, default=5)
    parser.add_argument("--dst_end_epoch", type=int, default=80)
    parser.add_argument("--dst_initial_drop_fraction", type=float, default=0.30)
    parser.add_argument("--dst_final_drop_fraction", type=float, default=0.05)
    parser.add_argument("--dst_candidate_multiplier", type=int, default=16)
    parser.add_argument("--dst_grow_init", choices=("zero",), default="zero")
    parser.add_argument("--dst_score", choices=("genotype_plus_mask_grad",), default="genotype_plus_mask_grad")
    parser.add_argument("--lambda_latent", type=float, default=1.0)
    parser.add_argument("--lambda_neighbor", type=float, default=0.1)
    parser.add_argument("--lambda_coverage", type=float, default=0.5)
    parser.add_argument("--snp_dropout", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--split_strategy",
        choices=("random", "country", "original_group", "low_coverage", "date"),
        default="random",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap for smoke/debug runs after teacher ID alignment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain_source", default="", help="Alias for --pretrain_source_meta_json")
    parser.add_argument("--pretrain_source_meta_json", default="")
    parser.add_argument("--pretrain_source_sample_stats_tsv", default="")
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--lambda_pretrain_consistency", type=float, default=1.0)
    parser.add_argument("--lambda_pretrain_structure", type=float, default=0.25)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb_project", default="adna-overlap-embed")
    parser.add_argument("--wandb_group", default="oos-encoder")
    parser.add_argument("--wandb_name", default="oos_encoder")
    parser.add_argument("--wandb_tags", default="oos-encoder")
    parser.add_argument("--wandb_log_every", type=int, default=10)
    parser.add_argument("--wandb_init_timeout", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = train_oos_encoder(
        meta_json=Path(args.meta_json),
        snp_manifest=Path(args.snp_manifest),
        teacher_latents_csv=Path(args.teacher_latents_csv),
        sample_stats_tsv=Path(args.sample_stats_tsv),
        output_dir=Path(args.output_dir),
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        model_type=args.model_type,
        deep_hidden_dim=args.deep_hidden_dim,
        deep_bottleneck_dim=args.deep_bottleneck_dim,
        sparse_fanout=args.sparse_fanout,
        sparse_init=args.sparse_init,
        dst_enable=args.dst_enable,
        dst_update_interval_steps=args.dst_update_interval_steps,
        dst_start_epoch=args.dst_start_epoch,
        dst_end_epoch=args.dst_end_epoch,
        dst_initial_drop_fraction=args.dst_initial_drop_fraction,
        dst_final_drop_fraction=args.dst_final_drop_fraction,
        dst_candidate_multiplier=args.dst_candidate_multiplier,
        dst_grow_init=args.dst_grow_init,
        dst_score=args.dst_score,
        lambda_latent=args.lambda_latent,
        lambda_neighbor=args.lambda_neighbor,
        lambda_coverage=args.lambda_coverage,
        snp_dropout=args.snp_dropout,
        val_ratio=args.val_ratio,
        split_strategy=args.split_strategy,
        max_samples=args.max_samples,
        seed=args.seed,
        pretrain_source_meta_json=(
            Path(args.pretrain_source_meta_json or args.pretrain_source)
            if (args.pretrain_source_meta_json or args.pretrain_source)
            else None
        ),
        pretrain_source_sample_stats_tsv=(
            Path(args.pretrain_source_sample_stats_tsv) if args.pretrain_source_sample_stats_tsv else None
        ),
        pretrain_epochs=args.pretrain_epochs,
        lambda_pretrain_consistency=args.lambda_pretrain_consistency,
        lambda_pretrain_structure=args.lambda_pretrain_structure,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_log_every=args.wandb_log_every,
        wandb_init_timeout=args.wandb_init_timeout,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
