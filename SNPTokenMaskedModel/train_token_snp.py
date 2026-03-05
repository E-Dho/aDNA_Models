#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from token_snp.train import TrainConfig, run_training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train a tokenized SNP masked model with chunked sequence encoding "
            "and export per-sample embeddings."
        )
    )
    p.add_argument("--meta_json", required=True, help="Path to memmap metadata JSON.")
    p.add_argument("--output_dir", required=True, help="Output run directory.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window_size", type=int, default=8192)
    p.add_argument("--windows_per_sample", type=int, default=1)

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps_per_epoch", type=int, default=250)
    p.add_argument("--val_steps", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--mask_prob", type=float, default=0.2)
    p.add_argument("--missingness_dropout", type=float, default=0.1)

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--local_layers", type=int, default=2)
    p.add_argument("--chunk_layers", type=int, default=2)
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--disable_obs_embedding", action="store_true")
    p.add_argument("--use_snp_id_embedding", action="store_true")

    p.add_argument("--embedding_batch_size", type=int, default=128)
    p.add_argument("--embedding_windows_per_sample", type=int, default=32)
    p.add_argument("--coverage_monitor_subset", type=int, default=512)
    p.add_argument("--monitor_every", type=int, default=1)
    p.add_argument(
        "--batch_labels_tsv",
        default=None,
        help="Optional two-column TSV/space file: sample_id batch_label",
    )

    p.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    p.add_argument("--wandb_project", default="token-snp-mask")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_tags", default=None, help="Comma-separated tags.")
    p.add_argument("--wandb_mode", choices=("offline", "online"), default="offline")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = TrainConfig(
        output_dir=args.output_dir,
        meta_json=args.meta_json,
        seed=args.seed,
        window_size=args.window_size,
        windows_per_sample=args.windows_per_sample,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        mask_prob=args.mask_prob,
        missingness_dropout=args.missingness_dropout,
        d_model=args.d_model,
        n_heads=args.n_heads,
        local_layers=args.local_layers,
        chunk_layers=args.chunk_layers,
        chunk_size=args.chunk_size,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        use_obs_embedding=not bool(args.disable_obs_embedding),
        use_snp_id_embedding=bool(args.use_snp_id_embedding),
        embedding_batch_size=args.embedding_batch_size,
        embedding_windows_per_sample=args.embedding_windows_per_sample,
        coverage_monitor_subset=args.coverage_monitor_subset,
        monitor_every=args.monitor_every,
        batch_labels_tsv=args.batch_labels_tsv,
        wandb_enable=bool(args.wandb),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_group=args.wandb_group,
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode,
    )
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
