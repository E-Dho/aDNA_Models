#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from hmae.train import TrainConfig, run_training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train a hierarchical masked autoencoder on windowed genotype data "
            "and export global embeddings."
        )
    )
    p.add_argument("--meta_json", required=True, help="Path to memmap metadata JSON.")
    p.add_argument("--output_dir", required=True, help="Output run directory.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window_size", type=int, default=8192)
    p.add_argument("--windows_per_step", type=int, default=8)

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps_per_epoch", type=int, default=250)
    p.add_argument("--val_steps", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument(
        "--coverage_corr_weight",
        type=float,
        default=0.0,
        help=(
            "Penalty weight for squared Pearson correlation between global latent norm "
            "and per-sample observed coverage in each training batch."
        ),
    )
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--mask_prob", type=float, default=0.2)
    p.add_argument("--observed_dropout", type=float, default=0.1)

    p.add_argument("--window_latent_dim", type=int, default=32)
    p.add_argument("--global_latent_dim", type=int, default=32)
    p.add_argument("--embed_dim", type=int, default=16)
    p.add_argument("--local_hidden_dim", type=int, default=128)
    p.add_argument(
        "--local_encoder_type",
        choices=("conv_attn", "meanpool"),
        default="conv_attn",
    )
    p.add_argument("--local_conv_layers", type=int, default=4)
    p.add_argument("--local_conv_kernel", type=int, default=7)
    p.add_argument("--local_attn_heads", type=int, default=4)
    p.add_argument("--local_dropout", type=float, default=0.1)
    p.add_argument("--global_model_dim", type=int, default=128)
    p.add_argument("--global_heads", type=int, default=4)
    p.add_argument("--global_layers", type=int, default=2)
    p.add_argument("--decoder_hidden_dim", type=int, default=128)

    p.add_argument("--include_window_coverage", action="store_true")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=None)
    p.add_argument("--variational", action="store_true")
    p.add_argument("--kl_beta_max", type=float, default=1.0)
    p.add_argument("--kl_warmup_fraction", type=float, default=0.3)

    p.add_argument("--embedding_batch_size", type=int, default=128)
    p.add_argument("--embedding_window_chunk", type=int, default=16)
    p.add_argument("--coverage_monitor_subset", type=int, default=512)
    p.add_argument("--monitor_every", type=int, default=1)
    p.add_argument(
        "--batch_labels_tsv",
        default=None,
        help="Optional two-column TSV/space file: sample_id batch_label",
    )
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    p.add_argument("--wandb_project", default="hmae-phase1")
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
        windows_per_step=args.windows_per_step,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        coverage_corr_weight=args.coverage_corr_weight,
        mask_prob=args.mask_prob,
        observed_dropout=args.observed_dropout,
        patience=args.patience,
        variational=args.variational,
        kl_beta_max=args.kl_beta_max,
        kl_warmup_fraction=args.kl_warmup_fraction,
        window_latent_dim=args.window_latent_dim,
        global_latent_dim=args.global_latent_dim,
        embed_dim=args.embed_dim,
        local_hidden_dim=args.local_hidden_dim,
        local_encoder_type=args.local_encoder_type,
        local_conv_layers=args.local_conv_layers,
        local_conv_kernel=args.local_conv_kernel,
        local_attn_heads=args.local_attn_heads,
        local_dropout=args.local_dropout,
        global_model_dim=args.global_model_dim,
        global_heads=args.global_heads,
        global_layers=args.global_layers,
        decoder_hidden_dim=args.decoder_hidden_dim,
        include_window_coverage=bool(args.include_window_coverage),
        require_cuda=bool(args.require_cuda),
        amp=args.amp,
        embedding_batch_size=args.embedding_batch_size,
        embedding_window_chunk=args.embedding_window_chunk,
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
