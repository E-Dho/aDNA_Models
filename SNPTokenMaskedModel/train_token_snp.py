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
    p.add_argument(
        "--use_coverage_equalization",
        action="store_true",
        help="Enable stochastic observed->missing downsampling toward a target observed fraction.",
    )
    p.add_argument(
        "--coverage_equalization_target",
        type=float,
        default=0.30,
        help="Target observed fraction used by stochastic coverage equalization.",
    )
    p.add_argument(
        "--coverage_equalization_scope",
        choices=("sample",),
        default="sample",
        help="Equalization granularity (currently only sample-level).",
    )
    p.add_argument(
        "--use_group_rus",
        action="store_true",
        help="Enable training-only balanced batch sampling by Group ID.",
    )
    p.add_argument(
        "--group_rus_mode",
        default="balanced_batch",
        help="Group-RUS mode. Only 'balanced_batch' is implemented.",
    )
    p.add_argument(
        "--group_rus_max_per_group_in_batch",
        type=int,
        default=None,
        help="Optional cap on samples drawn from one group within a train batch.",
    )
    p.add_argument(
        "--group_rus_groups_per_batch",
        type=int,
        default=None,
        help="Optional number of Group-ID classes selected per train batch.",
    )
    p.add_argument(
        "--group_rus_seed",
        type=int,
        default=42,
        help="Seed used by the balanced Group-ID batch sampler.",
    )
    p.add_argument(
        "--group_metadata_tsv",
        default=None,
        help="Metadata TSV/ANNO used to resolve sample_id -> Group ID for batch RUS.",
    )
    p.add_argument(
        "--group_metadata_col",
        default="Group ID",
        help="Metadata column used as the group label for batch RUS.",
    )
    p.add_argument(
        "--group_metadata_key",
        default=None,
        help="Metadata key column used to join sample IDs; defaults to the first metadata column.",
    )
    p.add_argument(
        "--use_country_loss",
        action="store_true",
        help="Enable supervised pairwise country loss on pooled latent z.",
    )
    p.add_argument(
        "--lambda_country_loss_target",
        type=float,
        default=0.1,
        help="Target weight for country_loss after ramp.",
    )
    p.add_argument(
        "--country_loss_pos_margin",
        type=float,
        default=0.25,
        help="Same-country hinge margin for pairwise country loss.",
    )
    p.add_argument(
        "--country_loss_neg_margin",
        type=float,
        default=1.0,
        help="Different-country hinge margin for pairwise country loss.",
    )
    p.add_argument(
        "--country_loss_normalize_z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize pooled z before pairwise country loss.",
    )
    p.add_argument(
        "--country_loss_ramp_start_epoch",
        type=int,
        default=1,
        help="Epoch where country-loss weight ramp starts.",
    )
    p.add_argument(
        "--country_loss_ramp_end_epoch",
        type=int,
        default=10,
        help="Epoch where country-loss weight reaches target.",
    )
    p.add_argument(
        "--country_loss_groups_per_batch",
        type=int,
        default=None,
        help="Expected Group-RUS groups per train batch for country-loss positive pairs.",
    )
    p.add_argument(
        "--country_loss_metadata_tsv",
        default=None,
        help="Metadata TSV/ANNO used to resolve sample_id -> country label for country loss.",
    )
    p.add_argument(
        "--country_loss_metadata_col",
        default="Group ID",
        help="Metadata column used as country label for country loss.",
    )
    p.add_argument(
        "--country_loss_metadata_key",
        default=None,
        help="Metadata key column for country loss; defaults to the first metadata column.",
    )

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--local_layers", type=int, default=2)
    p.add_argument("--chunk_layers", type=int, default=2)
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--latent_tokens", type=int, default=8)
    p.add_argument(
        "--latent_feedback_mode",
        choices=("cross_attn",),
        default="cross_attn",
        help="How latent tokens are fed into reconstruction.",
    )
    p.add_argument(
        "--use_strong_latent_reconstruction",
        action="store_true",
        help="Force reconstruction to depend on pooled latent z through decoder-side conditioning.",
    )
    p.add_argument(
        "--latent_recon_mode",
        choices=("film_only", "crossattn_only", "film_plus_crossattn", "latent_only"),
        default="film_plus_crossattn",
        help="Decoder-side strong latent reconstruction mode.",
    )
    p.add_argument(
        "--latent_recon_hidden_dim",
        type=int,
        default=None,
        help="Hidden size for latent FiLM MLP (default: d_model).",
    )
    p.add_argument(
        "--latent_recon_num_heads",
        type=int,
        default=4,
        help="Attention heads used by decoder-side latent cross-attention.",
    )
    p.add_argument(
        "--latent_recon_dropout",
        type=float,
        default=0.05,
        help="Dropout used by decoder-side latent cross-attention.",
    )
    p.add_argument(
        "--latent_recon_num_tokens",
        type=int,
        default=4,
        help="Number of latent decoder tokens derived from pooled z.",
    )
    p.add_argument(
        "--latent_cross_attn_residual_scale",
        type=float,
        default=0.1,
        help="Residual carry-through scale for decoder latent cross-attention (0=attn dominates).",
    )
    p.add_argument(
        "--latent_only_aux_weight",
        type=float,
        default=0.0,
        help="Auxiliary CE weight for a latent-only reconstruction branch.",
    )
    p.add_argument("--disable_obs_embedding", action="store_true")
    p.add_argument("--use_snp_id_embedding", action="store_true")

    p.add_argument("--embedding_batch_size", type=int, default=128)
    p.add_argument("--embedding_windows_per_sample", type=int, default=32)
    p.add_argument("--coverage_monitor_subset", type=int, default=512)
    p.add_argument("--monitor_every", type=int, default=1)
    p.add_argument(
        "--adv_coverage_enable",
        action="store_true",
        help="Enable gradient-reversal adversary predicting standardized coverage from z.",
    )
    p.add_argument(
        "--lambda_adv_target",
        type=float,
        default=0.0,
        help="Target GRL lambda value after warmup.",
    )
    p.add_argument(
        "--lambda_adv_warmup_epochs",
        type=int,
        default=10,
        help="Linear warmup epochs for lambda_adv from 0 to target.",
    )
    p.add_argument(
        "--lambda_cov_target",
        type=float,
        default=0.0,
        help="Target covariance penalty multiplier.",
    )
    p.add_argument(
        "--adv_steps_per_batch",
        type=int,
        default=1,
        help="Adversary-only optimization steps per main encoder step.",
    )
    p.add_argument(
        "--lambda_ramp_start_epoch",
        type=int,
        default=6,
        help="Epoch index where lambda_adv/lambda_cov ramp starts.",
    )
    p.add_argument(
        "--lambda_ramp_end_epoch",
        type=int,
        default=15,
        help="Epoch index where lambda_adv/lambda_cov reaches target.",
    )
    p.add_argument(
        "--adv_mlp_hidden_dim",
        type=int,
        default=128,
        help="Hidden size of coverage adversary MLP.",
    )
    p.add_argument(
        "--adv_mlp_dropout",
        type=float,
        default=0.1,
        help="Dropout in coverage adversary MLP.",
    )
    p.add_argument(
        "--coverage_conditioning_mode",
        choices=("none", "concat", "film", "film_concat"),
        default="none",
        help="Coverage conditioning mode applied to the reconstruction pathway.",
    )
    p.add_argument(
        "--coverage_embed_dim",
        type=int,
        default=0,
        help="Optional coverage embedding width for concat conditioning (0=scalar).",
    )
    p.add_argument(
        "--batch_labels_tsv",
        default=None,
        help="Optional two-column TSV/space file: sample_id batch_label",
    )
    p.add_argument(
        "--probe_eval_enable",
        action="store_true",
        help="Run ancestry probe evaluation after training.",
    )
    p.add_argument(
        "--probe_metadata_tsv",
        default=None,
        help="Metadata TSV/ANNO used for probe evaluation.",
    )
    p.add_argument(
        "--probe_target_col",
        default="Political Entity",
        help="Metadata column used as ancestry probe target.",
    )
    p.add_argument(
        "--probe_seed",
        type=int,
        default=42,
        help="Random seed used by probe train/val/test split.",
    )

    p.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    p.add_argument("--wandb_project", default="token-snp-mask")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_tags", default=None, help="Comma-separated tags.")
    p.add_argument("--wandb_mode", choices=("offline", "online"), default="offline")

    p.add_argument("--debug_mode", action="store_true", help="Enable verbose debug checks.")
    p.add_argument(
        "--debug_examples_per_batch",
        type=int,
        default=20,
        help="How many masked-position examples to print for debug batches.",
    )
    p.add_argument(
        "--debug_max_batches_per_phase",
        type=int,
        default=-1,
        help="If >=0, only print per-batch debug counts for this many batches per phase.",
    )
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
        use_coverage_equalization=bool(args.use_coverage_equalization),
        coverage_equalization_target=args.coverage_equalization_target,
        coverage_equalization_scope=args.coverage_equalization_scope,
        use_group_rus=bool(args.use_group_rus),
        group_rus_mode=args.group_rus_mode,
        group_rus_max_per_group_in_batch=args.group_rus_max_per_group_in_batch,
        group_rus_groups_per_batch=args.group_rus_groups_per_batch,
        group_rus_seed=args.group_rus_seed,
        group_metadata_tsv=args.group_metadata_tsv,
        group_metadata_col=args.group_metadata_col,
        group_metadata_key=args.group_metadata_key,
        use_country_loss=bool(args.use_country_loss),
        lambda_country_loss_target=args.lambda_country_loss_target,
        country_loss_pos_margin=args.country_loss_pos_margin,
        country_loss_neg_margin=args.country_loss_neg_margin,
        country_loss_normalize_z=bool(args.country_loss_normalize_z),
        country_loss_ramp_start_epoch=args.country_loss_ramp_start_epoch,
        country_loss_ramp_end_epoch=args.country_loss_ramp_end_epoch,
        country_loss_groups_per_batch=args.country_loss_groups_per_batch,
        country_loss_metadata_tsv=args.country_loss_metadata_tsv,
        country_loss_metadata_col=args.country_loss_metadata_col,
        country_loss_metadata_key=args.country_loss_metadata_key,
        d_model=args.d_model,
        n_heads=args.n_heads,
        local_layers=args.local_layers,
        chunk_layers=args.chunk_layers,
        chunk_size=args.chunk_size,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        latent_tokens=args.latent_tokens,
        latent_feedback_mode=args.latent_feedback_mode,
        use_strong_latent_reconstruction=bool(args.use_strong_latent_reconstruction),
        latent_recon_mode=args.latent_recon_mode,
        latent_recon_hidden_dim=args.latent_recon_hidden_dim,
        latent_recon_num_heads=args.latent_recon_num_heads,
        latent_recon_dropout=args.latent_recon_dropout,
        latent_recon_num_tokens=args.latent_recon_num_tokens,
        latent_cross_attn_residual_scale=args.latent_cross_attn_residual_scale,
        latent_only_aux_weight=args.latent_only_aux_weight,
        use_obs_embedding=not bool(args.disable_obs_embedding),
        use_snp_id_embedding=bool(args.use_snp_id_embedding),
        embedding_batch_size=args.embedding_batch_size,
        embedding_windows_per_sample=args.embedding_windows_per_sample,
        coverage_monitor_subset=args.coverage_monitor_subset,
        monitor_every=args.monitor_every,
        adv_coverage_enable=bool(args.adv_coverage_enable),
        lambda_adv_target=args.lambda_adv_target,
        lambda_adv_warmup_epochs=args.lambda_adv_warmup_epochs,
        lambda_cov_target=args.lambda_cov_target,
        adv_steps_per_batch=args.adv_steps_per_batch,
        lambda_ramp_start_epoch=args.lambda_ramp_start_epoch,
        lambda_ramp_end_epoch=args.lambda_ramp_end_epoch,
        adv_mlp_hidden_dim=args.adv_mlp_hidden_dim,
        adv_mlp_dropout=args.adv_mlp_dropout,
        coverage_conditioning_mode=args.coverage_conditioning_mode,
        coverage_embed_dim=args.coverage_embed_dim,
        batch_labels_tsv=args.batch_labels_tsv,
        probe_eval_enable=bool(args.probe_eval_enable),
        probe_metadata_tsv=args.probe_metadata_tsv,
        probe_target_col=args.probe_target_col,
        probe_seed=args.probe_seed,
        wandb_enable=bool(args.wandb),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_group=args.wandb_group,
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode,
        debug_mode=bool(args.debug_mode),
        debug_examples_per_batch=args.debug_examples_per_batch,
        debug_max_batches_per_phase=args.debug_max_batches_per_phase,
    )
    summary = run_training(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
