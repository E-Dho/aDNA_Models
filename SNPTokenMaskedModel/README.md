# Tokenized SNP Masked Model (Chunked Encoder)

This project implements a second self-supervised model for aDNA embeddings that treats SNPs as tokens instead of using a single ultra-wide dense input.

## What it does

- Represents SNP calls as discrete tokens.
  - Diploid: `0,1,2` plus `MISSING`
  - Pseudo-haploid: `0,1` plus `MISSING`
- Uses two masks during training:
  - `obs_mask`: naturally observed vs missing coverage
  - `train_mask`: intentionally masked targets for reconstruction
- Samples random contiguous genome windows each training step.
- Runs a chunked sequence encoder:
  - local Transformer over fixed chunks
  - global Transformer over chunk summaries
- Reconstructs only intentionally masked, observed SNP tokens via cross-entropy.
- Exports stable per-sample embeddings by averaging pooled window embeddings across many random windows.

## Layout

- `prepare_eigenstrat_memmap.py`: convert EIGENSTRAT `.geno` into int8 memmap
- `train_token_snp.py`: train and export embeddings
- `token_snp/data.py`: memmap I/O, random window batch sampling, masks
- `token_snp/model.py`: token embedding + chunked Transformer encoder + reconstruction head
- `token_snp/train.py`: training loop, validation, monitoring, embedding export

## Quick start

### 1) Build memmap

```bash
cd /Users/edo/Desktop/Hiwi/SNPTokenMaskedModel
python3 prepare_eigenstrat_memmap.py \
  --geno /path/to/europe_filtered_eigenstrat.geno \
  --ind /path/to/europe_filtered_eigenstrat.ind \
  --out_prefix /path/to/hmae/europe_tokenized \
  --ploidy diploid
```

### 2) Train model

```bash
cd /Users/edo/Desktop/Hiwi/SNPTokenMaskedModel
python3 train_token_snp.py \
  --meta_json /path/to/hmae/europe_tokenized.meta.json \
  --output_dir /path/to/projects/token_snp_europe \
  --window_size 8192 \
  --chunk_size 256 \
  --mask_prob 0.2 \
  --missingness_dropout 0.1 \
  --latent_dim 64 \
  --embedding_windows_per_sample 64
```

### 2b) Offline W&B (cluster-friendly)

```bash
python3 train_token_snp.py \
  --meta_json /path/to/hmae/europe_tokenized.meta.json \
  --output_dir /path/to/projects/token_snp_europe \
  --wandb \
  --wandb_mode offline \
  --wandb_project token-snp-mask
```

### 2c) Adversarial coverage removal (GRL)

The training loop supports adversarial removal of coverage signal from pooled latent `z`.
An MLP adversary predicts standardized `coverage_observed_fraction`; gradients are reversed
before reaching the encoder so the encoder is pushed to remove coverage information.

```bash
python3 train_token_snp.py \
  --meta_json /path/to/europe_tokenized.meta.json \
  --output_dir /path/to/projects/token_snp_europe_advcov \
  --adv_coverage_enable \
  --lambda_adv_target 0.05 \
  --lambda_adv_warmup_epochs 10 \
  --adv_mlp_hidden_dim 128 \
  --adv_mlp_dropout 0.1
```

Notes:
- Reconstruction loss and token masking behavior remain unchanged.
- Total training objective is `recon_loss + adv_loss + lambda_cov * cov_penalty`.
- Early stopping / best checkpoint selection remains based on `val_ce`.

### 2d) Branch A knobs (stronger adversary + covariance penalty)

Additional controls:

- `--lambda_cov_target`: target multiplier for covariance/correlation penalty.
- `--adv_steps_per_batch`: adversary-only update steps per main encoder step.
- `--lambda_ramp_start_epoch`, `--lambda_ramp_end_epoch`: shared ramp schedule for `lambda_adv` and `lambda_cov`.

Schedule behavior:

- epochs `< lambda_ramp_start_epoch`: both lambdas are `0`.
- ramp window: linearly increases to target.
- epochs `>= lambda_ramp_end_epoch`: fixed at target.

### 2e) Branch B knobs (coverage-conditioned reconstruction)

Coverage is provided explicitly to the reconstruction pathway while keeping latent extraction unchanged.

- `--coverage_conditioning_mode none|concat|film|film_concat`
- `--coverage_embed_dim` (used by concat mode; `0` means scalar coverage directly)

### 2f) Optional Group-ID random under-sampling (train only)

Training batches can be rebalanced on the fly so large `Group ID` classes do not dominate
individual batches. This does not alter the underlying dataset and does not affect validation
or test sampling.

- `--use_group_rus`
- `--group_rus_mode balanced_batch`
- `--group_rus_max_per_group_in_batch` (optional hard cap per group inside a batch)
- `--group_rus_seed`
- `--group_metadata_tsv`
- `--group_metadata_col` (default: `Group ID`)
- `--group_metadata_key` (defaults to the first metadata column)

Notes:
- only `balanced_batch` is implemented in this pass
- group labels are loaded from metadata, not from `--batch_labels_tsv`
- when disabled, training uses the original random batch sampling path unchanged

### 2g) Strong latent reconstruction (decoder-side)

The default baseline keeps the encoder unchanged and reconstructs directly from token contextual states.
The strong latent reconstruction variant adds a decoder-side forcing block immediately before the logits head
so reconstruction depends more directly on pooled latent `z`.

Flags:

- `--use_strong_latent_reconstruction`
- `--latent_recon_mode film_only|crossattn_only|film_plus_crossattn|latent_only`
- `--latent_recon_hidden_dim` (default: `d_model`)
- `--latent_recon_num_heads`
- `--latent_recon_dropout`
- `--latent_recon_num_tokens`
- `--latent_cross_attn_residual_scale` (default: `0.1`, lower means the decoder keeps less of the direct token-context shortcut during latent cross-attention)
- `--latent_only_aux_weight` (default: `0.0`, adds an auxiliary CE from the `latent_only` branch to push more information into pooled `z`)

Modes:

- `film_only`: FiLM-style modulation of decoder token states from pooled `z`
- `crossattn_only`: token queries attend to `z`-derived latent decoder tokens
- `film_plus_crossattn`: applies both, this is the recommended comparison mode
- `latent_only`: diagnostic ablation that reconstructs from structural token information plus `z`, not from encoder contextual token states

Additional pressure mechanisms:

- lower `latent_cross_attn_residual_scale` weakens the direct residual shortcut from encoder token states into the latent cross-attention path
- positive `latent_only_aux_weight` trains a second latent-only decoder branch alongside the main reconstruction loss
- increasing `mask_prob` makes local token context less sufficient, which further raises pressure on `z`

Validation additionally logs:

- `val_ce_with_latent`
- `val_ce_without_latent`
- `latent_usage_gap = val_ce_without_latent - val_ce_with_latent`

Interpretation:

- `latent_usage_gap ~= 0`: reconstruction is still largely bypassing `z`
- clearly positive `latent_usage_gap`: decoder-side forcing is making reconstruction use `z`

Recommended workflow:

1. Compare baseline vs `film_only`, `crossattn_only`, and `film_plus_crossattn` at the same latent size.
2. Once one mode shows a positive `latent_usage_gap`, rerun with smaller `latent_dim`.
3. Shrinking `latent_dim` should now hurt CE materially if the architecture is depending on `z` as intended.

### 2h) Country loss on pooled latent `z`

Country loss is an optional supervised latent-space objective that uses metadata labels, typically
`Group ID` from the dataset-specific `.anno`, to pull same-country latent vectors together and push
different-country latent vectors apart.

Flags:

- `--use_country_loss`
- `--lambda_country_loss_target`
- `--country_loss_pos_margin`
- `--country_loss_neg_margin`
- `--country_loss_normalize_z` / `--no-country_loss_normalize_z`
- `--country_loss_ramp_start_epoch`
- `--country_loss_ramp_end_epoch`
- `--country_loss_groups_per_batch`
- `--country_loss_metadata_tsv`
- `--country_loss_metadata_col` (default: `Group ID`)
- `--country_loss_metadata_key`
- `--group_rus_groups_per_batch`

The loss is:

- `country_loss = positive_country_loss + negative_country_loss`
- positive term: penalizes same-country pairs farther apart than the positive margin
- negative term: penalizes different-country pairs closer together than the negative margin

For country-loss runs with small batches, set Group-RUS to sample multiple examples per selected country,
for example `batch_size=6` and `group_rus_groups_per_batch=3`, so every training batch has positive pairs.

Country-loss metrics:

- `lambda_country_loss`
- `train_country_loss`, `train_positive_country_loss`, `train_negative_country_loss`
- `train_country_positive_pairs_mean`, `train_country_negative_pairs_mean`
- `val_country_loss`, `val_positive_country_loss`, `val_negative_country_loss`
- `val_country_positive_pairs_mean`, `val_country_negative_pairs_mean`

## Key outputs

- `metrics.jsonl`: epoch metrics (train/val CE, masked accuracy, leakage monitors)
- `best_model.pt`: best checkpoint by validation CE
- `run_summary.json`: run config + best epoch
- `global_latents.npy`: `[n_samples, latent_dim]`
- `global_latents.csv`: sample IDs with latent columns
- `coverage_observed_fraction.csv`: observed-coverage target per sample
- `probe_summary.tsv`: coverage probe summary (`coverage <- z`, `coverage <- z/||z||`)
- `pc_coverage_corr.tsv`: PC/coverage correlations
- `joined_data_<run_name>.csv`: per-sample coverage diagnostics
- `norm_vs_coverage_scatter_<run_name>.png`: norm vs coverage scatter
- `group_rus_epoch_counts.jsonl`: epoch-level sampled counts by group when Group-RUS is enabled

Coverage/batch confound monitors:

- `coverage_latent_norm_corr`
- `batch_latent_norm_r2` (if `--batch_labels_tsv` is provided)
- `train_adv_mse`, `val_adv_mse`, `lambda_adv` (when adversarial coverage is enabled)
- `lambda_cov`, `cov_penalty`, `val_cov_penalty`, `adv_steps_per_batch`
- `group_rus_unique_group_ids_mean`, `group_rus_max_group_fraction_mean`, `group_rus_group_entropy_mean`
- `lambda_country_loss`, `train_country_loss`, `val_country_loss` and positive/negative country-loss components
- `val_ce_with_latent`, `val_ce_without_latent`, `latent_usage_gap`
- final `R²(coverage <- z)` and `R²(coverage <- z / ||z||)` in `run_summary.json`
- final `corr_pc1_coverage` ... `corr_pc5_coverage` in `run_summary.json`

## Notes

- Loss is computed only for intentionally masked observed SNPs.
- Naturally missing SNPs are never supervised.
- Missingness augmentation drops extra observed SNPs to improve robustness to variable aDNA coverage.
