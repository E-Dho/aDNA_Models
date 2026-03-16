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

Coverage/batch confound monitors:

- `coverage_latent_norm_corr`
- `batch_latent_norm_r2` (if `--batch_labels_tsv` is provided)
- `train_adv_mse`, `val_adv_mse`, `lambda_adv` (when adversarial coverage is enabled)
- `lambda_cov`, `cov_penalty`, `val_cov_penalty`, `adv_steps_per_batch`
- final `R²(coverage <- z)` and `R²(coverage <- z / ||z||)` in `run_summary.json`
- final `corr_pc1_coverage` ... `corr_pc5_coverage` in `run_summary.json`

## Notes

- Loss is computed only for intentionally masked observed SNPs.
- Naturally missing SNPs are never supervised.
- Missingness augmentation drops extra observed SNPs to improve robustness to variable aDNA coverage.
