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

## Key outputs

- `metrics.jsonl`: epoch metrics (train/val CE, masked accuracy, leakage monitors)
- `best_model.pt`: best checkpoint by validation CE
- `run_summary.json`: run config + best epoch
- `global_latents.npy`: `[n_samples, latent_dim]`
- `global_latents.csv`: sample IDs with latent columns

Coverage/batch confound monitors:

- `coverage_latent_norm_corr`
- `batch_latent_norm_r2` (if `--batch_labels_tsv` is provided)

## Notes

- Loss is computed only for intentionally masked observed SNPs.
- Naturally missing SNPs are never supervised.
- Missingness augmentation drops extra observed SNPs to improve robustness to variable aDNA coverage.
