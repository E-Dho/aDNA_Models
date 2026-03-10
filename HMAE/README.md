# Hierarchical Masked Autoencoder (HMAE) for aDNA

This directory contains a scalable baseline for sparse aDNA dimensionality
reduction with:

- windowed SNP processing
- configurable local encoder across windows (`conv_attn` or legacy `meanpool`)
- configurable local encoder across windows (`conv_attn`,
  `conv_attn_multislot`, or legacy `meanpool`)
- masked denoising objective on observed SNPs
- attention-based global aggregation over window latents
- optional variational global latent

The implementation is designed for very large SNP sets (>1M SNPs) by avoiding
a single massive dense input layer.

## Layout

- `prepare_eigenstrat_memmap.py`: convert EIGENSTRAT `.geno` to int8 memmap
- `train_hmae.py`: train HMAE and export global embeddings
- `hmae/data.py`: windowing, memmap metadata, batching helpers
- `hmae/model.py`: local encoder, global attention aggregator, decoder
- `hmae/train.py`: training/evaluation/embedding export pipeline

## Data representation

- Genotypes are integer categories:
  - diploid mode: `0,1,2` (`-1` missing)
  - pseudo-haploid mode: `0,1` (`-1` missing)
- Observed mask is derived from `genotype >= 0`.
- Training mask is generated per step by masking a random subset of observed
  SNPs (`mask_prob`), separate from the missing-data mask.

## Quick start

### 1) Build memmap from EIGENSTRAT `.geno`

```bash
cd /Users/edo/Desktop/Hiwi/aDNA_Models/HMAE
python3 prepare_eigenstrat_memmap.py \
  --geno /path/to/data.geno \
  --ind /path/to/data.ind \
  --out_prefix /path/to/output/aadr_unfiltered \
  --ploidy diploid
```

This produces:

- `/path/to/output/aadr_unfiltered.genotypes.int8.memmap`
- `/path/to/output/aadr_unfiltered.meta.json`

### 2) Train deterministic baseline

```bash
cd /Users/edo/Desktop/Hiwi/aDNA_Models/HMAE
python3 train_hmae.py \
  --meta_json /path/to/output/aadr_unfiltered.meta.json \
  --output_dir /path/to/output/hmae_run \
  --window_size 8192 \
  --windows_per_step 8 \
  --window_latent_dim 32 \
  --global_latent_dim 32 \
  --local_encoder_type conv_attn \
  --window_latent_slots 4 \
  --local_conv_layers 4 \
  --local_conv_kernel 7 \
  --local_attn_heads 4 \
  --local_dropout 0.1 \
  --decoder_attn_heads 4 \
  --mask_prob 0.2 \
  --observed_dropout 0.1
```

### 2b) Train with offline W&B

```bash
cd /Users/edo/Desktop/Hiwi/aDNA_Models/HMAE
python3 train_hmae.py \
  --meta_json /path/to/output/aadr_unfiltered.meta.json \
  --output_dir /path/to/output/hmae_run \
  --wandb \
  --wandb_mode offline \
  --wandb_project hmae-phase1 \
  --wandb_name hmae-baseline
```

### 2c) GPU-first training flags

- `--require_cuda`: fail early if CUDA is unavailable.
- `--amp` / `--no-amp`: mixed precision override.
  - Default behavior is automatic: AMP on CUDA, off on CPU.
- `--window_latent_slots`: number of learned latent query tokens per window
  in `conv_attn_multislot` mode.
- `--decoder_attn_heads`: decoder cross-attention heads (multislot path).

### 2d) W&B online with low overhead

- `--wandb_mode online` to stream to W&B directly.
- `--wandb_log_every 2` to reduce payload/chattiness.
- `--wandb_init_timeout 45` to cap online init wait before fallback.
- `--wandb_online_fallback` (default) allows automatic fallback if online init
  or logging fails, so training continues and local files are still written.

### 3) Outputs

- `metrics.jsonl`: step/epoch metrics
- `best_model.pt`: best checkpoint by validation reconstruction
- `run_summary.json`: run configuration and final metrics
- `global_latents.npy`: global embeddings `[n_samples, global_dim]`
- `global_latents.csv`: sample IDs + global embedding columns

Coverage/batch confound checks are logged per epoch in `metrics.jsonl`:

- `coverage_latent_norm_corr`
- `batch_latent_norm_r2` (if `--batch_labels_tsv` is provided)
- W&B offline files under `output_dir/wandb/offline-run-*` when enabled.

You can later sync offline runs:

```bash
wandb sync /path/to/output/hmae_run/wandb/offline-run-*
```

### 4) Visualize exported latents

```bash
cd /Users/edo/Desktop/Hiwi
python3 plot_latents.py \
  --latents_csv /path/to/output/hmae_run/global_latents.csv \
  --metadata_tsv /Users/edo/Desktop/Hiwi/Daten/dataverse_files/v62.0_1240k_public.anno \
  --output_dir /path/to/output/hmae_run/plots \
  --model_label hmae \
  --color_by "Group ID" \
  --color_by "Political Entity"
```

This writes:

- `merged_latents_metadata.csv`
- `pca_coords.csv`
- one PNG plot per available method and requested metadata field

`umap` and `tsne` outputs are generated only if `umap-learn` and
`scikit-learn` are installed.

## Notes

- This baseline uses cross-entropy reconstruction on intentionally masked
  observed SNPs only.
- It uses random window sampling per step (not all windows at once).
- `conv_attn` local encoder is the default for v2.
- `conv_attn_multislot` uses learned latent slot queries per window and passes
  all slot tokens to the global transformer.
- Use `--local_encoder_type meanpool` for backward-compatible behavior.
- Use `--variational` to enable a global VAE latent with KL term.
