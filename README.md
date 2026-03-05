# aDNA_Models

This repository contains three active modeling pipelines for ancient DNA (aDNA)
representation learning:

- `VAE/`: Phase-1 masked VAE workflow on pre-existing Datamatrix `X`
- `HMAE/`: Hierarchical masked autoencoder with windowed local/global encoding
- `SNPTokenMaskedModel/`: Tokenized SNP masked model with chunked Transformer encoder

## Repository layout

- `README.md`: this overview
- `plot_latents.py`: shared latent-space plotting utility
- `VAE/README.md`: model-specific VAE details
- `HMAE/README.md`: model-specific HMAE details
- `SNPTokenMaskedModel/README.md`: model-specific token model details

## Quick start

1. Enter one model directory and install its requirements:

```bash
cd VAE && pip install -r requirements.txt
# or
cd HMAE && pip install -r requirements.txt
# or
cd SNPTokenMaskedModel && pip install -r requirements.txt
```

2. Run the corresponding preparation/training scripts described in each
   subdirectory README.

3. Visualize exported embeddings (example):

```bash
python3 plot_latents.py \
  --latents_csv /path/to/global_latents.csv \
  --metadata_tsv /path/to/metadata.anno \
  --output_dir /path/to/plots \
  --model_label model_name \
  --color_by "Group ID" \
  --color_by "Political Entity"
```

## Typical outputs

Across models, training runs commonly produce:

- `best_model.pt`
- `metrics.jsonl`
- `run_summary.json`
- `global_latents.npy`
- `global_latents.csv`

## Notes

- `HMAE/` and `SNPTokenMaskedModel/` train directly from EIGENSTRAT-derived
  memmaps.
- `VAE/` is a Datamatrix-based Phase-1 pipeline that assumes `X` already
  exists.
- Optional Weights & Biases logging is available in all three pipelines
  (including offline mode for cluster workflows).
