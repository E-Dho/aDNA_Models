# Phase 1 Masked VAE Pipeline (X Pre-Exists)

This folder implements a **VAE-first** workflow that assumes Datamatrix `X`
already exists and does **not** include any `X` generation step.

## What is implemented

1. `build_mask_from_x.py`
- Reads `X.sparse` + `X.info`
- Creates binary support mask `M.sparse` + `M.info`
- `M[row,col] = 1` for each coordinate present in `X.sparse`

2. `train_masked_vae.py`
- Optionally builds/rebuilds `M`
- Validates shape and coordinate coverage (`X` covered by `M`)
- Selects features from `X` (call-rate + variance filters)
- Builds dense memmaps for selected features
- Trains masked VAE (PyTorch)
- Writes metrics + review bundle + gate decision

## Outputs

Main artifacts in `--output_dir`:
- `M.sparse`, `M.info`
- `phase1_X_selected.float32.memmap`
- `phase1_M_selected.uint8.memmap`
- `splits.npz`
- `best_model.pt`
- `phase1_results.json`
- `gate_decision.json`
- `CAN_START_VQVAE.txt`
- `review_bundle.md`
- `val_latents.csv`

## CLI usage

### 1) Build `M` from existing `X`

```bash
python3 build_mask_from_x.py \
  --x_sparse /path/to/X.sparse \
  --x_info /path/to/X.info \
  --m_sparse /path/to/M.sparse \
  --m_info /path/to/M.info
```

### 2) Run Phase 1 (masked VAE + review gate)

```bash
python3 train_masked_vae.py \
  --x_sparse /path/to/X.sparse \
  --x_info /path/to/X.info \
  --x_ind /path/to/X.ind \
  --output_dir /path/to/output \
  --run_name phase1 \
  --rebuild_mask
```

## Key defaults

- `max_features=20000`
- `min_call_rate=0.05`
- `epochs=30`
- `latent_dim=32`
- Review gate:
  - `min_recon_improvement=0.01`
  - `min_active_latent_dims=4`
  - `active_latent_std_threshold=0.05`

## Dependencies

- Python 3.10+
- NumPy
- PyTorch (required for training)
- Weights & Biases (`wandb`, optional; required only when `--wandb` is used)

Mask creation and preprocessing checks run without PyTorch.

## W&B Monitoring

Enable W&B logging during training:

```bash
python3 train_masked_vae.py \
  --x_sparse /path/to/X.sparse \
  --x_info /path/to/X.info \
  --x_ind /path/to/X.ind \
  --output_dir /path/to/output \
  --run_name phase1 \
  --rebuild_mask \
  --wandb \
  --wandb_project your-project \
  --wandb_entity your-team \
  --wandb_name phase1-run-001 \
  --wandb_tags phase1,vae,baseline
```

Available W&B flags:
- `--wandb`
- `--wandb_project`
- `--wandb_entity`
- `--wandb_name`
- `--wandb_group`
- `--wandb_tags` (comma-separated)
- `--wandb_mode` (`online` or `offline`)
