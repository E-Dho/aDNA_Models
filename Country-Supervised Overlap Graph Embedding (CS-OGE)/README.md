# Country-Supervised Overlap Graph Embedding (CS-OGE)

This project learns a compact, coverage-robust embedding from an overlap-aware
sample graph instead of reconstructing per-sample SNP vectors. The current
variant adds country supervision during training to encourage stronger country
separation in the latent space.

## Model idea

- Convert 1240k Europe genotypes to an `int8` memmap.
- Collapse to `pseudohaploid` by default to reduce coverage-driven
  heterozygosity effects.
- Build pairwise similarities only over SNPs observed in both samples.
- Keep a sparse top-k sample graph and initialize with a spectral embedding.
- Train one learnable embedding vector per sample with:
  - positive attraction on graph edges
  - coverage-matched negative repulsion
  - variance whitening
  - covariance penalty
  - coverage leakage penalty
  - optional country supervision

The default target latent is `6D`.

## Local dataset

Filtered Europe dataset:

- `/Users/edo/Desktop/Hiwi/Daten/europe_country_ab_ge20_drop_premerge_gt90/europe_country_ab_ge20_drop_premerge_gt90.geno`
- `/Users/edo/Desktop/Hiwi/Daten/europe_country_ab_ge20_drop_premerge_gt90/europe_country_ab_ge20_drop_premerge_gt90.snp`
- `/Users/edo/Desktop/Hiwi/Daten/europe_country_ab_ge20_drop_premerge_gt90/europe_country_ab_ge20_drop_premerge_gt90.ind`
- `/Users/edo/Desktop/Hiwi/Daten/europe_country_ab_ge20_drop_premerge_gt90/europe_country_ab_ge20_drop_premerge_gt90.anno`

## Local workflow

### 1. Prepare memmap

```bash
cd "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)"
python3 prepare_memmap.py \
  --geno /Users/edo/Desktop/Hiwi/Daten/europe_country_ab_ge20_drop_premerge_gt90/europe_country_ab_ge20_drop_premerge_gt90.geno \
  --ind /Users/edo/Desktop/Hiwi/Daten/europe_country_ab_ge20_drop_premerge_gt90/europe_country_ab_ge20_drop_premerge_gt90.ind \
  --out_prefix "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/prepared_genotypes" \
  --ploidy pseudohaploid
```

### 2. Build overlap graph

```bash
python3 build_overlap_graph.py \
  --meta_json "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/prepared_genotypes.meta.json" \
  --anno /Users/edo/Desktop/Hiwi/Daten/europe_country_ab_ge20_drop_premerge_gt90/europe_country_ab_ge20_drop_premerge_gt90.anno \
  --out_dir "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90" \
  --ploidy pseudohaploid \
  --chunk_snps 4096 \
  --top_k 32 \
  --min_overlap 50000 \
  --overlap_shrinkage 100000 \
  --latent_dim 6
```

### 3. Train embedding

```bash
python3 train_overlap_embedding.py \
  --graph_npz "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/graph_edges.npz" \
  --spectral_init "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/spectral_init.npy" \
  --sample_stats_tsv "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/sample_stats.tsv" \
  --output_dir "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/train" \
  --latent_dim 6 \
  --epochs 400 \
  --wandb \
  --wandb_mode offline
```

### 4. Re-evaluate an exported latent

```bash
python3 evaluate_overlap_embedding.py \
  --latents_csv "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/train/final_latents.csv" \
  --sample_stats_tsv "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/sample_stats.tsv" \
  --output_dir "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/eval" \
  --label country
```

## Outputs

Graph build:

- `graph_edges.npz`
- `sample_stats.tsv`
- `spectral_init.npy`
- `build_summary.json`

Training:

- `best_model.pt`
- `metrics.jsonl`
- `latents.npy`
- `final_latents.csv`
- `coverage_dim_corr.tsv`
- `knn_report.tsv`
- `final_latent_pca.png`
- `final_latent_umap.png` when `umap-learn` is installed
- `run_summary.json`

## W&B behavior

The training script supports:

- `--wandb_mode online|offline|disabled`
- `--wandb_log_every`
- `--wandb_init_timeout`
- `--wandb_online_fallback`

HPC defaults are designed to avoid job instability:

- log every `10` epochs
- no `wandb.watch`
- no code or git snapshot upload
- automatic online-to-offline fallback on init failure
- only scalar metrics during training

## Tests

From the project root:

```bash
cd "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)"
python3 -m unittest discover -s tests -p 'test_*.py'
```

Some tests skip automatically if `torch` is unavailable.
