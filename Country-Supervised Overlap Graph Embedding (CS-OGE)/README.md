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

### 5. Render interactive latent HTML

For trained latents, render chosen latent axes directly:

```bash
python3 render_latent_html.py \
  --latents_csv "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/train/final_latents.csv" \
  --sample_stats_tsv "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/sample_stats.tsv" \
  --output_html "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/train/pc1_pc2_pc3_original_group_id.html" \
  --color_by original_group_id \
  --axes 1,2,3 \
  --projection direct
```

For spectral initialization or other high-dimensional matrices, PCA-project first:

```bash
python3 render_latent_html.py \
  --latents_npy "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/spectral_init.npy" \
  --sample_stats_tsv "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/sample_stats.tsv" \
  --output_html "/Users/edo/Desktop/Hiwi/aDNA_Models/Country-Supervised Overlap Graph Embedding (CS-OGE)/runs/europe_drop_premerge_gt90/spectral_init_pc1_pc2_pc3_country.html" \
  --color_by country \
  --axes 1,2,3 \
  --projection pca
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

## Out-of-sample encoder

The OOS encoder is a post-hoc distillation model. CS-OGE remains the teacher
that defines the coordinate frame; the encoder learns `genotype -> teacher
latent` so later samples can be inserted without rebuilding the full overlap
graph.

The first intended teacher is the HO run:

```text
csoge_ho_dropgt90_obs005_country0375_origgroup020_16910148
```

### 1. Prepare teacher reference

This writes the SNP order contract (`snp_manifest.tsv`) and copies teacher
latents/metadata into one reference directory. Use prefix-plus-extension paths,
not `Path.with_suffix`, because HO prefixes contain dots.

```bash
python3 prepare_encoder_reference.py \
  --data_prefix /work_beegfs/sunip956/Hiwi/Daten/v62.0_HO_public_europe_country_ab_ge20_drop_premerge_gt90/v62.0_HO_public_europe_country_ab_ge20_drop_premerge_gt90 \
  --teacher_run_dir /work_beegfs/sunip956/projects/overlap_embed_runs/csoge_ho_dropgt90_obs005_country0375_origgroup020_16910148 \
  --graph_dir /work_beegfs/sunip956/aadr/overlap_embed/ho_europe_country_ab_ge20_drop_premerge_gt90_graph_obs005 \
  --out_dir /work_beegfs/sunip956/aadr/overlap_embed/oos_encoder_refs/ho_origgroup020_16910148
```

### 2. Train encoder

Train against the teacher memmap and teacher `final_latents.csv`. The default
objective combines latent MSE/cosine alignment, mini-batch distance preservation,
coverage leakage penalty, and SNP dropout for sparse-data robustness.

```bash
python3 train_oos_encoder.py \
  --meta_json /work_beegfs/sunip956/aadr/overlap_embed/ho_europe_country_ab_ge20_drop_premerge_gt90.meta.json \
  --snp_manifest /work_beegfs/sunip956/aadr/overlap_embed/oos_encoder_refs/ho_origgroup020_16910148/snp_manifest.tsv \
  --teacher_latents_csv /work_beegfs/sunip956/projects/overlap_embed_runs/csoge_ho_dropgt90_obs005_country0375_origgroup020_16910148/final_latents.csv \
  --sample_stats_tsv /work_beegfs/sunip956/aadr/overlap_embed/ho_europe_country_ab_ge20_drop_premerge_gt90_graph_obs005/sample_stats.tsv \
  --output_dir /work_beegfs/sunip956/projects/overlap_embed_runs/oos_encoder_ho_origgroup020_16910148 \
  --latent_dim 6 \
  --epochs 100 \
  --batch_size 64 \
  --wandb_mode online
```

For a quick smoke, add `--max_samples 256 --epochs 2 --wandb_mode disabled`.

The deeper encoder variant is available with:

```bash
python3 train_oos_encoder.py \
  ... \
  --model_type sparse_deep_encoder \
  --deep_hidden_dim 100000 \
  --deep_bottleneck_dim 1024 \
  --sparse_fanout 2
```

This is intentionally not a dense `584,131 -> 100,000` layer. Each SNP connects
to a small number of deterministic hidden buckets, giving a sparse first layer,
then a dense GELU stack:

```text
genotype/mask vectors
  -> sparse 584,131 -> 100,000 bucket projection
  -> GELU
  -> dense 100,000 -> 1,024
  -> GELU
  -> dense 1,024 -> 6 teacher-frame latent
```

The sparse first layer now defaults to fan-in-aware initialization and can use
Bucket-RigL dynamic sparse training:

```bash
python3 train_oos_encoder.py \
  ... \
  --model_type sparse_deep_encoder \
  --sparse_init fan_in \
  --dst_enable \
  --dst_update_interval_steps 100 \
  --dst_start_epoch 5 \
  --dst_end_epoch 80 \
  --dst_initial_drop_fraction 0.30 \
  --dst_final_drop_fraction 0.05 \
  --dst_candidate_multiplier 16
```

Fan-in-aware initialization scales every active SNP-to-bucket connection by the
actual fan-in of its target hidden bucket. DST periodically prunes low-salience
active SNP-to-bucket connections and grows new ones from sampled inactive
candidates using hidden-gradient scores.

Outputs:

- `encoder.pt`
- `encoder_config.json`
- `snp_manifest.tsv`
- `metrics.jsonl`
- `heldout_predictions.csv`
- `training_summary.json`

### 3. Encode new samples

New samples must be aligned to the teacher SNP manifest. Missing teacher SNPs
are encoded as missing calls; extra source SNPs are ignored; allele-set
mismatches are reported and skipped.

```bash
python3 encode_samples.py \
  --encoder_dir /work_beegfs/sunip956/projects/overlap_embed_runs/oos_encoder_ho_origgroup020_16910148 \
  --geno /path/to/new_samples.geno \
  --snp /path/to/new_samples.snp \
  --ind /path/to/new_samples.ind \
  --out_dir /work_beegfs/sunip956/projects/overlap_embed_runs/oos_encoded_new_samples \
  --teacher_latents_csv /work_beegfs/sunip956/projects/overlap_embed_runs/csoge_ho_dropgt90_obs005_country0375_origgroup020_16910148/final_latents.csv \
  --teacher_sample_stats_tsv /work_beegfs/sunip956/aadr/overlap_embed/ho_europe_country_ab_ge20_drop_premerge_gt90_graph_obs005/sample_stats.tsv
```

Outputs:

- `encoded_latents.csv`
- `encoded_sample_stats.tsv`
- `nearest_neighbors.tsv` when teacher latents are supplied
- `encoding_report.json`

### 4. Evaluate held-out predictions

```bash
python3 evaluate_oos_encoder.py \
  --predicted_latents_csv /work_beegfs/sunip956/projects/overlap_embed_runs/oos_encoder_ho_origgroup020_16910148/heldout_predictions.csv \
  --teacher_latents_csv /work_beegfs/sunip956/projects/overlap_embed_runs/csoge_ho_dropgt90_obs005_country0375_origgroup020_16910148/final_latents.csv \
  --sample_stats_tsv /work_beegfs/sunip956/aadr/overlap_embed/ho_europe_country_ab_ge20_drop_premerge_gt90_graph_obs005/sample_stats.tsv \
  --output_dir /work_beegfs/sunip956/projects/overlap_embed_runs/oos_encoder_ho_origgroup020_16910148/oos_eval
```

Transfer-learning flags (`--pretrain_source_meta_json` and
`--pretrain_source_sample_stats_tsv`) enable a conservative pretraining stage
before teacher fine-tuning. The pretraining objective uses two SNP-dropout views
of the same source sample, metadata structure losses when country/original-group
or date labels exist, and then anchors the final head by fine-tuning to the
teacher CS-OGE latent coordinates.

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
