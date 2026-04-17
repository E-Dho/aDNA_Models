# Genetic Distortion Map

This directory contains a genetic distortion map generated from the CS-OGE
country-focused latent embedding.

The goal is to compare learned genetic similarity structure against known sample
geography. The latent space is reduced to 2D, aligned to projected geographic
coordinates, and the remaining residual vectors are interpreted as places where
genetic similarity disagrees with physical geography.

## Input Data

Latents:

```text
../final_latents.csv
```

Metadata:

```text
../distortion_metadata_countryfocus_obs005.tsv
```

The metadata file was derived from the original `.anno` table and contains:

- `sample_id`
- `latitude`
- `longitude`
- `country`
- `original_group_id`
- `date_mean_bp`
- `observed_fraction`
- `locality`
- `political_entity`

The analysis used `7694` samples after the CS-OGE low-coverage pruning step.

## Command

Run from the CS-OGE project root:

```bash
MPLCONFIGDIR=/tmp/csoge_geo_mpl .venv_geo/bin/python scripts/genetic_distortion_map.py \
  --latents_csv analysis/overlap_embed_countryfocus_obs005_16308834/final_latents.csv \
  --metadata_tsv analysis/overlap_embed_countryfocus_obs005_16308834/distortion_metadata_countryfocus_obs005.tsv \
  --lat_col latitude \
  --lon_col longitude \
  --id_col sample_id \
  --label_col country \
  --time_col date_mean_bp \
  --pc_indices 1,3 \
  --project_coords 1 \
  --out_dir analysis/overlap_embed_countryfocus_obs005_16308834/genetic_distortion_map_pc1_pc3 \
  --seed 0
```

## Method

### 1. Latent Reduction

The original CS-OGE latent has 6 dimensions. This run used PCA components:

```text
PC1 and PC3
```

These were selected because they were useful in the previous visual inspection.
Together they explain:

```text
41.32% of latent variance
```

Breakdown:

```text
PC1 = 22.90%
PC3 = 18.42%
```

No UMAP or t-SNE was used. The distortion map uses PCA only so that the geometry
is inspectable and distance distortions are not introduced by nonlinear
visualization methods.

### 2. Geographic Projection

The latitude and longitude columns were projected into a metric planar coordinate
system before alignment:

```text
EPSG:3035
```

This is a Europe-friendly projected CRS. It avoids fitting the Procrustes
alignment directly on raw latitude/longitude degrees.

### 3. Procrustes Alignment

The 2D latent coordinates were aligned to projected geography using an explicit
similarity transform:

```text
geography ~= scale * standardized_latent @ rotation + translation
```

The fitted transform includes:

- rotation
- optional reflection
- uniform scaling
- translation

The transform is fitted with centered matrices and SVD. It is not a black-box
wrapper.

### 4. Distortion Vectors

For each sample, the script computes:

```text
dx = aligned_latent_x - true_projected_x
dy = aligned_latent_y - true_projected_y
distortion_magnitude = sqrt(dx^2 + dy^2)
```

The arrows in `distortion_map.png` point:

```text
from true geography to aligned latent-implied geography
```

Interpretation:

- short arrow: latent position agrees well with geography
- long arrow: latent position disagrees with geography
- coherent regional arrows: possible systematic genetic-geographic distortion
- isolated long arrows: possible outlier, label issue, sparse data effect, or
  real long-range genetic affinity

## Output Files

### `distortion_vectors.tsv`

Per-sample distortion table.

Important columns:

- `sample_id`
- `latitude`
- `longitude`
- `projected_x`
- `projected_y`
- `aligned_x`
- `aligned_y`
- `dx`
- `dy`
- `distortion_magnitude`
- `unit_dx`
- `unit_dy`
- `country`
- `date_mean_bp`

Use this file for downstream statistical checks, sorting outliers, or joining
back to metadata.

### `distortion_metrics.json`

Quantitative diagnostics for the run.

Key values:

```text
n_samples = 7694
pairwise_latent_geo_pearson = 0.5199
pairwise_latent_geo_spearman = 0.5367
rms_alignment_error = 694107.78
median_distortion_magnitude = 462430.13
mean_distortion_magnitude = 555221.81
knn_geography_overlap@15 = 0.0732
```

Distance units are projected meters.

### `distortion_map.png`

Main distortion map.

It shows true geographic points and residual arrows toward the aligned latent
position.

### `aligned_vs_true.png`

Overlay of:

- true projected geography
- aligned latent-implied geography

This checks whether the global alignment is plausible.

### `distortion_histogram.png`

Histogram of per-sample distortion magnitudes.

Useful for identifying whether distortion is broadly distributed or driven by a
small number of extreme points.

### `distance_correlation.png`

Scatter plot of sampled pairwise distances:

```text
x-axis: latent 2D distance
y-axis: projected geographic distance
```

Annotated with Pearson and Spearman correlations.

### `merged_latent_geo.tsv`

Merged latent-geography table after filtering. This is mostly useful for
debugging joins and checking which samples entered the map.

## Interpretation Notes

This map does not show literal migration routes.

The residual vectors are relative distortions after a global best-fit alignment.
They indicate where the learned genetic representation pulls samples away from
their physical geography.

Important caveats:

- Distortion magnitudes are in projected meters, but they should be interpreted
  comparatively, not as exact historical distances.
- The current CS-OGE run includes strong country supervision, so country
  separation can dominate subtler temporal or within-country structure.
- PC1 and PC3 explain only part of the 6D latent variance.
- Samples with unusual dates, admixed ancestry, or atypical labels can produce
  large residuals.
- Residuals can also reflect model artifacts, metadata errors, or uneven sample
  density.

## Suggested Follow-Up Checks

Useful next analyses:

- sort `distortion_vectors.tsv` by `distortion_magnitude`
- compare distortion magnitude by `country`
- compare distortion magnitude by `date_mean_bp`
- check whether distortion correlates with `observed_fraction`
- rerun with different PCA component choices, especially `PC1,PC2`
- rerun with weaker country supervision to see whether distortions become more
  biologically gradual

