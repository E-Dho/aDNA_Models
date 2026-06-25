# Geolocation Model

This model predicts ancient DNA sample geolocation from LD-pruned genotype PCs plus sample age.

## Baseline Pipeline

1. Convert packed EIGENSTRAT to PLINK BED/BIM/FAM with `convertf`.
2. LD-prune with PLINK:

```bash
plink --indep-pairwise 200 25 0.2
```

3. Run PLINK PCA with `256` PCs on the pruned SNPs.
4. Join PCA eigenvectors to the dataset `.anno` metadata.
5. Train a PyTorch MLP on `256` PCs plus scaled date BP.
6. Predict an isotropic 2D Gaussian in projected Europe coordinates (`EPSG:3035`).

The final layer is `Linear(64, 3)` with output interpretation:

```text
mu_x, mu_y, log_sigma
```

`log_sigma` defines one isotropic uncertainty radius in projected meters.

## Dataset

Default cluster dataset:

```text
/work_beegfs/sunip956/Hiwi/Daten/v62.0_HO_public_europe_groupid_cov5to90_nonsingleton/
```

Required files:

```text
v62.0_HO_public_europe_groupid_cov5to90_nonsingleton.geno
v62.0_HO_public_europe_groupid_cov5to90_nonsingleton.snp
v62.0_HO_public_europe_groupid_cov5to90_nonsingleton.ind
v62.0_HO_public_europe_groupid_cov5to90_nonsingleton.anno
```

## Training Target

Coordinates are read from `.anno` columns:

```text
Lat.
Long.
```

Age is read from the AADR date mean column beginning with:

```text
Date mean in BP
```

The leave-site-out split uses:

```text
Locality
```

Training targets receive Gaussian coordinate jitter with sigma `25 km`. Validation and test targets are not jittered.

## Outputs

Runs live under:

```text
aDNA_Models/Geolocation_Model/runs/<run_name>/
```

Each run writes:

```text
hypothesis.md
config.yaml
results.md
links.md
metrics.csv
run_summary.json
evaluation_summary.json
sample_stats.tsv
split_sites.json
training_history.csv
test_predictions.csv
best_model.pt
plink_prune.prune.in
plink_pca.eigenvec
plink_pca.eigenval
```

## Local Invocation

```bash
python3 aDNA_Models/Geolocation_Model/geolocation_model/run_geolocation_model.py \
  --geno Daten/v62.0_HO_public_europe_groupid_cov5to90_nonsingleton/v62.0_HO_public_europe_groupid_cov5to90_nonsingleton.geno \
  --snp Daten/v62.0_HO_public_europe_groupid_cov5to90_nonsingleton/v62.0_HO_public_europe_groupid_cov5to90_nonsingleton.snp \
  --ind Daten/v62.0_HO_public_europe_groupid_cov5to90_nonsingleton/v62.0_HO_public_europe_groupid_cov5to90_nonsingleton.ind \
  --anno Daten/v62.0_HO_public_europe_groupid_cov5to90_nonsingleton/v62.0_HO_public_europe_groupid_cov5to90_nonsingleton.anno \
  --output_dir aDNA_Models/Geolocation_Model/runs
```

## Cluster Invocation

Use the CAU sbatch:

```bash
sbatch /work_beegfs/$USER/Hiwi/DataFilter/cluster/caucluster/submit_geolocation_model_ho.sbatch
```

Smoke test:

```bash
MAX_SAMPLES=300 MAX_SNPS=20000 PCA_DIMS=32 EPOCHS=1 RUN_NAME=smoke_geo_mlp sbatch \
  /work_beegfs/$USER/Hiwi/DataFilter/cluster/caucluster/submit_geolocation_model_ho.sbatch
```

## Precision Fusion Variants

The original uncertainty-based precision fusion runner remains unchanged:

```bash
python3 aDNA_Models/Geolocation_Model/geolocation_model/run_precision_fusion.py --help
```

The supervised kNN-calibration variant lives beside it as a separate experiment type:

```bash
python3 aDNA_Models/Geolocation_Model/geolocation_model/run_precision_fusion_supervised_knn_calibration.py --help
```

This variant is logged as `precision_fusion_supervised_knn_calibration`. It keeps the train-only weighted kNN expert and global MLP expert, but trains the kNN sigma calibrator directly against `log(true_kNN_error + 1 km)` before evaluating precision fusion. It writes additional calibration diagnostics:

```text
knn_sigma_correlation.tsv
knn_sigma_calibration_deciles.tsv
calibrator_training_history.csv
mlp_training_history.csv
```

CAU submission script:

```bash
sbatch /work_beegfs/$USER/Hiwi/DataFilter/cluster/caucluster/submit_precision_fusion_supervised_knn_calibration_ho.sbatch
```

Smoke test:

```bash
MAX_SAMPLES=300 MAX_SNPS=20000 PCA_DIMS=32 EPOCHS=2 CALIBRATOR_EPOCHS=2 RUN_NAME=smoke_supervised_knn_sigma sbatch \
  /work_beegfs/$USER/Hiwi/DataFilter/cluster/caucluster/submit_precision_fusion_supervised_knn_calibration_ho.sbatch
```
