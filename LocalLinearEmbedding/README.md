# LocalLinearEmbedding

This folder contains experiments that test Locally Linear Embedding (LLE) on ancient DNA datasets.

The primary run uses the HO Europe Group_ID-filtered dataset:

- `v62.0_HO_public_europe_groupid_cov5to90_nonsingleton`

The production pipeline is cluster-first:

1. Convert packed EIGENSTRAT to PLINK `PACKEDPED`/BED with `convertf`.
2. Run PLINK LD pruning with `--indep-pairwise 200 25 0.2`.
3. Run PLINK PCA with `--pca 100` on pruned SNPs.
4. Run sklearn LLE with `n_neighbors=32` and `n_components=6` on the 100 PCA dimensions.
5. Evaluate original-group purity, date locality, and coverage predictability.

Experiment outputs live in:

- Local: `aDNA_Models/LocalLinearEmbedding/runs/`
- HPC: `/work_beegfs/sunip956/Hiwi/aDNA_Models/LocalLinearEmbedding/runs/`

Each run follows the repository experiment logging standard and includes `hypothesis.md`, `config.yaml`, `results.md`, `links.md`, `metrics.csv`, `evaluation_summary.json`, and latent/plot artifacts.
