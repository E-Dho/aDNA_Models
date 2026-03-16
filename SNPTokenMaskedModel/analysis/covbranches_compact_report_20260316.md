# Covbranches Compact Report

- Source aggregate: `/tmp/covbranches_chain_20260313_112054_jobs_aggregate.tsv`

## Overall

- Best CE: `covcond_film` (`best_val_ce=0.390949`, `final_val_ce=0.394438`)
- Lowest |coverage_corr_mean|: `advcov_cov_l050_c001` (`coverage_corr_mean=-0.007029`)

## Branch Stats

| Branch | n | mean_best_ce | best_ce | mean_abs_covcorr | best_abs_covcorr | mean_r2_z | mean_r2_z_unit |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 6 | 0.396531 | 0.391672 | 0.208214 | 0.007029 | 0.973852 | 0.970243 |
| B | 4 | 0.391759 | 0.390949 | 0.288032 | 0.029768 | 0.973368 | 0.966380 |
| B+adv | 2 | 0.392525 | 0.392078 | 0.059137 | 0.037007 | 0.972923 | 0.971425 |

## Top 5 by best_val_ce

| run_name | branch | best_val_ce | coverage_corr_mean | r2_coverage_from_z |
|---|---|---:|---:|---:|
| covcond_film | B | 0.390949 | 0.029768 | 0.978377 |
| advcov_cov_l020_c001 | A | 0.391672 | 0.313438 | 0.987921 |
| covcond_film_concat | B | 0.391733 | 0.214679 | 0.974806 |
| covcond_concat_smallcov | B | 0.392053 | 0.392865 | 0.970059 |
| covcond_film_advcov | B+adv | 0.392078 | -0.037007 | 0.956888 |

## Top 5 by |coverage_corr_mean|

| run_name | branch | abs_cov_corr | coverage_corr_mean | best_val_ce | r2_coverage_from_z |
|---|---|---:|---:|---:|---:|
| advcov_cov_l050_c001 | A | 0.007029 | -0.007029 | 0.395260 | 0.972861 |
| covcond_film | B | 0.029768 | 0.029768 | 0.390949 | 0.978377 |
| covcond_film_advcov | B+adv | 0.037007 | -0.037007 | 0.392078 | 0.956888 |
| covcond_concat_advcov | B+adv | 0.081267 | -0.081267 | 0.392972 | 0.988958 |
| advcov_cov_l050_c001_adv3 | A | 0.095533 | -0.095533 | 0.395757 | 0.975793 |
