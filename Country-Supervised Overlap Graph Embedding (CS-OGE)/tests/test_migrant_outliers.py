from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from overlap_embed.data import write_tsv
from overlap_embed.eval import write_latents_csv
from overlap_embed.outlier import (
    add_geo_candidate_filters,
    calibrate_epsilon,
    expand_candidate_knn_rows,
    haversine_km,
    score_migrant_outliers,
    spherical_mean_latlon,
)


def write_anno(path: Path, rows: list[dict[str, object]]) -> None:
    header = ["Genetic ID", "Lat.", "Long."]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(col, "")) for col in header) + "\n")


class MigrantOutlierTests(unittest.TestCase):
    def test_haversine_known_distance(self) -> None:
        distance = haversine_km(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(distance, 111.195, places=2)

    def test_spherical_mean_ignores_missing_coordinates(self) -> None:
        lat, lon = spherical_mean_latlon(
            np.asarray([0.0, np.nan, 0.0]),
            np.asarray([0.0, 10.0, 2.0]),
        )
        self.assertAlmostEqual(lat, 0.0, places=6)
        self.assertAlmostEqual(lon, 1.0, places=2)

    def test_calibrate_epsilon_uses_only_high_purity_samples(self) -> None:
        distances = np.asarray([10.0, 20.0, 1000.0, 2000.0])
        purity = np.asarray([0.9, 0.8, 0.79, np.nan])
        epsilon, table, n = calibrate_epsilon(
            distances,
            purity,
            adaptive_purity_min=0.80,
            epsilon_quantile=0.95,
        )
        self.assertEqual(n, 2)
        self.assertAlmostEqual(epsilon, 19.5)
        p95 = table.loc[table["quantile"] == 0.95, "distance_km"].iloc[0]
        self.assertAlmostEqual(float(p95), 19.5)

    def test_original_group_isolation_flags_only_non_singletons(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_ids = ["a1", "a2", "a3", "a4", "b1", "b2", "c1"]
            latents = np.asarray(
                [
                    [0.0, 0.0],
                    [10.0, 0.0],
                    [10.1, 0.0],
                    [10.2, 0.0],
                    [0.1, 0.0],
                    [0.2, 0.0],
                    [0.3, 0.0],
                ],
                dtype=np.float32,
            )
            write_latents_csv(root / "latents.csv", sample_ids, latents)
            write_tsv(
                root / "sample_stats.tsv",
                [
                    {"sample_id": "a1", "observed_fraction": 0.5, "country": "A", "original_group_id": "A", "date_mean_bp": 1000},
                    {"sample_id": "a2", "observed_fraction": 0.5, "country": "A", "original_group_id": "A", "date_mean_bp": 1000},
                    {"sample_id": "a3", "observed_fraction": 0.5, "country": "A", "original_group_id": "A", "date_mean_bp": 1000},
                    {"sample_id": "a4", "observed_fraction": 0.5, "country": "A", "original_group_id": "A", "date_mean_bp": 1000},
                    {"sample_id": "b1", "observed_fraction": 0.5, "country": "B", "original_group_id": "B", "date_mean_bp": 1000},
                    {"sample_id": "b2", "observed_fraction": 0.5, "country": "B", "original_group_id": "B", "date_mean_bp": 1000},
                    {"sample_id": "c1", "observed_fraction": 0.5, "country": "C", "original_group_id": "C", "date_mean_bp": 1000},
                ],
                ["sample_id", "observed_fraction", "country", "original_group_id", "date_mean_bp"],
            )
            write_anno(
                root / "data.anno",
                [{"Genetic ID": sid, "Lat.": 0.0, "Long.": 0.0} for sid in sample_ids],
            )
            scores, _, _ = score_migrant_outliers(
                latents_csv=root / "latents.csv",
                sample_stats_tsv=root / "sample_stats.tsv",
                anno_path=root / "data.anno",
                k=3,
                epsilon_km=1000.0,
            )
            by_id = scores.set_index("sample_id")
            self.assertTrue(bool(by_id.loc["a1", "flag_original_group_isolation"]))
            self.assertFalse(bool(by_id.loc["a2", "flag_original_group_isolation"]))
            self.assertFalse(bool(by_id.loc["a3", "flag_original_group_isolation"]))
            self.assertFalse(bool(by_id.loc["a4", "flag_original_group_isolation"]))
            self.assertFalse(bool(by_id.loc["c1", "flag_original_group_isolation"]))

    def test_geo_displacement_skips_missing_sample_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_ids = ["s1", "s2", "s3"]
            latents = np.asarray([[0.0], [0.1], [0.2]], dtype=np.float32)
            write_latents_csv(root / "latents.csv", sample_ids, latents)
            write_tsv(
                root / "sample_stats.tsv",
                [
                    {"sample_id": sid, "observed_fraction": 0.5, "country": "A", "original_group_id": "A", "date_mean_bp": 1000}
                    for sid in sample_ids
                ],
                ["sample_id", "observed_fraction", "country", "original_group_id", "date_mean_bp"],
            )
            write_anno(
                root / "data.anno",
                [
                    {"Genetic ID": "s1", "Lat.": "..", "Long.": ".."},
                    {"Genetic ID": "s2", "Lat.": 0.0, "Long.": 0.0},
                    {"Genetic ID": "s3", "Lat.": 0.0, "Long.": 1.0},
                ],
            )
            scores, _, _ = score_migrant_outliers(
                latents_csv=root / "latents.csv",
                sample_stats_tsv=root / "sample_stats.tsv",
                anno_path=root / "data.anno",
                k=1,
                epsilon_km=1.0,
            )
            row = scores.set_index("sample_id").loc["s1"]
            self.assertTrue(np.isnan(float(row["knn_geo_distance_km"])))
            self.assertFalse(bool(row["flag_geo_displacement"]))

    def test_geo_candidate_filters_drop_group_prefix_and_far_rank1(self) -> None:
        rows = [
            {
                "sample_id": "prefix",
                "country": "A",
                "original_group_id": "A_1",
                "original_group_size": 3,
                "same_original_group_neighbors": 2,
                "knn_geo_distance_km": 2000.0,
                "adaptive_original_group_purity": 1.0,
                "flag_original_group_isolation": False,
                "flag_geo_displacement": True,
                "neighbor_sample_ids": "p1;p2;p3",
                "neighbor_original_group_ids": "A_1;A_1;B_1",
                "neighbor_countries": "A;A;B",
                "neighbor_latent_distances": "0.1;0.2;0.3",
            },
            {
                "sample_id": "far",
                "country": "B",
                "original_group_id": "B_1",
                "original_group_size": 1,
                "same_original_group_neighbors": 0,
                "knn_geo_distance_km": 2100.0,
                "adaptive_original_group_purity": 0.0,
                "flag_original_group_isolation": False,
                "flag_geo_displacement": True,
                "neighbor_sample_ids": "f1;f2;f3",
                "neighbor_original_group_ids": "C_1;D_1;E_1",
                "neighbor_countries": "C;D;E",
                "neighbor_latent_distances": "0.61;0.7;0.8",
            },
            {
                "sample_id": "keep",
                "country": "C",
                "original_group_id": "C_1",
                "original_group_size": 1,
                "same_original_group_neighbors": 0,
                "knn_geo_distance_km": 2200.0,
                "adaptive_original_group_purity": 0.0,
                "flag_original_group_isolation": False,
                "flag_geo_displacement": True,
                "neighbor_sample_ids": "k1;k2;k3",
                "neighbor_original_group_ids": "D_1;E_1;F_1",
                "neighbor_countries": "D;E;F",
                "neighbor_latent_distances": "0.6;0.7;0.8",
            },
        ]
        import pandas as pd

        filtered = add_geo_candidate_filters(pd.DataFrame(rows))
        by_id = filtered.set_index("sample_id")
        self.assertTrue(bool(by_id.loc["prefix", "drop_original_group_prefix"]))
        self.assertTrue(bool(by_id.loc["far", "drop_rank1_neighbor_distance"]))
        self.assertTrue(bool(by_id.loc["keep", "flag_geo_candidate_filtered"]))

        expanded = expand_candidate_knn_rows(filtered, candidate_flag="flag_geo_candidate_filtered")
        self.assertEqual(expanded["sample_id"].unique().tolist(), ["keep"])
        self.assertEqual(expanded["neighbor_rank"].tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
