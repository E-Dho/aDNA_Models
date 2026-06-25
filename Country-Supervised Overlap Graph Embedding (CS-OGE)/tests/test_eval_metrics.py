from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from overlap_embed.data import write_tsv
from overlap_embed.eval import (
    adaptive_group_knn_purity,
    evaluate_embedding,
    macro_knn_purity,
    nearest_neighbors,
)


class EvalMetricTests(unittest.TestCase):
    def test_macro_knn_purity(self) -> None:
        latents = np.asarray(
            [
                [-1.0, 0.0],
                [-0.8, 0.0],
                [0.8, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
        neighbors = nearest_neighbors(latents, 1)
        purity = macro_knn_purity(["A", "A", "B", "B"], neighbors)
        self.assertAlmostEqual(purity, 1.0)

    def test_macro_knn_purity_can_exclude_singletons(self) -> None:
        labels = ["A", "A", "B"]
        neighbors = np.asarray([[1], [0], [0]], dtype=np.int64)
        self.assertAlmostEqual(macro_knn_purity(labels, neighbors), 0.5)
        self.assertAlmostEqual(macro_knn_purity(labels, neighbors, min_group_size=2), 1.0)

    def test_adaptive_group_knn_purity_trims_farther_non_group_neighbors(self) -> None:
        labels = ["A", "A", "A", "B", "B"]
        neighbors = np.asarray(
            [
                [1, 2, 3, 4],
                [0, 2, 3, 4],
                [0, 1, 3, 4],
                [4, 0, 1, 2],
                [3, 0, 1, 2],
            ],
            dtype=np.int64,
        )
        neighbor_dist2 = np.asarray(
            [
                [1.0, 2.0, 5.0, 6.0],
                [1.0, 2.0, 5.0, 6.0],
                [1.0, 2.0, 5.0, 6.0],
                [1.0, 3.0, 4.0, 5.0],
                [1.0, 3.0, 4.0, 5.0],
            ],
            dtype=np.float64,
        )
        purity = adaptive_group_knn_purity(labels, neighbors, neighbor_dist2)
        self.assertAlmostEqual(purity, 1.0)

    def test_adaptive_group_knn_purity_keeps_closer_non_group_neighbors(self) -> None:
        labels = ["A", "A", "A", "B", "B"]
        neighbors = np.asarray(
            [
                [1, 3, 2, 4],
                [0, 3, 2, 4],
                [0, 3, 1, 4],
                [4, 0, 1, 2],
                [3, 0, 1, 2],
            ],
            dtype=np.int64,
        )
        neighbor_dist2 = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            dtype=np.float64,
        )
        purity = adaptive_group_knn_purity(labels, neighbors, neighbor_dist2)
        self.assertAlmostEqual(purity, 5.0 / 6.0)

    def test_adaptive_group_knn_purity_matches_standard_when_group_exceeds_k(self) -> None:
        labels = ["A", "A", "A", "A", "B", "B", "B", "B"]
        neighbors = np.asarray(
            [
                [1, 4],
                [0, 4],
                [3, 4],
                [2, 4],
                [5, 0],
                [4, 0],
                [7, 0],
                [6, 0],
            ],
            dtype=np.int64,
        )
        neighbor_dist2 = np.asarray(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
            ],
            dtype=np.float64,
        )
        standard = macro_knn_purity(labels, neighbors)
        adaptive = adaptive_group_knn_purity(labels, neighbors, neighbor_dist2)
        self.assertAlmostEqual(adaptive, standard)

    def test_adaptive_group_knn_purity_can_exclude_singletons(self) -> None:
        labels = ["A", "A", "B"]
        neighbors = np.asarray([[1], [0], [0]], dtype=np.int64)
        neighbor_dist2 = np.asarray([[1.0], [1.0], [2.0]], dtype=np.float64)
        self.assertAlmostEqual(adaptive_group_knn_purity(labels, neighbors, neighbor_dist2), 0.5)
        self.assertAlmostEqual(
            adaptive_group_knn_purity(labels, neighbors, neighbor_dist2, min_group_size=2),
            1.0,
        )

    def test_evaluate_embedding_outputs_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_stats = root / "sample_stats.tsv"
            write_tsv(
                sample_stats,
                [
                    {"sample_id": "s1", "observed_fraction": 0.2, "country": "A", "original_group_id": "A_1", "date_mean_bp": 1000.0},
                    {"sample_id": "s2", "observed_fraction": 0.21, "country": "A", "original_group_id": "A_1", "date_mean_bp": 980.0},
                    {"sample_id": "s3", "observed_fraction": 0.8, "country": "B", "original_group_id": "B_1", "date_mean_bp": 2000.0},
                    {"sample_id": "s4", "observed_fraction": 0.79, "country": "B", "original_group_id": "B_1", "date_mean_bp": 2020.0},
                ],
                fieldnames=["sample_id", "observed_fraction", "country", "original_group_id", "date_mean_bp"],
            )
            latents = np.asarray(
                [
                    [-1.0, 0.0],
                    [-0.9, 0.0],
                    [0.9, 0.0],
                    [1.0, 0.0],
                ],
                dtype=np.float32,
            )
            metrics, corr_rows, neighbors, merged = evaluate_embedding(latents, sample_stats, k=1)
            self.assertIn("coverage_r2", metrics)
            self.assertIn("original_group_adaptive_knn_purity@15", metrics)
            self.assertIn("original_group_knn_purity@15_non_singleton", metrics)
            self.assertIn("original_group_adaptive_knn_purity@15_non_singleton", metrics)
            self.assertEqual(len(corr_rows), 2)
            self.assertEqual(neighbors.shape, (4, 1))
            self.assertEqual(len(merged), 4)


if __name__ == "__main__":
    unittest.main()
