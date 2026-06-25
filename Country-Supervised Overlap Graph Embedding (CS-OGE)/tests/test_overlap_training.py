from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from overlap_embed.data import write_tsv
from overlap_embed.train import TrainConfig, train_overlap_embedding


try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is required for training tests")
class OverlapTrainingTests(unittest.TestCase):
    def test_training_produces_summary_and_latents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_npz = root / "graph_edges.npz"
            spectral_init = root / "spectral_init.npy"
            sample_stats = root / "sample_stats.tsv"
            out_dir = root / "run"

            src = np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32)
            dst = np.asarray([1, 0, 3, 2, 2, 0], dtype=np.int32)
            weight = np.asarray([1.0, 1.0, 1.0, 1.0, 0.2, 0.2], dtype=np.float32)
            overlap = np.asarray([100, 100, 100, 100, 90, 90], dtype=np.int32)
            np.savez_compressed(graph_npz, src=src, dst=dst, weight=weight, overlap=overlap)
            np.save(
                spectral_init,
                np.asarray(
                    [
                        [-1.0, 0.0],
                        [-0.8, 0.0],
                        [0.8, 0.0],
                        [1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
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
            config = TrainConfig(
                latent_dim=2,
                epochs=20,
                batch_size=4,
                patience=5,
                lambda_country=0.5,
                country_negatives_per_anchor=2,
                wandb=False,
            )
            summary = train_overlap_embedding(
                graph_npz=graph_npz,
                spectral_init=spectral_init,
                sample_stats_tsv=sample_stats,
                output_dir=out_dir,
                config=config,
            )
            self.assertTrue((out_dir / "run_summary.json").exists())
            self.assertTrue((out_dir / "final_latents.csv").exists())
            self.assertIn("evaluation", summary)
            self.assertTrue(np.isfinite(summary["evaluation"]["coverage_r2"]))
            saved = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["config"]["lambda_country"], 0.5)
            self.assertIn("spectral_init_evaluation", saved)

    def test_global_original_group_supervision_can_cross_country_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_npz = root / "graph_edges.npz"
            spectral_init = root / "spectral_init.npy"
            sample_stats = root / "sample_stats.tsv"
            out_dir = root / "run"

            src = np.asarray([0, 1, 2, 3, 0, 2, 1, 3], dtype=np.int32)
            dst = np.asarray([1, 0, 3, 2, 2, 0, 3, 1], dtype=np.int32)
            weight = np.asarray([1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
            overlap = np.asarray([100, 100, 100, 100, 95, 95, 95, 95], dtype=np.int32)
            np.savez_compressed(graph_npz, src=src, dst=dst, weight=weight, overlap=overlap)
            np.save(
                spectral_init,
                np.asarray(
                    [
                        [-1.0, 0.0],
                        [-0.8, 0.0],
                        [0.8, 0.0],
                        [1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
            write_tsv(
                sample_stats,
                [
                    {"sample_id": "s1", "observed_fraction": 0.20, "country": "A", "original_group_id": "g1", "date_mean_bp": 1000.0},
                    {"sample_id": "s2", "observed_fraction": 0.21, "country": "B", "original_group_id": "g1", "date_mean_bp": 1020.0},
                    {"sample_id": "s3", "observed_fraction": 0.80, "country": "A", "original_group_id": "g2", "date_mean_bp": 2000.0},
                    {"sample_id": "s4", "observed_fraction": 0.79, "country": "B", "original_group_id": "g2", "date_mean_bp": 1980.0},
                ],
                fieldnames=["sample_id", "observed_fraction", "country", "original_group_id", "date_mean_bp"],
            )
            config = TrainConfig(
                latent_dim=2,
                epochs=25,
                batch_size=4,
                patience=6,
                lambda_country=0.0,
                lambda_original_group=0.2,
                original_group_negatives_per_anchor=1,
                wandb=False,
            )
            summary = train_overlap_embedding(
                graph_npz=graph_npz,
                spectral_init=spectral_init,
                sample_stats_tsv=sample_stats,
                output_dir=out_dir,
                config=config,
            )
            saved = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["config"]["lambda_country"], 0.0)
            self.assertEqual(saved["config"]["lambda_original_group"], 0.2)
            self.assertGreater(saved["evaluation"]["original_group_knn_purity@15"], 0.0)
            self.assertTrue(np.isfinite(summary["evaluation"]["coverage_r2"]))

    def test_snp_similarity_mode_ignores_conflicting_label_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_npz = root / "graph_edges.npz"
            spectral_init = root / "spectral_init.npy"
            sample_stats = root / "sample_stats.tsv"
            out_dir = root / "run"

            src = np.asarray([0, 1, 2, 3, 0, 2], dtype=np.int32)
            dst = np.asarray([1, 0, 3, 2, 2, 0], dtype=np.int32)
            weight = np.asarray([1.0, 1.0, 1.0, 1.0, 0.1, 0.1], dtype=np.float32)
            overlap = np.asarray([100, 100, 100, 100, 90, 90], dtype=np.int32)
            np.savez_compressed(graph_npz, src=src, dst=dst, weight=weight, overlap=overlap)
            np.save(
                spectral_init,
                np.asarray(
                    [
                        [-1.0, 0.0],
                        [-0.8, 0.0],
                        [0.8, 0.0],
                        [1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
            write_tsv(
                sample_stats,
                [
                    {"sample_id": "s1", "observed_fraction": 0.20, "country": "A", "original_group_id": "g1", "date_mean_bp": 1000.0},
                    {"sample_id": "s2", "observed_fraction": 0.21, "country": "B", "original_group_id": "g2", "date_mean_bp": 1020.0},
                    {"sample_id": "s3", "observed_fraction": 0.80, "country": "A", "original_group_id": "g1", "date_mean_bp": 2000.0},
                    {"sample_id": "s4", "observed_fraction": 0.79, "country": "B", "original_group_id": "g2", "date_mean_bp": 1980.0},
                ],
                fieldnames=["sample_id", "observed_fraction", "country", "original_group_id", "date_mean_bp"],
            )
            config = TrainConfig(
                objective_mode="snp_similarity",
                latent_dim=2,
                epochs=5,
                batch_size=4,
                patience=3,
                lambda_country=100.0,
                lambda_original_group=100.0,
                lambda_graph_similarity=0.5,
                wandb=False,
            )
            train_overlap_embedding(
                graph_npz=graph_npz,
                spectral_init=spectral_init,
                sample_stats_tsv=sample_stats,
                output_dir=out_dir,
                config=config,
            )
            saved = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
            first_metric = json.loads((out_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(saved["config"]["objective_mode"], "snp_similarity")
            self.assertEqual(first_metric["effective_lambda_country"], 0.0)
            self.assertEqual(first_metric["effective_lambda_original_group"], 0.0)
            self.assertIn("graph_knn_recall@15", saved["evaluation"])
            self.assertIn("weighted_graph_edge_spearman", saved["evaluation"])


if __name__ == "__main__":
    unittest.main()
