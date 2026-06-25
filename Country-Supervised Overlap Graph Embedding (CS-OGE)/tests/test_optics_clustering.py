from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from overlap_embed.data import write_tsv
from overlap_embed.eval import write_latents_csv
from overlap_embed.optics import (
    OpticsConfig,
    OpticsRun,
    assignments_frame,
    load_latents_with_stats,
    run_optics,
    run_optics_sweep,
    summarize_clusters,
    sweep_summary,
)

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class OpticsClusteringTests(unittest.TestCase):
    def test_load_latents_with_stats_aligns_sample_ids_and_latent_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_latents_csv(
                root / "latents.csv",
                ["s2", "s1"],
                np.asarray([[2.0, 20.0], [1.0, 10.0]], dtype=np.float32),
            )
            write_tsv(
                root / "stats.tsv",
                [
                    {"sample_id": "s1", "country": "A", "original_group_id": "A1", "date_mean_bp": 1, "observed_fraction": 0.1},
                    {"sample_id": "s2", "country": "B", "original_group_id": "B1", "date_mean_bp": 2, "observed_fraction": 0.2},
                ],
                ["sample_id", "country", "original_group_id", "date_mean_bp", "observed_fraction"],
            )
            frame, values, cols = load_latents_with_stats(root / "latents.csv", root / "stats.tsv")
            self.assertEqual(frame["sample_id"].tolist(), ["s2", "s1"])
            self.assertEqual(cols, ["z1", "z2"])
            self.assertEqual(values.shape, (2, 2))

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn unavailable")
    def test_optics_finds_clusters_and_noise_on_toy_data(self) -> None:
        rng = np.random.default_rng(3)
        values = np.vstack(
            [
                rng.normal(loc=(-2.0, 0.0), scale=0.05, size=(30, 2)),
                rng.normal(loc=(2.0, 0.0), scale=0.05, size=(30, 2)),
                np.asarray([[0.0, 3.0], [0.0, -3.0]]),
            ]
        )
        run = run_optics(values, OpticsConfig(min_samples=5, min_cluster_size=5, xi=0.05))
        self.assertGreaterEqual(int(run.diagnostics["n_clusters"]), 2)
        self.assertGreaterEqual(int(run.diagnostics["n_noise"]), 1)

    def test_sweep_summary_marks_chosen_and_cluster_summary_uses_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c", "d"],
                "z1": [0.0, 0.1, 1.0, 1.1],
                "z2": [0.0, 0.1, 1.0, 1.1],
                "country": ["A", "A", "B", "B"],
                "original_group_id": ["A1", "A1", "B1", "B2"],
                "date_mean_bp": [1000, 1010, 2000, 2010],
                "observed_fraction": [0.1, 0.2, 0.3, 0.4],
            }
        )
        config = OpticsConfig()
        run = OpticsRun(
            config=config,
            labels=np.asarray([0, 0, 1, -1]),
            reachability=np.asarray([0.1, 0.1, 0.2, 0.3]),
            core_distances=np.asarray([0.1, 0.1, 0.2, 0.3]),
            ordering=np.asarray([0, 1, 2, 3]),
            diagnostics={"selection_score": 1.0, "degenerate": False, "n_clusters": 2},
        )
        assignments = assignments_frame(frame, ["z1", "z2"], run)
        summary = summarize_clusters(assignments)
        sweep = sweep_summary([run], run)
        self.assertEqual(summary.loc[summary["cluster_id"] == 0, "country_top5"].iloc[0], "A:2")
        self.assertTrue(bool(sweep["chosen"].iloc[0]))

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn unavailable")
    def test_sweep_ranking_avoids_degenerate_run(self) -> None:
        rng = np.random.default_rng(9)
        values = np.vstack(
            [
                rng.normal(loc=(-1.0, 0.0), scale=0.04, size=(25, 2)),
                rng.normal(loc=(1.0, 0.0), scale=0.04, size=(25, 2)),
            ]
        )
        chosen, runs = run_optics_sweep(
            values,
            [
                OpticsConfig(min_samples=45, min_cluster_size=45, xi=0.05),
                OpticsConfig(min_samples=5, min_cluster_size=5, xi=0.05),
            ],
        )
        self.assertEqual(len(runs), 2)
        self.assertFalse(bool(chosen.diagnostics["degenerate"]))


if __name__ == "__main__":
    unittest.main()
