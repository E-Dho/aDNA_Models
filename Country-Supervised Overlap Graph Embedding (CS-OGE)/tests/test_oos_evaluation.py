from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from overlap_embed.data import write_tsv
from overlap_embed.eval import write_latents_csv
from overlap_embed.oos_encoder import evaluate_predictions


class OosEvaluationTests(unittest.TestCase):
    def test_evaluation_metrics_are_finite_on_toy_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_ids = [f"s{i}" for i in range(6)]
            target = np.asarray(
                [[-1.0, 0.0], [-0.8, 0.1], [-1.2, -0.1], [1.0, 0.0], [0.8, -0.1], [1.2, 0.1]],
                dtype=np.float32,
            )
            predicted = target + 0.01
            coverage = np.asarray([0.1, 0.2, 0.15, 0.7, 0.8, 0.75], dtype=np.float32)
            stats_rows = []
            for i, sample_id in enumerate(sample_ids):
                stats_rows.append(
                    {
                        "sample_id": sample_id,
                        "observed_fraction": coverage[i],
                        "country": "A" if i < 3 else "B",
                        "original_group_id": f"{'A' if i < 3 else 'B'}1",
                        "date_mean_bp": 1000 + i * 100,
                    }
                )
            stats_tsv = root / "sample_stats.tsv"
            write_tsv(stats_tsv, stats_rows, ["sample_id", "observed_fraction", "country", "original_group_id", "date_mean_bp"])
            write_latents_csv(root / "pred.csv", sample_ids, predicted)
            write_latents_csv(root / "target.csv", sample_ids, target)

            metrics = evaluate_predictions(
                predicted=predicted,
                target=target,
                sample_ids=sample_ids,
                sample_stats_tsv=stats_tsv,
                coverage=coverage,
                k=2,
            )
            self.assertLess(metrics["latent_mse"], 0.001)
            self.assertGreater(metrics["mean_cosine_similarity"], 0.99)
            self.assertIn("knn_overlap@2", metrics)
            self.assertIn("pred_country_macro_knn_purity@15", metrics)


if __name__ == "__main__":
    unittest.main()
