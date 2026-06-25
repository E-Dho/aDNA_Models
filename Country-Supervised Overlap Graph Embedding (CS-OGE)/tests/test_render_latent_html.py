from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from render_latent_html import _parse_axes, _project_axes, _read_latents


class RenderLatentHtmlTests(unittest.TestCase):
    def test_parse_axes_accepts_two_or_three_axes(self) -> None:
        self.assertEqual(_parse_axes("1,3"), [1, 3])
        self.assertEqual(_parse_axes("1, 2, 4"), [1, 2, 4])

    def test_project_axes_direct_and_pca(self) -> None:
        latents = np.asarray(
            [
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 3.0],
                [1.0, 1.0, 4.0],
            ],
            dtype=np.float32,
        )
        coords_direct, labels_direct, explained_direct = _project_axes(latents, [1, 3], "direct")
        np.testing.assert_allclose(coords_direct, latents[:, [0, 2]])
        self.assertEqual(labels_direct, ["z1", "z3"])
        self.assertIsNone(explained_direct)

        coords_pca, labels_pca, explained_pca = _project_axes(latents, [1, 2], "pca")
        self.assertEqual(coords_pca.shape, (3, 2))
        self.assertEqual(len(labels_pca), 2)
        self.assertEqual(len(explained_pca), 2)
        self.assertTrue(all(value >= 0.0 for value in explained_pca))

    def test_read_latents_npy_uses_sample_stats_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latents_npy = root / "spectral_init.npy"
            np.save(latents_npy, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
            stats = pd.DataFrame(
                [
                    {"sample_id": "s1", "country": "A", "original_group_id": "g1", "date_mean_bp": 1000.0},
                    {"sample_id": "s2", "country": "B", "original_group_id": "g2", "date_mean_bp": 2000.0},
                ]
            )
            sample_ids, latents = _read_latents(None, latents_npy, stats)
            self.assertEqual(sample_ids, ["s1", "s2"])
            np.testing.assert_allclose(latents, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    def test_excluding_samples_filters_frame(self) -> None:
        frame = pd.DataFrame(
            [
                {"sample_id": "s1", "country": "A", "original_group_id": "g1"},
                {"sample_id": "s2", "country": "B", "original_group_id": "g2"},
                {"sample_id": "s3", "country": "C", "original_group_id": "g3"},
            ]
        )
        excluded = {"s2", "s3"}
        filtered = frame[~frame["sample_id"].astype(str).isin(excluded)].copy()
        self.assertEqual(filtered["sample_id"].tolist(), ["s1"])


if __name__ == "__main__":
    unittest.main()
