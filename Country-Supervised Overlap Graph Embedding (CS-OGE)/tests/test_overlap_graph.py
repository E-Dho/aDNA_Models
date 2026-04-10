from __future__ import annotations

import unittest

import numpy as np

from overlap_embed.data import compute_observed_fraction
from overlap_embed.graph import (
    build_topk_symmetric_graph,
    similarity_from_components,
    spectral_initialization,
)


class OverlapGraphTests(unittest.TestCase):
    def test_similarity_and_topk_graph(self) -> None:
        numerator = np.asarray(
            [
                [0.0, 3.0, 1.0],
                [3.0, 0.0, 2.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        overlap = np.asarray(
            [
                [0, 100, 20],
                [100, 0, 80],
                [20, 80, 0],
            ],
            dtype=np.uint32,
        )
        sim = similarity_from_components(numerator, overlap, overlap_shrinkage=10.0)
        src, dst, weight, ov = build_topk_symmetric_graph(sim, overlap, min_overlap=50, top_k=1)
        self.assertEqual(src.tolist(), [0, 1, 1, 2])
        self.assertEqual(dst.tolist(), [1, 0, 2, 1])
        self.assertEqual(ov.tolist(), [100, 100, 80, 80])
        self.assertTrue(np.all(weight > 0))

    def test_small_graph_spectral_init_shape(self) -> None:
        src = np.asarray([0, 1, 1, 2], dtype=np.int32)
        dst = np.asarray([1, 0, 2, 1], dtype=np.int32)
        weight = np.asarray([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
        init = spectral_initialization(src, dst, weight, n_nodes=3, latent_dim=2)
        self.assertEqual(init.shape, (3, 2))
        self.assertTrue(np.isfinite(init).all())

    def test_compute_observed_fraction_supports_subsets(self) -> None:
        geno = np.asarray(
            [
                [0, -1, 1, -1],
                [0, 1, 1, 1],
                [-1, -1, 0, 0],
            ],
            dtype=np.int8,
        )
        frac = compute_observed_fraction(geno, sample_indices=np.asarray([0, 2]), max_snps=3)
        self.assertTrue(np.allclose(frac, np.asarray([2.0 / 3.0, 1.0 / 3.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
