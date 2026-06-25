from __future__ import annotations

import unittest

import numpy as np

from overlap_embed.train import (
    OriginalGroupConstraintSampler,
    original_group_constraint_loss,
)


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class OriginalGroupSamplerTests(unittest.TestCase):
    def test_sampler_uses_global_group_positives_and_global_group_negatives(self) -> None:
        groups = ["g1", "g1", "g2", "g3", "g1", "g2", "g1", ""]
        coverage = np.asarray([0.10, 0.11, 0.12, 0.40, 0.15, 0.16, 0.20, 0.21], dtype=np.float32)
        sampler = OriginalGroupConstraintSampler(groups, coverage, seed=7)

        pos_src, pos_dst, neg_src, neg_dst = sampler.sample(
            np.asarray([0, 1, 2, 4, 6, 7], dtype=np.int64),
            negatives_per_anchor=2,
            window=8,
        )

        self.assertGreater(pos_src.size, 0)
        for src, dst in zip(pos_src.tolist(), pos_dst.tolist()):
            self.assertEqual(groups[src], groups[dst])
            self.assertNotEqual(src, dst)

        self.assertGreater(neg_src.size, 0)
        for src, dst in zip(neg_src.tolist(), neg_dst.tolist()):
            self.assertNotEqual(groups[src], groups[dst])
            self.assertTrue(groups[src])
            self.assertTrue(groups[dst])

        self.assertNotIn(7, pos_src.tolist())

    def test_sampler_returns_empty_arrays_when_no_valid_candidates_exist(self) -> None:
        sampler = OriginalGroupConstraintSampler(
            original_groups=["g1", "g2", "g3", ""],
            coverage=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            seed=11,
        )

        outputs = sampler.sample(np.arange(4, dtype=np.int64), negatives_per_anchor=2)
        for arr in outputs:
            self.assertEqual(arr.dtype, np.int64)
            self.assertEqual(arr.size, 0)

    def test_sampler_can_link_same_group_across_countries(self) -> None:
        groups = ["shared", "shared", "other", "other"]
        coverage = np.asarray([0.10, 0.12, 0.11, 0.13], dtype=np.float32)
        sampler = OriginalGroupConstraintSampler(groups, coverage, seed=13)

        pos_src, pos_dst, neg_src, neg_dst = sampler.sample(
            np.asarray([0, 1], dtype=np.int64),
            negatives_per_anchor=1,
            window=4,
        )

        self.assertEqual(pos_src.tolist(), [0, 1])
        self.assertEqual(sorted(pos_dst.tolist()), [0, 1])
        self.assertTrue(np.all(groups[src] != groups[dst] for src, dst in zip(neg_src.tolist(), neg_dst.tolist())))


@unittest.skipIf(torch is None, "torch is required for original-group loss tests")
class OriginalGroupLossTests(unittest.TestCase):
    def test_same_group_attraction_is_zero_for_equal_embeddings(self) -> None:
        latents = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
        loss = original_group_constraint_loss(
            latents,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            temperature=1.0,
        )
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_soft_repulsion_is_larger_when_different_groups_are_close(self) -> None:
        close = torch.tensor([[0.0, 0.0], [0.1, 0.0]], dtype=torch.float32)
        far = torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
        src = torch.tensor([0], dtype=torch.long)
        dst = torch.tensor([1], dtype=torch.long)
        empty = torch.tensor([], dtype=torch.long)

        close_loss = original_group_constraint_loss(close, empty, empty, src, dst, temperature=1.0)
        far_loss = original_group_constraint_loss(far, empty, empty, src, dst, temperature=1.0)

        self.assertGreater(float(close_loss), float(far_loss))
        self.assertGreater(float(far_loss), 0.0)


if __name__ == "__main__":
    unittest.main()
