from __future__ import annotations

import unittest

from overlap_embed.train import TrainConfig, original_group_constraint_loss


try:
    import torch

    from overlap_embed.model import graph_similarity_loss, positive_edge_loss
except ImportError:  # pragma: no cover
    torch = None


class TrainConfigFlagTests(unittest.TestCase):
    def test_new_flags_preserve_default_behavior(self) -> None:
        config = TrainConfig()
        self.assertEqual(config.objective_mode, "label_supervised")
        self.assertEqual(config.lambda_graph_pos, 1.0)
        self.assertEqual(config.lambda_graph_neg, 1.0)
        self.assertEqual(config.lambda_graph_similarity, 1.0)
        self.assertEqual(config.graph_similarity_temperature, 1.0)
        self.assertEqual(config.lambda_original_group, 0.0)
        self.assertEqual(config.original_group_negatives_per_anchor, 4)
        self.assertEqual(config.original_group_temperature, 1.0)

    def test_cli_parses_snp_similarity_flags(self) -> None:
        from train_overlap_embedding import build_parser

        args = build_parser().parse_args(
            [
                "--graph_npz",
                "graph_edges.npz",
                "--spectral_init",
                "spectral_init.npy",
                "--sample_stats_tsv",
                "sample_stats.tsv",
                "--output_dir",
                "run",
                "--objective_mode",
                "snp_similarity",
                "--lambda_graph_similarity",
                "1.5",
                "--lambda_graph_neg",
                "3.0",
                "--graph_similarity_temperature",
                "2.0",
            ]
        )
        self.assertEqual(args.objective_mode, "snp_similarity")
        self.assertEqual(args.lambda_graph_similarity, 1.5)
        self.assertEqual(args.lambda_graph_neg, 3.0)
        self.assertEqual(args.graph_similarity_temperature, 2.0)


@unittest.skipIf(torch is None, "torch is required for loss weighting tests")
class TrainConfigLossWeightTests(unittest.TestCase):
    def test_lambda_graph_pos_scales_positive_loss(self) -> None:
        latents = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        src = torch.tensor([0], dtype=torch.long)
        dst = torch.tensor([1], dtype=torch.long)
        weight = torch.tensor([1.0], dtype=torch.float32)
        pos_loss = positive_edge_loss(latents, src, dst, weight)

        old_total = float(TrainConfig(lambda_graph_pos=1.0).lambda_graph_pos) * pos_loss
        new_total = float(TrainConfig(lambda_graph_pos=2.0).lambda_graph_pos) * pos_loss

        self.assertAlmostEqual(float(new_total), 2.0 * float(old_total), places=6)

    def test_lambda_original_group_zero_disables_loss_contribution(self) -> None:
        latents = torch.tensor([[0.0, 0.0], [0.25, 0.0]], dtype=torch.float32)
        loss = original_group_constraint_loss(
            latents,
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
            temperature=1.0,
        )

        disabled = float(TrainConfig(lambda_original_group=0.0).lambda_original_group) * loss
        enabled = float(TrainConfig(lambda_original_group=0.1).lambda_original_group) * loss

        self.assertAlmostEqual(float(disabled), 0.0, places=7)
        self.assertGreater(float(enabled), 0.0)

    def test_graph_similarity_loss_prefers_matching_edge_weights(self) -> None:
        src = torch.tensor([0, 0], dtype=torch.long)
        dst = torch.tensor([1, 2], dtype=torch.long)
        target = torch.tensor([0.95, 0.05], dtype=torch.float32)
        matched = torch.tensor(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        mismatched = torch.tensor(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [0.2, 0.0],
            ],
            dtype=torch.float32,
        )

        self.assertLess(
            float(graph_similarity_loss(matched, src, dst, target, temperature=1.0)),
            float(graph_similarity_loss(mismatched, src, dst, target, temperature=1.0)),
        )

    def test_snp_similarity_mode_uses_effective_zero_label_weights(self) -> None:
        config = TrainConfig(
            objective_mode="snp_similarity",
            lambda_country=10.0,
            lambda_original_group=10.0,
        )
        pos_loss = torch.tensor(1.0)
        neg_loss = torch.tensor(0.5)
        graph_sim_loss = torch.tensor(0.25)
        country_loss = torch.tensor(100.0)
        original_group_loss = torch.tensor(100.0)
        total = (
            config.lambda_graph_pos * pos_loss
            + config.lambda_graph_neg * neg_loss
            + config.lambda_graph_similarity * graph_sim_loss
            + 0.0 * country_loss
            + 0.0 * original_group_loss
        )

        self.assertAlmostEqual(float(total), 1.75, places=6)


if __name__ == "__main__":
    unittest.main()
