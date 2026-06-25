from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from overlap_embed.data import build_memmap_from_eigenstrat, write_tsv
from overlap_embed.eval import read_latents_csv, write_latents_csv
from overlap_embed.oos_encoder import (
    build_encoder_model,
    encode_aligned_memmap,
    load_encoder,
    read_snp_records,
    train_oos_encoder,
    write_snp_manifest,
)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_torch_available(), "torch is required for OOS encoder tests")
class OosEncoderTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        geno = root / "toy.geno"
        ind = root / "toy.ind"
        snp = root / "toy.snp"
        geno.write_text(
            "\n".join(
                [
                    "00111199",
                    "01101109",
                    "00011101",
                    "10101010",
                    "99900111",
                    "01010101",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        ind.write_text("\n".join([f"s{i} U pop{i % 2}" for i in range(8)]) + "\n", encoding="utf-8")
        snp.write_text(
            "\n".join([f"rs{i} 1 0.0 {100 + i} A G" for i in range(6)]) + "\n",
            encoding="utf-8",
        )
        meta = build_memmap_from_eigenstrat(geno, root / "toy", "pseudohaploid", ind)
        manifest = root / "snp_manifest.tsv"
        write_snp_manifest(manifest, read_snp_records(snp))
        sample_ids = [f"s{i}" for i in range(8)]
        latents = np.asarray(
            [[-1.0, -0.5], [-0.8, -0.4], [-1.1, -0.6], [-0.9, -0.3], [1.0, 0.5], [0.9, 0.4], [1.1, 0.6], [0.8, 0.3]],
            dtype=np.float32,
        )
        latents_csv = root / "teacher_latents.csv"
        write_latents_csv(latents_csv, sample_ids, latents)
        stats_rows = []
        for i, sample_id in enumerate(sample_ids):
            stats_rows.append(
                {
                    "sample_id": sample_id,
                    "observed_fraction": 0.5 + 0.02 * i,
                    "country": "A" if i < 4 else "B",
                    "original_group_id": f"{'A' if i < 4 else 'B'}{i % 2}",
                    "date_mean_bp": 1000 + 100 * i,
                }
            )
        stats_tsv = root / "sample_stats.tsv"
        write_tsv(stats_tsv, stats_rows, ["sample_id", "observed_fraction", "country", "original_group_id", "date_mean_bp"])
        return {
            "meta_json": root / "toy.meta.json",
            "manifest": manifest,
            "latents_csv": latents_csv,
            "stats_tsv": stats_tsv,
        }

    def test_train_artifacts_and_label_free_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._fixture(root)
            out_dir = root / "encoder"
            summary = train_oos_encoder(
                meta_json=paths["meta_json"],
                snp_manifest=paths["manifest"],
                teacher_latents_csv=paths["latents_csv"],
                sample_stats_tsv=paths["stats_tsv"],
                output_dir=out_dir,
                latent_dim=2,
                epochs=2,
                batch_size=4,
                hidden_dim=4,
                wandb_mode="disabled",
                seed=7,
            )
            self.assertTrue((out_dir / "encoder.pt").exists())
            self.assertTrue((out_dir / "encoder_config.json").exists())
            self.assertTrue((out_dir / "snp_manifest.tsv").exists())
            self.assertTrue((out_dir / "heldout_predictions.csv").exists())
            self.assertEqual(summary["training"]["n_epochs_completed"], 2)
            model, config, _ = load_encoder(out_dir)
            self.assertEqual(config["latent_dim"], 2)
            self.assertIsNotNone(model)

            report = encode_aligned_memmap(
                encoder_dir=out_dir,
                meta_json=paths["meta_json"],
                out_dir=root / "encoded",
                batch_size=4,
            )
            self.assertEqual(report["n_samples"], 8)
            encoded_ids, encoded_latents = read_latents_csv(root / "encoded" / "encoded_latents.csv")
            self.assertEqual(encoded_ids[0], "s0")
            self.assertEqual(encoded_latents.shape, (8, 2))
            self.assertTrue((root / "encoded" / "encoded_sample_stats.tsv").exists())

    def test_mismatched_latent_dimension_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._fixture(root)
            with self.assertRaises(ValueError):
                train_oos_encoder(
                    meta_json=paths["meta_json"],
                    snp_manifest=paths["manifest"],
                    teacher_latents_csv=paths["latents_csv"],
                    sample_stats_tsv=paths["stats_tsv"],
                    output_dir=root / "bad_encoder",
                    latent_dim=3,
                    epochs=1,
                    batch_size=4,
                    hidden_dim=4,
                    wandb_mode="disabled",
                )

    def test_sparse_deep_encoder_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = self._fixture(root)
            out_dir = root / "deep_encoder"
            summary = train_oos_encoder(
                meta_json=paths["meta_json"],
                snp_manifest=paths["manifest"],
                teacher_latents_csv=paths["latents_csv"],
                sample_stats_tsv=paths["stats_tsv"],
                output_dir=out_dir,
                latent_dim=2,
                epochs=1,
                batch_size=4,
                model_type="sparse_deep_encoder",
                deep_hidden_dim=16,
                deep_bottleneck_dim=4,
                sparse_fanout=2,
                wandb_mode="disabled",
                seed=11,
            )
            self.assertTrue((out_dir / "encoder.pt").exists())
            self.assertEqual(summary["config"]["model_type"], "sparse_deep_encoder")
            self.assertEqual(summary["config"]["deep_hidden_dim"], 16)
            _, config, _ = load_encoder(out_dir)
            self.assertEqual(config["deep_bottleneck_dim"], 4)

    def test_sparse_fan_in_initialization_scales_by_bucket_fan_in(self) -> None:
        import torch

        torch.manual_seed(3)
        model = build_encoder_model(
            24,
            2,
            4,
            model_type="sparse_deep_encoder",
            deep_hidden_dim=7,
            deep_bottleneck_dim=3,
            sparse_fanout=2,
            sparse_init="fan_in",
        )
        counts = torch.bincount(model.bucket_idx.reshape(-1), minlength=7).to(torch.float32)
        expected = torch.sqrt(2.0 / (2.0 * counts).clamp_min(1.0)[model.bucket_idx])
        torch.testing.assert_close(model.connection_init_scale, expected)
        self.assertGreater(float(model.connection_init_scale.max()), float(model.connection_init_scale.min()))

    def test_dynamic_sparse_update_preserves_count_and_prevents_duplicates(self) -> None:
        import torch

        torch.manual_seed(5)
        model = build_encoder_model(
            12,
            2,
            4,
            model_type="sparse_deep_encoder",
            deep_hidden_dim=9,
            deep_bottleneck_dim=3,
            sparse_fanout=2,
            sparse_init="fan_in",
            dst_enable=True,
            dst_candidate_multiplier=4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        values = torch.randint(0, 2, (5, 12), dtype=torch.float32)
        mask = torch.ones_like(values)
        pred = model(values, mask)
        loss = pred.pow(2).mean()
        loss.backward()
        optimizer.step()
        before_slots = int(model.bucket_idx.numel())
        result = model.dynamic_sparse_update(
            values=values,
            mask=mask,
            optimizer=optimizer,
            drop_fraction=0.25,
            candidate_multiplier=4,
        )
        self.assertEqual(int(model.bucket_idx.numel()), before_slots)
        self.assertEqual(result["pruned"], result["grown"])
        for snp_idx in range(model.bucket_idx.shape[1]):
            buckets = model.bucket_idx[:, snp_idx].tolist()
            self.assertEqual(len(buckets), len(set(buckets)))

    def test_growth_candidate_scoring_prefers_highest_gradient_bucket(self) -> None:
        import torch

        model = build_encoder_model(
            4,
            2,
            4,
            model_type="sparse_deep_encoder",
            deep_hidden_dim=5,
            deep_bottleneck_dim=3,
            sparse_fanout=1,
            sparse_init="fan_in",
        )
        values = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        mask = torch.ones_like(values)
        model._last_hidden_grad = torch.tensor(
            [
                [0.0, 0.1, 3.0, 0.2, 0.0],
                [0.0, 0.1, 2.0, 0.2, 0.0],
            ]
        )
        snp_idx = torch.tensor([0])
        candidates = torch.tensor([[1, 2, 3]])
        scores = model._score_growth_candidates(values, mask, snp_idx, candidates)
        self.assertEqual(int(scores.argmax(dim=1)[0]), 1)


if __name__ == "__main__":
    unittest.main()
