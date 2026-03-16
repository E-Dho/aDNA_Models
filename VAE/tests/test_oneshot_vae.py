import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oneshot_vae.train import build_parser, compute_coverage_audit, run_oneshot_vae_training

if importlib.util.find_spec("torch") is not None:
    import torch

    from oneshot_vae.model import OneShotMaskedVAE, OneShotMaskedVAEConfig
    from oneshot_vae.train import create_training_masked_batch
else:  # pragma: no cover
    torch = None
    OneShotMaskedVAE = None
    OneShotMaskedVAEConfig = None
    create_training_masked_batch = None


def write_synthetic_meta(tmp: Path, n_samples: int = 10, n_snps: int = 33) -> Path:
    memmap_path = tmp / "toy.genotypes.int8.memmap"
    sample_ids_path = tmp / "toy.sample_ids.txt"
    mm = np.memmap(memmap_path, dtype=np.int8, mode="w+", shape=(n_samples, n_snps))
    rng = np.random.default_rng(11)
    data = rng.integers(0, 3, size=(n_samples, n_snps), dtype=np.int8)
    missing_mask = rng.random(size=(n_samples, n_snps)) < 0.25
    data[missing_mask] = -1
    data[-1, :] = -1
    mm[:, :] = data
    mm.flush()

    sample_ids_path.write_text("\n".join(f"sample_{i}" for i in range(n_samples)) + "\n", encoding="utf-8")
    meta_path = tmp / "toy.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "memmap_path": str(memmap_path),
                "n_samples": n_samples,
                "n_snps": n_snps,
                "dtype": "int8",
                "ploidy": "diploid",
                "n_classes": 3,
                "missing_value": -1,
                "sample_ids_path": str(sample_ids_path),
            }
        ),
        encoding="utf-8",
    )
    return meta_path


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch not installed")
class TestOneShotVAE(unittest.TestCase):
    def test_training_mask_never_targets_natural_missing(self):
        raw = torch.tensor(
            [
                [0, 1, -1, 2, -1, 0],
                [-1, 2, 1, -1, 0, 1],
            ],
            dtype=torch.long,
        )
        tokens, targets, valid_mask = create_training_masked_batch(
            raw=raw,
            mask_prob=0.5,
            missing_token_id=3,
            mask_token_id=4,
        )
        self.assertTrue(torch.all(~valid_mask[raw < 0]))
        self.assertTrue(torch.all(tokens[raw < 0] == 3))
        self.assertTrue(torch.all(targets[valid_mask] >= 0))

    def test_model_trims_padding(self):
        model = OneShotMaskedVAE(
            OneShotMaskedVAEConfig(
                n_snps=37,
                chunk_size=8,
                token_embed_dim=4,
                chunk_dim=32,
                encoder_heads=4,
                latent_dim=8,
                decoder_hidden_dim=32,
            )
        )
        tokens = torch.randint(low=0, high=5, size=(2, 37), dtype=torch.long)
        out = model(tokens)
        self.assertEqual(tuple(out["logits"].shape), (2, 37, 3))
        self.assertEqual(tuple(out["mu"].shape), (2, 8))

    def test_smoke_training_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            meta_path = write_synthetic_meta(tmp, n_samples=12, n_snps=41)
            out_dir = tmp / "out"
            args = types.SimpleNamespace(
                meta_json=str(meta_path),
                output_dir=str(out_dir),
                run_name="toy_run",
                seed=7,
                train_ratio=0.7,
                val_ratio=0.15,
                mask_prob=0.2,
                eval_mask_prob=0.2,
                epochs=2,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=1e-2,
                patience=2,
                grad_clip_norm=1.0,
                latent_dim=8,
                chunk_size=8,
                token_embed_dim=4,
                chunk_dim=32,
                encoder_layers=1,
                encoder_heads=4,
                encoder_ff_mult=2,
                decoder_hidden_dim=32,
                dropout=0.1,
                beta_max=0.01,
                beta_warmup_epochs=1,
                num_workers=0,
                monitor_every=1,
                coverage_monitor_subset=4,
                wandb=False,
                wandb_project="vae-oneshot",
                wandb_entity=None,
                wandb_name=None,
                wandb_group=None,
                wandb_tags=None,
                wandb_mode="offline",
            )

            payload = run_oneshot_vae_training(args)
            self.assertIn("coverage_audit", payload)
            for rel in (
                "config.json",
                "splits.npz",
                "metrics.jsonl",
                "best_model.pt",
                "run_summary.json",
                "all_latents.csv",
                "coverage_audit.json",
                "pc_coverage_corr.tsv",
            ):
                self.assertTrue((out_dir / rel).exists(), rel)


class TestOneShotUtilities(unittest.TestCase):
    def test_coverage_audit_detects_relationship(self):
        latents = np.array(
            [
                [0.1, 0.0],
                [0.2, 0.1],
                [0.3, 0.1],
                [0.4, 0.2],
            ],
            dtype=np.float32,
        )
        observed_fraction = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        audit = compute_coverage_audit(latents, observed_fraction)
        self.assertGreater(audit["r2_cov_from_z"], 0.9)

    def test_cli_parser_accepts_defaults(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--meta_json",
                "/tmp/example.meta.json",
                "--output_dir",
                "/tmp/vae_out",
            ]
        )
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.chunk_size, 512)


if __name__ == "__main__":
    unittest.main()
