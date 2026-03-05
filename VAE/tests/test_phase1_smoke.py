import importlib.util
import tempfile
import types
import unittest
from pathlib import Path

import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1_pipeline import run_phase1_vae_training


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch not installed")
class TestPhase1Smoke(unittest.TestCase):
    def test_one_epoch_smoke(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            x_sparse = tmp / "X.sparse"
            x_info = tmp / "X.info"
            x_ind = tmp / "X.ind"
            out_dir = tmp / "out"

            # Small synthetic matrix: 8 rows x 6 cols
            x_sparse.write_text(
                "0 0 0.5\n"
                "0 2 0.1\n"
                "1 0 0.4\n"
                "1 3 -0.2\n"
                "2 1 0.7\n"
                "3 2 0.2\n"
                "4 4 -0.5\n"
                "5 5 1.0\n"
                "6 1 0.3\n"
                "7 3 0.6\n",
                encoding="utf-8",
            )
            x_info.write_text("format: COO\nnrows: 8\nncols: 6\nnnz: 10\n", encoding="utf-8")
            x_ind.write_text("\n".join(f"sample{i} U POP" for i in range(8)) + "\n", encoding="utf-8")

            args = types.SimpleNamespace(
                x_sparse=str(x_sparse),
                x_info=str(x_info),
                x_ind=str(x_ind),
                m_sparse=None,
                m_info=None,
                rebuild_mask=True,
                output_dir=str(out_dir),
                run_name="smoke",
                min_call_rate=0.0,
                min_variance=0.0,
                max_features=6,
                train_ratio=0.75,
                val_ratio=0.125,
                seed=7,
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                latent_dim=4,
                hidden_1=16,
                hidden_2=8,
                beta_max=1.0,
                beta_warmup_fraction=1.0,
                patience=2,
                min_recon_improvement=0.0,
                min_active_latent_dims=1,
                active_latent_std_threshold=0.0,
            )

            payload = run_phase1_vae_training(args)
            self.assertIn("gate", payload)

            results = out_dir / "phase1_results.json"
            self.assertTrue(results.exists())
            saved = json.loads(results.read_text(encoding="utf-8"))
            self.assertIn("metrics", saved)


if __name__ == "__main__":
    unittest.main()
