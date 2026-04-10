from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from overlap_embed.graph import compute_overlap_numerator_and_counts


try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None


class OverlapGraphRuntimeTests(unittest.TestCase):
    def test_numpy_backend_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            geno = np.asarray(
                [
                    [0, 1, -1, 1],
                    [1, 1, 0, -1],
                    [0, -1, 1, 1],
                ],
                dtype=np.int8,
            )
            meta_json = root / "toy.meta.json"
            meta_json.write_text("{}", encoding="utf-8")
            numerator, overlap, stats = compute_overlap_numerator_and_counts(
                geno,
                out_dir=root,
                meta_json_path=meta_json,
                latent_dim=2,
                min_overlap=1,
                top_k=2,
                overlap_shrinkage=10.0,
                ploidy="pseudohaploid",
                max_samples=-1,
                chunk_snps=2,
                checkpoint_every=1,
                compute_backend="numpy",
                progress_every=0,
                resume_existing=False,
            )
            self.assertEqual(numerator.shape, (3, 3))
            self.assertEqual(overlap.shape, (3, 3))
            self.assertEqual(stats["last_completed_chunk"], 2)
            self.assertTrue((root / "graph_checkpoint.json").exists())
            self.assertTrue((root / "partial_numerator.float32.memmap").exists())
            self.assertTrue((root / "partial_overlap.uint32.memmap").exists())
            numerator_resume, overlap_resume, stats_resume = compute_overlap_numerator_and_counts(
                geno,
                out_dir=root,
                meta_json_path=meta_json,
                latent_dim=2,
                min_overlap=1,
                top_k=2,
                overlap_shrinkage=10.0,
                ploidy="pseudohaploid",
                max_samples=-1,
                chunk_snps=2,
                checkpoint_every=1,
                compute_backend="numpy",
                progress_every=0,
                resume_existing=True,
            )
            np.testing.assert_allclose(numerator, numerator_resume, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(overlap, overlap_resume)
            self.assertEqual(stats_resume["last_completed_chunk"], 2)

    @unittest.skipIf(torch is None, "torch is required for torch backend graph test")
    def test_torch_backend_matches_numpy_on_toy_matrix(self) -> None:
        geno = np.asarray(
            [
                [0, 1, -1, 1],
                [1, 1, 0, -1],
                [0, -1, 1, 1],
            ],
            dtype=np.int8,
        )
        with tempfile.TemporaryDirectory() as numpy_dir, tempfile.TemporaryDirectory() as torch_dir:
            meta_numpy = Path(numpy_dir) / "toy.meta.json"
            meta_torch = Path(torch_dir) / "toy.meta.json"
            meta_numpy.write_text("{}", encoding="utf-8")
            meta_torch.write_text("{}", encoding="utf-8")
            numerator_np, overlap_np, _ = compute_overlap_numerator_and_counts(
                geno,
                out_dir=Path(numpy_dir),
                meta_json_path=meta_numpy,
                latent_dim=2,
                min_overlap=1,
                top_k=2,
                overlap_shrinkage=10.0,
                ploidy="pseudohaploid",
                max_samples=-1,
                chunk_snps=2,
                checkpoint_every=1,
                compute_backend="numpy",
                progress_every=0,
                resume_existing=False,
            )
            numerator_t, overlap_t, _ = compute_overlap_numerator_and_counts(
                geno,
                out_dir=Path(torch_dir),
                meta_json_path=meta_torch,
                latent_dim=2,
                min_overlap=1,
                top_k=2,
                overlap_shrinkage=10.0,
                ploidy="pseudohaploid",
                max_samples=-1,
                chunk_snps=2,
                checkpoint_every=1,
                compute_backend="torch",
                compute_device="cpu",
                progress_every=0,
                resume_existing=False,
            )
            np.testing.assert_allclose(numerator_np, numerator_t, rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(overlap_np, overlap_t)


if __name__ == "__main__":
    unittest.main()
