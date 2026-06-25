from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from overlap_embed.geo import (
    _matched_point_colors,
    _with_alpha,
    apply_similarity_transform,
    compute_arrow_density_alpha,
    compute_distortion_metrics,
    compute_distortion_vectors,
    fit_similarity_transform,
    load_and_merge_latent_geo,
    reduce_latent_to_2d,
    run_distortion_pipeline,
)


class GeoAlignmentTests(unittest.TestCase):
    def test_similarity_transform_recovers_known_transform(self) -> None:
        rng = np.random.default_rng(123)
        source = rng.normal(size=(80, 2))
        theta = np.deg2rad(37.0)
        rotation = np.asarray(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float64,
        )
        target = 2.75 * (source @ rotation) + np.asarray([300.0, -120.0])
        transform = fit_similarity_transform(source, target)
        aligned = apply_similarity_transform(source, transform)
        self.assertLess(np.max(np.abs(aligned - target)), 1e-10)
        self.assertLess(transform.disparity, 1e-20)

    def test_near_zero_residual_for_transformed_copy(self) -> None:
        source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
        target = -1.5 * source + np.asarray([10.0, 5.0])
        transform = fit_similarity_transform(source, target, allow_reflection=True)
        aligned = apply_similarity_transform(source, transform)
        residual = np.sqrt(((aligned - target) ** 2).sum(axis=1))
        self.assertLess(float(residual.max()), 1e-10)

    def test_merge_and_filter_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latents = root / "latents.csv"
            meta = root / "meta.tsv"
            latents.write_text(
                "sample_id,z0,z1,z2\n"
                "a,0,1,2\n"
                "b,1,2,3\n"
                "c,2,,4\n"
                "x,9,9,9\n",
                encoding="utf-8",
            )
            meta.write_text(
                "sample_id\tlatitude\tlongitude\tcountry\n"
                "a\t50\t10\tA\n"
                "b\t51\t11\tB\n"
                "c\t\t12\tC\n",
                encoding="utf-8",
            )
            merged, values, report = load_and_merge_latent_geo(
                latents,
                meta,
                label_col="country",
                min_samples=2,
            )
            self.assertEqual(merged["sample_id"].tolist(), ["a", "b"])
            self.assertEqual(values.shape, (2, 3))
            self.assertEqual(report.n_latents, 4)
            self.assertEqual(report.n_joined, 3)
            self.assertEqual(report.n_after_filter, 2)

    def test_pca_reduction_shape_and_manual_components(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=(30, 6))
        result = reduce_latent_to_2d(x, reduce_method="pca", component_indices="1,3")
        self.assertEqual(result.coords.shape, (30, 2))
        self.assertEqual(result.selected_components, [1, 3])
        self.assertGreaterEqual(len(result.explained_variance_ratio), 3)

    def test_distortion_magnitude_correctness(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
                "country": ["A", "B"],
            }
        )
        geo = np.asarray([[0.0, 0.0], [3.0, 4.0]])
        aligned = np.asarray([[0.0, 0.0], [6.0, 8.0]])
        out = compute_distortion_vectors(
            frame,
            id_col="sample_id",
            lat_col="latitude",
            lon_col="longitude",
            geo_xy=geo,
            aligned_xy=aligned,
            label_col="country",
        )
        self.assertAlmostEqual(float(out.loc[1, "dx"]), 3.0)
        self.assertAlmostEqual(float(out.loc[1, "dy"]), 4.0)
        self.assertAlmostEqual(float(out.loc[1, "distortion_magnitude"]), 5.0)
        self.assertAlmostEqual(float(out.loc[1, "unit_dx"]), 0.6)
        self.assertAlmostEqual(float(out.loc[1, "unit_dy"]), 0.8)

    def test_metrics_keys_present_and_finite(self) -> None:
        latent = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
        geo = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [4.0, 2.0]])
        dist = pd.DataFrame(
            {
                "distortion_magnitude": [0.1, 0.2, 0.3, 0.4],
                "country": ["A", "A", "B", "B"],
            }
        )
        metrics = compute_distortion_metrics(latent, geo, dist, label_col="country", knn_k=2)
        for key in (
            "pairwise_latent_geo_pearson",
            "pairwise_latent_geo_spearman",
            "rms_alignment_error",
            "median_distortion_magnitude",
            "knn_geography_overlap@2",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))
        self.assertIn("per_label_mean_distortion_magnitude", metrics)

    def test_pipeline_writes_expected_artifacts_on_synthetic_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latents = root / "latents.csv"
            meta = root / "meta.tsv"
            out = root / "out"
            with latents.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_id", "z0", "z1", "z2"])
                for idx in range(12):
                    writer.writerow([f"s{idx}", idx, idx % 3, idx * 0.5])
            with meta.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["sample_id", "latitude", "longitude", "country"])
                for idx in range(12):
                    writer.writerow([f"s{idx}", 45.0 + idx * 0.1, 8.0 + idx * 0.2, "A" if idx < 6 else "B"])
            result = run_distortion_pipeline(
                latents_csv=latents,
                metadata_tsv=meta,
                out_dir=out,
                label_col="country",
                project_coords=0,
                min_samples=10,
                seed=7,
            )
            self.assertEqual(result["n_samples"], 12)
            for name in (
                "distortion_vectors.tsv",
                "distortion_metrics.json",
                "distortion_map.png",
                "aligned_vs_true.png",
                "distortion_histogram.png",
                "distance_correlation.png",
            ):
                self.assertTrue((out / name).exists())
            metrics = json.loads((out / "distortion_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["n_samples"], 12)

    def test_arrow_density_alpha_increases_for_crowded_midpoints(self) -> None:
        x = np.asarray([0.0, 0.02, 0.04, 10.0])
        y = np.asarray([0.0, 0.02, 0.04, 10.0])
        dx = np.asarray([0.2, 0.2, 0.2, 0.2])
        dy = np.zeros_like(dx)
        alpha = compute_arrow_density_alpha(
            x,
            y,
            dx,
            dy,
            alpha_min=0.05,
            alpha_max=0.65,
            grid_size=4,
        )
        self.assertEqual(alpha.shape, (4,))
        self.assertGreaterEqual(float(alpha.min()), 0.05)
        self.assertLessEqual(float(alpha.max()), 0.65)
        self.assertGreater(float(alpha[0]), float(alpha[-1]))

    def test_matched_sample_colors_keep_true_and_aligned_rgb_equal(self) -> None:
        frame = pd.DataFrame({"sample_id": [f"s{i}" for i in range(6)]})
        colors = _matched_point_colors(frame, mode="sample")
        true_colors = _with_alpha(colors, 0.82)
        aligned_colors = _with_alpha(colors, 0.55)
        self.assertEqual(colors.shape, (6, 4))
        np.testing.assert_allclose(true_colors[:, :3], aligned_colors[:, :3])
        self.assertTrue(np.allclose(true_colors[:, 3], 0.82))
        self.assertTrue(np.allclose(aligned_colors[:, 3], 0.55))

    def test_matched_label_colors_are_shared_within_country(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": ["a1", "a2", "b1", "b2"],
                "country": ["A", "A", "B", "B"],
            }
        )
        colors = _matched_point_colors(frame, mode="label", label_col="country")
        np.testing.assert_allclose(colors[0], colors[1])
        np.testing.assert_allclose(colors[2], colors[3])
        self.assertGreater(float(np.abs(colors[0] - colors[2]).sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
