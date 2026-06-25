import math
import unittest

import numpy as np

from geolocation_model.data import make_leave_site_split
from geolocation_model.geo import (
    calibration_containment,
    gaussian_jitter_xy,
    isotropic_gaussian_nll_numpy,
    isotropic_radius_for_probability,
    project_latlon_to_xy,
)


class GeoUtilsTests(unittest.TestCase):
    def test_projection_returns_metric_coordinates(self):
        result = project_latlon_to_xy([54.3, 54.4], [10.1, 10.2])
        self.assertEqual(result.xy.shape, (2, 2))
        self.assertTrue(np.isfinite(result.xy).all())
        self.assertGreater(np.linalg.norm(result.xy[1] - result.xy[0]), 1000.0)

    def test_isotropic_gaussian_nll_known_value(self):
        pred = np.array([[0.0, 0.0, math.log(2.0)]])
        target = np.array([[0.0, 0.0]])
        value = isotropic_gaussian_nll_numpy(pred, target, math.log(0.1), math.log(100.0))
        expected = math.log(2.0 * math.pi) + 2.0 * math.log(2.0)
        self.assertAlmostEqual(value, expected, places=8)

    def test_jitter_has_expected_shape_and_scale(self):
        xy = np.zeros((20000, 2), dtype=float)
        jittered = gaussian_jitter_xy(xy, sigma_m=25000.0, seed=7)
        self.assertEqual(jittered.shape, xy.shape)
        self.assertAlmostEqual(float(jittered.std()), 25000.0, delta=600.0)

    def test_leave_site_split_has_no_leakage(self):
        localities = [f"site_{i // 4}" for i in range(60)]
        split = make_leave_site_split(localities, seed=11)
        self.assertFalse(set(split.train_sites) & set(split.val_sites))
        self.assertFalse(set(split.train_sites) & set(split.test_sites))
        self.assertFalse(set(split.val_sites) & set(split.test_sites))
        self.assertEqual(len(split.train) + len(split.val) + len(split.test), 60)

    def test_calibration_radius_thresholds(self):
        sigma = np.array([10.0, 10.0])
        r50 = isotropic_radius_for_probability(sigma, 0.50)
        self.assertTrue(np.allclose(r50, 10.0 * math.sqrt(-2.0 * math.log(0.5))))
        cal = calibration_containment(np.array([1.0, 1000.0]), sigma, probabilities=(0.50,))
        self.assertEqual(cal["containment_50"], 0.5)


if __name__ == "__main__":
    unittest.main()
