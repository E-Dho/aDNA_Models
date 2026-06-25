from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from overlap_embed.cluster_compare import (
    AssignmentSpec,
    compare_pair,
    contingency_matrix,
    load_assignment,
    row_normalize,
    variation_of_information_from_counts,
)


def _frame(labels: list[int], noise: list[bool] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(labels))],
            "cluster_id": labels,
            "is_optics_noise": [label == -1 for label in labels] if noise is None else noise,
        }
    )


class TestOpticsClusterCompare(unittest.TestCase):
    def test_identical_clusterings_have_perfect_ari_and_zero_vi(self) -> None:
        frame = _frame([0, 0, 1, 1, -1])
        result = compare_pair("a", frame, "b", frame, mode="include_noise")
        self.assertAlmostEqual(result.ari, 1.0)
        self.assertAlmostEqual(result.vi, 0.0, places=12)
        self.assertAlmostEqual(result.normalized_vi, 0.0, places=12)

    def test_different_clusterings_have_positive_vi(self) -> None:
        a = _frame([0, 0, 1, 1])
        b = _frame([0, 1, 0, 1])
        result = compare_pair("a", a, "b", b, mode="include_noise")
        self.assertGreater(result.vi, 0.0)
        self.assertGreaterEqual(result.normalized_vi, 0.0)

    def test_noise_modes_change_sample_count(self) -> None:
        a = _frame([0, 0, 1, -1])
        b = _frame([0, -1, 1, 1])
        include = compare_pair("a", a, "b", b, mode="include_noise")
        exclude = compare_pair("a", a, "b", b, mode="exclude_noise_either")
        self.assertEqual(include.n_samples, 4)
        self.assertEqual(exclude.n_samples, 2)
        self.assertEqual(exclude.n_dropped_for_noise, 2)

    def test_overlap_matrix_counts_match_known_intersections(self) -> None:
        counts = contingency_matrix([0, 0, 1, 1, -1], [2, 2, 2, 3, -1])
        self.assertEqual(counts.loc[-1, -1], 1)
        self.assertEqual(counts.loc[0, 2], 2)
        self.assertEqual(counts.loc[1, 2], 1)
        self.assertEqual(counts.loc[1, 3], 1)

    def test_row_normalized_overlap_rows_sum_to_one(self) -> None:
        counts = pd.DataFrame([[2, 2], [0, 3]], index=[0, 1], columns=[0, 1])
        fractions = row_normalize(counts)
        np.testing.assert_allclose(fractions.sum(axis=1).to_numpy(), np.ones(2))
        self.assertAlmostEqual(fractions.loc[0, 0], 0.5)
        self.assertAlmostEqual(fractions.loc[1, 1], 1.0)

    def test_vi_for_identical_counts_is_zero(self) -> None:
        counts = contingency_matrix([0, 0, 1, 1], [0, 0, 1, 1])
        vi, normalized = variation_of_information_from_counts(counts)
        self.assertTrue(math.isclose(vi, 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(normalized, 0.0, abs_tol=1e-12))

    def test_load_assignment_rejects_duplicate_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignments.tsv"
            pd.DataFrame({"sample_id": ["s1", "s1"], "cluster_id": [0, 1]}).to_csv(
                path,
                sep="\t",
                index=False,
            )
            with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
                load_assignment(AssignmentSpec("dup", path))


if __name__ == "__main__":
    unittest.main()
