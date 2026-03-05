import tempfile
import unittest
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1_pipeline import (
    build_selected_dense_memmaps,
    compute_column_stats,
    select_features,
)


class TestFeaturePipeline(unittest.TestCase):
    def test_stats_select_and_memmaps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            x_sparse = tmp / "X.sparse"
            m_sparse = tmp / "M.sparse"

            # 4 rows, 5 cols
            x_sparse.write_text(
                "0 0 1.0\n"
                "0 2 0.4\n"
                "1 0 0.9\n"
                "1 3 0.2\n"
                "2 1 -0.5\n"
                "3 0 0.8\n"
                "3 1 -0.2\n",
                encoding="utf-8",
            )
            m_sparse.write_text(
                "0 0 1\n"
                "0 2 1\n"
                "1 0 1\n"
                "1 3 1\n"
                "2 1 1\n"
                "3 0 1\n"
                "3 1 1\n",
                encoding="utf-8",
            )

            counts, sums, sq_sums = compute_column_stats(x_sparse, nrows=4, ncols=5)
            feature_data = select_features(
                counts=counts,
                sums=sums,
                sq_sums=sq_sums,
                nrows=4,
                min_call_rate=0.25,
                min_variance=1e-8,
                max_features=3,
            )
            selected = feature_data["selected_indices"]
            self.assertLessEqual(selected.size, 3)

            dense_info = build_selected_dense_memmaps(
                x_sparse_path=x_sparse,
                m_sparse_path=m_sparse,
                nrows=4,
                selected_indices=selected,
                output_dir=tmp,
                prefix="unit",
            )

            x_mm = np.memmap(
                dense_info["x_memmap_path"],
                dtype=np.float32,
                mode="r",
                shape=(4, int(dense_info["n_features"])),
            )
            m_mm = np.memmap(
                dense_info["m_memmap_path"],
                dtype=np.uint8,
                mode="r",
                shape=(4, int(dense_info["n_features"])),
            )

            # Mask must be binary and non-empty for selected features.
            self.assertTrue(np.all((m_mm == 0) | (m_mm == 1)))
            self.assertGreater(int(m_mm.sum()), 0)
            self.assertEqual(x_mm.shape, m_mm.shape)


if __name__ == "__main__":
    unittest.main()
