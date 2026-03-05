import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase1_pipeline import build_mask_from_x, parse_info_file, verify_x_covered_by_m


class TestMaskBuilder(unittest.TestCase):
    def test_build_mask_from_x(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            x_sparse = tmp / "X.sparse"
            x_info = tmp / "X.info"
            m_sparse = tmp / "M.sparse"
            m_info = tmp / "M.info"

            x_info.write_text("format: COO\nnrows: 3\nncols: 4\nnnz: 4\n", encoding="utf-8")
            x_sparse.write_text(
                "0 0 0.1\n"
                "0 0 0.1\n"  # duplicate consecutive entry
                "1 2 -0.5\n"
                "2 3 1.7\n",
                encoding="utf-8",
            )

            stats = build_mask_from_x(x_sparse, x_info, m_sparse, m_info)

            self.assertEqual(stats["x_nnz_input"], 4)
            self.assertEqual(stats["m_nnz_written"], 3)
            self.assertEqual(stats["deduped_consecutive_entries"], 1)

            info = parse_info_file(m_info)
            self.assertEqual(int(info["nrows"]), 3)
            self.assertEqual(int(info["ncols"]), 4)
            self.assertEqual(int(info["nnz"]), 3)

            lines = m_sparse.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines, ["0 0 1", "1 2 1", "2 3 1"])

    def test_coverage_check(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            x_sparse = tmp / "X.sparse"
            m_sparse = tmp / "M.sparse"

            x_sparse.write_text("0 0 0.1\n0 2 -0.3\n1 1 0.4\n", encoding="utf-8")
            m_sparse.write_text("0 0 1\n0 2 1\n1 1 1\n", encoding="utf-8")

            result = verify_x_covered_by_m(x_sparse, m_sparse)
            self.assertTrue(result["covered"])


if __name__ == "__main__":
    unittest.main()
