from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from overlap_embed.data import build_memmap_from_eigenstrat, load_meta, open_genotype_memmap
from overlap_embed.data import build_sample_metadata_index, read_anno_rows, build_sample_stats_rows


class PrepareMemmapTests(unittest.TestCase):
    def test_pseudohaploid_mapping_and_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            geno = root / "toy.geno"
            ind = root / "toy.ind"
            geno.write_text("0129\n2210\n", encoding="utf-8")
            ind.write_text("s1 U A\ns2 U B\ns3 U C\ns4 U D\n", encoding="utf-8")
            out_prefix = root / "toy"
            meta = build_memmap_from_eigenstrat(
                geno_path=geno,
                ind_path=ind,
                out_prefix=out_prefix,
                ploidy="pseudohaploid",
            )
            loaded = load_meta(out_prefix.with_suffix(".meta.json"))
            mm = open_genotype_memmap(loaded)
            expected = np.asarray(
                [
                    [0, 1],
                    [1, 1],
                    [1, 1],
                    [-1, 0],
                ],
                dtype=np.int8,
            )
            np.testing.assert_array_equal(np.asarray(mm), expected)
            self.assertEqual(meta.n_classes, 2)
            self.assertEqual(meta.ploidy, "pseudohaploid")

    def test_anno_header_with_long_genetic_id_name_is_joined(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            anno = root / "toy.anno"
            anno.write_text(
                (
                    '"Genetic ID (suffixes: "".DG"" is a high coverage shotgun genome)"\t'
                    "Group ID\tOriginal Group ID\t"
                    "Date mean in BP in years before 1950 CE [OxCal mu for a direct radiocarbon date, and average of range for a contextual date]\n"
                    "s1\tA\tA_1\t1000\n"
                    "s2\tB\tB_1\t2000\n"
                ),
                encoding="utf-8",
            )
            rows = read_anno_rows(anno)
            index = build_sample_metadata_index(rows)
            self.assertIn("s1", index)
            stats = build_sample_stats_rows(["s1", "s2"], np.asarray([0.1, 0.2], dtype=np.float32), anno)
            self.assertEqual(stats[0]["country"], "A")
            self.assertEqual(stats[1]["original_group_id"], "B_1")
            self.assertEqual(stats[0]["date_mean_bp"], 1000.0)


if __name__ == "__main__":
    unittest.main()
