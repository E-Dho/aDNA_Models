from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from overlap_embed.data import load_meta, open_genotype_memmap
from overlap_embed.oos_encoder import (
    align_eigenstrat_to_manifest,
    load_snp_manifest,
    prepare_encoder_reference,
    read_snp_records,
    write_snp_manifest,
)


class OosSnpAlignmentTests(unittest.TestCase):
    def _write_teacher_manifest(self, root: Path) -> Path:
        teacher_snp = root / "teacher.snp"
        teacher_snp.write_text(
            "\n".join(
                [
                    "rs1 1 0.0 100 A G",
                    "rs2 1 0.0 200 C T",
                    "rs3 1 0.0 300 G A",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = root / "snp_manifest.tsv"
        write_snp_manifest(manifest, read_snp_records(teacher_snp))
        return manifest

    def test_alignment_reorders_missing_and_ignores_extra_snps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = self._write_teacher_manifest(root)
            geno = root / "source.geno"
            snp = root / "source.snp"
            ind = root / "source.ind"
            snp.write_text(
                "\n".join(
                    [
                        "rs2 1 0.0 200 C T",
                        "rs1 1 0.0 100 A G",
                        "rsX 1 0.0 999 A C",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            geno.write_text("01\n29\n11\n", encoding="utf-8")
            ind.write_text("s1 U A\ns2 U B\n", encoding="utf-8")

            _, report = align_eigenstrat_to_manifest(
                geno_path=geno,
                snp_path=snp,
                ind_path=ind,
                manifest_path=manifest,
                out_prefix=root / "aligned",
            )
            meta = load_meta(root / "aligned.meta.json")
            mm = open_genotype_memmap(meta)
            expected = np.asarray(
                [
                    [1, 0, -1],
                    [-1, 1, -1],
                ],
                dtype=np.int8,
            )
            np.testing.assert_array_equal(np.asarray(mm), expected)
            self.assertEqual(report.n_aligned_snps, 2)
            self.assertEqual(report.n_missing_snps, 1)
            self.assertEqual(report.n_extra_source_snps, 1)

    def test_allele_set_mismatch_is_reported_and_not_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = self._write_teacher_manifest(root)
            geno = root / "source.geno"
            snp = root / "source.snp"
            ind = root / "source.ind"
            snp.write_text("rs1 1 0.0 100 C G\nrs2 1 0.0 200 T C\n", encoding="utf-8")
            geno.write_text("01\n10\n", encoding="utf-8")
            ind.write_text("s1 U A\ns2 U B\n", encoding="utf-8")

            _, report = align_eigenstrat_to_manifest(
                geno_path=geno,
                snp_path=snp,
                ind_path=ind,
                manifest_path=manifest,
                out_prefix=root / "aligned",
            )
            meta = load_meta(root / "aligned.meta.json")
            mm = open_genotype_memmap(meta)
            self.assertTrue(np.all(np.asarray(mm)[:, 0] == -1))
            self.assertEqual(report.n_allele_set_mismatch, 1)
            self.assertEqual(report.n_allele_order_mismatch, 1)

    def test_prepare_reference_uses_prefix_not_path_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_prefix = root / "v62.0_HO_public_europe_filtered"
            Path(f"{data_prefix}.snp").write_text("rs1 1 0.0 100 A G\n", encoding="utf-8")
            teacher_run = root / "teacher"
            graph = root / "graph"
            teacher_run.mkdir()
            graph.mkdir()
            (teacher_run / "final_latents.csv").write_text("sample_id,z1\ns1,0.0\n", encoding="utf-8")
            (graph / "sample_stats.tsv").write_text(
                "sample_id\tobserved_fraction\tcountry\toriginal_group_id\tdate_mean_bp\ns1\t0.1\tA\tA1\t100\n",
                encoding="utf-8",
            )
            summary = prepare_encoder_reference(
                data_prefix=data_prefix,
                teacher_run_dir=teacher_run,
                graph_dir=graph,
                out_dir=root / "ref",
            )
            self.assertTrue(Path(summary["snp_manifest"]).exists())
            self.assertEqual(load_snp_manifest(root / "ref" / "snp_manifest.tsv")[0].snp_id, "rs1")


if __name__ == "__main__":
    unittest.main()
