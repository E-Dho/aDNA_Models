"""Coverage-robust overlap graph embedding for aDNA."""

from .data import (
    GenotypeMemmapMeta,
    build_memmap_from_eigenstrat,
    compute_observed_fraction,
    load_meta,
    load_sample_ids_from_meta,
    open_genotype_memmap,
    read_anno_rows,
    save_meta,
)
from .oos_encoder import (
    align_eigenstrat_to_manifest,
    load_snp_manifest,
    prefixed_path,
    write_snp_manifest,
)

__all__ = [
    "GenotypeMemmapMeta",
    "build_memmap_from_eigenstrat",
    "compute_observed_fraction",
    "load_meta",
    "load_sample_ids_from_meta",
    "open_genotype_memmap",
    "read_anno_rows",
    "save_meta",
    "align_eigenstrat_to_manifest",
    "load_snp_manifest",
    "prefixed_path",
    "write_snp_manifest",
]
