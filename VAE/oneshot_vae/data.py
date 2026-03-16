from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = object


MISSING_VALUE = -1


@dataclass(frozen=True)
class GenotypeMemmapMeta:
    memmap_path: str
    n_samples: int
    n_snps: int
    dtype: str
    ploidy: str
    n_classes: int
    missing_value: int
    sample_ids_path: Optional[str]


def load_meta(meta_path: Path) -> GenotypeMemmapMeta:
    with meta_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return GenotypeMemmapMeta(
        memmap_path=str(data["memmap_path"]),
        n_samples=int(data["n_samples"]),
        n_snps=int(data["n_snps"]),
        dtype=str(data["dtype"]),
        ploidy=str(data["ploidy"]),
        n_classes=int(data["n_classes"]),
        missing_value=int(data["missing_value"]),
        sample_ids_path=data.get("sample_ids_path"),
    )


def load_sample_ids(sample_ids_path: Optional[Path], expected_n: int) -> List[str]:
    if sample_ids_path is None:
        return [f"sample_{i}" for i in range(expected_n)]
    sample_ids: List[str] = []
    with sample_ids_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                sample_ids.append(line)
    if len(sample_ids) != expected_n:
        raise ValueError(
            f"Sample count mismatch: sample ID file has {len(sample_ids)} rows, expected {expected_n}"
        )
    return sample_ids


def open_genotype_memmap(meta: GenotypeMemmapMeta, mode: str = "r") -> np.memmap:
    return np.memmap(
        meta.memmap_path,
        dtype=np.dtype(meta.dtype),
        mode=mode,
        shape=(meta.n_samples, meta.n_snps),
    )


def compute_observed_fraction(geno_mm: np.memmap, chunk_size: int = 256) -> np.ndarray:
    n_samples = int(geno_mm.shape[0])
    out = np.zeros(n_samples, dtype=np.float32)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        block = np.asarray(geno_mm[start:end, :], dtype=np.int16)
        out[start:end] = (block >= 0).mean(axis=1).astype(np.float32)
    return out


def build_stratified_splits(
    eligible_indices: np.ndarray,
    observed_fraction: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios. Need train>0, val>0, train+val<1")
    if eligible_indices.size == 0:
        raise ValueError("No eligible samples remain after filtering zero-observed rows.")

    rng = np.random.default_rng(seed)
    eligible_indices = np.asarray(eligible_indices, dtype=np.int64)
    cov = np.asarray(observed_fraction[eligible_indices], dtype=np.float64)

    if eligible_indices.size < 10:
        perm = rng.permutation(eligible_indices)
        train_end = int(len(perm) * train_ratio)
        val_end = train_end + int(len(perm) * val_ratio)
        return {
            "train": perm[:train_end],
            "val": perm[train_end:val_end],
            "test": perm[val_end:],
        }

    quantiles = np.quantile(cov, np.linspace(0.0, 1.0, num=min(n_bins, eligible_indices.size) + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    bins = np.digitize(cov, quantiles[1:-1], right=True)

    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []

    for bin_id in np.unique(bins):
        members = eligible_indices[bins == bin_id]
        members = rng.permutation(members)
        n = int(members.size)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        train_parts.append(members[:train_end])
        val_parts.append(members[train_end:val_end])
        test_parts.append(members[val_end:])

    splits = {
        "train": np.concatenate(train_parts) if train_parts else np.zeros(0, dtype=np.int64),
        "val": np.concatenate(val_parts) if val_parts else np.zeros(0, dtype=np.int64),
        "test": np.concatenate(test_parts) if test_parts else np.zeros(0, dtype=np.int64),
    }

    if min(len(splits["train"]), len(splits["val"]), len(splits["test"])) == 0:
        perm = rng.permutation(eligible_indices)
        train_end = int(len(perm) * train_ratio)
        val_end = train_end + int(len(perm) * val_ratio)
        splits = {
            "train": perm[:train_end],
            "val": perm[train_end:val_end],
            "test": perm[val_end:],
        }

    return splits


class OneShotMemmapDataset(Dataset):
    def __init__(
        self,
        meta: GenotypeMemmapMeta,
        indices: Sequence[int],
        observed_fraction: np.ndarray,
        sample_ids: Sequence[str],
    ) -> None:
        self.meta = meta
        self.indices = np.asarray(indices, dtype=np.int64)
        self.observed_fraction = np.asarray(observed_fraction, dtype=np.float32)
        self.sample_ids = list(sample_ids)
        self._geno_mm: Optional[np.memmap] = None

    def _ensure_memmap(self) -> np.memmap:
        if self._geno_mm is None:
            self._geno_mm = open_genotype_memmap(self.meta, mode="r")
        return self._geno_mm

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int):
        row_idx = int(self.indices[item])
        geno_mm = self._ensure_memmap()
        row = np.asarray(geno_mm[row_idx], dtype=np.int16)
        return row, row_idx, self.sample_ids[row_idx], float(self.observed_fraction[row_idx])
