from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch


MISSING_VALUE = -1


@dataclass(frozen=True)
class Window:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


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


def build_windows(n_snps: int, window_size: int) -> List[Window]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    windows: List[Window] = []
    start = 0
    while start < n_snps:
        end = min(start + window_size, n_snps)
        windows.append(Window(start=start, end=end))
        start = end
    return windows


def load_sample_ids(ind_path: Optional[Path], expected_n: int) -> List[str]:
    if ind_path is None:
        return [f"sample_{i}" for i in range(expected_n)]
    sample_ids: List[str] = []
    with ind_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            sample_ids.append(line.split()[0])
    if len(sample_ids) != expected_n:
        raise ValueError(
            f"Sample count mismatch: .ind has {len(sample_ids)} IDs, expected {expected_n}"
        )
    return sample_ids


def save_meta(meta_path: Path, meta: GenotypeMemmapMeta) -> None:
    payload = {
        "memmap_path": meta.memmap_path,
        "n_samples": meta.n_samples,
        "n_snps": meta.n_snps,
        "dtype": meta.dtype,
        "ploidy": meta.ploidy,
        "n_classes": meta.n_classes,
        "missing_value": meta.missing_value,
        "sample_ids_path": meta.sample_ids_path,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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


def open_genotype_memmap(meta: GenotypeMemmapMeta, mode: str = "r") -> np.memmap:
    return np.memmap(
        meta.memmap_path,
        dtype=np.dtype(meta.dtype),
        mode=mode,
        shape=(meta.n_samples, meta.n_snps),
    )


def _count_nonempty_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.strip():
                n += 1
    return n


def _first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                return line
    raise ValueError(f"No non-empty lines in file: {path}")


def _map_genotypes(raw_ascii: np.ndarray, ploidy: str) -> np.ndarray:
    vals = raw_ascii.astype(np.int16) - ord("0")
    out = np.full(vals.shape, MISSING_VALUE, dtype=np.int8)
    observed = vals != 9
    if ploidy == "diploid":
        valid = observed & ((vals == 0) | (vals == 1) | (vals == 2))
        out[valid] = vals[valid].astype(np.int8)
    elif ploidy == "pseudohaploid":
        valid = observed & ((vals == 0) | (vals == 1) | (vals == 2))
        # Collapse any non-zero genotype to 1.
        out[valid] = (vals[valid] > 0).astype(np.int8)
    else:
        raise ValueError("ploidy must be one of: diploid, pseudohaploid")
    return out


def build_memmap_from_eigenstrat(
    geno_path: Path,
    out_prefix: Path,
    ploidy: str,
    ind_path: Optional[Path] = None,
) -> GenotypeMemmapMeta:
    if ploidy not in {"diploid", "pseudohaploid"}:
        raise ValueError("ploidy must be one of: diploid, pseudohaploid")

    first = _first_nonempty_line(geno_path)
    n_samples = len(first)
    n_snps = _count_nonempty_lines(geno_path)

    if ind_path is not None:
        ind_count = _count_nonempty_lines(ind_path)
        if ind_count != n_samples:
            raise ValueError(
                f".ind count ({ind_count}) does not match .geno sample width ({n_samples})"
            )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    memmap_path = out_prefix.with_suffix(".genotypes.int8.memmap")
    mm = np.memmap(
        memmap_path,
        dtype=np.int8,
        mode="w+",
        shape=(n_samples, n_snps),
    )

    snp_idx = 0
    with geno_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if len(line) != n_samples:
                raise ValueError(
                    f"Inconsistent line width in {geno_path} at SNP row {snp_idx}: "
                    f"{len(line)} vs expected {n_samples}"
                )
            ascii_vals = np.frombuffer(line.encode("ascii"), dtype=np.uint8)
            mm[:, snp_idx] = _map_genotypes(ascii_vals, ploidy)
            snp_idx += 1

    mm.flush()

    sample_ids_path = None
    if ind_path is not None:
        sample_ids_path = str(out_prefix.with_suffix(".sample_ids.txt"))
        sample_ids = load_sample_ids(ind_path, expected_n=n_samples)
        with Path(sample_ids_path).open("w", encoding="utf-8") as handle:
            for sid in sample_ids:
                handle.write(f"{sid}\n")

    meta = GenotypeMemmapMeta(
        memmap_path=str(memmap_path),
        n_samples=n_samples,
        n_snps=n_snps,
        dtype="int8",
        ploidy=ploidy,
        n_classes=3 if ploidy == "diploid" else 2,
        missing_value=MISSING_VALUE,
        sample_ids_path=sample_ids_path,
    )
    save_meta(out_prefix.with_suffix(".meta.json"), meta)
    return meta


def make_splits(
    n_samples: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, np.ndarray]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios. Need train>0, val>0, train+val<1")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(indices)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def compute_observed_fraction(
    geno_mm: np.memmap,
    chunk_size: int = 256,
) -> np.ndarray:
    n_samples = geno_mm.shape[0]
    out = np.zeros(n_samples, dtype=np.float32)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        block = np.asarray(geno_mm[start:end, :], dtype=np.int16)
        out[start:end] = (block >= 0).mean(axis=1).astype(np.float32)
    return out


def build_window_batch(
    geno_mm: np.memmap,
    sample_indices: np.ndarray,
    windows: Sequence[Window],
    window_ids: np.ndarray,
    window_size: int,
    n_classes: int,
    rng: np.random.Generator,
    mask_prob: float,
    observed_dropout: float,
    training: bool,
    device: "torch.device",
) -> Dict[str, "torch.Tensor"]:
    import torch
    bsz = int(sample_indices.shape[0])
    n_w = int(window_ids.shape[0])

    input_idx = np.full((bsz, n_w, window_size), n_classes, dtype=np.int64)
    targets = np.full((bsz, n_w, window_size), -100, dtype=np.int64)
    obs_mask = np.zeros((bsz, n_w, window_size), dtype=np.float32)
    coverage = np.zeros((bsz, n_w, 1), dtype=np.float32)

    for wi, wid in enumerate(window_ids.tolist()):
        w = windows[wid]
        wlen = w.length
        block = np.asarray(geno_mm[sample_indices, w.start : w.end], dtype=np.int16)
        observed = block >= 0

        input_idx[:, wi, :wlen][observed] = block[observed].astype(np.int64)
        targets[:, wi, :wlen][observed] = block[observed].astype(np.int64)
        obs_mask[:, wi, :wlen] = observed.astype(np.float32)
        coverage[:, wi, 0] = observed.mean(axis=1).astype(np.float32)

    if training:
        observed = obs_mask > 0.5
        corrupt = observed & (rng.random(observed.shape) < mask_prob)
        dropped = observed & (~corrupt) & (rng.random(observed.shape) < observed_dropout)

        input_idx[corrupt | dropped] = n_classes
        obs_mask[corrupt | dropped] = 0.0
        targets = np.where(corrupt, targets, -100)

    batch = {
        "input_idx": torch.from_numpy(input_idx).to(device=device, dtype=torch.long),
        "obs_mask": torch.from_numpy(obs_mask).to(device=device, dtype=torch.float32),
        "targets": torch.from_numpy(targets).to(device=device, dtype=torch.long),
        "coverage": torch.from_numpy(coverage).to(device=device, dtype=torch.float32),
        "window_ids": torch.from_numpy(window_ids).to(device=device, dtype=torch.long),
    }
    return batch
