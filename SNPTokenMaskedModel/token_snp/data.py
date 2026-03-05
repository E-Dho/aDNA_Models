from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    import torch


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


def load_sample_ids(sample_ids_path: Optional[str], expected_n: int) -> List[str]:
    if sample_ids_path is None:
        return [f"sample_{i}" for i in range(expected_n)]
    path = Path(sample_ids_path)
    if not path.exists():
        return [f"sample_{i}" for i in range(expected_n)]

    sample_ids: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                sample_ids.append(line)
    if len(sample_ids) != expected_n:
        raise ValueError(
            f"Sample count mismatch in {path}: {len(sample_ids)} != expected {expected_n}"
        )
    return sample_ids


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


def _read_ids_from_ind(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            ids.append(line.split()[0])
    return ids


def _map_genotypes(raw_ascii: np.ndarray, ploidy: str) -> np.ndarray:
    vals = raw_ascii.astype(np.int16) - ord("0")
    out = np.full(vals.shape, MISSING_VALUE, dtype=np.int8)
    observed = vals != 9

    if ploidy == "diploid":
        valid = observed & ((vals == 0) | (vals == 1) | (vals == 2))
        out[valid] = vals[valid].astype(np.int8)
    elif ploidy == "pseudohaploid":
        valid = observed & ((vals == 0) | (vals == 1) | (vals == 2))
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

    first_line = _first_nonempty_line(geno_path)
    n_samples = len(first_line)
    n_snps = _count_nonempty_lines(geno_path)

    if ind_path is not None:
        n_ind = _count_nonempty_lines(ind_path)
        if n_ind != n_samples:
            raise ValueError(
                f".ind count ({n_ind}) does not match .geno sample width ({n_samples})"
            )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    memmap_path = out_prefix.with_suffix(".genotypes.int8.memmap")
    mm = np.memmap(memmap_path, dtype=np.int8, mode="w+", shape=(n_samples, n_snps))

    snp_idx = 0
    with geno_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if len(line) != n_samples:
                raise ValueError(
                    f"Inconsistent line width at SNP row {snp_idx}: "
                    f"{len(line)} vs expected {n_samples}"
                )
            ascii_vals = np.frombuffer(line.encode("ascii"), dtype=np.uint8)
            mm[:, snp_idx] = _map_genotypes(ascii_vals, ploidy)
            snp_idx += 1

    mm.flush()

    sample_ids_path = None
    if ind_path is not None:
        ids = _read_ids_from_ind(ind_path)
        if len(ids) != n_samples:
            raise ValueError(
                f"Sample ID count mismatch in {ind_path}: {len(ids)} != {n_samples}"
            )
        sample_ids_path = str(out_prefix.with_suffix(".sample_ids.txt"))
        with Path(sample_ids_path).open("w", encoding="utf-8") as handle:
            for sid in ids:
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
    idx = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(idx)

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    return {
        "train": idx[:train_end],
        "val": idx[train_end:val_end],
        "test": idx[val_end:],
    }


def compute_observed_fraction(geno_mm: np.memmap, chunk_size: int = 128) -> np.ndarray:
    n_samples = geno_mm.shape[0]
    out = np.zeros(n_samples, dtype=np.float32)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        block = np.asarray(geno_mm[start:end, :], dtype=np.int16)
        out[start:end] = (block >= 0).mean(axis=1).astype(np.float32)
    return out


def build_random_window_batch(
    geno_mm: np.memmap,
    sample_indices: np.ndarray,
    window_size: int,
    windows_per_sample: int,
    n_classes: int,
    rng: np.random.Generator,
    mask_prob: float,
    missingness_dropout: float,
    apply_mask: bool,
    apply_missingness_dropout: bool,
    device: "torch.device",
    window_starts: Optional[np.ndarray] = None,
) -> Dict[str, "torch.Tensor"]:
    import torch

    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if windows_per_sample <= 0:
        raise ValueError("windows_per_sample must be > 0")

    n_snps = int(geno_mm.shape[1])
    bsz = int(sample_indices.shape[0])
    max_start = max(1, n_snps - window_size + 1)

    if window_starts is None:
        starts = rng.integers(
            low=0,
            high=max_start,
            size=(bsz, windows_per_sample),
            dtype=np.int64,
        )
    else:
        starts = np.asarray(window_starts, dtype=np.int64)
        if starts.shape != (bsz, windows_per_sample):
            raise ValueError(
                "window_starts shape mismatch: "
                f"got {starts.shape}, expected {(bsz, windows_per_sample)}"
            )

    missing_token = n_classes
    mask_token = n_classes + 1

    tokens = np.full((bsz, windows_per_sample, window_size), missing_token, dtype=np.int64)
    genotypes = np.full((bsz, windows_per_sample, window_size), MISSING_VALUE, dtype=np.int16)
    obs_mask = np.zeros((bsz, windows_per_sample, window_size), dtype=np.float32)
    train_mask = np.zeros((bsz, windows_per_sample, window_size), dtype=np.float32)
    snp_idx = np.zeros((bsz, windows_per_sample, window_size), dtype=np.int64)

    for bi, sample_idx in enumerate(sample_indices.tolist()):
        for wi, start in enumerate(starts[bi].tolist()):
            end = min(start + window_size, n_snps)
            block = np.asarray(geno_mm[sample_idx, start:end], dtype=np.int16)
            wlen = int(end - start)

            observed = block >= 0
            if wlen > 0:
                token_view = tokens[bi, wi, :wlen]
                geno_view = genotypes[bi, wi, :wlen]
                obs_view = obs_mask[bi, wi, :wlen]
                snp_view = snp_idx[bi, wi, :wlen]

                token_view[observed] = block[observed].astype(np.int64)
                geno_view[observed] = block[observed]
                obs_view[:] = observed.astype(np.float32)
                snp_view[:] = start + np.arange(wlen, dtype=np.int64)

    if apply_mask:
        observed = obs_mask > 0.5
        mask_draw = rng.random(observed.shape)
        mask_targets = observed & (mask_draw < mask_prob)

        train_mask[mask_targets] = 1.0
        tokens[mask_targets] = mask_token

        if apply_missingness_dropout and missingness_dropout > 0:
            drop_draw = rng.random(observed.shape)
            dropped = observed & (~mask_targets) & (drop_draw < missingness_dropout)
            tokens[dropped] = missing_token
            obs_mask[dropped] = 0.0

    targets = np.full(tokens.shape, -100, dtype=np.int64)
    target_positions = train_mask > 0.5
    targets[target_positions] = genotypes[target_positions].astype(np.int64)

    n_seq = bsz * windows_per_sample
    sample_rep = np.repeat(sample_indices.astype(np.int64), windows_per_sample)

    batch = {
        "tokens": torch.from_numpy(tokens.reshape(n_seq, window_size)).to(
            device=device,
            dtype=torch.long,
        ),
        "targets": torch.from_numpy(targets.reshape(n_seq, window_size)).to(
            device=device,
            dtype=torch.long,
        ),
        "obs_mask": torch.from_numpy(obs_mask.reshape(n_seq, window_size)).to(
            device=device,
            dtype=torch.float32,
        ),
        "train_mask": torch.from_numpy(train_mask.reshape(n_seq, window_size)).to(
            device=device,
            dtype=torch.float32,
        ),
        "snp_idx": torch.from_numpy(snp_idx.reshape(n_seq, window_size)).to(
            device=device,
            dtype=torch.long,
        ),
        "sample_idx": torch.from_numpy(sample_rep).to(device=device, dtype=torch.long),
        "window_start": torch.from_numpy(starts.reshape(n_seq)).to(device=device, dtype=torch.long),
    }
    return batch
