from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


MISSING_VALUE = -1
DATE_MEAN_COLUMN = (
    "Date mean in BP in years before 1950 CE [OxCal mu for a direct radiocarbon "
    "date, and average of range for a contextual date]"
)
GENETIC_ID_PREFIX = "Genetic ID"
GROUP_ID_COLUMN = "Group ID"
ORIGINAL_GROUP_ID_COLUMN = "Original Group ID"


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


def load_sample_ids_from_meta(meta: GenotypeMemmapMeta) -> List[str]:
    if meta.sample_ids_path is None:
        return [f"sample_{i}" for i in range(meta.n_samples)]
    path = Path(meta.sample_ids_path)
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def open_genotype_memmap(meta: GenotypeMemmapMeta, mode: str = "r") -> np.memmap:
    return np.memmap(
        meta.memmap_path,
        dtype=np.dtype(meta.dtype),
        mode=mode,
        shape=(meta.n_samples, meta.n_snps),
    )


def _map_genotypes(raw_ascii: np.ndarray, ploidy: str) -> np.ndarray:
    vals = raw_ascii.astype(np.int16) - ord("0")
    out = np.full(vals.shape, MISSING_VALUE, dtype=np.int8)
    observed = vals != 9
    valid = observed & ((vals == 0) | (vals == 1) | (vals == 2))
    if ploidy == "diploid":
        out[valid] = vals[valid].astype(np.int8)
    elif ploidy == "pseudohaploid":
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


def compute_observed_fraction(
    geno_mm: np.memmap,
    *,
    sample_indices: Optional[np.ndarray] = None,
    max_snps: int = -1,
    chunk_size: int = 256,
) -> np.ndarray:
    total_samples = int(geno_mm.shape[0])
    total_snps = int(geno_mm.shape[1])
    n_snps = total_snps if max_snps <= 0 else min(int(max_snps), total_snps)
    if sample_indices is None:
        selected = np.arange(total_samples, dtype=np.int64)
    else:
        selected = np.asarray(sample_indices, dtype=np.int64)
    out = np.zeros(int(selected.shape[0]), dtype=np.float32)
    for start in range(0, selected.shape[0], chunk_size):
        end = min(start + chunk_size, selected.shape[0])
        block = np.asarray(geno_mm[selected[start:end], :n_snps], dtype=np.int16)
        out[start:end] = (block >= 0).mean(axis=1).astype(np.float32)
    return out


def _normalize_anno_header(header: Sequence[str]) -> List[str]:
    normalized = [cell.strip() for cell in header]
    if normalized:
        first = normalized[0]
        if first == "" or first.startswith(GENETIC_ID_PREFIX):
            normalized[0] = "Genetic ID"
    return normalized


def read_anno_rows(anno_path: Path) -> List[Dict[str, str]]:
    with anno_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = _normalize_anno_header(next(reader))
        except StopIteration as exc:
            raise ValueError(f"Empty anno file: {anno_path}") from exc
        rows: List[Dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            rows.append({header[i]: row[i] for i in range(len(header))})
    return rows


def _clean_value(value: Optional[str]) -> str:
    if value is None:
        return ""
    stripped = value.strip()
    return "" if stripped == ".." else stripped


def _resolve_column_name(row: Dict[str, str], preferred: str, prefix: Optional[str] = None) -> Optional[str]:
    if preferred in row:
        return preferred
    if prefix is not None:
        for key in row:
            if key.startswith(prefix):
                return key
    return None


def _parse_optional_float(value: Optional[str]) -> float:
    cleaned = _clean_value(value)
    if not cleaned:
        return float("nan")
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def build_sample_metadata_index(anno_rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for row in anno_rows:
        sample_key = _resolve_column_name(row, "Genetic ID", GENETIC_ID_PREFIX)
        sample_id = _clean_value(row.get(sample_key) if sample_key else None)
        if sample_id:
            index[sample_id] = row
    return index


def build_sample_stats_rows(
    sample_ids: Sequence[str],
    observed_fraction: np.ndarray,
    anno_path: Path,
) -> List[Dict[str, object]]:
    if len(sample_ids) != int(observed_fraction.shape[0]):
        raise ValueError("sample_ids and observed_fraction length mismatch")
    anno_index = build_sample_metadata_index(read_anno_rows(anno_path))
    rows: List[Dict[str, object]] = []
    for idx, sample_id in enumerate(sample_ids):
        meta = anno_index.get(sample_id, {})
        group_key = _resolve_column_name(meta, GROUP_ID_COLUMN)
        original_group_key = _resolve_column_name(meta, ORIGINAL_GROUP_ID_COLUMN)
        date_key = _resolve_column_name(meta, DATE_MEAN_COLUMN, "Date mean in BP")
        rows.append(
            {
                "sample_id": sample_id,
                "observed_fraction": float(observed_fraction[idx]),
                "country": _clean_value(meta.get(group_key) if group_key else None),
                "original_group_id": _clean_value(meta.get(original_group_key) if original_group_key else None),
                "date_mean_bp": _parse_optional_float(meta.get(date_key) if date_key else None),
            }
        )
    return rows


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_sample_stats_tsv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: List[Dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "observed_fraction": float(row["observed_fraction"]),
                    "country": row.get("country", ""),
                    "original_group_id": row.get("original_group_id", ""),
                    "date_mean_bp": _parse_optional_float(row.get("date_mean_bp")),
                }
            )
    return rows
