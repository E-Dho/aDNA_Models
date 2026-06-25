from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .geo import project_latlon_to_xy
from .plink import fail

GENETIC_ID_PREFIX = "Genetic ID"
DATE_MEAN_PREFIX = "Date mean in BP"
GROUP_ID_COLUMN = "Group ID"
ORIGINAL_GROUP_ID_COLUMN = "Original Group ID"
LOCALITY_COLUMN = "Locality"
POLITICAL_ENTITY_COLUMN = "Political Entity"
LAT_COLUMN = "Lat."
LON_COLUMN = "Long."


@dataclass(frozen=True)
class MetadataTable:
    sample_ids: List[str]
    original_group_id: List[str]
    group_id: List[str]
    locality: List[str]
    political_entity: List[str]
    date_bp: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    xy: np.ndarray
    projection: Dict[str, str]


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_sites: List[str]
    val_sites: List[str]
    test_sites: List[str]


def clean_value(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    return "" if value == ".." else value


def parse_float(value: str | None) -> float:
    value = clean_value(value)
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def normalize_anno_header(header: Sequence[str]) -> List[str]:
    out = [h.strip() for h in header]
    if out and (out[0] == "" or out[0].startswith(GENETIC_ID_PREFIX)):
        out[0] = "Genetic ID"
    return out


def find_key(row: Dict[str, str], exact: str | None = None, prefix: str | None = None) -> str | None:
    if exact and exact in row:
        return exact
    if prefix:
        for key in row:
            if key.startswith(prefix):
                return key
    return None


def read_anno_index(anno: Path) -> Dict[str, Dict[str, str]]:
    with anno.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = normalize_anno_header(next(reader))
        except StopIteration:
            fail(f"Empty anno file: {anno}")
        rows: Dict[str, Dict[str, str]] = {}
        for raw in reader:
            if not raw:
                continue
            if len(raw) < len(header):
                raw += [""] * (len(header) - len(raw))
            row = {header[i]: raw[i] for i in range(len(header))}
            sid = row.get("Genetic ID", "").strip()
            if not sid:
                continue
            if sid in rows:
                fail(f"Duplicate Genetic ID in anno: {sid}")
            rows[sid] = row
    return rows


def build_metadata(sample_ids: Sequence[str], anno: Path) -> MetadataTable:
    anno_index = read_anno_index(anno)
    missing = [sid for sid in sample_ids if sid not in anno_index]
    if missing:
        fail(f"Missing anno rows for {len(missing)} samples. Examples: {', '.join(missing[:10])}")

    original_group_id: List[str] = []
    group_id: List[str] = []
    locality: List[str] = []
    political_entity: List[str] = []
    date_bp: List[float] = []
    lat: List[float] = []
    lon: List[float] = []
    dropped: List[str] = []

    for sid in sample_ids:
        row = anno_index[sid]
        date_key = find_key(row, prefix=DATE_MEAN_PREFIX)
        original = clean_value(row.get(ORIGINAL_GROUP_ID_COLUMN)) or clean_value(row.get(GROUP_ID_COLUMN))
        group = clean_value(row.get(GROUP_ID_COLUMN))
        loc = clean_value(row.get(LOCALITY_COLUMN))
        pol = clean_value(row.get(POLITICAL_ENTITY_COLUMN))
        d = parse_float(row.get(date_key) if date_key else None)
        la = parse_float(row.get(LAT_COLUMN))
        lo = parse_float(row.get(LON_COLUMN))
        if not group or not loc or not np.isfinite(d) or not np.isfinite(la) or not np.isfinite(lo):
            dropped.append(sid)
        original_group_id.append(original)
        group_id.append(group)
        locality.append(loc)
        political_entity.append(pol)
        date_bp.append(d)
        lat.append(la)
        lon.append(lo)

    if dropped:
        fail(f"Samples with missing Group ID, Locality, date, or coordinates: {len(dropped)}. Examples: {', '.join(dropped[:10])}")

    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    projection = project_latlon_to_xy(lat_arr, lon_arr, crs="EPSG:3035")
    return MetadataTable(
        sample_ids=list(sample_ids),
        original_group_id=original_group_id,
        group_id=group_id,
        locality=locality,
        political_entity=political_entity,
        date_bp=np.asarray(date_bp, dtype=np.float64),
        lat=lat_arr,
        lon=lon_arr,
        xy=projection.xy,
        projection={"crs": projection.crs, "method": projection.method},
    )


def make_leave_site_split(
    localities: Sequence[str],
    seed: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> SplitIndices:
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")
    sites = sorted(set(localities))
    if len(sites) < 3:
        raise ValueError("Need at least three Locality values for train/val/test leave-site split")
    rng = np.random.default_rng(seed)
    shuffled = sites.copy()
    rng.shuffle(shuffled)
    n_sites = len(shuffled)
    n_train = max(1, int(round(n_sites * train_fraction)))
    n_val = max(1, int(round(n_sites * val_fraction)))
    if n_train + n_val >= n_sites:
        n_train = max(1, n_sites - 2)
        n_val = 1
    train_sites = sorted(shuffled[:n_train])
    val_sites = sorted(shuffled[n_train : n_train + n_val])
    test_sites = sorted(shuffled[n_train + n_val :])
    site_to_split = {s: "train" for s in train_sites}
    site_to_split.update({s: "val" for s in val_sites})
    site_to_split.update({s: "test" for s in test_sites})

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for i, site in enumerate(localities):
        split = site_to_split[site]
        if split == "train":
            train_idx.append(i)
        elif split == "val":
            val_idx.append(i)
        else:
            test_idx.append(i)
    if not train_idx or not val_idx or not test_idx:
        raise ValueError("leave-site split produced an empty train/val/test split")
    if set(train_sites) & set(val_sites) or set(train_sites) & set(test_sites) or set(val_sites) & set(test_sites):
        raise ValueError("leave-site split has locality leakage")
    return SplitIndices(
        train=np.asarray(train_idx, dtype=np.int64),
        val=np.asarray(val_idx, dtype=np.int64),
        test=np.asarray(test_idx, dtype=np.int64),
        train_sites=train_sites,
        val_sites=val_sites,
        test_sites=test_sites,
    )


def standardize_train_apply(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64)
    mean = arr[train_idx].mean(axis=0, keepdims=True)
    std = arr[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return ((arr - mean) / std).astype(np.float32), mean.reshape(-1), std.reshape(-1)


def write_sample_stats(path: Path, meta: MetadataTable, splits: SplitIndices, features: np.ndarray) -> None:
    split_names = np.full(len(meta.sample_ids), "", dtype=object)
    split_names[splits.train] = "train"
    split_names[splits.val] = "val"
    split_names[splits.test] = "test"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "sample_id",
            "split",
            "original_group_id",
            "group_id",
            "locality",
            "political_entity",
            "date_mean_bp",
            "latitude",
            "longitude",
            "x_m",
            "y_m",
            "age_scaled",
        ])
        for i, sid in enumerate(meta.sample_ids):
            writer.writerow([
                sid,
                split_names[i],
                meta.original_group_id[i],
                meta.group_id[i],
                meta.locality[i],
                meta.political_entity[i],
                f"{meta.date_bp[i]:.6f}",
                f"{meta.lat[i]:.8f}",
                f"{meta.lon[i]:.8f}",
                f"{meta.xy[i, 0]:.6f}",
                f"{meta.xy[i, 1]:.6f}",
                f"{features[i, -1]:.8f}",
            ])


def write_split_sites(path: Path, splits: SplitIndices) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_sites": splits.train_sites,
        "val_sites": splits.val_sites,
        "test_sites": splits.test_sites,
        "n_train_sites": len(splits.train_sites),
        "n_val_sites": len(splits.val_sites),
        "n_test_sites": len(splits.test_sites),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
