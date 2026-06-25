#!/usr/bin/env python3
"""Run LD-pruned PLINK PCA followed by LLE on an EIGENSTRAT dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

GENETIC_ID_PREFIX = "Genetic ID"
GROUP_ID_COLUMN = "Group ID"
ORIGINAL_GROUP_ID_COLUMN = "Original Group ID"
POLITICAL_ENTITY_COLUMN = "Political Entity"
DATE_MEAN_PREFIX = "Date mean in BP"


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"Missing {label}: {path}")


def resolve_executable(explicit: str | None, env_name: str, names: Sequence[str]) -> str:
    if explicit:
        path = Path(explicit)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        fail(f"Executable not found or not executable for --{env_name.lower()}: {path}")
    env_value = os.environ.get(env_name)
    if env_value:
        path = Path(env_value)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        found = shutil.which(env_value)
        if found:
            return found
        fail(f"{env_name} is set but not executable/found: {env_value}")
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    fail(f"Could not find executable. Tried env {env_name} and names: {', '.join(names)}")


def run_cmd(cmd: Sequence[str], *, cwd: Path | None = None, log_path: Path | None = None) -> None:
    log("+ " + " ".join(str(x) for x in cmd))
    started = time.time()
    result = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")
    if result.stdout:
        print(result.stdout[-6000:], flush=True)
    if result.returncode != 0:
        where = f" See {log_path}" if log_path else ""
        fail(f"Command failed with exit code {result.returncode} after {elapsed:.1f}s.{where}")
    log(f"Command completed in {elapsed:.1f}s")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def read_ind(path: Path) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw in enumerate(handle, start=1):
            parts = raw.split()
            if not parts:
                continue
            if len(parts) < 3:
                fail(f"Malformed .ind line {line_no} in {path}")
            rows.append((parts[0], parts[1], parts[2]))
    if not rows:
        fail(f"No samples found in .ind: {path}")
    return rows


def write_subset_ind(src_ind: Path, dst_ind: Path, max_samples: int) -> List[str]:
    rows = read_ind(src_ind)
    if max_samples > 0:
        rows = rows[:max_samples]
    dst_ind.parent.mkdir(parents=True, exist_ok=True)
    with dst_ind.open("w", encoding="utf-8") as handle:
        for sid, sex, group in rows:
            handle.write(f"{sid}\t{sex}\t{group}\n")
    return [sid for sid, _, _ in rows]


def write_subset_snp(src_snp: Path, dst_snp: Path, max_snps: int) -> int:
    n = 0
    dst_snp.parent.mkdir(parents=True, exist_ok=True)
    with src_snp.open("r", encoding="utf-8", errors="ignore") as src, dst_snp.open("w", encoding="utf-8") as dst:
        for raw in src:
            if not raw.strip():
                continue
            if max_snps > 0 and n >= max_snps:
                break
            dst.write(raw)
            n += 1
    if n == 0:
        fail(f"SNP subset is empty from {src_snp}")
    return n


def convert_eigenstrat(
    *,
    convertf_bin: str,
    geno: Path,
    snp: Path,
    ind: Path,
    out_prefix: Path,
    output_format: str,
    packedped: bool = False,
    newindivname: Path | None = None,
    newsnpname: Path | None = None,
) -> Tuple[Path, Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if packedped:
        geno_out = out_prefix.with_suffix(".bed")
        snp_out = out_prefix.with_suffix(".bim")
        ind_out = out_prefix.with_suffix(".fam")
    else:
        geno_out = out_prefix.with_suffix(".geno")
        snp_out = out_prefix.with_suffix(".snp")
        ind_out = out_prefix.with_suffix(".ind")
    par = out_prefix.with_suffix(f".{output_format.lower()}.par")
    lines = [
        f"genotypename: {geno}",
        f"snpname: {snp}",
        f"indivname: {ind}",
        f"outputformat: {output_format}",
        f"genooutfilename: {geno_out}",
        f"snpoutfilename: {snp_out}",
        f"indoutfilename: {ind_out}",
        f"genotypeoutname: {geno_out}",
        f"snpoutname: {snp_out}",
        f"indivoutname: {ind_out}",
        "familynames: NO",
        "outputgroup: YES",
        "checksizemode: NO",
    ]
    if newindivname is not None:
        lines.append(f"newindivname: {newindivname}")
    if newsnpname is not None:
        lines.append(f"newsnpname: {newsnpname}")
    par.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_cmd([convertf_bin, "-p", par], log_path=out_prefix.with_suffix(".convertf.log"))
    for produced in (geno_out, snp_out, ind_out):
        require_file(produced, f"convertf {output_format} output")
    return geno_out, snp_out, ind_out


def prepare_plink_bed(
    *,
    convertf_bin: str,
    plink_bin: str,
    geno: Path,
    snp: Path,
    ind: Path,
    work_dir: Path,
    max_samples: int,
    max_snps: int,
) -> Tuple[Path, List[str], int, int, Dict[str, str]]:
    original_n_samples = count_lines(ind)
    original_n_snps = count_lines(snp)
    all_sample_ids = [sid for sid, _, _ in read_ind(ind)]
    sample_ids = list(all_sample_ids)
    conversion_inputs = {"geno": str(geno), "snp": str(snp), "ind": str(ind)}

    source_geno, source_snp, source_ind = geno, snp, ind
    if max_snps > 0:
        subset_dir = work_dir / "subset_inputs"
        subset_snp = subset_dir / "subset.snp"
        subset_snps = write_subset_snp(snp, subset_snp, max_snps)
        subset_prefix = subset_dir / "subset_packed"
        log(f"Creating SNP smoke subset: samples={len(sample_ids)} snps={subset_snps}")
        source_geno, source_snp, source_ind = convert_eigenstrat(
            convertf_bin=convertf_bin,
            geno=geno,
            snp=snp,
            ind=ind,
            out_prefix=subset_prefix,
            output_format="PACKEDANCESTRYMAP",
            newsnpname=subset_snp,
        )
        conversion_inputs = {"geno": str(source_geno), "snp": str(source_snp), "ind": str(source_ind)}

    plink_prefix = work_dir / "plink_input" / "dataset"
    bed, bim, fam = convert_eigenstrat(
        convertf_bin=convertf_bin,
        geno=source_geno,
        snp=source_snp,
        ind=source_ind,
        out_prefix=plink_prefix,
        output_format="PACKEDPED",
        packedped=True,
    )
    if max_samples > 0:
        sample_ids = all_sample_ids[:max_samples]
        keep_path = work_dir / "plink_input" / "sample_keep.txt"
        fam_pairs: Dict[str, Tuple[str, str]] = {}
        with fam.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                parts = raw.split()
                if len(parts) >= 2:
                    fam_pairs[parts[1]] = (parts[0], parts[1])
        missing_keep = [sid for sid in sample_ids if sid not in fam_pairs]
        if missing_keep:
            fail(f"PLINK FAM missing keep samples. Examples: {', '.join(missing_keep[:10])}")
        with keep_path.open("w", encoding="utf-8") as handle:
            for sid in sample_ids:
                fid, iid = fam_pairs[sid]
                handle.write(f"{fid}\t{iid}\n")
        kept_prefix = work_dir / "plink_input" / "dataset_kept"
        run_cmd([plink_bin, "--bfile", plink_prefix, "--allow-no-sex", "--keep", keep_path, "--make-bed", "--out", kept_prefix], log_path=kept_prefix.with_suffix(".plink_keep.log"))
        plink_prefix = kept_prefix
        bed, bim, fam = kept_prefix.with_suffix(".bed"), kept_prefix.with_suffix(".bim"), kept_prefix.with_suffix(".fam")
        for produced in (bed, bim, fam):
            require_file(produced, "PLINK kept output")

    return plink_prefix, sample_ids, len(sample_ids), count_lines(source_snp), {
        **conversion_inputs,
        "original_n_samples": str(original_n_samples),
        "original_n_snps": str(original_n_snps),
        "bed": str(bed),
        "bim": str(bim),
        "fam": str(fam),
    }


def run_plink_ld_pca(
    *,
    plink_bin: str,
    bfile_prefix: Path,
    out_dir: Path,
    window: int,
    step: int,
    r2: float,
    pca_dims: int,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prune_prefix = out_dir / "plink_prune"
    run_cmd(
        [plink_bin, "--bfile", bfile_prefix, "--allow-no-sex", "--indep-pairwise", window, step, r2, "--out", prune_prefix],
        log_path=out_dir / "plink_prune.log",
    )
    prune_in = prune_prefix.with_suffix(".prune.in")
    require_file(prune_in, "PLINK prune.in")
    if count_lines(prune_in) == 0:
        fail(f"PLINK prune list is empty: {prune_in}")
    pca_prefix = out_dir / "plink_pca"
    run_cmd(
        [
            plink_bin,
            "--bfile",
            bfile_prefix,
            "--allow-no-sex",
            "--extract",
            prune_in,
            "--pca",
            pca_dims,
            "--out",
            pca_prefix,
        ],
        log_path=out_dir / "plink_pca.log",
    )
    eigenvec = pca_prefix.with_suffix(".eigenvec")
    eigenval = pca_prefix.with_suffix(".eigenval")
    require_file(eigenvec, "PLINK PCA eigenvec")
    require_file(eigenval, "PLINK PCA eigenval")
    return prune_in, eigenvec, eigenval


def load_plink_eigenvec(eigenvec: Path, sample_ids: Sequence[str], pca_dims: int) -> np.ndarray:
    rows: Dict[str, List[float]] = {}
    with eigenvec.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline()
        if not first:
            fail(f"Empty eigenvec file: {eigenvec}")
        pending = [first]
        for raw in pending + list(handle):
            parts = raw.strip().split()
            if not parts:
                continue
            if parts[0].upper() in {"FID", "#FID"}:
                continue
            if len(parts) < 2 + pca_dims:
                fail(f"PCA row has {len(parts)-2} PCs, expected {pca_dims}: {raw[:120]}")
            iid = parts[1]
            vals = [float(x) for x in parts[2 : 2 + pca_dims]]
            rows[iid] = vals
    missing = [sid for sid in sample_ids if sid not in rows]
    if missing:
        preview = ", ".join(missing[:10])
        fail(f"PCA eigenvec missing {len(missing)} sample IDs. Examples: {preview}")
    arr = np.asarray([rows[sid] for sid in sample_ids], dtype=np.float64)
    if arr.shape != (len(sample_ids), pca_dims):
        fail(f"Unexpected PCA feature shape: {arr.shape}, expected {(len(sample_ids), pca_dims)}")
    if not np.isfinite(arr).all():
        fail("PCA feature matrix contains non-finite values")
    return arr


def normalize_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (x - mean) / std


def run_lle(features: np.ndarray, n_neighbors: int, n_components: int, seed: int) -> np.ndarray:
    try:
        from sklearn.manifold import LocallyLinearEmbedding
    except ImportError as exc:
        fail("scikit-learn is required for LLE but is not installed in the active environment")
        raise exc
    n_samples = features.shape[0]
    if n_samples <= n_neighbors:
        fail(f"Need more samples than n_neighbors for LLE: samples={n_samples}, n_neighbors={n_neighbors}")

    def fit_with_solver(solver: str) -> np.ndarray:
        log(f"Running LLE with eigen_solver={solver}")
        reducer = LocallyLinearEmbedding(
            n_neighbors=n_neighbors,
            n_components=n_components,
            method="standard",
            eigen_solver=solver,
            random_state=seed,
            n_jobs=-1,
        )
        return reducer.fit_transform(features).astype(np.float32)

    try:
        latents = fit_with_solver("auto")
    except ValueError as exc:
        message = str(exc)
        if "ARPACK" not in message and "null-space" not in message and "singular" not in message:
            raise
        log(f"LLE auto/ARPACK failed with a singular null-space solve; retrying dense solver. Original error: {message}")
        latents = fit_with_solver("dense")

    if not np.isfinite(latents).all():
        fail("LLE produced non-finite values")
    return latents


def parse_packed_header(geno_path: Path, n_individuals: int) -> Tuple[str, int, int]:
    rlen = max((n_individuals * 2 + 7) // 8, 48)
    with geno_path.open("rb") as handle:
        header = handle.read(rlen).decode("ascii", errors="ignore").strip("\x00").strip()
    parts = header.split()
    if len(parts) < 3 or parts[0] != "GENO":
        fail(f"Input .geno does not have packed GENO header: {geno_path}")
    return header, int(parts[1]), int(parts[2])


def build_unpack_lookup() -> np.ndarray:
    lookup = np.empty((256, 4), dtype=np.uint8)
    for byte in range(256):
        lookup[byte, 0] = (byte >> 6) & 3
        lookup[byte, 1] = (byte >> 4) & 3
        lookup[byte, 2] = (byte >> 2) & 3
        lookup[byte, 3] = byte & 3
    return lookup


def compute_observed_fraction(geno_path: Path, n_individuals: int, n_snps: int, chunk_size: int, max_snps: int) -> np.ndarray:
    use_snps = n_snps if max_snps <= 0 else min(max_snps, n_snps)
    rlen = max((n_individuals * 2 + 7) // 8, 48)
    expected = rlen * (n_snps + 1)
    actual = geno_path.stat().st_size
    if actual != expected:
        fail(f"Packed geno size mismatch for {geno_path}: expected={expected}, actual={actual}")
    lookup = build_unpack_lookup()
    packed = np.memmap(geno_path, dtype=np.uint8, mode="r", offset=rlen, shape=(n_snps, rlen))
    full_bytes = n_individuals // 4
    rem = n_individuals % 4
    counts = np.zeros(n_individuals, dtype=np.uint32)
    for start in range(0, use_snps, chunk_size):
        end = min(start + chunk_size, use_snps)
        block = packed[start:end]
        if full_bytes:
            expanded = lookup[block[:, :full_bytes]]
            counts[: full_bytes * 4] += (expanded != 3).sum(axis=0, dtype=np.uint32).reshape(-1)
        if rem:
            tail = lookup[block[:, full_bytes], :rem]
            counts[full_bytes * 4 :] += (tail != 3).sum(axis=0, dtype=np.uint32)
        if end == use_snps or end % 100000 == 0:
            log(f"Coverage progress: {end:,}/{use_snps:,} SNPs")
    return counts.astype(np.float64) / float(use_snps)


def normalize_anno_header(header: Sequence[str]) -> List[str]:
    out = [h.strip() for h in header]
    if out and (out[0] == "" or out[0].startswith(GENETIC_ID_PREFIX)):
        out[0] = "Genetic ID"
    return out


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
            if sid:
                rows[sid] = row
    return rows


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


def find_key(row: Dict[str, str], exact: str | None = None, prefix: str | None = None) -> str | None:
    if exact and exact in row:
        return exact
    if prefix:
        for key in row:
            if key.startswith(prefix):
                return key
    return None


def build_sample_stats(sample_ids: Sequence[str], observed: np.ndarray, anno: Path, output_path: Path) -> List[Dict[str, object]]:
    anno_index = read_anno_index(anno)
    missing = [sid for sid in sample_ids if sid not in anno_index]
    if missing:
        fail(f"Missing anno rows for {len(missing)} samples. Examples: {', '.join(missing[:10])}")
    rows: List[Dict[str, object]] = []
    for idx, sid in enumerate(sample_ids):
        meta = anno_index[sid]
        group = clean_value(meta.get(GROUP_ID_COLUMN))
        original = clean_value(meta.get(ORIGINAL_GROUP_ID_COLUMN)) or group
        political = clean_value(meta.get(POLITICAL_ENTITY_COLUMN))
        country = political or group
        date_key = find_key(meta, prefix=DATE_MEAN_PREFIX)
        rows.append(
            {
                "sample_id": sid,
                "observed_fraction": float(observed[idx]),
                "country": country,
                "original_group_id": original,
                "date_mean_bp": parse_float(meta.get(date_key) if date_key else None),
            }
        )
    write_tsv(output_path, rows, ["sample_id", "observed_fraction", "country", "original_group_id", "date_mean_bp"])
    return rows


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pairwise_sq_dist(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x2 = (x * x).sum(axis=1, keepdims=True)
    dist = np.maximum(x2 + x2.T - 2.0 * x @ x.T, 0.0)
    np.fill_diagonal(dist, np.inf)
    return dist


def nearest_neighbors_with_distances(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    dist = pairwise_sq_dist(x)
    k = min(k, x.shape[0] - 1)
    idx = np.argpartition(dist, kth=k, axis=1)[:, :k]
    row = np.arange(x.shape[0])[:, None]
    order = np.argsort(dist[row, idx], axis=1)
    idx = idx[row, order]
    return idx, dist[row, idx]


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    x = x[finite].astype(np.float64)
    y = y[finite].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    return pearson_corr(rankdata_average(x[finite]), rankdata_average(y[finite]))


def ridge_probe_r2(x: np.ndarray, y: np.ndarray, seed: int = 42) -> float:
    finite = np.isfinite(y)
    x = np.asarray(x[finite], dtype=np.float64)
    y = np.asarray(y[finite], dtype=np.float64)
    if x.shape[0] < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    split = max(int(0.8 * x.shape[0]), 1)
    train, test = order[:split], order[split:]
    if test.size == 0:
        test = train
    x_mean = x[train].mean(axis=0, keepdims=True)
    x_std = x[train].std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    xt = (x[train] - x_mean) / x_std
    xv = (x[test] - x_mean) / x_std
    yt = y[train]
    yv = y[test]
    y_mean = yt.mean()
    best = float("-inf")
    for alpha in (0.01, 0.1, 1.0, 10.0):
        coef = np.linalg.solve(xt.T @ xt + np.eye(xt.shape[1]) * alpha, xt.T @ (yt - y_mean))
        pred = xv @ coef + y_mean
        ss_res = float(((yv - pred) ** 2).sum())
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        if np.isfinite(r2) and r2 > best:
            best = float(r2)
    return best


def macro_knn_purity(labels: Sequence[str], neighbors: np.ndarray, min_group_size: int = 1) -> float:
    arr = np.asarray(labels, dtype=object)
    counts = Counter(str(x) for x in arr if str(x))
    per_label: Dict[str, List[float]] = {}
    for i, label_raw in enumerate(arr):
        label = str(label_raw)
        if not label or counts[label] < min_group_size:
            continue
        per_label.setdefault(label, []).append(float((arr[neighbors[i]] == label).mean()))
    return float(np.mean([np.mean(v) for v in per_label.values()])) if per_label else float("nan")


def adaptive_group_knn_purity(labels: Sequence[str], neighbors: np.ndarray, dist2: np.ndarray, min_group_size: int = 1) -> float:
    arr = np.asarray(labels, dtype=object)
    counts = Counter(str(x) for x in arr if str(x))
    per_label: Dict[str, List[float]] = {}
    for i, label_raw in enumerate(arr):
        label = str(label_raw)
        if not label or counts[label] < min_group_size:
            continue
        same = arr[neighbors[i]] == label
        same_count = int(same.sum())
        available_same = max(counts[label] - 1, 0)
        purity = float(same_count / max(neighbors.shape[1], 1))
        if 0 < available_same <= neighbors.shape[1] and same_count == available_same:
            furthest_same = float(np.max(dist2[i][same]))
            keep = same | (dist2[i] <= furthest_same)
            if int(keep.sum()) > 0:
                purity = float(same_count / int(keep.sum()))
        per_label.setdefault(label, []).append(purity)
    return float(np.mean([np.mean(v) for v in per_label.values()])) if per_label else float("nan")


def date_neighbor_metrics(dates: np.ndarray, neighbors: np.ndarray) -> Tuple[float, float]:
    maes: List[float] = []
    source: List[float] = []
    neigh_mean: List[float] = []
    for i, date in enumerate(dates):
        if not np.isfinite(date):
            continue
        nd = dates[neighbors[i]]
        nd = nd[np.isfinite(nd)]
        if nd.size == 0:
            continue
        maes.append(float(np.mean(np.abs(nd - date))))
        source.append(float(date))
        neigh_mean.append(float(nd.mean()))
    if not maes:
        return float("nan"), float("nan")
    return float(np.mean(maes)), spearman_corr(np.asarray(source), np.asarray(neigh_mean))


def evaluate_latents(latents: np.ndarray, sample_stats: Sequence[Dict[str, object]], ks: Sequence[int], seed: int) -> Dict[str, float]:
    coverage = np.asarray([float(row["observed_fraction"]) for row in sample_stats], dtype=np.float64)
    country = [str(row.get("country", "")) for row in sample_stats]
    original = [str(row.get("original_group_id", "")) for row in sample_stats]
    dates = np.asarray([float(row.get("date_mean_bp", float("nan"))) for row in sample_stats], dtype=np.float64)
    metrics: Dict[str, float] = {"coverage_r2": ridge_probe_r2(latents, coverage, seed=seed)}
    for k in ks:
        neighbors, dist2 = nearest_neighbors_with_distances(latents, k)
        metrics[f"country_macro_knn_purity@{k}"] = macro_knn_purity(country, neighbors)
        metrics[f"original_group_knn_purity@{k}"] = macro_knn_purity(original, neighbors)
        metrics[f"original_group_knn_purity@{k}_non_singleton"] = macro_knn_purity(original, neighbors, min_group_size=2)
        metrics[f"original_group_adaptive_knn_purity@{k}"] = adaptive_group_knn_purity(original, neighbors, dist2)
        metrics[f"original_group_adaptive_knn_purity@{k}_non_singleton"] = adaptive_group_knn_purity(original, neighbors, dist2, min_group_size=2)
        date_mae, date_spear = date_neighbor_metrics(dates, neighbors)
        metrics[f"date_neighbor_mae@{k}"] = date_mae
        metrics[f"date_neighbor_spearman@{k}"] = date_spear
    return metrics


def write_latents_csv(path: Path, sample_ids: Sequence[str], latents: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id"] + [f"z{i+1}" for i in range(latents.shape[1])])
        for sid, row in zip(sample_ids, latents):
            writer.writerow([sid] + [f"{float(x):.9g}" for x in row])


def write_metrics_csv(path: Path, metrics: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])


def write_plot_html(path: Path, latents: np.ndarray, sample_stats: Sequence[Dict[str, object]], color_key: str, title: str) -> None:
    coords = latents[:, :3]
    labels = [row.get(color_key, "") for row in sample_stats]
    if color_key == "date_mean_bp" or color_key == "observed_fraction":
        marker = {"size": 3, "color": labels, "colorscale": "Viridis", "showscale": True}
    else:
        unique = {str(v): i for i, v in enumerate(sorted({str(v) for v in labels}))}
        marker = {"size": 3, "color": [unique[str(v)] for v in labels], "colorscale": "Turbo", "showscale": False}
    custom = [str(v) for v in labels]
    payload = {
        "x": coords[:, 0].astype(float).tolist(),
        "y": coords[:, 1].astype(float).tolist(),
        "z": coords[:, 2].astype(float).tolist(),
        "mode": "markers",
        "type": "scatter3d",
        "marker": marker,
        "text": custom,
        "hovertemplate": "%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
    }
    html = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>{title}</title><script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script></head>
<body><div id=\"plot\" style=\"width:100%;height:95vh;\"></div>
<script>Plotly.newPlot('plot', [{json.dumps(payload)}], {{title: {json.dumps(title)}, margin: {{l:0,r:0,b:0,t:40}}}}, {{responsive: true}});</script>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def write_experiment_docs(run_dir: Path, config: Dict[str, object], metrics: Dict[str, float], artifacts: Dict[str, str]) -> None:
    hypothesis = """# Hypothesis

## Problem

Spectral analysis alone does not yield good enough results without contrastive learning using labels.

## Motivation

The initial LD-pruned PCA + LLE run looked weak, with original-group purity scores that were comparably bad to prior non-contrastive runs. This grid search tests whether the failure is caused by hyperparameters, or whether LLE is the wrong method for this sparse aDNA setting. Direct LLE on the SNP matrix remains computationally expensive, so every grid point first uses LD pruning and PCA to make the input feasible.

## Hypothesis

If the first LLE result failed because of preprocessing or neighborhood hyperparameters, then changing PCA dimensionality, LD-pruning strength, or LLE neighborhood size should substantially improve original-group locality.

## Success Criteria

A grid point shows a significant increase in `original_group_adaptive_knn_purity`, especially at the primary evaluation neighborhood, relative to the initial PCA100 / LD r2 0.2 / LLE k32 run.

## Failure Criteria

The grid points show equal or worse `original_group_adaptive_knn_purity` than the initial run, supporting the interpretation that LLE is likely the wrong method rather than merely poorly tuned.

## Next Planned Experiments

If a grid point improves clearly, inspect the embedding manually and consider OPTICS clustering. If the grid remains weak, deprioritize LLE and return to label-aware or graph/contrastive methods.
"""
    (run_dir / "hypothesis.md").write_text(hypothesis, encoding="utf-8")
    (run_dir / "config.yaml").write_text(to_simple_yaml(config), encoding="utf-8")
    links = ["# Links", "", "## Core Artifacts", ""]
    for name, value in artifacts.items():
        links.append(f"- `{name}`: `{value}`")
    (run_dir / "links.md").write_text("\n".join(links) + "\n", encoding="utf-8")
    lines = ["# Results", "", "## Status", "", "Completed LLE run on LD-pruned PCA features.", "", "## Metrics", ""]
    for key in sorted(metrics):
        val = metrics[key]
        lines.append(f"- `{key}`: {val:.6g}" if np.isfinite(val) else f"- `{key}`: NaN")
    (run_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_simple_yaml(data: object, indent: int = 0) -> str:
    sp = "  " * indent
    if isinstance(data, dict):
        lines: List[str] = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{sp}{key}:")
                lines.append(to_simple_yaml(value, indent + 1).rstrip())
            else:
                lines.append(f"{sp}{key}: {json.dumps(value) if isinstance(value, str) else value}")
        return "\n".join(lines) + "\n"
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(to_simple_yaml(item, indent + 1).rstrip())
            else:
                lines.append(f"{sp}- {json.dumps(item) if isinstance(item, str) else item}")
        return "\n".join(lines) + "\n"
    return f"{sp}{data}\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geno", required=True)
    parser.add_argument("--snp", required=True)
    parser.add_argument("--ind", required=True)
    parser.add_argument("--anno", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--plink_bin", default=None)
    parser.add_argument("--convertf_bin", default=None)
    parser.add_argument("--ld_window", type=int, default=200)
    parser.add_argument("--ld_step", type=int, default=25)
    parser.add_argument("--ld_r2", type=float, default=0.2)
    parser.add_argument("--pca_dims", type=int, default=100)
    parser.add_argument("--lle_neighbors", type=int, default=32)
    parser.add_argument("--lle_components", type=int, default=6)
    parser.add_argument("--feature_dim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_snps", type=int, default=-1)
    parser.add_argument("--coverage_chunk_size", type=int, default=20000)
    parser.add_argument("--dataset_label", default="v62.0_HO_public_europe_groupid_cov5to90_nonsingleton")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    geno = Path(args.geno).resolve()
    snp = Path(args.snp).resolve()
    ind = Path(args.ind).resolve()
    anno = Path(args.anno).resolve()
    for path, label in ((geno, "geno"), (snp, "snp"), (ind, "ind"), (anno, "anno")):
        require_file(path, label)
    if args.feature_dim != args.pca_dims:
        fail("feature_dim must match pca_dims because PLINK PCA output is the only LLE feature matrix")

    run_name = args.run_name or f"{datetime.now().strftime('%Y-%m-%d')}_ho-groupid-cov5to90-nonsingleton-ldpca{args.pca_dims}-lle-k{args.lle_neighbors}"
    output_root = Path(args.output_dir).resolve()
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    plink_bin = resolve_executable(args.plink_bin, "PLINK", ["plink", "plink1.9"])
    convertf_bin = resolve_executable(args.convertf_bin, "EIG_CONVERTF", ["convertf_local", "convertf"])
    log(f"Run directory: {run_dir}")
    log(f"PLINK: {plink_bin}")
    log(f"convertf: {convertf_bin}")

    plink_prefix, sample_ids, n_samples, n_snps, conversion_meta = prepare_plink_bed(
        convertf_bin=convertf_bin,
        plink_bin=plink_bin,
        geno=geno,
        snp=snp,
        ind=ind,
        work_dir=work_dir,
        max_samples=args.max_samples,
        max_snps=args.max_snps,
    )
    prune_in, eigenvec, eigenval = run_plink_ld_pca(
        plink_bin=plink_bin,
        bfile_prefix=plink_prefix,
        out_dir=run_dir,
        window=args.ld_window,
        step=args.ld_step,
        r2=args.ld_r2,
        pca_dims=args.pca_dims,
    )
    for src, dst_name in ((prune_in, "plink_prune.prune.in"), (eigenvec, "plink_pca.eigenvec"), (eigenval, "plink_pca.eigenval")):
        dst = run_dir / dst_name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    features = load_plink_eigenvec(eigenvec, sample_ids, args.pca_dims)
    np.save(run_dir / "pca_features.npy", features.astype(np.float32))
    latents = run_lle(normalize_features(features), args.lle_neighbors, args.lle_components, args.seed)
    np.save(run_dir / "lle_latents.npy", latents)
    write_latents_csv(run_dir / "lle_latents.csv", sample_ids, latents)

    cov_geno = Path(conversion_meta["geno"])
    cov_ind = Path(conversion_meta["ind"])
    cov_n_samples = count_lines(cov_ind)
    header, header_nind, header_nsnp = parse_packed_header(cov_geno, cov_n_samples)
    if header_nind != cov_n_samples:
        fail(f"Packed header sample count mismatch: header={header_nind}, ind={cov_n_samples}")
    observed = compute_observed_fraction(cov_geno, cov_n_samples, header_nsnp, args.coverage_chunk_size, -1)
    sample_stats = build_sample_stats(sample_ids, observed, anno, run_dir / "sample_stats.tsv")

    metrics = evaluate_latents(latents, sample_stats, ks=[args.lle_neighbors, 15], seed=args.seed)
    with (run_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    write_metrics_csv(run_dir / "metrics.csv", metrics)

    write_plot_html(run_dir / "lle_3d_original_group_id.html", latents, sample_stats, "original_group_id", "LLE colored by Original Group ID")
    write_plot_html(run_dir / "lle_3d_date_mean_bp.html", latents, sample_stats, "date_mean_bp", "LLE colored by Date BP")
    write_plot_html(run_dir / "lle_3d_observed_fraction.html", latents, sample_stats, "observed_fraction", "LLE colored by observed SNP fraction")

    config = {
        "experiment_name": run_name,
        "experiment_kind": "LD-pruned PCA + LLE",
        "status": "completed",
        "dataset_label": args.dataset_label,
        "inputs": {"geno": str(geno), "snp": str(snp), "ind": str(ind), "anno": str(anno)},
        "ld_pruning": {"window": args.ld_window, "step": args.ld_step, "r2": args.ld_r2},
        "pca_dims": args.pca_dims,
        "lle": {"n_neighbors": args.lle_neighbors, "n_components": args.lle_components, "feature_dim": args.feature_dim, "method": "standard"},
        "cluster": {"plink_bin": plink_bin, "convertf_bin": convertf_bin, "hostname": os.uname().nodename},
        "limits": {"max_samples": args.max_samples, "max_snps": args.max_snps},
    }
    artifacts = {
        "run_dir": str(run_dir),
        "plink_prune.prune.in": str(run_dir / "plink_prune.prune.in"),
        "plink_pca.eigenvec": str(run_dir / "plink_pca.eigenvec"),
        "lle_latents.npy": str(run_dir / "lle_latents.npy"),
        "evaluation_summary.json": str(run_dir / "evaluation_summary.json"),
    }
    write_experiment_docs(run_dir, config, metrics, artifacts)

    summary = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "n_snps_input": n_snps,
        "n_pruned_snps": count_lines(run_dir / "plink_prune.prune.in"),
        "pca_feature_shape": list(features.shape),
        "lle_latent_shape": list(latents.shape),
        "packed_header": header,
        "conversion": conversion_meta,
        "metrics": metrics,
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    log(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
