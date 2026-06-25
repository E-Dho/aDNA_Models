#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geolocation_model.data import (  # noqa: E402
    build_metadata,
    find_key,
    make_leave_site_split,
    read_anno_index,
    standardize_train_apply,
    write_sample_stats,
    write_split_sites,
)
from geolocation_model.plink import (  # noqa: E402
    fail,
    load_plink_eigenvec,
    prepare_plink_bed,
    require_file,
    resolve_executable,
    run_plink_ld_pca,
)

AGE_BINS = [
    (0.0, 1000.0, "0-1000 BP"),
    (1000.0, 2500.0, "1000-2500 BP"),
    (2500.0, 4000.0, "2500-4000 BP"),
    (4000.0, 6000.0, "4000-6000 BP"),
    (6000.0, float("inf"), ">6000 BP"),
]
BASE_SEQ_TYPES = ["1240k", "Shotgun", "HumanOrigins", "BigYoruba", "ArchaicAdmixture", "WholeGenomeCapture"]


def log(message: str) -> None:
    print(message, flush=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def age_bin(age_bp: float) -> str:
    for lo, hi, label in AGE_BINS:
        if lo <= age_bp < hi:
            return label
    return "missing"


def error_km(pred_xy_km: np.ndarray, true_xy_km: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(pred_xy_km) - np.asarray(true_xy_km), axis=1)


def baseline_metrics(name: str, pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    err = error_km(pred, true)
    return {
        f"{name}_mean_km": float(err.mean()),
        f"{name}_median_km": float(np.median(err)),
        f"{name}_p90_km": float(np.percentile(err, 90)),
    }


def parse_seq_tokens(data_type: str) -> List[str]:
    toks = [x.strip() for x in data_type.replace(";", ",").split(",") if x.strip()]
    return toks or ["missing"]


def load_anno_features(sample_ids: Sequence[str], anno: Path) -> tuple[Dict[str, str], Dict[str, float], List[str]]:
    anno_index = read_anno_index(anno)
    data_type: Dict[str, str] = {}
    coverage: Dict[str, float] = {}
    found_cov_cols: List[str] = []
    for sid in sample_ids:
        row = anno_index.get(sid)
        if row is None:
            fail(f"Missing anno row for feature lookup: {sid}")
        dt_key = find_key(row, exact="Data type")
        data_type[sid] = (row.get(dt_key, "") if dt_key else "").strip() or "missing"
        cov_keys = [k for k in row if k.startswith("SNPs hit on autosomal targets")]
        if cov_keys and not found_cov_cols:
            found_cov_cols = cov_keys
        vals = []
        for key in cov_keys:
            raw = row.get(key, "").strip()
            try:
                val = float(raw)
                if np.isfinite(val):
                    vals.append(val)
            except ValueError:
                pass
        coverage[sid] = max(vals) if vals else float("nan")
    return data_type, coverage, found_cov_cols


def build_seq_multihot(data_types: Dict[str, str], sample_ids: Sequence[str]) -> tuple[np.ndarray, List[str]]:
    token_set = set(BASE_SEQ_TYPES)
    for sid in sample_ids:
        token_set.update(parse_seq_tokens(data_types[sid]))
    tokens = [t for t in BASE_SEQ_TYPES if t in token_set] + sorted(t for t in token_set if t not in set(BASE_SEQ_TYPES))
    arr = np.zeros((len(sample_ids), len(tokens)), dtype=np.float32)
    idx = {t: i for i, t in enumerate(tokens)}
    for r, sid in enumerate(sample_ids):
        for tok in parse_seq_tokens(data_types[sid]):
            arr[r, idx[tok]] = 1.0
    if not np.isfinite(arr).all():
        fail("Sequencing multi-hot contains non-finite values")
    return arr, tokens


def inverse_project_xy_km(xy_km: np.ndarray, projection: Dict[str, str]) -> np.ndarray:
    xy_m = np.asarray(xy_km, dtype=np.float64) * 1000.0
    crs = projection.get("crs", "")
    method = projection.get("method", "")
    if method != "pyproj" or not crs.startswith("EPSG"):
        return np.full((xy_m.shape[0], 2), np.nan, dtype=np.float64)
    try:
        from pyproj import Transformer
    except ImportError:
        return np.full((xy_m.shape[0], 2), np.nan, dtype=np.float64)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xy_m[:, 0], xy_m[:, 1])
    return np.column_stack([np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)])


def compute_knn_features(
    *,
    query_x: np.ndarray,
    query_global_idx: np.ndarray,
    ref_x: np.ndarray,
    ref_y_km: np.ndarray,
    ref_age: np.ndarray,
    ref_global_idx: np.ndarray,
    k: int,
    eps: float,
    power: float,
    exclude_self: bool,
    chunk_size: int = 512,
) -> Dict[str, np.ndarray]:
    if k < 1:
        fail("k must be >= 1")
    if k > 20:
        log("WARNING: d20 is defined as the 20th neighbor distance; for k < 20 it uses the furthest available neighbor")
    extra = 1 if exclude_self else 0
    if k + extra > len(ref_x):
        fail(f"Need at least k+1 training samples for leave-self-out kNN; k={k}, ref={len(ref_x)}")
    n = len(query_x)
    mu = np.empty((n, 2), dtype=np.float64)
    raw_sigma = np.empty(n, dtype=np.float64)
    mean_dist = np.empty(n, dtype=np.float64)
    min_dist = np.empty(n, dtype=np.float64)
    max_dist = np.empty(n, dtype=np.float64)
    std_dist = np.empty(n, dtype=np.float64)
    d1 = np.empty(n, dtype=np.float64)
    d5 = np.empty(n, dtype=np.float64)
    d10 = np.empty(n, dtype=np.float64)
    d20 = np.empty(n, dtype=np.float64)
    spatial_var = np.empty(n, dtype=np.float64)
    age_mean = np.empty(n, dtype=np.float64)
    age_std = np.empty(n, dtype=np.float64)
    first_neighbor_global = np.empty(n, dtype=np.int64)
    kth = k + extra - 1
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = query_x[start:end].astype(np.float64)
        dist2 = ((block[:, None, :] - ref_x[None, :, :].astype(np.float64)) ** 2).sum(axis=2)
        cand_idx = np.argpartition(dist2, kth=kth, axis=1)[:, : k + extra]
        for j in range(end - start):
            qi_global = int(query_global_idx[start + j])
            candidates = cand_idx[j]
            d = np.sqrt(dist2[j, candidates])
            if exclude_self:
                keep = ref_global_idx[candidates] != qi_global
                candidates = candidates[keep]
                d = d[keep]
            if len(candidates) < k:
                fail("Leave-self-out kNN produced fewer than k neighbors")
            order = np.argsort(d)[:k]
            nn = candidates[order]
            d = d[order]
            if exclude_self and np.any(ref_global_idx[nn] == qi_global):
                fail("Leakage: train kNN includes query sample as its own neighbor")
            weights = 1.0 / np.power(d + eps, power)
            weights = weights / weights.sum()
            coords = ref_y_km[nn]
            pred = (weights[:, None] * coords).sum(axis=0)
            delta = coords - pred[None, :]
            var = float((weights * np.sum(delta * delta, axis=1)).sum())
            ages = ref_age[nn]
            mu[start + j] = pred
            spatial_var[start + j] = var
            raw_sigma[start + j] = math.sqrt(max(var, 0.0) + eps)
            mean_dist[start + j] = float(d.mean())
            min_dist[start + j] = float(d.min())
            max_dist[start + j] = float(d.max())
            std_dist[start + j] = float(d.std())
            d1[start + j] = float(d[min(0, len(d) - 1)])
            d5[start + j] = float(d[min(4, len(d) - 1)])
            d10[start + j] = float(d[min(9, len(d) - 1)])
            d20[start + j] = float(d[min(19, len(d) - 1)])
            age_mean[start + j] = float(ages.mean())
            age_std[start + j] = float(ages.std())
            first_neighbor_global[start + j] = int(ref_global_idx[nn[0]])
    out = {
        "mu": mu,
        "raw_sigma": raw_sigma,
        "min_neighbor_distance": min_dist,
        "mean_neighbor_distance": mean_dist,
        "max_neighbor_distance": max_dist,
        "std_neighbor_distance": std_dist,
        "d1": d1,
        "d5": d5,
        "d10": d10,
        "d20": d20,
        "weighted_neighbor_age_mean": age_mean,
        "weighted_neighbor_age_std": age_std,
        "weighted_spatial_variance": spatial_var,
        "first_neighbor_global": first_neighbor_global,
    }
    for name, arr in out.items():
        if not np.isfinite(arr).all():
            fail(f"kNN feature contains non-finite values: {name}")
    return out


def standardize_from_train(train_values: np.ndarray, all_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((all_values - mean) / std).astype(np.float32), mean.reshape(-1), std.reshape(-1)


def write_logging_files(run_dir: Path, args: argparse.Namespace) -> None:
    write_text(
        run_dir / "hypothesis.md",
        """
# Hypothesis

## Problem

The previous learned-gate model successfully combined local kNN and global MLP information, but alpha was directly learned and therefore less interpretable as uncertainty.

## Motivation

Replace the direct sigmoid gate with uncertainty-based expert fusion. The model should learn each expert's uncertainty and combine the local kNN and global MLP by precision weighting.

## Hypothesis

The kNN uncertainty calibrator should assign high sigma_knn_corrected when the neighbor cloud is diffuse or unreliable and low sigma_knn_corrected when neighbors are geographically coherent. The MLP should learn high sigma_mlp when global prediction is uncertain. The derived alpha_precision should approach 1 when kNN is more reliable and 0 when MLP is more reliable.

## Success Criteria

Improve median error over weighted kNN while keeping p90 close to or better than the MLP expert.

## Failure Criteria

The precision fusion model is worse than weighted kNN on median error or worse than MLP on p90, or alpha_precision collapses without meaningful uncertainty behavior.

## Next Planned Experiments

Ablate sequencing features, coverage features, k values, and jitter sigma; compare k=5/10/20/50 and jitter 0/25/50.
""",
    )
    write_text(
        run_dir / "config.yaml",
        f"""
experiment_type: precision_weighted_knn_mlp
model_name: GeoPrecisionFusion
aliases: [precision_fusion]
dataset:
  geno: {args.geno}
  snp: {args.snp}
  ind: {args.ind}
  anno: {args.anno}
preprocessing:
  ld_pruning:
    window: {args.ld_window}
    step: {args.ld_step}
    r2: {args.ld_r2}
  pca_dims: {args.pca_dims}
  coordinate_units: projected_km
knn:
  k: {args.knn_k}
  eps: {args.knn_eps}
  power: {args.knn_power}
  reference: train_only
  train_mode: leave_self_out
features:
  use_sequencing_in_global: {args.use_sequencing_in_global}
  use_sequencing_in_calibrator: {args.use_sequencing_in_calibrator}
  use_coverage_in_calibrator: {args.use_coverage_in_calibrator}
sigma:
  sigma_min_km: {args.sigma_min_km}
  sigma_max_km: {args.sigma_max_km}
training:
  optimizer: AdamW
  lr: {args.lr}
  weight_decay: {args.weight_decay}
  batch_size: {args.batch_size}
  max_epochs: {args.epochs}
  patience: {args.patience}
  jitter_sigma_km: {args.jitter_sigma_km}
""",
    )
    write_text(run_dir / "links.md", "# Links\n\nNo external links recorded yet.")
    write_text(run_dir / "results.md", "# Results\n\nRun started; results will be appended after training.\n")


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precision-weighted kNN + MLP geolocation model")
    parser.add_argument("--geno", required=True)
    parser.add_argument("--snp", required=True)
    parser.add_argument("--ind", required=True)
    parser.add_argument("--anno", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default=datetime.now().strftime("%Y-%m-%d_precision-fusion_%H%M%S"))
    parser.add_argument("--plink_bin", default=None)
    parser.add_argument("--convertf_bin", default=None)
    parser.add_argument("--ld_window", type=int, default=200)
    parser.add_argument("--ld_step", type=int, default=25)
    parser.add_argument("--ld_r2", type=float, default=0.2)
    parser.add_argument("--pca_dims", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_snps", type=int, default=-1)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--knn_eps", type=float, default=1e-6)
    parser.add_argument("--knn_power", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--calibrator_dropout", type=float, default=0.1)
    parser.add_argument("--jitter_sigma_km", type=float, default=25.0)
    parser.add_argument("--sigma_min_km", type=float, default=25.0)
    parser.add_argument("--sigma_max_km", type=float, default=3000.0)
    parser.add_argument("--use_sequencing_in_global", type=str_to_bool, default=True)
    parser.add_argument("--use_sequencing_in_calibrator", type=str_to_bool, default=True)
    parser.add_argument("--use_coverage_in_calibrator", type=str_to_bool, default=True)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geno = Path(args.geno).resolve(); snp = Path(args.snp).resolve(); ind = Path(args.ind).resolve(); anno = Path(args.anno).resolve()
    for path, label in [(geno, "geno"), (snp, "snp"), (ind, "ind"), (anno, "anno")]:
        require_file(path, label)
    output_root = Path(args.output_dir).resolve(); run_dir = output_root / args.run_name; run_dir.mkdir(parents=True, exist_ok=True)
    write_logging_files(run_dir, args)
    plink_bin = resolve_executable(args.plink_bin, "PLINK_BIN", ["plink"])
    convertf_bin = resolve_executable(args.convertf_bin, "CONVERTF_BIN", ["convertf_local", "convertf"])
    log(f"Run directory: {run_dir}")
    log(f"Inputs: geno={geno} snp={snp} ind={ind} anno={anno}")

    plink_prefix, sample_ids, conversion_meta = prepare_plink_bed(convertf_bin=convertf_bin, plink_bin=plink_bin, geno=geno, snp=snp, ind=ind, work_dir=run_dir / "work", max_samples=args.max_samples, max_snps=args.max_snps)
    prune_in, eigenvec, eigenval = run_plink_ld_pca(plink_bin=plink_bin, bfile_prefix=plink_prefix, out_dir=run_dir, window=args.ld_window, step=args.ld_step, r2=args.ld_r2, pca_dims=args.pca_dims)
    features_pc = load_plink_eigenvec(eigenvec, sample_ids, args.pca_dims)
    np.save(run_dir / "pca_features_raw.npy", features_pc)
    meta = build_metadata(sample_ids, anno)
    data_type_by_id, coverage_by_id, cov_cols = load_anno_features(sample_ids, anno)
    if args.use_coverage_in_calibrator and cov_cols:
        log("Using coverage/call-rate proxy columns in calibrator: " + "; ".join(cov_cols))
    elif args.use_coverage_in_calibrator:
        log("WARNING: no coverage/call-rate proxy columns found; skipping coverage calibrator feature")
    seq_multihot, seq_tokens = build_seq_multihot(data_type_by_id, sample_ids)
    write_json(run_dir / "sequencing_encoding.json", {"tokens": seq_tokens})
    splits = make_leave_site_split(meta.locality, seed=args.seed, train_fraction=args.train_fraction, val_fraction=args.val_fraction)
    write_split_sites(run_dir / "split_sites.json", splits)
    pc_scaled, pc_mean, pc_std = standardize_train_apply(features_pc, splits.train)
    age_scaled, age_mean, age_std = standardize_train_apply(meta.date_bp.reshape(-1, 1), splits.train)
    features = np.column_stack([pc_scaled, age_scaled]).astype(np.float32)
    global_features = features if not args.use_sequencing_in_global else np.column_stack([features, seq_multihot]).astype(np.float32)
    np.save(run_dir / "pca_age_features.npy", features); np.save(run_dir / "global_features.npy", global_features)
    write_sample_stats(run_dir / "sample_stats.tsv", meta, splits, features)

    xy_km = meta.xy.astype(np.float64) / 1000.0
    target_center_km = xy_km[splits.train].mean(axis=0).astype(np.float64)
    centered_xy_km = xy_km - target_center_km[None, :]
    train_ref_x = features[splits.train]; train_ref_y = centered_xy_km[splits.train]; train_ref_age = meta.date_bp[splits.train]; train_ref_global = splits.train.astype(np.int64)
    log(f"Samples: total={len(sample_ids)} train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")
    log(f"kNN reference size train={len(train_ref_x)} val_ref={len(train_ref_x)} test_ref={len(train_ref_x)}")
    log(f"Computing train-only kNN features, k={args.knn_k}")
    knn_by_split = {}
    for name, idx, excl in [("train", splits.train, True), ("val", splits.val, False), ("test", splits.test, False)]:
        knn_by_split[name] = compute_knn_features(query_x=features[idx], query_global_idx=idx.astype(np.int64), ref_x=train_ref_x, ref_y_km=train_ref_y, ref_age=train_ref_age, ref_global_idx=train_ref_global, k=args.knn_k, eps=args.knn_eps, power=args.knn_power, exclude_self=excl)
    knn_mu = np.empty((len(sample_ids), 2), dtype=np.float64)
    raw_sigma = np.empty(len(sample_ids), dtype=np.float64)
    diag = np.empty((len(sample_ids), 12), dtype=np.float64)
    first_neighbor = np.empty(len(sample_ids), dtype=np.int64)
    for name, idx in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        kf = knn_by_split[name]
        knn_mu[idx] = kf["mu"]; raw_sigma[idx] = kf["raw_sigma"]
        diag[idx] = np.column_stack([kf["raw_sigma"], kf["min_neighbor_distance"], kf["mean_neighbor_distance"], kf["max_neighbor_distance"], kf["std_neighbor_distance"], kf["d1"], kf["d5"], kf["d10"], kf["d20"], kf["weighted_spatial_variance"], kf["weighted_neighbor_age_mean"], kf["weighted_neighbor_age_std"]])
        first_neighbor[idx] = kf["first_neighbor_global"]
    if np.any(first_neighbor[splits.train] == splits.train):
        fail("Self-neighbor exclusion check failed")
    if not np.isfinite(diag).all() or not np.isfinite(knn_mu).all() or not np.isfinite(raw_sigma).all():
        fail("NaNs in kNN features")
    diag_scaled, diag_mean, diag_std = standardize_from_train(diag[splits.train], diag)
    cov_feature = np.asarray([coverage_by_id[sid] for sid in sample_ids], dtype=np.float64).reshape(-1, 1)
    use_cov = bool(args.use_coverage_in_calibrator and np.isfinite(cov_feature).any())
    if use_cov:
        finite = np.isfinite(cov_feature[:, 0])
        fill = np.nanmedian(cov_feature[:, 0])
        cov_feature[~finite, 0] = fill
        cov_scaled, cov_mean, cov_std = standardize_from_train(cov_feature[splits.train], cov_feature)
    else:
        cov_scaled = np.zeros((len(sample_ids), 0), dtype=np.float32)
        cov_mean = np.zeros(0); cov_std = np.zeros(0)
    calib_parts = [diag_scaled, age_scaled.astype(np.float32)]
    if args.use_sequencing_in_calibrator:
        calib_parts.append(seq_multihot)
    if use_cov:
        calib_parts.append(cov_scaled)
    calibrator_features = np.column_stack(calib_parts).astype(np.float32)
    np.save(run_dir / "knn_diag_features.npy", diag.astype(np.float32)); np.save(run_dir / "calibrator_features.npy", calibrator_features)

    train_model(args=args, run_dir=run_dir, sample_ids=sample_ids, global_features=global_features, calibrator_features=calibrator_features, knn_mu=knn_mu.astype(np.float32), raw_sigma=raw_sigma.astype(np.float32), centered_xy_km=centered_xy_km.astype(np.float32), xy_km=xy_km, target_center_km=target_center_km, splits=splits, meta=meta, data_type_by_id=data_type_by_id, seq_tokens=seq_tokens, projection=meta.projection, conversion_meta=conversion_meta, prune_in=prune_in, eigenvec=eigenvec, eigenval=eigenval)


def train_model(**kw) -> None:
    args=kw["args"]; run_dir=kw["run_dir"]; sample_ids=kw["sample_ids"]; global_features=kw["global_features"]; calibrator_features=kw["calibrator_features"]; knn_mu=kw["knn_mu"]; raw_sigma=kw["raw_sigma"]; centered_xy_km=kw["centered_xy_km"]; xy_km=kw["xy_km"]; target_center_km=kw["target_center_km"]; splits=kw["splits"]; meta=kw["meta"]; data_type_by_id=kw["data_type_by_id"]
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        fail(f"PyTorch is required but unavailable: {exc}")

    class GeoPrecisionFusion(nn.Module):
        def __init__(self, global_dim: int, calib_dim: int):
            super().__init__(); self.sigma_min=float(args.sigma_min_km); self.sigma_max=float(args.sigma_max_km); self.softplus=nn.Softplus()
            self.global_mlp=nn.Sequential(nn.Linear(global_dim,512),nn.GELU(),nn.BatchNorm1d(512),nn.Dropout(args.dropout),nn.Linear(512,256),nn.GELU(),nn.BatchNorm1d(256),nn.Dropout(args.dropout),nn.Linear(256,128),nn.GELU(),nn.BatchNorm1d(128),nn.Dropout(args.dropout),nn.Linear(128,64),nn.GELU(),nn.Linear(64,3))
            self.knn_calibrator=nn.Sequential(nn.Linear(calib_dim,64),nn.GELU(),nn.Dropout(args.calibrator_dropout),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
        def sigma_from_raw(self, raw):
            return (self.sigma_min + self.softplus(raw)).clamp(min=self.sigma_min, max=self.sigma_max)
        def forward(self, gx, cx, kmu):
            mout=self.global_mlp(gx); mu_mlp=mout[:,:2]; sigma_mlp=self.sigma_from_raw(mout[:,2])
            sigma_knn=self.sigma_from_raw(self.knn_calibrator(cx).squeeze(1))
            var_knn=sigma_knn*sigma_knn; var_mlp=sigma_mlp*sigma_mlp
            w_knn=1.0/var_knn; w_mlp=1.0/var_mlp; denom=w_knn+w_mlp
            mu_final=(w_knn[:,None]*kmu + w_mlp[:,None]*mu_mlp)/denom[:,None]
            sigma_final=torch.sqrt(1.0/denom).clamp(min=self.sigma_min, max=self.sigma_max)
            alpha_precision=(w_knn/denom).clamp(min=0.0, max=1.0)
            return {"mu_final":mu_final,"sigma_final":sigma_final,"alpha_precision":alpha_precision,"mu_knn":kmu,"sigma_knn_corrected":sigma_knn,"mu_mlp":mu_mlp,"sigma_mlp":sigma_mlp}

    def nll(out, target, train_mode):
        sigma2=out["sigma_final"]*out["sigma_final"]
        if train_mode and args.jitter_sigma_km>0:
            sigma2=sigma2+float(args.jitter_sigma_km)**2
        dist2=torch.sum((target-out["mu_final"])**2, dim=1)
        return (0.5*dist2/sigma2 + 0.5*torch.log(sigma2)).mean()

    seed=int(args.seed); np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    log(f"Training GeoPrecisionFusion on device: {device}")
    gx=torch.tensor(global_features,dtype=torch.float32); cx=torch.tensor(calibrator_features,dtype=torch.float32); kmu=torch.tensor(knn_mu,dtype=torch.float32); y=torch.tensor(centered_xy_km,dtype=torch.float32)
    train_ds=TensorDataset(gx[splits.train],cx[splits.train],kmu[splits.train],y[splits.train])
    loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,drop_last=False)
    model=GeoPrecisionFusion(global_features.shape[1],calibrator_features.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    val_pack=[gx[splits.val].to(device),cx[splits.val].to(device),kmu[splits.val].to(device),y[splits.val].to(device)]
    best=float("inf"); best_epoch=0; patience=int(args.patience); history=[]; ckpt=run_dir/"best_precision_fusion.pt"; stop_epoch=0
    for epoch in range(1,int(args.epochs)+1):
        model.train(); losses=[]
        for bg,bc,bk,by in loader:
            bg,bc,bk,by=bg.to(device),bc.to(device),bk.to(device),by.to(device)
            target=by + torch.randn_like(by)*float(args.jitter_sigma_km) if args.jitter_sigma_km>0 else by
            opt.zero_grad(set_to_none=True); out=model(bg,bc,bk); loss=nll(out,target,True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out=model(*val_pack[:3]); val_loss=float(nll(out,val_pack[3],False).detach().cpu()); val_err=error_km(out["mu_final"].detach().cpu().numpy()+target_center_km[None,:],xy_km[splits.val]); val_med=float(np.median(val_err))
        train_loss=float(np.mean(losses)); history.append({"epoch":epoch,"train_nll":train_loss,"val_nll":val_loss,"val_median_km":val_med})
        if epoch==1 or epoch%10==0: log(f"epoch={epoch} train_nll={train_loss:.6f} val_nll={val_loss:.6f} val_median_km={val_med:.3f}")
        if val_loss < best - 1e-6:
            best=val_loss; best_epoch=epoch; patience=int(args.patience); torch.save({"model_state":model.state_dict(),"best_val_nll":best,"best_epoch":best_epoch},ckpt)
        else:
            patience-=1
            if patience<=0:
                stop_epoch=epoch; log(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}"); break
    if stop_epoch==0: stop_epoch=epoch
    write_csv(run_dir/"training_history.csv",history,["epoch","train_nll","val_nll","val_median_km"])
    saved=torch.load(ckpt,map_location=device); model.load_state_dict(saved["model_state"]); model.eval()

    def predict(name, idx):
        with torch.no_grad(): out=model(gx[idx].to(device),cx[idx].to(device),kmu[idx].to(device))
        arr={k:v.detach().cpu().numpy().astype(np.float64) for k,v in out.items()}
        mu_final_abs=arr["mu_final"]+target_center_km[None,:]; mu_knn_abs=arr["mu_knn"]+target_center_km[None,:]; mu_mlp_abs=arr["mu_mlp"]+target_center_km[None,:]
        true=xy_km[idx]; err=error_km(mu_final_abs,true); nllv=0.5*np.sum((true-mu_final_abs)**2,axis=1)/(arr["sigma_final"]**2)+np.log(arr["sigma_final"])
        pred_latlon=inverse_project_xy_km(mu_final_abs, kw["projection"])
        rows=[]
        for pos,gi in enumerate(idx):
            sid=sample_ids[int(gi)]
            rows.append({"sample_id":sid,"split":name,"true_x":f"{true[pos,0]:.6f}","true_y":f"{true[pos,1]:.6f}","true_lat":f"{meta.lat[int(gi)]:.8f}","true_lon":f"{meta.lon[int(gi)]:.8f}","pred_x":f"{mu_final_abs[pos,0]:.6f}","pred_y":f"{mu_final_abs[pos,1]:.6f}","pred_lat":f"{pred_latlon[pos,0]:.8f}","pred_lon":f"{pred_latlon[pos,1]:.8f}","error_km":f"{err[pos]:.6f}","nll":f"{nllv[pos]:.8f}","mu_knn_x":f"{mu_knn_abs[pos,0]:.6f}","mu_knn_y":f"{mu_knn_abs[pos,1]:.6f}","raw_sigma_knn":f"{raw_sigma[int(gi)]:.6f}","sigma_knn_corrected":f"{arr['sigma_knn_corrected'][pos]:.6f}","mu_mlp_x":f"{mu_mlp_abs[pos,0]:.6f}","mu_mlp_y":f"{mu_mlp_abs[pos,1]:.6f}","sigma_mlp":f"{arr['sigma_mlp'][pos]:.6f}","sigma_final":f"{arr['sigma_final'][pos]:.6f}","alpha_precision":f"{arr['alpha_precision'][pos]:.8f}","sequencing_type":data_type_by_id.get(sid,"missing"),"age":f"{meta.date_bp[int(gi)]:.6f}","age_bin":age_bin(float(meta.date_bp[int(gi)])),"locality_id":meta.locality[int(gi)]})
        write_csv(run_dir/f"{name}_predictions.tsv",rows,list(rows[0].keys()),delimiter="\t")
        return {"rows":rows,"err":err,"nll":nllv,"alpha":arr["alpha_precision"],"mu_final":mu_final_abs,"mu_mlp":mu_mlp_abs,"mu_knn":mu_knn_abs}

    valp=predict("val",splits.val); testp=predict("test",splits.test); test_true=xy_km[splits.test]
    k1=np.empty_like(test_true)
    for i,gi in enumerate(splits.test):
        d2=((features_for_k1(global_features, splits.train, args, kw) - features_for_k1(global_features, np.array([gi]), args, kw)[0])**2).sum(axis=1)
        k1[i]=xy_km[splits.train[int(np.argmin(d2))]]
    weighted_knn_mu=knn_mu[splits.test].astype(np.float64)+target_center_km[None,:]
    baseline={}; baseline.update(baseline_metrics("k1_nn",k1,test_true)); baseline.update(baseline_metrics("weighted_knn",weighted_knn_mu,test_true)); baseline.update(baseline_metrics("mlp_expert",testp["mu_mlp"],test_true)); baseline.update(baseline_metrics("precision_fusion",testp["mu_final"],test_true))
    alpha=testp["alpha"]
    alpha_summary={"alpha_precision_mean":float(alpha.mean()),"alpha_precision_median":float(np.median(alpha)),"alpha_precision_q10":float(np.percentile(alpha,10)),"alpha_precision_q25":float(np.percentile(alpha,25)),"alpha_precision_q50":float(np.percentile(alpha,50)),"alpha_precision_q75":float(np.percentile(alpha,75)),"alpha_precision_q90":float(np.percentile(alpha,90))}
    high=testp["err"][alpha>0.75]; low=testp["err"][alpha<0.25]
    alpha_summary.update({"median_error_alpha_precision_gt_0_75":float(np.median(high)) if high.size else float("nan"),"n_alpha_precision_gt_0_75":int(high.size),"median_error_alpha_precision_lt_0_25":float(np.median(low)) if low.size else float("nan"),"n_alpha_precision_lt_0_25":int(low.size)})
    write_csv(run_dir/"alpha_precision_summary.tsv",[alpha_summary],list(alpha_summary.keys()),delimiter="\t")

    def grouped(rows,key):
        groups=defaultdict(list)
        for r in rows: groups[str(r[key])].append(r)
        out=[]
        for g,rs in groups.items():
            vals=np.asarray([float(r["error_km"]) for r in rs]); al=np.asarray([float(r["alpha_precision"]) for r in rs])
            out.append({key:g,"n_test":len(rs),"median_error_km":f"{np.median(vals):.6f}","mean_error_km":f"{vals.mean():.6f}","p90_error_km":f"{np.percentile(vals,90):.6f}","median_alpha_precision":f"{np.median(al):.8f}","mean_alpha_precision":f"{al.mean():.8f}"})
        return out
    age_rows=grouped(testp["rows"],"age_bin"); order={label:i for i,_,label in AGE_BINS}; age_rows.sort(key=lambda r:order.get(str(r["age_bin"]),99)); write_csv(run_dir/"median_error_by_age_bin.tsv",age_rows,["age_bin","n_test","median_error_km","mean_error_km","p90_error_km","median_alpha_precision","mean_alpha_precision"],delimiter="\t")
    seq_rows=grouped(testp["rows"],"sequencing_type"); seq_rows.sort(key=lambda r:(-int(r["n_test"]),str(r["sequencing_type"]))); write_csv(run_dir/"median_error_by_sequencing_method.tsv",seq_rows,["sequencing_type","n_test","median_error_km","mean_error_km","p90_error_km","median_alpha_precision","mean_alpha_precision"],delimiter="\t")
    alpha_bin_rows=[]
    for lo,hi,label in [(0,.25,"0-0.25"),(.25,.5,"0.25-0.5"),(.5,.75,"0.5-0.75"),(.75,1.0000001,"0.75-1")]:
        m=(alpha>=lo)&(alpha<hi); vals=testp["err"][m]; alpha_bin_rows.append({"alpha_precision_bin":label,"n_test":int(m.sum()),"median_error_km":f"{np.median(vals):.6f}" if vals.size else "nan","mean_error_km":f"{vals.mean():.6f}" if vals.size else "nan"})
    write_csv(run_dir/"error_by_alpha_precision_bin.tsv",alpha_bin_rows,["alpha_precision_bin","n_test","median_error_km","mean_error_km"],delimiter="\t")

    # Include prior learned-gate run metrics if present locally on the same filesystem.
    learned_gate={}
    prior=run_dir.parent/"2026-06-09_ho-gated-knn-mlp-k20-ldpca256"/"evaluation_summary.json"
    if prior.is_file():
        try:
            p=json.loads(prior.read_text()); learned_gate={"learned_gated_median_km":p.get("gated_median_km"),"learned_gated_mean_km":p.get("gated_mean_km"),"learned_gated_p90_km":p.get("gated_p90_km")}
        except Exception: pass
    summary={"status":"completed","run_name":args.run_name,"timestamp":datetime.now(timezone.utc).isoformat(),"n_samples":len(sample_ids),"n_train":int(len(splits.train)),"n_val":int(len(splits.val)),"n_test":int(len(splits.test)),"best_epoch":int(saved["best_epoch"]),"early_stopping_epoch":int(stop_epoch),"best_val_nll":float(saved["best_val_nll"]),"test_gaussian_nll":float(np.mean(testp["nll"])),"val_gaussian_nll":float(np.mean(valp["nll"])),"knn_k":int(args.knn_k),"pca_dims":int(args.pca_dims),"target_center_km":target_center_km.tolist(),"sequencing_tokens":kw["seq_tokens"],"conversion":kw["conversion_meta"],"prune_in":str(kw["prune_in"]),"eigenvec":str(kw["eigenvec"]),"eigenval":str(kw["eigenval"]),**baseline,**learned_gate,**alpha_summary}
    write_json(run_dir/"evaluation_summary.json",summary); write_json(run_dir/"run_summary.json",summary)
    fields=[k for k,v in summary.items() if not isinstance(v,(dict,list))]; write_csv(run_dir/"metrics.csv",[{k:summary[k] for k in fields}],fields)
    learned_row=""
    if learned_gate:
        learned_row=f"| learned gated two-expert | {learned_gate['learned_gated_median_km']:.3f} | {learned_gate['learned_gated_mean_km']:.3f} | {learned_gate['learned_gated_p90_km']:.3f} |\n"
    write_text(run_dir/"results.md",f"""
# Results

## Test Comparison

| model | median km | mean km | p90 km |
| --- | ---: | ---: | ---: |
| k=1 NN | {summary['k1_nn_median_km']:.3f} | {summary['k1_nn_mean_km']:.3f} | {summary['k1_nn_p90_km']:.3f} |
| weighted kNN k={args.knn_k} | {summary['weighted_knn_median_km']:.3f} | {summary['weighted_knn_mean_km']:.3f} | {summary['weighted_knn_p90_km']:.3f} |
| MLP expert | {summary['mlp_expert_median_km']:.3f} | {summary['mlp_expert_mean_km']:.3f} | {summary['mlp_expert_p90_km']:.3f} |
{learned_row}| precision-weighted kNN+MLP | {summary['precision_fusion_median_km']:.3f} | {summary['precision_fusion_mean_km']:.3f} | {summary['precision_fusion_p90_km']:.3f} |

## Precision Fusion

Mean alpha_precision: {summary['alpha_precision_mean']:.4f}

Median alpha_precision: {summary['alpha_precision_median']:.4f}

Alpha q10/q25/q50/q75/q90: {summary['alpha_precision_q10']:.4f}, {summary['alpha_precision_q25']:.4f}, {summary['alpha_precision_q50']:.4f}, {summary['alpha_precision_q75']:.4f}, {summary['alpha_precision_q90']:.4f}

## Files

- `test_predictions.tsv`
- `val_predictions.tsv`
- `alpha_precision_summary.tsv`
- `median_error_by_age_bin.tsv`
- `median_error_by_sequencing_method.tsv`
- `error_by_alpha_precision_bin.tsv`
""")
    log(f"Final precision fusion median={summary['precision_fusion_median_km']:.3f}km p90={summary['precision_fusion_p90_km']:.3f}km alpha_precision_median={summary['alpha_precision_median']:.3f}")


def features_for_k1(global_features: np.ndarray, idx: np.ndarray, args, kw) -> np.ndarray:
    # k=1 baseline should use the same scaled PCA+age space as kNN, not sequencing columns.
    pca_age_dim = int(args.pca_dims) + 1
    return global_features[idx, :pca_age_dim].astype(np.float64)


if __name__ == "__main__":
    main()
