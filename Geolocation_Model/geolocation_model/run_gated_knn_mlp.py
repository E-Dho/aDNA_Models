#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
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


def log(message: str) -> None:
    print(message, flush=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def summarize(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90))}


def error_km(pred_xy_km: np.ndarray, true_xy_km: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(pred_xy_km) - np.asarray(true_xy_km), axis=1)


def load_data_type(sample_ids: Sequence[str], anno: Path) -> Dict[str, str]:
    anno_index = read_anno_index(anno)
    out: Dict[str, str] = {}
    for sid in sample_ids:
        row = anno_index.get(sid)
        if row is None:
            fail(f"Missing anno row for data type lookup: {sid}")
        key = find_key(row, exact="Data type")
        out[sid] = (row.get(key, "") if key else "").strip() or "missing"
    return out


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
    extra = 1 if exclude_self else 0
    if k + extra > len(ref_x):
        fail(f"Need at least k+1 training samples for leave-self-out kNN; k={k}, ref={len(ref_x)}")
    n = len(query_x)
    mu = np.empty((n, 2), dtype=np.float64)
    sigma = np.empty(n, dtype=np.float64)
    mean_dist = np.empty(n, dtype=np.float64)
    min_dist = np.empty(n, dtype=np.float64)
    max_dist = np.empty(n, dtype=np.float64)
    std_dist = np.empty(n, dtype=np.float64)
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
            sigma[start + j] = math.sqrt(max(var, 0.0) + eps)
            mean_dist[start + j] = float(d.mean())
            min_dist[start + j] = float(d.min())
            max_dist[start + j] = float(d.max())
            std_dist[start + j] = float(d.std())
            age_mean[start + j] = float(ages.mean())
            age_std[start + j] = float(ages.std())
            first_neighbor_global[start + j] = int(ref_global_idx[nn[0]])
    out = {
        "mu": mu,
        "sigma": sigma,
        "mean_neighbor_distance": mean_dist,
        "min_neighbor_distance": min_dist,
        "max_neighbor_distance": max_dist,
        "std_neighbor_distance": std_dist,
        "weighted_spatial_variance": spatial_var,
        "neighbor_age_mean": age_mean,
        "neighbor_age_std": age_std,
        "first_neighbor_global": first_neighbor_global,
    }
    for name, arr in out.items():
        if not np.isfinite(arr).all():
            fail(f"kNN feature contains non-finite values: {name}")
    return out


def standardize_diag(train_diag: np.ndarray, all_diag: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_diag.mean(axis=0, keepdims=True)
    std = train_diag.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((all_diag - mean) / std).astype(np.float32), mean.reshape(-1), std.reshape(-1)


def baseline_metrics(name: str, pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    err = error_km(pred, true)
    return {
        f"{name}_mean_km": float(err.mean()),
        f"{name}_median_km": float(np.median(err)),
        f"{name}_p90_km": float(np.percentile(err, 90)),
    }


def write_logging_files(run_dir: Path, args: argparse.Namespace) -> None:
    write_text(
        run_dir / "hypothesis.md",
        """
# Hypothesis

## Problem

The pure MLP geolocation model is global and smooth, while weighted kNN is local but can fail when the neighbor cloud is diffuse or misleading.

## Motivation

Combine a fixed local weighted-kNN expert in PCA+age space with a learned global Gaussian MLP expert. A learned gate should decide per sample whether local interpolation or the global model is more reliable.

## Hypothesis

The gated two-expert model will learn high kNN weight for samples with tight, reliable neighbor clouds and low kNN weight when local PCA+age neighborhoods are contradictory.

## Success Criteria

The gated model improves median geolocation error and p90 error relative to both the standalone weighted-kNN baseline and the standalone MLP expert.

## Failure Criteria

The gated model is equal or worse than either expert, or alpha diagnostics show collapse to a single expert without performance gain.

## Next Planned Experiments

Tune k, gate diagnostic features, sigma calibration, and potentially train the MLP expert first before joint gate fine-tuning.
""",
    )
    write_text(
        run_dir / "config.yaml",
        f"""
experiment_type: gated_knn_mlp
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
model:
  input_dim: {args.pca_dims + 1}
  expert: GeoMixtureOfExperts
  gate_hidden: [64, 32]
  gate_dropout: {args.gate_dropout}
knn:
  k: {args.knn_k}
  weighting: inverse_distance
  eps: {args.knn_eps}
  power: {args.knn_power}
  train_reference: train_split_only
  train_query_mode: leave_self_out
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
split:
  type: leave-site-out
  site_column: Locality
  train_fraction: {args.train_fraction}
  val_fraction: {args.val_fraction}
""",
    )
    write_text(run_dir / "links.md", "# Links\n\nNo external links recorded yet.")
    write_text(run_dir / "results.md", "# Results\n\nRun started; results will be appended after training.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gated weighted-kNN + MLP geolocation model")
    parser.add_argument("--geno", required=True)
    parser.add_argument("--snp", required=True)
    parser.add_argument("--ind", required=True)
    parser.add_argument("--anno", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default=datetime.now().strftime("%Y-%m-%d_gated-knn-mlp_%H%M%S"))
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
    parser.add_argument("--gate_dropout", type=float, default=0.1)
    parser.add_argument("--jitter_sigma_km", type=float, default=25.0)
    parser.add_argument("--sigma_min_km", type=float, default=25.0)
    parser.add_argument("--sigma_max_km", type=float, default=3000.0)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geno = Path(args.geno).resolve()
    snp = Path(args.snp).resolve()
    ind = Path(args.ind).resolve()
    anno = Path(args.anno).resolve()
    for path, label in [(geno, "geno"), (snp, "snp"), (ind, "ind"), (anno, "anno")]:
        require_file(path, label)

    output_root = Path(args.output_dir).resolve()
    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_logging_files(run_dir, args)

    plink_bin = resolve_executable(args.plink_bin, "PLINK_BIN", ["plink"])
    convertf_bin = resolve_executable(args.convertf_bin, "CONVERTF_BIN", ["convertf_local", "convertf"])
    args.plink_bin = plink_bin
    args.convertf_bin = convertf_bin
    log(f"Run directory: {run_dir}")
    log(f"Inputs: geno={geno} snp={snp} ind={ind} anno={anno}")
    log(f"Executables: plink={plink_bin} convertf={convertf_bin}")

    work_dir = run_dir / "work"
    plink_prefix, sample_ids, conversion_meta = prepare_plink_bed(
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
    features_pc = load_plink_eigenvec(eigenvec, sample_ids, args.pca_dims)
    np.save(run_dir / "pca_features_raw.npy", features_pc)

    meta = build_metadata(sample_ids, anno)
    data_type_by_id = load_data_type(sample_ids, anno)
    splits = make_leave_site_split(meta.locality, seed=args.seed, train_fraction=args.train_fraction, val_fraction=args.val_fraction)
    write_split_sites(run_dir / "split_sites.json", splits)

    pc_scaled, pc_mean, pc_std = standardize_train_apply(features_pc, splits.train)
    age_scaled, age_mean, age_std = standardize_train_apply(meta.date_bp.reshape(-1, 1), splits.train)
    features = np.column_stack([pc_scaled, age_scaled]).astype(np.float32)
    np.save(run_dir / "pca_age_features.npy", features)
    write_sample_stats(run_dir / "sample_stats.tsv", meta, splits, features)

    target_center_km = (meta.xy[splits.train].mean(axis=0) / 1000.0).astype(np.float64)
    xy_km = meta.xy.astype(np.float64) / 1000.0
    centered_xy_km = xy_km - target_center_km[None, :]
    train_ref_x = features[splits.train]
    train_ref_y = centered_xy_km[splits.train]
    train_ref_age = meta.date_bp[splits.train]
    train_ref_global = splits.train.astype(np.int64)

    log(f"Samples: total={len(sample_ids)} train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")
    log(f"Computing kNN expert features with train-only reference, k={args.knn_k}")
    knn_by_split = {}
    for split_name, idx, excl in [("train", splits.train, True), ("val", splits.val, False), ("test", splits.test, False)]:
        knn_by_split[split_name] = compute_knn_features(
            query_x=features[idx],
            query_global_idx=idx.astype(np.int64),
            ref_x=train_ref_x,
            ref_y_km=train_ref_y,
            ref_age=train_ref_age,
            ref_global_idx=train_ref_global,
            k=args.knn_k,
            eps=args.knn_eps,
            power=args.knn_power,
            exclude_self=excl,
        )
    if len(train_ref_global) != len(splits.train):
        fail("kNN reference size does not equal train split size")

    knn_mu = np.empty((len(sample_ids), 2), dtype=np.float64)
    knn_sigma = np.empty(len(sample_ids), dtype=np.float64)
    diag = np.empty((len(sample_ids), 7), dtype=np.float64)
    first_neighbor = np.empty(len(sample_ids), dtype=np.int64)
    for split_name, idx in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        kf = knn_by_split[split_name]
        knn_mu[idx] = kf["mu"]
        knn_sigma[idx] = kf["sigma"]
        diag[idx] = np.column_stack([
            kf["mu"][:, 0],
            kf["mu"][:, 1],
            kf["sigma"],
            kf["mean_neighbor_distance"],
            kf["min_neighbor_distance"],
            kf["std_neighbor_distance"],
            kf["weighted_spatial_variance"],
        ])
        first_neighbor[idx] = kf["first_neighbor_global"]
    if np.any(first_neighbor[splits.train] == splits.train):
        fail("Sanity check failed: train kNN first neighbor includes self")
    diag_scaled, diag_mean, diag_std = standardize_diag(diag[splits.train], diag)
    np.save(run_dir / "knn_diag_features.npy", diag.astype(np.float32))
    np.save(run_dir / "knn_diag_features_scaled.npy", diag_scaled)

    train_model(
        args=args,
        run_dir=run_dir,
        sample_ids=sample_ids,
        features=features,
        diag_scaled=diag_scaled,
        knn_mu=knn_mu.astype(np.float32),
        knn_sigma=knn_sigma.astype(np.float32),
        centered_xy_km=centered_xy_km.astype(np.float32),
        xy_km=xy_km,
        target_center_km=target_center_km,
        splits=splits,
        meta=meta,
        data_type_by_id=data_type_by_id,
        conversion_meta=conversion_meta,
        prune_in=prune_in,
        eigenvec=eigenvec,
        eigenval=eigenval,
        pc_mean=pc_mean,
        pc_std=pc_std,
        age_mean=age_mean,
        age_std=age_std,
        diag_mean=diag_mean,
        diag_std=diag_std,
    )


def train_model(**kwargs) -> None:
    args: argparse.Namespace = kwargs["args"]
    run_dir: Path = kwargs["run_dir"]
    sample_ids: Sequence[str] = kwargs["sample_ids"]
    features: np.ndarray = kwargs["features"]
    diag_scaled: np.ndarray = kwargs["diag_scaled"]
    knn_mu: np.ndarray = kwargs["knn_mu"]
    knn_sigma: np.ndarray = kwargs["knn_sigma"]
    centered_xy_km: np.ndarray = kwargs["centered_xy_km"]
    xy_km: np.ndarray = kwargs["xy_km"]
    target_center_km: np.ndarray = kwargs["target_center_km"]
    splits = kwargs["splits"]
    meta = kwargs["meta"]
    data_type_by_id: Dict[str, str] = kwargs["data_type_by_id"]

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        fail(f"PyTorch is required for gated model training but is unavailable: {exc}")

    class GeoMixtureOfExperts(nn.Module):
        def __init__(self, x_dim: int, diag_dim: int):
            super().__init__()
            self.sigma_min = float(args.sigma_min_km)
            self.sigma_max = float(args.sigma_max_km)
            self.mlp = nn.Sequential(
                nn.Linear(x_dim, 512), nn.GELU(), nn.BatchNorm1d(512), nn.Dropout(args.dropout),
                nn.Linear(512, 256), nn.GELU(), nn.BatchNorm1d(256), nn.Dropout(args.dropout),
                nn.Linear(256, 128), nn.GELU(), nn.BatchNorm1d(128), nn.Dropout(args.dropout),
                nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3),
            )
            self.gate = nn.Sequential(
                nn.Linear(x_dim + diag_dim, 64), nn.GELU(), nn.Dropout(args.gate_dropout),
                nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid(),
            )
            self.softplus = nn.Softplus()

        def forward(self, x, diag_x, k_mu, k_sigma):
            mlp_out = self.mlp(x)
            mu_mlp = mlp_out[:, :2]
            sigma_mlp = self.sigma_min + self.softplus(mlp_out[:, 2])
            sigma_mlp = sigma_mlp.clamp(min=self.sigma_min, max=self.sigma_max)
            alpha = self.gate(torch.cat([x, diag_x], dim=1)).squeeze(1)
            k_sigma = k_sigma.clamp(min=self.sigma_min, max=self.sigma_max)
            mu_final = alpha[:, None] * k_mu + (1.0 - alpha[:, None]) * mu_mlp
            sigma_final = alpha * k_sigma + (1.0 - alpha) * sigma_mlp
            sigma_final = sigma_final.clamp(min=self.sigma_min, max=self.sigma_max)
            return {
                "mu_final": mu_final,
                "sigma_final": sigma_final,
                "alpha": alpha,
                "mu_mlp": mu_mlp,
                "sigma_mlp": sigma_mlp,
                "mu_knn": k_mu,
                "sigma_knn": k_sigma,
            }

    def nll(out, target, train_mode: bool):
        sigma = out["sigma_final"]
        if train_mode and args.jitter_sigma_km > 0:
            sigma = torch.sqrt(sigma * sigma + float(args.jitter_sigma_km) ** 2)
        dist2 = torch.sum((target - out["mu_final"]) ** 2, dim=1)
        return (0.5 * dist2 / (sigma * sigma) + torch.log(sigma)).mean()

    def tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    log(f"Training gated two-expert model on device: {device}")

    x_t = tensor(features)
    d_t = tensor(diag_scaled)
    kmu_t = tensor(knn_mu)
    ksig_t = tensor(knn_sigma)
    y_t = tensor(centered_xy_km)
    train_ds = TensorDataset(x_t[splits.train], d_t[splits.train], kmu_t[splits.train], ksig_t[splits.train], y_t[splits.train])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = GeoMixtureOfExperts(features.shape[1], diag_scaled.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    val_pack = [x_t[splits.val].to(device), d_t[splits.val].to(device), kmu_t[splits.val].to(device), ksig_t[splits.val].to(device), y_t[splits.val].to(device)]
    best_val = float("inf")
    best_epoch = 0
    patience_left = int(args.patience)
    checkpoint = run_dir / "best_gated_model.pt"
    history: list[dict[str, object]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for bx, bd, bkmu, bks, by in loader:
            bx, bd, bkmu, bks, by = bx.to(device), bd.to(device), bkmu.to(device), bks.to(device), by.to(device)
            if args.jitter_sigma_km > 0:
                by_loss = by + torch.randn_like(by) * float(args.jitter_sigma_km)
            else:
                by_loss = by
            opt.zero_grad(set_to_none=True)
            out = model(bx, bd, bkmu, bks)
            loss = nll(out, by_loss, train_mode=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vout = model(*val_pack[:4])
            val_loss = float(nll(vout, val_pack[4], train_mode=False).detach().cpu())
            val_err = error_km((vout["mu_final"].detach().cpu().numpy() + target_center_km[None, :]), xy_km[splits.val])
            val_median = float(np.median(val_err))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss, "val_median_km": val_median})
        if epoch == 1 or epoch % 10 == 0:
            log(f"epoch={epoch} train_nll={train_loss:.6f} val_nll={val_loss:.6f} val_median_km={val_median:.3f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            patience_left = int(args.patience)
            torch.save({"model_state": model.state_dict(), "best_val_nll": best_val, "best_epoch": best_epoch}, checkpoint)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break
    write_csv(run_dir / "training_history.csv", history, ["epoch", "train_nll", "val_nll", "val_median_km"])
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state"])
    model.eval()

    def predict_split(split_name: str, idx: np.ndarray) -> Dict[str, object]:
        pack = [x_t[idx].to(device), d_t[idx].to(device), kmu_t[idx].to(device), ksig_t[idx].to(device)]
        with torch.no_grad():
            out = model(*pack)
        arr = {key: val.detach().cpu().numpy().astype(np.float64) for key, val in out.items()}
        mu_final_abs = arr["mu_final"] + target_center_km[None, :]
        mu_mlp_abs = arr["mu_mlp"] + target_center_km[None, :]
        mu_knn_abs = arr["mu_knn"] + target_center_km[None, :]
        true_abs = xy_km[idx]
        final_err = error_km(mu_final_abs, true_abs)
        mlp_err = error_km(mu_mlp_abs, true_abs)
        knn_err = error_km(mu_knn_abs, true_abs)
        final_nll = 0.5 * np.sum((true_abs - mu_final_abs) ** 2, axis=1) / (arr["sigma_final"] ** 2) + np.log(arr["sigma_final"])
        rows = []
        for pos, gi in enumerate(idx):
            sid = sample_ids[int(gi)]
            rows.append({
                "sample_index": int(gi),
                "sample_id": sid,
                "split": split_name,
                "data_type": data_type_by_id.get(sid, "missing"),
                "age_bin": age_bin(float(meta.date_bp[int(gi)])),
                "date_mean_bp": f"{float(meta.date_bp[int(gi)]):.6f}",
                "locality": meta.locality[int(gi)],
                "mu_final_x_km": f"{mu_final_abs[pos, 0]:.6f}",
                "mu_final_y_km": f"{mu_final_abs[pos, 1]:.6f}",
                "sigma_final_km": f"{arr['sigma_final'][pos]:.6f}",
                "alpha": f"{arr['alpha'][pos]:.8f}",
                "mu_knn_x_km": f"{mu_knn_abs[pos, 0]:.6f}",
                "mu_knn_y_km": f"{mu_knn_abs[pos, 1]:.6f}",
                "knn_sigma_km": f"{arr['sigma_knn'][pos]:.6f}",
                "mu_mlp_x_km": f"{mu_mlp_abs[pos, 0]:.6f}",
                "mu_mlp_y_km": f"{mu_mlp_abs[pos, 1]:.6f}",
                "sigma_mlp_km": f"{arr['sigma_mlp'][pos]:.6f}",
                "true_x_km": f"{true_abs[pos, 0]:.6f}",
                "true_y_km": f"{true_abs[pos, 1]:.6f}",
                "final_error_km": f"{final_err[pos]:.6f}",
                "mlp_error_km": f"{mlp_err[pos]:.6f}",
                "knn_error_km": f"{knn_err[pos]:.6f}",
                "final_nll": f"{final_nll[pos]:.8f}",
            })
        fields = list(rows[0].keys()) if rows else []
        write_csv(run_dir / f"{split_name}_predictions.tsv", rows, fields, delimiter="\t")
        return {"rows": rows, "final_err": final_err, "mlp_err": mlp_err, "knn_err": knn_err, "nll": final_nll, "alpha": arr["alpha"]}

    val_pred = predict_split("val", splits.val)
    test_pred = predict_split("test", splits.test)
    test_true = xy_km[splits.test]
    k1_mu = np.empty_like(test_true)
    for i, gi in enumerate(splits.test):
        # k=1 baseline from train-only reference, no validation/test leakage.
        q = features[int(gi)][None, :].astype(np.float64)
        d2 = ((q[:, None, :] - features[splits.train][None, :, :].astype(np.float64)) ** 2).sum(axis=2)[0]
        k1_mu[i] = xy_km[splits.train[int(np.argmin(d2))]]
    baseline = {}
    baseline.update(baseline_metrics("k1_nn", k1_mu, test_true))
    baseline.update(baseline_metrics("weighted_knn", (knn_mu[splits.test].astype(np.float64) + target_center_km[None, :]), test_true))
    baseline.update(baseline_metrics("mlp_expert", np.asarray([[float(r["mu_mlp_x_km"]), float(r["mu_mlp_y_km"])] for r in test_pred["rows"]]), test_true))
    baseline.update(baseline_metrics("gated", np.asarray([[float(r["mu_final_x_km"]), float(r["mu_final_y_km"])] for r in test_pred["rows"]]), test_true))

    alpha = test_pred["alpha"]
    alpha_summary = {
        "alpha_mean": float(alpha.mean()),
        "alpha_median": float(np.median(alpha)),
        "alpha_q10": float(np.percentile(alpha, 10)),
        "alpha_q25": float(np.percentile(alpha, 25)),
        "alpha_q50": float(np.percentile(alpha, 50)),
        "alpha_q75": float(np.percentile(alpha, 75)),
        "alpha_q90": float(np.percentile(alpha, 90)),
    }
    high = test_pred["final_err"][alpha > 0.75]
    low = test_pred["final_err"][alpha < 0.25]
    alpha_summary["median_error_alpha_gt_0_75"] = float(np.median(high)) if high.size else float("nan")
    alpha_summary["n_alpha_gt_0_75"] = int(high.size)
    alpha_summary["median_error_alpha_lt_0_25"] = float(np.median(low)) if low.size else float("nan")
    alpha_summary["n_alpha_lt_0_25"] = int(low.size)
    write_csv(run_dir / "alpha_summary.tsv", [alpha_summary], list(alpha_summary.keys()), delimiter="\t")

    def grouped(rows, key):
        groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        out = []
        for group, rs in groups.items():
            vals = np.asarray([float(r["final_error_km"]) for r in rs])
            al = np.asarray([float(r["alpha"]) for r in rs])
            out.append({key: group, "n_test": len(rs), "median_error_km": f"{np.median(vals):.6f}", "mean_error_km": f"{vals.mean():.6f}", "p90_error_km": f"{np.percentile(vals, 90):.6f}", "median_alpha": f"{np.median(al):.8f}", "mean_alpha": f"{al.mean():.8f}"})
        out.sort(key=lambda r: str(r[key]))
        return out

    age_rows = grouped(test_pred["rows"], "age_bin")
    age_order = {label: i for i, _, label in AGE_BINS}
    age_rows.sort(key=lambda r: age_order.get(str(r["age_bin"]), 99))
    write_csv(run_dir / "median_error_by_age_bin.tsv", age_rows, ["age_bin", "n_test", "median_error_km", "mean_error_km", "p90_error_km", "median_alpha", "mean_alpha"], delimiter="\t")
    seq_rows = grouped(test_pred["rows"], "data_type")
    seq_rows.sort(key=lambda r: (-int(r["n_test"]), str(r["data_type"])))
    write_csv(run_dir / "median_error_by_sequencing_method.tsv", seq_rows, ["data_type", "n_test", "median_error_km", "mean_error_km", "p90_error_km", "median_alpha", "mean_alpha"], delimiter="\t")

    alpha_bin_rows = []
    for lo, hi, label in [(0, .25, "0-0.25"), (.25, .5, "0.25-0.5"), (.5, .75, "0.5-0.75"), (.75, 1.0000001, "0.75-1")]:
        mask = (alpha >= lo) & (alpha < hi)
        vals = test_pred["final_err"][mask]
        alpha_bin_rows.append({"alpha_bin": label, "n_test": int(mask.sum()), "median_error_km": f"{np.median(vals):.6f}" if vals.size else "nan", "mean_error_km": f"{vals.mean():.6f}" if vals.size else "nan"})
    write_csv(run_dir / "error_by_alpha_bin.tsv", alpha_bin_rows, ["alpha_bin", "n_test", "median_error_km", "mean_error_km"], delimiter="\t")

    summary = {
        "status": "completed",
        "run_name": args.run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(sample_ids),
        "n_train": int(len(splits.train)),
        "n_val": int(len(splits.val)),
        "n_test": int(len(splits.test)),
        "best_epoch": int(saved["best_epoch"]),
        "best_val_nll": float(saved["best_val_nll"]),
        "test_gaussian_nll": float(np.mean(test_pred["nll"])),
        "val_gaussian_nll": float(np.mean(val_pred["nll"])),
        "knn_k": int(args.knn_k),
        "pca_dims": int(args.pca_dims),
        "target_center_km": target_center_km.tolist(),
        "conversion": kwargs["conversion_meta"],
        "prune_in": str(kwargs["prune_in"]),
        "eigenvec": str(kwargs["eigenvec"]),
        "eigenval": str(kwargs["eigenval"]),
        **baseline,
        **alpha_summary,
    }
    write_json(run_dir / "evaluation_summary.json", summary)
    write_json(run_dir / "run_summary.json", summary)
    metric_fields = [k for k, v in summary.items() if not isinstance(v, (dict, list))]
    write_csv(run_dir / "metrics.csv", [{k: summary[k] for k in metric_fields}], metric_fields)
    write_text(
        run_dir / "results.md",
        f"""
# Results

## Test Comparison

| model | median km | mean km | p90 km |
| --- | ---: | ---: | ---: |
| k=1 NN | {summary['k1_nn_median_km']:.3f} | {summary['k1_nn_mean_km']:.3f} | {summary['k1_nn_p90_km']:.3f} |
| weighted kNN k={args.knn_k} | {summary['weighted_knn_median_km']:.3f} | {summary['weighted_knn_mean_km']:.3f} | {summary['weighted_knn_p90_km']:.3f} |
| MLP expert | {summary['mlp_expert_median_km']:.3f} | {summary['mlp_expert_mean_km']:.3f} | {summary['mlp_expert_p90_km']:.3f} |
| gated two-expert | {summary['gated_median_km']:.3f} | {summary['gated_mean_km']:.3f} | {summary['gated_p90_km']:.3f} |

## Gate

Mean alpha: {summary['alpha_mean']:.4f}

Median alpha: {summary['alpha_median']:.4f}

Alpha q10/q25/q50/q75/q90: {summary['alpha_q10']:.4f}, {summary['alpha_q25']:.4f}, {summary['alpha_q50']:.4f}, {summary['alpha_q75']:.4f}, {summary['alpha_q90']:.4f}

## Files

- `test_predictions.tsv`
- `val_predictions.tsv`
- `alpha_summary.tsv`
- `median_error_by_age_bin.tsv`
- `median_error_by_sequencing_method.tsv`
- `error_by_alpha_bin.tsv`
""",
    )
    log(f"Final gated median={summary['gated_median_km']:.3f}km p90={summary['gated_p90_km']:.3f}km alpha_median={summary['alpha_median']:.3f}")


if __name__ == "__main__":
    main()
