#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geolocation_model.data import (  # noqa: E402
    build_metadata,
    make_leave_site_split,
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
from geolocation_model.run_precision_fusion import (  # noqa: E402
    AGE_BINS,
    age_bin,
    baseline_metrics,
    build_seq_multihot,
    compute_knn_features,
    error_km,
    inverse_project_xy_km,
    load_anno_features,
    standardize_from_train,
    str_to_bool,
)


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


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    return corrcoef_safe(rankdata_average(a[mask]), rankdata_average(b[mask]))


def write_logging_files(run_dir: Path, args: argparse.Namespace) -> None:
    write_text(
        run_dir / "hypothesis.md",
        """
# Hypothesis

## Problem

The previous precision-fusion model learned kNN uncertainty only indirectly through the final Gaussian NLL. The resulting alpha_precision behavior suggested that the kNN expert remained overconfident in some high-error cases.

## Motivation

Test whether directly supervising kNN uncertainty against the observed train-set kNN residual error produces a better calibrated local expert. The existing weighted kNN, MLP expert, learned gated model, and old precision_fusion implementation remain available unchanged for comparison.

## Hypothesis

Directly supervising kNN uncertainty should produce better calibrated sigma_knn_corrected, more meaningful alpha_precision, and better precision fusion. If successful, alpha_precision can be interpreted as how much a sample should trust local genetic analogues versus the global ancestry manifold.

## Success Criteria

Improve median error over weighted kNN and old precision_fusion, while reducing the previous failure mode where alpha_precision > 0.75 is worse than alpha_precision < 0.25.

## Failure Criteria

The supervised calibrator does not correlate with true kNN residuals, fusion does not improve, or high alpha_precision remains concentrated in poor predictions.

## Next Planned Experiments

If this works, try joint fine-tuning after supervised pretraining. If it fails, revisit the kNN feature space or replace local analogues with a different locality prior.
""",
    )
    write_text(
        run_dir / "config.yaml",
        f"""
experiment_type: precision_fusion_supervised_knn_calibration
model_name: GeoPrecisionFusionSupervisedKnnCalibration
aliases: [precision_fusion_supervised_knn, precision_fusion_error_calibrated, supervised_knn_sigma_fusion]
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
supervised_knn_calibration:
  epsilon_error_km: {args.epsilon_error_km}
  loss: {args.knn_sigma_loss}
  epochs: {args.calibrator_epochs}
  patience: {args.calibrator_patience}
  lr: {args.calibrator_lr}
  dropout: {args.calibrator_dropout}
features:
  use_sequencing_in_global: {args.use_sequencing_in_global}
  use_sequencing_in_calibrator: {args.use_sequencing_in_calibrator}
  use_coverage_in_calibrator: {args.use_coverage_in_calibrator}
sigma:
  sigma_min_km: {args.sigma_min_km}
  sigma_max_km: {args.sigma_max_km}
training:
  train_jointly: {args.train_jointly}
  global_mlp_epochs: {args.epochs}
  global_mlp_patience: {args.patience}
  optimizer: AdamW
  lr: {args.lr}
  weight_decay: {args.weight_decay}
  batch_size: {args.batch_size}
  jitter_sigma_km: {args.jitter_sigma_km}
""",
    )
    write_text(run_dir / "links.md", "# Links\n\nNo external links recorded yet.")
    write_text(run_dir / "results.md", "# Results\n\nRun started; results will be appended after training.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precision fusion with directly supervised kNN sigma calibration")
    parser.add_argument("--geno", required=True)
    parser.add_argument("--snp", required=True)
    parser.add_argument("--ind", required=True)
    parser.add_argument("--anno", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default=datetime.now().strftime("%Y-%m-%d_precision-fusion-supervised-knn_%H%M%S"))
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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--calibrator_epochs", type=int, default=500)
    parser.add_argument("--calibrator_patience", type=int, default=50)
    parser.add_argument("--calibrator_lr", type=float, default=1e-3)
    parser.add_argument("--calibrator_dropout", type=float, default=0.1)
    parser.add_argument("--knn_sigma_loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--epsilon_error_km", type=float, default=1.0)
    parser.add_argument("--jitter_sigma_km", type=float, default=25.0)
    parser.add_argument("--sigma_min_km", type=float, default=25.0)
    parser.add_argument("--sigma_max_km", type=float, default=3000.0)
    parser.add_argument("--use_sequencing_in_global", type=str_to_bool, default=True)
    parser.add_argument("--use_sequencing_in_calibrator", type=str_to_bool, default=True)
    parser.add_argument("--use_coverage_in_calibrator", type=str_to_bool, default=True)
    parser.add_argument("--train_jointly", type=str_to_bool, default=False)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_jointly:
        fail("train_jointly=True is reserved for a future experiment; this runner implements the requested two-stage training.")
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
    log(f"Run directory: {run_dir}")
    log(f"Inputs: geno={geno} snp={snp} ind={ind} anno={anno}")

    plink_prefix, sample_ids, conversion_meta = prepare_plink_bed(
        convertf_bin=convertf_bin,
        plink_bin=plink_bin,
        geno=geno,
        snp=snp,
        ind=ind,
        work_dir=run_dir / "work",
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
    data_type_by_id, coverage_by_id, cov_cols = load_anno_features(sample_ids, anno)
    if args.use_coverage_in_calibrator and cov_cols:
        log("Using coverage/call-rate proxy columns in calibrator: " + "; ".join(cov_cols))
    elif args.use_coverage_in_calibrator:
        log("WARNING: no coverage/call-rate proxy columns found; skipping coverage calibrator feature")
    seq_multihot, seq_tokens = build_seq_multihot(data_type_by_id, sample_ids)
    write_json(run_dir / "sequencing_encoding.json", {"tokens": seq_tokens})
    splits = make_leave_site_split(meta.locality, seed=args.seed, train_fraction=args.train_fraction, val_fraction=args.val_fraction)
    write_split_sites(run_dir / "split_sites.json", splits)
    pc_scaled, _, _ = standardize_train_apply(features_pc, splits.train)
    age_scaled, _, _ = standardize_train_apply(meta.date_bp.reshape(-1, 1), splits.train)
    features = np.column_stack([pc_scaled, age_scaled]).astype(np.float32)
    global_features = features if not args.use_sequencing_in_global else np.column_stack([features, seq_multihot]).astype(np.float32)
    np.save(run_dir / "pca_age_features.npy", features)
    np.save(run_dir / "global_features.npy", global_features)
    write_sample_stats(run_dir / "sample_stats.tsv", meta, splits, features)

    xy_km = meta.xy.astype(np.float64) / 1000.0
    target_center_km = xy_km[splits.train].mean(axis=0).astype(np.float64)
    centered_xy_km = xy_km - target_center_km[None, :]
    train_ref_x = features[splits.train]
    train_ref_y = centered_xy_km[splits.train]
    train_ref_age = meta.date_bp[splits.train]
    train_ref_global = splits.train.astype(np.int64)
    log(f"Samples: total={len(sample_ids)} train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")
    log(f"kNN reference size train={len(train_ref_x)} val_ref={len(train_ref_x)} test_ref={len(train_ref_x)}")
    log(f"Computing train-only kNN features, k={args.knn_k}")

    knn_by_split = {}
    for name, idx, exclude_self in [("train", splits.train, True), ("val", splits.val, False), ("test", splits.test, False)]:
        knn_by_split[name] = compute_knn_features(
            query_x=features[idx],
            query_global_idx=idx.astype(np.int64),
            ref_x=train_ref_x,
            ref_y_km=train_ref_y,
            ref_age=train_ref_age,
            ref_global_idx=train_ref_global,
            k=args.knn_k,
            eps=args.knn_eps,
            power=args.knn_power,
            exclude_self=exclude_self,
        )

    n = len(sample_ids)
    knn_mu = np.empty((n, 2), dtype=np.float64)
    raw_sigma = np.empty(n, dtype=np.float64)
    diag = np.empty((n, 12), dtype=np.float64)
    first_neighbor = np.empty(n, dtype=np.int64)
    for name, idx in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        kf = knn_by_split[name]
        knn_mu[idx] = kf["mu"]
        raw_sigma[idx] = kf["raw_sigma"]
        diag[idx] = np.column_stack([
            kf["raw_sigma"],
            kf["min_neighbor_distance"],
            kf["mean_neighbor_distance"],
            kf["max_neighbor_distance"],
            kf["std_neighbor_distance"],
            kf["d1"],
            kf["d5"],
            kf["d10"],
            kf["d20"],
            kf["weighted_spatial_variance"],
            kf["weighted_neighbor_age_mean"],
            kf["weighted_neighbor_age_std"],
        ])
        first_neighbor[idx] = kf["first_neighbor_global"]
    if np.any(first_neighbor[splits.train] == splits.train):
        fail("Self-neighbor exclusion check failed")
    if not np.isfinite(diag).all() or not np.isfinite(knn_mu).all() or not np.isfinite(raw_sigma).all():
        fail("NaNs in kNN features")

    diag_scaled, _, _ = standardize_from_train(diag[splits.train], diag)
    cov_feature = np.asarray([coverage_by_id[sid] for sid in sample_ids], dtype=np.float64).reshape(-1, 1)
    use_cov = bool(args.use_coverage_in_calibrator and np.isfinite(cov_feature).any())
    if use_cov:
        finite = np.isfinite(cov_feature[:, 0])
        fill = np.nanmedian(cov_feature[:, 0])
        cov_feature[~finite, 0] = fill
        cov_scaled, _, _ = standardize_from_train(cov_feature[splits.train], cov_feature)
    else:
        cov_scaled = np.zeros((n, 0), dtype=np.float32)
    calib_parts = [diag_scaled, age_scaled.astype(np.float32)]
    if args.use_sequencing_in_calibrator:
        calib_parts.append(seq_multihot)
    if use_cov:
        calib_parts.append(cov_scaled)
    calibrator_features = np.column_stack(calib_parts).astype(np.float32)
    if not np.isfinite(calibrator_features).all():
        fail("NaNs in supervised kNN calibrator features")
    np.save(run_dir / "knn_diag_features.npy", diag.astype(np.float32))
    np.save(run_dir / "calibrator_features.npy", calibrator_features)

    train_model(
        args=args,
        run_dir=run_dir,
        sample_ids=sample_ids,
        global_features=global_features,
        calibrator_features=calibrator_features,
        knn_mu=knn_mu.astype(np.float32),
        raw_sigma=raw_sigma.astype(np.float32),
        centered_xy_km=centered_xy_km.astype(np.float32),
        xy_km=xy_km,
        target_center_km=target_center_km,
        splits=splits,
        meta=meta,
        data_type_by_id=data_type_by_id,
        seq_tokens=seq_tokens,
        projection=meta.projection,
        conversion_meta=conversion_meta,
        prune_in=prune_in,
        eigenvec=eigenvec,
        eigenval=eigenval,
    )


def train_model(**kw) -> None:
    args = kw["args"]
    run_dir = kw["run_dir"]
    sample_ids = kw["sample_ids"]
    global_features = kw["global_features"]
    calibrator_features = kw["calibrator_features"]
    knn_mu = kw["knn_mu"]
    raw_sigma = kw["raw_sigma"]
    centered_xy_km = kw["centered_xy_km"]
    xy_km = kw["xy_km"]
    target_center_km = kw["target_center_km"]
    splits = kw["splits"]
    meta = kw["meta"]
    data_type_by_id = kw["data_type_by_id"]
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        fail(f"PyTorch is required but unavailable: {exc}")

    class KnnSigmaCalibrator(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.GELU(),
                nn.Dropout(args.calibrator_dropout),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(1)

    class GlobalMlpExpert(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.softplus = nn.Softplus()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.GELU(),
                nn.BatchNorm1d(512),
                nn.Dropout(args.dropout),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.BatchNorm1d(256),
                nn.Dropout(args.dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.BatchNorm1d(128),
                nn.Dropout(args.dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 3),
            )

        def sigma_from_raw(self, raw):
            return (float(args.sigma_min_km) + self.softplus(raw)).clamp(min=float(args.sigma_min_km), max=float(args.sigma_max_km))

        def forward(self, x):
            out = self.net(x)
            return out[:, :2], self.sigma_from_raw(out[:, 2])

    def mlp_nll(mu, sigma, target, train_mode: bool):
        sigma2 = sigma * sigma
        if train_mode and args.jitter_sigma_km > 0:
            sigma2 = sigma2 + float(args.jitter_sigma_km) ** 2
        dist2 = torch.sum((target - mu) ** 2, dim=1)
        return (0.5 * dist2 / sigma2 + 0.5 * torch.log(sigma2)).mean()

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    log(f"Training supervised kNN calibrator and MLP expert on device: {device}")

    gx = torch.tensor(global_features, dtype=torch.float32)
    cx = torch.tensor(calibrator_features, dtype=torch.float32)
    kmu = torch.tensor(knn_mu, dtype=torch.float32)
    y = torch.tensor(centered_xy_km, dtype=torch.float32)

    true_knn_error = np.linalg.norm(centered_xy_km.astype(np.float64) - knn_mu.astype(np.float64), axis=1)
    target_log_sigma = np.log(true_knn_error + float(args.epsilon_error_km)).astype(np.float32)
    np.save(run_dir / "knn_true_error_km.npy", true_knn_error.astype(np.float32))
    np.save(run_dir / "knn_target_log_sigma.npy", target_log_sigma)

    # Stage A: supervised kNN sigma calibration.
    calibrator = KnnSigmaCalibrator(calibrator_features.shape[1]).to(device)
    opt_cal = torch.optim.AdamW(calibrator.parameters(), lr=float(args.calibrator_lr), weight_decay=float(args.weight_decay))
    cal_loss_fn = nn.SmoothL1Loss() if args.knn_sigma_loss == "huber" else nn.MSELoss()
    cal_train_ds = TensorDataset(cx[splits.train], torch.tensor(target_log_sigma[splits.train], dtype=torch.float32))
    cal_loader = DataLoader(cal_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    cal_val_x = cx[splits.val].to(device)
    cal_val_y = torch.tensor(target_log_sigma[splits.val], dtype=torch.float32).to(device)
    best_cal = float("inf")
    best_cal_epoch = 0
    cal_stop_epoch = 0
    cal_patience = int(args.calibrator_patience)
    cal_history = []
    cal_ckpt = run_dir / "best_supervised_knn_calibrator.pt"
    for epoch in range(1, int(args.calibrator_epochs) + 1):
        calibrator.train()
        losses = []
        for bx, by in cal_loader:
            bx, by = bx.to(device), by.to(device)
            opt_cal.zero_grad(set_to_none=True)
            loss = cal_loss_fn(calibrator(bx), by)
            loss.backward()
            opt_cal.step()
            losses.append(float(loss.detach().cpu()))
        calibrator.eval()
        with torch.no_grad():
            val_loss = float(cal_loss_fn(calibrator(cal_val_x), cal_val_y).detach().cpu())
        train_loss = float(np.mean(losses))
        cal_history.append({"epoch": epoch, "train_calibration_loss": train_loss, "val_calibration_loss": val_loss})
        if epoch == 1 or epoch % 25 == 0:
            log(f"calibrator_epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_cal - 1e-7:
            best_cal = val_loss
            best_cal_epoch = epoch
            cal_patience = int(args.calibrator_patience)
            torch.save({"model_state": calibrator.state_dict(), "best_val_loss": best_cal, "best_epoch": best_cal_epoch}, cal_ckpt)
        else:
            cal_patience -= 1
            if cal_patience <= 0:
                cal_stop_epoch = epoch
                log(f"Calibrator early stopping at epoch={epoch}; best_epoch={best_cal_epoch}")
                break
    if cal_stop_epoch == 0:
        cal_stop_epoch = epoch
    write_csv(run_dir / "calibrator_training_history.csv", cal_history, ["epoch", "train_calibration_loss", "val_calibration_loss"])
    cal_saved = torch.load(cal_ckpt, map_location=device)
    calibrator.load_state_dict(cal_saved["model_state"])
    calibrator.eval()
    for param in calibrator.parameters():
        param.requires_grad_(False)

    # Stage B: global MLP expert.
    mlp = GlobalMlpExpert(global_features.shape[1]).to(device)
    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    mlp_train_ds = TensorDataset(gx[splits.train], y[splits.train])
    mlp_loader = DataLoader(mlp_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_x = gx[splits.val].to(device)
    val_y = y[splits.val].to(device)
    best_mlp = float("inf")
    best_mlp_epoch = 0
    mlp_stop_epoch = 0
    mlp_patience = int(args.patience)
    mlp_history = []
    mlp_ckpt = run_dir / "best_global_mlp_expert.pt"
    for epoch in range(1, int(args.epochs) + 1):
        mlp.train()
        losses = []
        for bx, by in mlp_loader:
            bx, by = bx.to(device), by.to(device)
            target = by + torch.randn_like(by) * float(args.jitter_sigma_km) if args.jitter_sigma_km > 0 else by
            opt_mlp.zero_grad(set_to_none=True)
            mu, sigma = mlp(bx)
            loss = mlp_nll(mu, sigma, target, True)
            loss.backward()
            opt_mlp.step()
            losses.append(float(loss.detach().cpu()))
        mlp.eval()
        with torch.no_grad():
            mu, sigma = mlp(val_x)
            val_loss = float(mlp_nll(mu, sigma, val_y, False).detach().cpu())
            val_err = error_km(mu.detach().cpu().numpy() + target_center_km[None, :], xy_km[splits.val])
            val_med = float(np.median(val_err))
        train_loss = float(np.mean(losses))
        mlp_history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss, "val_median_km": val_med})
        if epoch == 1 or epoch % 25 == 0:
            log(f"mlp_epoch={epoch} train_nll={train_loss:.6f} val_nll={val_loss:.6f} val_median_km={val_med:.3f}")
        if val_loss < best_mlp - 1e-6:
            best_mlp = val_loss
            best_mlp_epoch = epoch
            mlp_patience = int(args.patience)
            torch.save({"model_state": mlp.state_dict(), "best_val_nll": best_mlp, "best_epoch": best_mlp_epoch}, mlp_ckpt)
        else:
            mlp_patience -= 1
            if mlp_patience <= 0:
                mlp_stop_epoch = epoch
                log(f"MLP early stopping at epoch={epoch}; best_epoch={best_mlp_epoch}")
                break
    if mlp_stop_epoch == 0:
        mlp_stop_epoch = epoch
    write_csv(run_dir / "mlp_training_history.csv", mlp_history, ["epoch", "train_nll", "val_nll", "val_median_km"])
    mlp_saved = torch.load(mlp_ckpt, map_location=device)
    mlp.load_state_dict(mlp_saved["model_state"])
    mlp.eval()

    def forward_split(idx: np.ndarray) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            pred_log_sigma = calibrator(cx[idx].to(device)).detach().cpu().numpy().astype(np.float64)
            mu_mlp_t, sigma_mlp_t = mlp(gx[idx].to(device))
            mu_mlp = mu_mlp_t.detach().cpu().numpy().astype(np.float64)
            sigma_mlp = sigma_mlp_t.detach().cpu().numpy().astype(np.float64)
        log_sigma_min = math.log(float(args.sigma_min_km))
        log_sigma_max = math.log(float(args.sigma_max_km))
        sigma_knn = np.exp(np.clip(pred_log_sigma, log_sigma_min, log_sigma_max))
        sigma_knn = np.clip(sigma_knn, float(args.sigma_min_km), float(args.sigma_max_km))
        sigma_mlp = np.clip(sigma_mlp, float(args.sigma_min_km), float(args.sigma_max_km))
        var_knn = sigma_knn * sigma_knn
        var_mlp = sigma_mlp * sigma_mlp
        w_knn = 1.0 / var_knn
        w_mlp = 1.0 / var_mlp
        denom = w_knn + w_mlp
        mu_final = (w_knn[:, None] * knn_mu[idx].astype(np.float64) + w_mlp[:, None] * mu_mlp) / denom[:, None]
        sigma_final = np.sqrt(1.0 / denom)
        sigma_final = np.clip(sigma_final, float(args.sigma_min_km), float(args.sigma_max_km))
        alpha = np.clip(w_knn / denom, 0.0, 1.0)
        if not np.isfinite(alpha).all() or not ((alpha >= -1e-6) & (alpha <= 1.0 + 1e-6)).all():
            fail("alpha_precision contains invalid values")
        return {
            "pred_log_sigma_knn": pred_log_sigma,
            "sigma_knn_corrected": sigma_knn,
            "mu_mlp": mu_mlp,
            "sigma_mlp": sigma_mlp,
            "mu_final": mu_final,
            "sigma_final": sigma_final,
            "alpha_precision": alpha,
        }

    def predict(name: str, idx: np.ndarray) -> Dict[str, object]:
        arr = forward_split(idx)
        mu_final_abs = arr["mu_final"] + target_center_km[None, :]
        mu_knn_abs = knn_mu[idx].astype(np.float64) + target_center_km[None, :]
        mu_mlp_abs = arr["mu_mlp"] + target_center_km[None, :]
        true = xy_km[idx]
        err = error_km(mu_final_abs, true)
        knn_err = error_km(mu_knn_abs, true)
        nllv = 0.5 * np.sum((true - mu_final_abs) ** 2, axis=1) / (arr["sigma_final"] ** 2) + np.log(arr["sigma_final"])
        pred_latlon = inverse_project_xy_km(mu_final_abs, kw["projection"])
        rows = []
        for pos, gi in enumerate(idx):
            sid = sample_ids[int(gi)]
            row = {
                "sample_id": sid,
                "split": name,
                "true_x": f"{true[pos, 0]:.6f}",
                "true_y": f"{true[pos, 1]:.6f}",
                "true_lat": f"{meta.lat[int(gi)]:.8f}",
                "true_lon": f"{meta.lon[int(gi)]:.8f}",
                "pred_x": f"{mu_final_abs[pos, 0]:.6f}",
                "pred_y": f"{mu_final_abs[pos, 1]:.6f}",
                "pred_lat": f"{pred_latlon[pos, 0]:.8f}",
                "pred_lon": f"{pred_latlon[pos, 1]:.8f}",
                "error_km": f"{err[pos]:.6f}",
                "nll": f"{nllv[pos]:.8f}",
                "mu_knn_x": f"{mu_knn_abs[pos, 0]:.6f}",
                "mu_knn_y": f"{mu_knn_abs[pos, 1]:.6f}",
                "raw_sigma_knn": f"{raw_sigma[int(gi)]:.6f}",
                "sigma_knn_corrected": f"{arr['sigma_knn_corrected'][pos]:.6f}",
                "predicted_log_sigma_knn": f"{arr['pred_log_sigma_knn'][pos]:.8f}",
                "mu_mlp_x": f"{mu_mlp_abs[pos, 0]:.6f}",
                "mu_mlp_y": f"{mu_mlp_abs[pos, 1]:.6f}",
                "sigma_mlp": f"{arr['sigma_mlp'][pos]:.6f}",
                "sigma_final": f"{arr['sigma_final'][pos]:.6f}",
                "alpha_precision": f"{arr['alpha_precision'][pos]:.8f}",
                "sequencing_type": data_type_by_id.get(sid, "missing"),
                "age": f"{meta.date_bp[int(gi)]:.6f}",
                "age_bin": age_bin(float(meta.date_bp[int(gi)])),
                "locality_id": meta.locality[int(gi)],
                "true_knn_error_km": f"{knn_err[pos]:.6f}" if name == "train" else "",
            }
            rows.append(row)
        write_csv(run_dir / f"{name}_predictions.tsv", rows, list(rows[0].keys()), delimiter="\t")
        return {
            "rows": rows,
            "err": err,
            "nll": nllv,
            "alpha": arr["alpha_precision"],
            "mu_final": mu_final_abs,
            "mu_mlp": mu_mlp_abs,
            "mu_knn": mu_knn_abs,
            "sigma_knn": arr["sigma_knn_corrected"],
            "knn_err": knn_err,
        }

    trainp = predict("train", splits.train)
    valp = predict("val", splits.val)
    testp = predict("test", splits.test)
    test_true = xy_km[splits.test]

    k1 = np.empty_like(test_true)
    pca_age_dim = int(args.pca_dims) + 1
    ref_x = global_features[splits.train, :pca_age_dim].astype(np.float64)
    for i, gi in enumerate(splits.test):
        q = global_features[np.array([gi]), :pca_age_dim].astype(np.float64)[0]
        d2 = ((ref_x - q[None, :]) ** 2).sum(axis=1)
        k1[i] = xy_km[splits.train[int(np.argmin(d2))]]

    weighted_knn_mu = knn_mu[splits.test].astype(np.float64) + target_center_km[None, :]
    baseline = {}
    baseline.update(baseline_metrics("k1_nn", k1, test_true))
    baseline.update(baseline_metrics("weighted_knn", weighted_knn_mu, test_true))
    baseline.update(baseline_metrics("mlp_expert", testp["mu_mlp"], test_true))
    baseline.update(baseline_metrics("supervised_precision_fusion", testp["mu_final"], test_true))

    alpha = testp["alpha"]
    alpha_summary = {
        "alpha_precision_mean": float(alpha.mean()),
        "alpha_precision_median": float(np.median(alpha)),
        "alpha_precision_q10": float(np.percentile(alpha, 10)),
        "alpha_precision_q25": float(np.percentile(alpha, 25)),
        "alpha_precision_q50": float(np.percentile(alpha, 50)),
        "alpha_precision_q75": float(np.percentile(alpha, 75)),
        "alpha_precision_q90": float(np.percentile(alpha, 90)),
    }
    high = testp["err"][alpha > 0.75]
    low = testp["err"][alpha < 0.25]
    alpha_summary.update({
        "median_error_alpha_precision_gt_0_75": float(np.median(high)) if high.size else float("nan"),
        "n_alpha_precision_gt_0_75": int(high.size),
        "median_error_alpha_precision_lt_0_25": float(np.median(low)) if low.size else float("nan"),
        "n_alpha_precision_lt_0_25": int(low.size),
    })
    write_csv(run_dir / "alpha_precision_summary.tsv", [alpha_summary], list(alpha_summary.keys()), delimiter="\t")

    cal_rows = []
    for split_name, pred in [("train", trainp), ("val", valp), ("test", testp)]:
        cal_rows.append({
            "split": split_name,
            "pearson_sigma_vs_true_knn_error": corrcoef_safe(pred["sigma_knn"], pred["knn_err"]),
            "spearman_sigma_vs_true_knn_error": spearman_safe(pred["sigma_knn"], pred["knn_err"]),
            "mean_sigma_knn_corrected": float(np.mean(pred["sigma_knn"])),
            "mean_true_knn_error": float(np.mean(pred["knn_err"])),
            "median_sigma_knn_corrected": float(np.median(pred["sigma_knn"])),
            "median_true_knn_error": float(np.median(pred["knn_err"])),
        })
    write_csv(run_dir / "knn_sigma_correlation.tsv", cal_rows, list(cal_rows[0].keys()), delimiter="\t")

    decile_rows = []
    for split_name, pred in [("train", trainp), ("val", valp), ("test", testp)]:
        sigma = np.asarray(pred["sigma_knn"], dtype=np.float64)
        err = np.asarray(pred["knn_err"], dtype=np.float64)
        edges = np.percentile(sigma, np.arange(0, 101, 10))
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            mask = (sigma >= lo) & (sigma <= hi if i == 9 else sigma < hi)
            vals = err[mask]
            sig = sigma[mask]
            decile_rows.append({
                "split": split_name,
                "sigma_decile": i + 1,
                "n": int(mask.sum()),
                "sigma_min": f"{lo:.6f}",
                "sigma_max": f"{hi:.6f}",
                "mean_predicted_sigma": f"{sig.mean():.6f}" if sig.size else "nan",
                "median_predicted_sigma": f"{np.median(sig):.6f}" if sig.size else "nan",
                "mean_observed_knn_error": f"{vals.mean():.6f}" if vals.size else "nan",
                "median_observed_knn_error": f"{np.median(vals):.6f}" if vals.size else "nan",
            })
    write_csv(run_dir / "knn_sigma_calibration_deciles.tsv", decile_rows, list(decile_rows[0].keys()), delimiter="\t")

    def grouped(rows, key):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        out = []
        for group, rs in groups.items():
            vals = np.asarray([float(r["error_km"]) for r in rs])
            al = np.asarray([float(r["alpha_precision"]) for r in rs])
            out.append({
                key: group,
                "n_test": len(rs),
                "median_error_km": f"{np.median(vals):.6f}",
                "mean_error_km": f"{vals.mean():.6f}",
                "p90_error_km": f"{np.percentile(vals, 90):.6f}",
                "median_alpha_precision": f"{np.median(al):.8f}",
                "mean_alpha_precision": f"{al.mean():.8f}",
            })
        return out

    age_rows = grouped(testp["rows"], "age_bin")
    order = {label: i for i, _, label in AGE_BINS}
    age_rows.sort(key=lambda r: order.get(str(r["age_bin"]), 99))
    write_csv(run_dir / "median_error_by_age_bin.tsv", age_rows, ["age_bin", "n_test", "median_error_km", "mean_error_km", "p90_error_km", "median_alpha_precision", "mean_alpha_precision"], delimiter="\t")
    seq_rows = grouped(testp["rows"], "sequencing_type")
    seq_rows.sort(key=lambda r: (-int(r["n_test"]), str(r["sequencing_type"])))
    write_csv(run_dir / "median_error_by_sequencing_method.tsv", seq_rows, ["sequencing_type", "n_test", "median_error_km", "mean_error_km", "p90_error_km", "median_alpha_precision", "mean_alpha_precision"], delimiter="\t")
    alpha_bin_rows = []
    for lo, hi, label in [(0, 0.25, "0-0.25"), (0.25, 0.5, "0.25-0.5"), (0.5, 0.75, "0.5-0.75"), (0.75, 1.0000001, "0.75-1")]:
        mask = (alpha >= lo) & (alpha < hi)
        vals = testp["err"][mask]
        alpha_bin_rows.append({
            "alpha_precision_bin": label,
            "n_test": int(mask.sum()),
            "median_error_km": f"{np.median(vals):.6f}" if vals.size else "nan",
            "mean_error_km": f"{vals.mean():.6f}" if vals.size else "nan",
        })
    write_csv(run_dir / "error_by_alpha_precision_bin.tsv", alpha_bin_rows, ["alpha_precision_bin", "n_test", "median_error_km", "mean_error_km"], delimiter="\t")

    learned_gate = {}
    prior_gate = run_dir.parent / "2026-06-09_ho-gated-knn-mlp-k20-ldpca256" / "evaluation_summary.json"
    if prior_gate.is_file():
        try:
            p = json.loads(prior_gate.read_text())
            learned_gate = {"learned_gated_median_km": p.get("gated_median_km"), "learned_gated_mean_km": p.get("gated_mean_km"), "learned_gated_p90_km": p.get("gated_p90_km")}
        except Exception:
            pass
    old_precision = {}
    prior_precision = run_dir.parent / "2026-06-09_ho-precision-fusion-k20-ldpca256-epochs400-patience50" / "evaluation_summary.json"
    if prior_precision.is_file():
        try:
            p = json.loads(prior_precision.read_text())
            old_precision = {"old_precision_fusion_median_km": p.get("precision_fusion_median_km"), "old_precision_fusion_mean_km": p.get("precision_fusion_mean_km"), "old_precision_fusion_p90_km": p.get("precision_fusion_p90_km")}
        except Exception:
            pass

    summary = {
        "status": "completed",
        "experiment_type": "precision_fusion_supervised_knn_calibration",
        "run_name": args.run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(sample_ids),
        "n_train": int(len(splits.train)),
        "n_val": int(len(splits.val)),
        "n_test": int(len(splits.test)),
        "calibrator_best_epoch": int(cal_saved["best_epoch"]),
        "calibrator_early_stopping_epoch": int(cal_stop_epoch),
        "calibrator_best_val_loss": float(cal_saved["best_val_loss"]),
        "mlp_best_epoch": int(mlp_saved["best_epoch"]),
        "mlp_early_stopping_epoch": int(mlp_stop_epoch),
        "mlp_best_val_nll": float(mlp_saved["best_val_nll"]),
        "test_gaussian_nll": float(np.mean(testp["nll"])),
        "val_gaussian_nll": float(np.mean(valp["nll"])),
        "knn_k": int(args.knn_k),
        "pca_dims": int(args.pca_dims),
        "target_center_km": target_center_km.tolist(),
        "sequencing_tokens": kw["seq_tokens"],
        "conversion": kw["conversion_meta"],
        "prune_in": str(kw["prune_in"]),
        "eigenvec": str(kw["eigenvec"]),
        "eigenval": str(kw["eigenval"]),
        **baseline,
        **learned_gate,
        **old_precision,
        **alpha_summary,
        "test_knn_sigma_pearson": corrcoef_safe(testp["sigma_knn"], testp["knn_err"]),
        "test_knn_sigma_spearman": spearman_safe(testp["sigma_knn"], testp["knn_err"]),
    }
    write_json(run_dir / "evaluation_summary.json", summary)
    write_json(run_dir / "run_summary.json", summary)
    fields = [k for k, v in summary.items() if not isinstance(v, (dict, list))]
    write_csv(run_dir / "metrics.csv", [{k: summary[k] for k in fields}], fields)

    learned_row = ""
    if learned_gate:
        learned_row = f"| learned gated two-expert | {learned_gate['learned_gated_median_km']:.3f} | {learned_gate['learned_gated_mean_km']:.3f} | {learned_gate['learned_gated_p90_km']:.3f} |\n"
    old_precision_row = ""
    if old_precision:
        old_precision_row = f"| old precision_fusion | {old_precision['old_precision_fusion_median_km']:.3f} | {old_precision['old_precision_fusion_mean_km']:.3f} | {old_precision['old_precision_fusion_p90_km']:.3f} |\n"
    write_text(run_dir / "results.md", f"""
# Results

## Test Comparison

| model | median km | mean km | p90 km |
| --- | ---: | ---: | ---: |
| k=1 NN | {summary['k1_nn_median_km']:.3f} | {summary['k1_nn_mean_km']:.3f} | {summary['k1_nn_p90_km']:.3f} |
| weighted kNN k={args.knn_k} | {summary['weighted_knn_median_km']:.3f} | {summary['weighted_knn_mean_km']:.3f} | {summary['weighted_knn_p90_km']:.3f} |
| MLP expert | {summary['mlp_expert_median_km']:.3f} | {summary['mlp_expert_mean_km']:.3f} | {summary['mlp_expert_p90_km']:.3f} |
{learned_row}{old_precision_row}| supervised precision_fusion | {summary['supervised_precision_fusion_median_km']:.3f} | {summary['supervised_precision_fusion_mean_km']:.3f} | {summary['supervised_precision_fusion_p90_km']:.3f} |

## Supervised kNN Sigma Calibration

Test Pearson sigma vs true kNN error: {summary['test_knn_sigma_pearson']:.4f}

Test Spearman sigma vs true kNN error: {summary['test_knn_sigma_spearman']:.4f}

Calibrator best epoch: {summary['calibrator_best_epoch']} / stopped at {summary['calibrator_early_stopping_epoch']}

## Precision Fusion Alpha

Mean alpha_precision: {summary['alpha_precision_mean']:.4f}

Median alpha_precision: {summary['alpha_precision_median']:.4f}

Alpha q10/q25/q50/q75/q90: {summary['alpha_precision_q10']:.4f}, {summary['alpha_precision_q25']:.4f}, {summary['alpha_precision_q50']:.4f}, {summary['alpha_precision_q75']:.4f}, {summary['alpha_precision_q90']:.4f}

Median error alpha_precision > 0.75: {summary['median_error_alpha_precision_gt_0_75']:.3f} km (n={summary['n_alpha_precision_gt_0_75']})

Median error alpha_precision < 0.25: {summary['median_error_alpha_precision_lt_0_25']:.3f} km (n={summary['n_alpha_precision_lt_0_25']})

## Files

- `train_predictions.tsv`
- `val_predictions.tsv`
- `test_predictions.tsv`
- `knn_sigma_correlation.tsv`
- `knn_sigma_calibration_deciles.tsv`
- `alpha_precision_summary.tsv`
- `median_error_by_age_bin.tsv`
- `median_error_by_sequencing_method.tsv`
- `error_by_alpha_precision_bin.tsv`
""")
    log(f"Final supervised precision fusion median={summary['supervised_precision_fusion_median_km']:.3f}km p90={summary['supervised_precision_fusion_p90_km']:.3f}km alpha_precision_median={summary['alpha_precision_median']:.3f}")


if __name__ == "__main__":
    main()
