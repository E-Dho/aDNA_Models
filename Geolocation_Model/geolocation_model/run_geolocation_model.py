#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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
from geolocation_model.geo import calibration_containment, euclidean_km  # noqa: E402
from geolocation_model.plink import (  # noqa: E402
    fail,
    load_plink_eigenvec,
    prepare_plink_bed,
    require_file,
    resolve_executable,
    run_plink_ld_pca,
)


def log(message: str) -> None:
    print(message, flush=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_logging_files(run_dir: Path, args: argparse.Namespace) -> None:
    write_text(
        run_dir / "hypothesis.md",
        """
# Hypothesis

## Problem

Current methods rely too much on contrastive label learning; this model should rely on genotype and date BP to infer geolocation signal.

## Motivation

Create a baseline geolocation model to test feasibility.

## Hypothesis

A geolocation model may work but still needs tuning. Missingness may hurt the signal, but PCA may smooth it enough for a useful baseline.

## Success Criteria

Median geolocation error: 150-250 km.

90th percentile geolocation error: 300-500 km.

## Failure Criteria

Median geolocation error greater than 300 km, or not better than the nearest-neighbor PCA baseline around 250 km.

## Next Planned Experiments

Fine tune the model depending on the results.
""",
    )
    write_text(
        run_dir / "config.yaml",
        f"""
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
  coordinate_crs: EPSG:3035
model:
  input_dim: {args.pca_dims + 1}
  hidden_layers: [512, 256, 128, 64]
  output_dim: 3
  output_interpretation: [mu_x, mu_y, log_sigma]
  dropout: {args.dropout}
training:
  optimizer: AdamW
  lr: {args.lr}
  weight_decay: {args.weight_decay}
  batch_size: {args.batch_size}
  max_epochs: {args.epochs}
  patience: {args.patience}
  coordinate_jitter_sigma_km: {args.jitter_sigma_km}
  min_sigma_km: {args.min_sigma_km}
  max_sigma_km: {args.max_sigma_km}
split:
  type: leave-site-out
  site_column: Locality
  train_fraction: {args.train_fraction}
  val_fraction: {args.val_fraction}
cluster:
  run_name: {args.run_name}
  plink_bin: {args.plink_bin or os.environ.get('PLINK_BIN', 'plink')}
  convertf_bin: {args.convertf_bin or os.environ.get('CONVERTF_BIN', '')}
""",
    )
    write_text(run_dir / "links.md", "# Links\n\nNo external links recorded yet.")
    write_text(run_dir / "results.md", "# Results\n\nRun started; results will be appended after training.\n")


def nearest_neighbor_baseline(features: np.ndarray, xy: np.ndarray, splits) -> Dict[str, float]:
    train_x = features[splits.train].astype(np.float64)
    train_y = xy[splits.train].astype(np.float64)
    test_x = features[splits.test].astype(np.float64)
    test_y = xy[splits.test].astype(np.float64)
    preds = np.empty_like(test_y)
    chunk = 512
    for start in range(0, len(test_x), chunk):
        end = min(start + chunk, len(test_x))
        block = test_x[start:end]
        d = ((block[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d, axis=1)
        preds[start:end] = train_y[idx]
    errors = euclidean_km(preds, test_y)
    return {
        "nn_pca_age_median_km": float(np.median(errors)),
        "nn_pca_age_p90_km": float(np.percentile(errors, 90)),
        "nn_pca_age_mean_km": float(np.mean(errors)),
    }


def run_training(features: np.ndarray, xy: np.ndarray, splits, args: argparse.Namespace, run_dir: Path) -> Dict[str, object]:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from geolocation_model.model import GeoMLP, isotropic_gaussian_nll
    except ImportError as exc:
        fail(f"PyTorch is required for training but is unavailable: {exc}")

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    log(f"Training on device: {device}")

    target_center = xy[splits.train].mean(axis=0, keepdims=True).astype(np.float32)
    centered_xy = (xy.astype(np.float32) - target_center).astype(np.float32)
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(centered_xy, dtype=torch.float32)

    train_ds = TensorDataset(x_tensor[splits.train], y_tensor[splits.train])
    val_x = x_tensor[splits.val].to(device)
    val_y = y_tensor[splits.val].to(device)
    test_x = x_tensor[splits.test].to(device)
    test_y_centered = y_tensor[splits.test].to(device)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = GeoMLP(input_dim=features.shape[1], dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    min_log_sigma = math.log(float(args.min_sigma_km) * 1000.0)
    max_log_sigma = math.log(float(args.max_sigma_km) * 1000.0)
    jitter_sigma_m = float(args.jitter_sigma_km) * 1000.0

    best_val = float("inf")
    best_epoch = 0
    patience_left = int(args.patience)
    history: list[dict[str, object]] = []
    checkpoint = run_dir / "best_model.pt"

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            if jitter_sigma_m > 0:
                by = by + torch.randn_like(by) * jitter_sigma_m
            opt.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = isotropic_gaussian_nll(pred, by, min_log_sigma, max_log_sigma)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(isotropic_gaussian_nll(val_pred, val_y, min_log_sigma, max_log_sigma).detach().cpu())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss})
        if epoch == 1 or epoch % 10 == 0:
            log(f"epoch={epoch} train_nll={train_loss:.6f} val_nll={val_loss:.6f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            patience_left = int(args.patience)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "target_center": target_center.reshape(-1).tolist(),
                    "input_dim": int(features.shape[1]),
                    "best_val_nll": best_val,
                    "best_epoch": best_epoch,
                    "min_log_sigma": min_log_sigma,
                    "max_log_sigma": max_log_sigma,
                },
                checkpoint,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break

    write_csv(run_dir / "training_history.csv", history, ["epoch", "train_nll", "val_nll"])
    if not checkpoint.is_file():
        fail("No best model checkpoint was written")
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    with torch.no_grad():
        test_pred_centered = model(test_x)
        test_nll = float(isotropic_gaussian_nll(test_pred_centered, test_y_centered, min_log_sigma, max_log_sigma).detach().cpu())
        pred_np = test_pred_centered.detach().cpu().numpy().astype(np.float64)
    target_center_np = np.asarray(saved["target_center"], dtype=np.float64).reshape(1, 2)
    pred_xy = pred_np[:, :2] + target_center_np
    true_xy = xy[splits.test].astype(np.float64)
    log_sigma = np.clip(pred_np[:, 2], min_log_sigma, max_log_sigma)
    sigma_m = np.exp(log_sigma)
    errors_km = euclidean_km(pred_xy, true_xy)
    containment = calibration_containment(errors_km * 1000.0, sigma_m)

    pred_rows = []
    for j, idx in enumerate(splits.test):
        pred_rows.append(
            {
                "sample_index": int(idx),
                "pred_x_m": f"{pred_xy[j, 0]:.6f}",
                "pred_y_m": f"{pred_xy[j, 1]:.6f}",
                "true_x_m": f"{true_xy[j, 0]:.6f}",
                "true_y_m": f"{true_xy[j, 1]:.6f}",
                "sigma_km": f"{sigma_m[j] / 1000.0:.6f}",
                "error_km": f"{errors_km[j]:.6f}",
            }
        )
    write_csv(run_dir / "test_predictions.csv", pred_rows, ["sample_index", "pred_x_m", "pred_y_m", "true_x_m", "true_y_m", "sigma_km", "error_km"])

    return {
        "device": str(device),
        "best_epoch": int(saved["best_epoch"]),
        "best_val_nll": float(saved["best_val_nll"]),
        "test_gaussian_nll": test_nll,
        "median_km": float(np.median(errors_km)),
        "p90_km": float(np.percentile(errors_km, 90)),
        "mean_km": float(np.mean(errors_km)),
        "sigma_median_km": float(np.median(sigma_m) / 1000.0),
        "sigma_mean_km": float(np.mean(sigma_m) / 1000.0),
        **containment,
    }


def append_index(model_root: Path, run_name: str, summary: Dict[str, object]) -> None:
    index = model_root / "runs" / "index.csv"
    index.parent.mkdir(parents=True, exist_ok=True)
    exists = index.is_file()
    fields = ["run_name", "timestamp", "status", "n_samples", "pca_dims", "median_km", "p90_km", "test_gaussian_nll", "nn_pca_age_median_km"]
    row = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": summary.get("status", "completed"),
        "n_samples": summary.get("n_samples", ""),
        "pca_dims": summary.get("pca_dims", ""),
        "median_km": summary.get("median_km", ""),
        "p90_km": summary.get("p90_km", ""),
        "test_gaussian_nll": summary.get("test_gaussian_nll", ""),
        "nn_pca_age_median_km": summary.get("nn_pca_age_median_km", ""),
    }
    with index.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LD-pruned PCA plus age-conditioned Gaussian MLP geolocation model")
    parser.add_argument("--geno", required=True)
    parser.add_argument("--snp", required=True)
    parser.add_argument("--ind", required=True)
    parser.add_argument("--anno", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default=datetime.now().strftime("%Y-%m-%d_geo-mlp_%H%M%S"))
    parser.add_argument("--plink_bin", default=None)
    parser.add_argument("--convertf_bin", default=None)
    parser.add_argument("--ld_window", type=int, default=200)
    parser.add_argument("--ld_step", type=int, default=25)
    parser.add_argument("--ld_r2", type=float, default=0.2)
    parser.add_argument("--pca_dims", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_snps", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--jitter_sigma_km", type=float, default=25.0)
    parser.add_argument("--min_sigma_km", type=float, default=1.0)
    parser.add_argument("--max_sigma_km", type=float, default=2000.0)
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
    splits = make_leave_site_split(meta.locality, seed=args.seed, train_fraction=args.train_fraction, val_fraction=args.val_fraction)
    write_split_sites(run_dir / "split_sites.json", splits)

    pc_scaled, pc_mean, pc_std = standardize_train_apply(features_pc, splits.train)
    age_scaled, age_mean, age_std = standardize_train_apply(meta.date_bp.reshape(-1, 1), splits.train)
    features = np.column_stack([pc_scaled, age_scaled]).astype(np.float32)
    np.save(run_dir / "pca_age_features.npy", features)
    write_sample_stats(run_dir / "sample_stats.tsv", meta, splits, features)

    log(f"Samples: total={len(sample_ids)} train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")
    log(f"Localities: train={len(splits.train_sites)} val={len(splits.val_sites)} test={len(splits.test_sites)}")
    baseline = nearest_neighbor_baseline(features, meta.xy, splits)
    log(f"NN PCA+age baseline median={baseline['nn_pca_age_median_km']:.3f}km p90={baseline['nn_pca_age_p90_km']:.3f}km")
    train_metrics = run_training(features, meta.xy, splits, args, run_dir)

    summary: Dict[str, object] = {
        "status": "completed",
        "run_name": args.run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(sample_ids),
        "n_train": int(len(splits.train)),
        "n_val": int(len(splits.val)),
        "n_test": int(len(splits.test)),
        "n_train_sites": len(splits.train_sites),
        "n_val_sites": len(splits.val_sites),
        "n_test_sites": len(splits.test_sites),
        "pca_dims": args.pca_dims,
        "ld_window": args.ld_window,
        "ld_step": args.ld_step,
        "ld_r2": args.ld_r2,
        "prune_in": str(prune_in),
        "eigenvec": str(eigenvec),
        "eigenval": str(eigenval),
        "projection": meta.projection,
        "age_mean_train": float(age_mean[0]),
        "age_std_train": float(age_std[0]),
        "pc_mean_train_shape": list(pc_mean.shape),
        "pc_std_train_shape": list(pc_std.shape),
        "conversion": conversion_meta,
        **baseline,
        **train_metrics,
    }
    write_json(run_dir / "evaluation_summary.json", summary)
    write_json(run_dir / "run_summary.json", summary)
    write_csv(
        run_dir / "metrics.csv",
        [{k: v for k, v in summary.items() if not isinstance(v, (dict, list))}],
        [k for k, v in summary.items() if not isinstance(v, (dict, list))],
    )
    write_text(
        run_dir / "results.md",
        f"""
# Results

## Primary Test Metrics

| metric | value |
| --- | ---: |
| median_km | {summary['median_km']:.3f} |
| p90_km | {summary['p90_km']:.3f} |
| gaussian_nll | {summary['test_gaussian_nll']:.6f} |
| containment_50 | {summary['containment_50']:.3f} |
| containment_68 | {summary['containment_68']:.3f} |
| containment_90 | {summary['containment_90']:.3f} |
| containment_95 | {summary['containment_95']:.3f} |

## Nearest-Neighbor PCA+Age Baseline

| metric | value |
| --- | ---: |
| median_km | {summary['nn_pca_age_median_km']:.3f} |
| p90_km | {summary['nn_pca_age_p90_km']:.3f} |
| mean_km | {summary['nn_pca_age_mean_km']:.3f} |

## Split

Train samples: {summary['n_train']} across {summary['n_train_sites']} Locality values.

Validation samples: {summary['n_val']} across {summary['n_val_sites']} Locality values.

Test samples: {summary['n_test']} across {summary['n_test_sites']} Locality values.
""",
    )
    append_index(run_dir.parents[1], args.run_name, summary)
    log(f"Final median={summary['median_km']:.3f}km p90={summary['p90_km']:.3f}km NLL={summary['test_gaussian_nll']:.6f}")
    log(f"Outputs written to: {run_dir}")


if __name__ == "__main__":
    main()
