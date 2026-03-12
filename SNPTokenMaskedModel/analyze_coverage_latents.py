#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Avoid matplotlib font/cache warnings on systems where $HOME/.matplotlib is not writable.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplcfg_")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunInput:
    name: str
    latents_csv: Path


@dataclass(frozen=True)
class RunResult:
    run_name: str
    n_samples: int
    latent_dim: int
    r2_cov_from_z: float
    r2_cov_from_z_unit: float
    pearson_pred_from_z: float
    pearson_pred_from_z_unit: float
    pearson_norm_vs_cov: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coverage-signal diagnostics for latent vectors.")
    p.add_argument(
        "--coverage_csv",
        required=True,
        help="CSV with columns sample_id,coverage_observed_fraction",
    )
    p.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form NAME=/path/to/global_latents.csv (repeat exactly twice).",
    )
    p.add_argument("--pc_count", type=int, default=10, help="Number of top PCs to test vs coverage.")
    p.add_argument("--output_dir", required=True, help="Directory for outputs.")
    p.add_argument("--random_state", type=int, default=42, help="Reserved for reproducibility options.")
    return p.parse_args()


def parse_run_specs(raw_specs: List[str]) -> List[RunInput]:
    if len(raw_specs) != 2:
        raise ValueError(f"Expected exactly 2 --run entries, got {len(raw_specs)}")
    out: List[RunInput] = []
    seen_names = set()
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --run format: '{spec}' (expected NAME=/path/file.csv)")
        name, path_str = spec.split("=", 1)
        name = name.strip()
        path = Path(path_str).expanduser().resolve()
        if not name:
            raise ValueError(f"Run name is empty in spec: '{spec}'")
        if name in seen_names:
            raise ValueError(f"Duplicate run name: '{name}'")
        if not path.exists():
            raise FileNotFoundError(f"Latents CSV not found for run '{name}': {path}")
        seen_names.add(name)
        out.append(RunInput(name=name, latents_csv=path))
    return out


def detect_latent_columns(frame: pd.DataFrame) -> List[str]:
    zcols = [c for c in frame.columns if c.startswith("z")]
    if not zcols:
        raise ValueError("No latent columns found; expected columns like z0, z1, ...")
    return zcols


def load_coverage(coverage_csv: Path) -> pd.DataFrame:
    cov = pd.read_csv(coverage_csv)
    required = {"sample_id", "coverage_observed_fraction"}
    missing = required - set(cov.columns)
    if missing:
        raise ValueError(f"Coverage CSV missing required columns: {sorted(missing)}")
    cov = cov[["sample_id", "coverage_observed_fraction"]].copy()
    cov["sample_id"] = cov["sample_id"].astype(str)
    cov["coverage_observed_fraction"] = pd.to_numeric(cov["coverage_observed_fraction"], errors="coerce")
    cov = cov.dropna(subset=["sample_id", "coverage_observed_fraction"])

    if cov["sample_id"].duplicated().any():
        dup_n = int(cov["sample_id"].duplicated().sum())
        raise ValueError(f"Coverage CSV has duplicate sample_id rows: {dup_n}")
    return cov


def load_latents(latents_csv: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(latents_csv)
    if "sample_id" not in df.columns:
        raise ValueError(f"{latents_csv} missing sample_id column")
    zcols = detect_latent_columns(df)
    use = df[["sample_id"] + zcols].copy()
    use["sample_id"] = use["sample_id"].astype(str)
    if use["sample_id"].duplicated().any():
        dup_n = int(use["sample_id"].duplicated().sum())
        raise ValueError(f"{latents_csv} has duplicate sample_id rows: {dup_n}")
    for col in zcols:
        use[col] = pd.to_numeric(use[col], errors="coerce")
    use = use.dropna(subset=zcols)
    return use, zcols


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    r, p = pearsonr(x, y)
    return float(r), float(p)


def fit_ols_with_intercept(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    beta, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
    return x_aug @ beta


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def analyze_run(
    run_name: str,
    merged: pd.DataFrame,
    zcols: List[str],
    pc_count: int,
    out_dir: Path,
) -> Tuple[RunResult, List[Dict[str, object]], pd.DataFrame]:
    x = merged[zcols].to_numpy(dtype=np.float64, copy=False)
    y = merged["coverage_observed_fraction"].to_numpy(dtype=np.float64, copy=False)

    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError(f"{run_name}: NaN detected after merge/cleaning")

    eps = 1e-12
    norm = np.linalg.norm(x, axis=1)
    x_unit = x / np.maximum(norm[:, None], eps)

    pred_from_z = fit_ols_with_intercept(x=x, y=y)
    pred_from_z_unit = fit_ols_with_intercept(x=x_unit, y=y)

    r2_z = compute_r2(y_true=y, y_pred=pred_from_z)
    r2_z_unit = compute_r2(y_true=y, y_pred=pred_from_z_unit)
    pred_r_z, _ = safe_pearson(pred_from_z, y)
    pred_r_z_unit, _ = safe_pearson(pred_from_z_unit, y)
    norm_r, _ = safe_pearson(norm, y)

    x_center = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_center, full_matrices=False)
    scores = x_center @ vt.T
    n_pc = min(pc_count, scores.shape[1])
    if n_pc < pc_count:
        print(f"[INFO] {run_name}: pc_count clamped from {pc_count} to {n_pc} (latent_dim={scores.shape[1]})")

    pc_rows: List[Dict[str, object]] = []
    for pc_idx in range(n_pc):
        r, p = safe_pearson(scores[:, pc_idx], y)
        pc_rows.append(
            {
                "run_name": run_name,
                "pc_index": pc_idx + 1,
                "pc_coverage_r": r,
                "pc_coverage_p": p,
            }
        )

    joined = merged[["sample_id", "coverage_observed_fraction"]].copy()
    joined["norm"] = norm
    joined["pred_cov_from_z"] = pred_from_z
    joined["pred_cov_from_z_unit"] = pred_from_z_unit
    joined.to_csv(out_dir / f"joined_data_{run_name}.csv", index=False)

    res = RunResult(
        run_name=run_name,
        n_samples=int(len(merged)),
        latent_dim=int(len(zcols)),
        r2_cov_from_z=float(r2_z),
        r2_cov_from_z_unit=float(r2_z_unit),
        pearson_pred_from_z=float(pred_r_z),
        pearson_pred_from_z_unit=float(pred_r_z_unit),
        pearson_norm_vs_cov=float(norm_r),
    )
    return res, pc_rows, joined


def write_probe_summary(path: Path, rows: List[RunResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "run_name",
                "n_samples",
                "latent_dim",
                "r2_cov_from_z",
                "r2_cov_from_z_unit",
                "pearson_pred_from_z",
                "pearson_pred_from_z_unit",
                "pearson_norm_vs_cov",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.run_name,
                    row.n_samples,
                    row.latent_dim,
                    row.r2_cov_from_z,
                    row.r2_cov_from_z_unit,
                    row.pearson_pred_from_z,
                    row.pearson_pred_from_z_unit,
                    row.pearson_norm_vs_cov,
                ]
            )


def write_pc_table(path: Path, rows: List[Dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, sep="\t", index=False)


def make_scatter_plots(
    joined_by_run: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    runs = list(joined_by_run.keys())
    x_all = np.concatenate([joined_by_run[r]["coverage_observed_fraction"].to_numpy() for r in runs])
    y_all = np.concatenate([joined_by_run[r]["norm"].to_numpy() for r in runs])

    x_min = float(np.nanmin(x_all))
    x_max = float(np.nanmax(x_all))
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    y_pad = 0.03 * (y_max - y_min if y_max > y_min else 1.0)
    y_lim = (y_min - y_pad, y_max + y_pad)

    plt.figure(figsize=(9, 7))
    for run_name, frame in joined_by_run.items():
        plt.scatter(
            frame["coverage_observed_fraction"],
            frame["norm"],
            s=9,
            alpha=0.45,
            label=run_name,
            linewidths=0,
        )
    plt.xlabel("coverage_observed_fraction")
    plt.ylabel("||z||")
    plt.title("||z|| vs coverage (S7 vs MT8)")
    plt.xlim(x_min, x_max)
    plt.ylim(y_lim)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "norm_vs_coverage_scatter_both.png", dpi=220)
    plt.close()

    for run_name, frame in joined_by_run.items():
        plt.figure(figsize=(9, 7))
        plt.scatter(
            frame["coverage_observed_fraction"],
            frame["norm"],
            s=9,
            alpha=0.55,
            linewidths=0,
        )
        plt.xlabel("coverage_observed_fraction")
        plt.ylabel("||z||")
        plt.title(f"||z|| vs coverage ({run_name})")
        plt.xlim(x_min, x_max)
        plt.ylim(y_lim)
        plt.tight_layout()
        plt.savefig(out_dir / f"norm_vs_coverage_scatter_{run_name}.png", dpi=220)
        plt.close()


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)

    coverage_csv = Path(args.coverage_csv).expanduser().resolve()
    if not coverage_csv.exists():
        raise FileNotFoundError(f"Coverage CSV not found: {coverage_csv}")

    runs = parse_run_specs(args.run)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage = load_coverage(coverage_csv)
    cov_ids = set(coverage["sample_id"].tolist())

    run_frames: Dict[str, pd.DataFrame] = {}
    run_zcols: Dict[str, List[str]] = {}
    shared_ids = set(cov_ids)

    for run in runs:
        latents, zcols = load_latents(run.latents_csv)
        run_frames[run.name] = latents
        run_zcols[run.name] = zcols
        shared_ids &= set(latents["sample_id"].tolist())

    if not shared_ids:
        raise ValueError("No shared sample IDs across runs and coverage table")
    shared_sorted = sorted(shared_ids)
    print(f"[INFO] shared_samples={len(shared_sorted)}")

    cov_shared = coverage[coverage["sample_id"].isin(shared_sorted)].copy()
    cov_shared = cov_shared.set_index("sample_id").loc[shared_sorted].reset_index()

    summary_rows: List[RunResult] = []
    pc_rows: List[Dict[str, object]] = []
    joined_by_run: Dict[str, pd.DataFrame] = {}

    for run in runs:
        latents = run_frames[run.name]
        latents = latents[latents["sample_id"].isin(shared_sorted)].copy()
        latents = latents.set_index("sample_id").loc[shared_sorted].reset_index()
        merged = latents.merge(cov_shared, on="sample_id", how="inner", validate="one_to_one")
        if len(merged) != len(shared_sorted):
            raise ValueError(
                f"{run.name}: merged rows mismatch ({len(merged)} != {len(shared_sorted)}) after alignment"
            )

        row, pc_for_run, joined = analyze_run(
            run_name=run.name,
            merged=merged,
            zcols=run_zcols[run.name],
            pc_count=args.pc_count,
            out_dir=out_dir,
        )
        summary_rows.append(row)
        pc_rows.extend(pc_for_run)
        joined_by_run[run.name] = joined

    write_probe_summary(out_dir / "probe_summary.tsv", summary_rows)
    write_pc_table(out_dir / "pc_coverage_corr.tsv", pc_rows)
    make_scatter_plots(joined_by_run=joined_by_run, out_dir=out_dir)

    print(f"[OK] wrote {out_dir / 'probe_summary.tsv'}")
    print(f"[OK] wrote {out_dir / 'pc_coverage_corr.tsv'}")
    print(f"[OK] wrote {out_dir / 'norm_vs_coverage_scatter_both.png'}")
    for run in runs:
        print(f"[OK] wrote {out_dir / f'norm_vs_coverage_scatter_{run.name}.png'}")


if __name__ == "__main__":
    main()
