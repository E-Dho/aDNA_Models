#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

AGE_BINS = [
    (0.0, 1000.0, "0-1000 BP"),
    (1000.0, 2500.0, "1000-2500 BP"),
    (2500.0, 4000.0, "2500-4000 BP"),
    (4000.0, 6000.0, "4000-6000 BP"),
    (6000.0, float("inf"), ">6000 BP"),
]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"Missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"Missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_tsv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def age_bin(age_bp: float) -> str:
    for lo, hi, label in AGE_BINS:
        if lo <= age_bp < hi:
            return label
    return "missing"


def euclidean_km(pred_xy: np.ndarray, true_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred_xy - true_xy, axis=1) / 1000.0


def weighted_knn_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    k: int,
    eps: float = 1e-8,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if k < 1:
        fail("k must be >= 1")
    if k > len(train_x):
        fail(f"k={k} exceeds train sample count {len(train_x)}")
    preds = np.empty((len(test_x), 2), dtype=np.float64)
    mean_neighbor_dist = np.empty(len(test_x), dtype=np.float64)
    for start in range(0, len(test_x), chunk_size):
        end = min(start + chunk_size, len(test_x))
        block = test_x[start:end].astype(np.float64)
        dist2 = ((block[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
        nn_idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        nn_dist = np.take_along_axis(dist2, nn_idx, axis=1) ** 0.5
        weights = 1.0 / (nn_dist + eps)
        weights = weights / weights.sum(axis=1, keepdims=True)
        preds[start:end] = np.einsum("nk,nkd->nd", weights, train_y[nn_idx])
        mean_neighbor_dist[start:end] = nn_dist.mean(axis=1)
    return preds, mean_neighbor_dist


def summarize(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def grouped_medians(rows: list[dict[str, object]], group_key: str, value_keys: Sequence[str]) -> list[dict[str, object]]:
    grouped: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    out: list[dict[str, object]] = []
    for group, group_rows in grouped.items():
        entry: dict[str, object] = {group_key: group, "n_test_samples": len(group_rows)}
        for key in value_keys:
            vals = np.asarray([float(r[key]) for r in group_rows], dtype=np.float64)
            entry[f"{key}_median"] = f"{np.median(vals):.6f}"
            entry[f"{key}_mean"] = f"{np.mean(vals):.6f}"
            entry[f"{key}_p90"] = f"{np.percentile(vals, 90):.6f}"
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-run geolocation error analysis")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=512)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    sample_rows = read_tsv(run_dir / "sample_stats.tsv")
    pred_rows = read_csv(run_dir / "test_predictions.csv")
    features_path = run_dir / "pca_age_features.npy"
    if not features_path.is_file():
        fail(f"Missing feature matrix: {features_path}")
    features = np.load(features_path)
    if len(sample_rows) != features.shape[0]:
        fail(f"sample_stats rows ({len(sample_rows)}) do not match features rows ({features.shape[0]})")

    split = np.asarray([row["split"] for row in sample_rows], dtype=object)
    train_idx = np.where(split == "train")[0]
    test_idx = np.where(split == "test")[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        fail("Need non-empty train and test splits")

    xy = np.asarray([[parse_float(row, "x_m"), parse_float(row, "y_m")] for row in sample_rows], dtype=np.float64)
    ages = np.asarray([parse_float(row, "date_mean_bp") for row in sample_rows], dtype=np.float64)
    locality = [row["locality"] for row in sample_rows]
    locality_sizes = Counter(locality)

    pred_by_idx: dict[int, dict[str, str]] = {int(row["sample_index"]): row for row in pred_rows}
    missing_pred = [int(i) for i in test_idx if int(i) not in pred_by_idx]
    if missing_pred:
        fail(f"test_predictions.csv misses {len(missing_pred)} test indices. Examples: {missing_pred[:10]}")

    knn_pred, mean_neighbor_dist = weighted_knn_predict(
        features[train_idx],
        xy[train_idx],
        features[test_idx],
        k=args.k,
        chunk_size=args.chunk_size,
    )
    knn_error = euclidean_km(knn_pred, xy[test_idx])

    per_sample: list[dict[str, object]] = []
    for pos, idx in enumerate(test_idx):
        mlp_error = float(pred_by_idx[int(idx)]["error_km"])
        row = sample_rows[int(idx)]
        per_sample.append(
            {
                "sample_index": int(idx),
                "sample_id": row["sample_id"],
                "age_bin": age_bin(ages[int(idx)]),
                "date_mean_bp": f"{ages[int(idx)]:.6f}",
                "locality": row["locality"],
                "locality_size_all_samples": int(locality_sizes[row["locality"]]),
                "original_group_id": row["original_group_id"],
                "mlp_error_km": f"{mlp_error:.6f}",
                "weighted_knn_error_km": f"{knn_error[pos]:.6f}",
                "weighted_knn_mean_neighbor_dist": f"{mean_neighbor_dist[pos]:.8f}",
                "weighted_knn_pred_x_m": f"{knn_pred[pos, 0]:.6f}",
                "weighted_knn_pred_y_m": f"{knn_pred[pos, 1]:.6f}",
                "true_x_m": row["x_m"],
                "true_y_m": row["y_m"],
            }
        )

    write_tsv(
        run_dir / "weighted_knn_test_predictions.tsv",
        per_sample,
        [
            "sample_index",
            "sample_id",
            "age_bin",
            "date_mean_bp",
            "locality",
            "locality_size_all_samples",
            "original_group_id",
            "mlp_error_km",
            "weighted_knn_error_km",
            "weighted_knn_mean_neighbor_dist",
            "weighted_knn_pred_x_m",
            "weighted_knn_pred_y_m",
            "true_x_m",
            "true_y_m",
        ],
    )

    age_rows = grouped_medians(per_sample, "age_bin", ["mlp_error_km", "weighted_knn_error_km"])
    order = {label: i for i, (_, _, label) in enumerate(AGE_BINS)}
    age_rows.sort(key=lambda r: order.get(str(r["age_bin"]), 99))
    write_tsv(
        run_dir / "median_error_by_age_bin.tsv",
        age_rows,
        [
            "age_bin",
            "n_test_samples",
            "mlp_error_km_median",
            "mlp_error_km_mean",
            "mlp_error_km_p90",
            "weighted_knn_error_km_median",
            "weighted_knn_error_km_mean",
            "weighted_knn_error_km_p90",
        ],
    )

    size_rows = grouped_medians(per_sample, "locality_size_all_samples", ["mlp_error_km", "weighted_knn_error_km"])
    size_rows.sort(key=lambda r: int(r["locality_size_all_samples"]))
    write_tsv(
        run_dir / "median_error_by_locality_size.tsv",
        size_rows,
        [
            "locality_size_all_samples",
            "n_test_samples",
            "mlp_error_km_median",
            "mlp_error_km_mean",
            "mlp_error_km_p90",
            "weighted_knn_error_km_median",
            "weighted_knn_error_km_mean",
            "weighted_knn_error_km_p90",
        ],
    )

    summary = {
        "k": args.k,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "weighted_knn_error_km": summarize(knn_error),
        "mlp_error_km": summarize(np.asarray([float(r["mlp_error_km"]) for r in per_sample], dtype=np.float64)),
        "outputs": {
            "weighted_knn_test_predictions": str(run_dir / "weighted_knn_test_predictions.tsv"),
            "median_error_by_age_bin": str(run_dir / "median_error_by_age_bin.tsv"),
            "median_error_by_locality_size": str(run_dir / "median_error_by_locality_size.tsv"),
        },
    }
    write_json(run_dir / "posthoc_error_analysis_summary.json", summary)

    with (run_dir / "results.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## Posthoc Error Analysis\n\n")
        handle.write(f"Weighted KNN baseline in PCA+age space with k={args.k}: median {summary['weighted_knn_error_km']['median']:.3f} km, p90 {summary['weighted_knn_error_km']['p90']:.3f} km.\n\n")
        handle.write("Additional tables: `median_error_by_age_bin.tsv`, `median_error_by_locality_size.tsv`, `weighted_knn_test_predictions.tsv`.\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
