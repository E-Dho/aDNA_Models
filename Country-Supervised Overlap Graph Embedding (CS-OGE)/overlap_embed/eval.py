from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .data import load_sample_stats_tsv


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    if denom <= 0.0:
        return float("nan")
    return float((x_centered * y_centered).sum() / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    return pearson_corr(rankdata_average(np.asarray(x)[finite]), rankdata_average(np.asarray(y)[finite]))


def ridge_probe_r2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 42,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = x.shape[0]
    if n < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    split = max(int(0.8 * n), 1)
    train_idx = order[:split]
    test_idx = order[split:]
    if test_idx.size == 0:
        test_idx = train_idx
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    x_train_n = (x_train - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std
    y_mean = y_train.mean()
    y_train_c = y_train - y_mean
    best_r2 = float("-inf")
    for alpha in alphas:
        xtx = x_train_n.T @ x_train_n
        reg = xtx + np.eye(xtx.shape[0], dtype=np.float64) * float(alpha)
        coef = np.linalg.solve(reg, x_train_n.T @ y_train_c)
        pred = x_test_n @ coef + y_mean
        ss_res = ((y_test - pred) ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        if np.isfinite(r2) and r2 > best_r2:
            best_r2 = float(r2)
    return best_r2


def pairwise_sq_dist(latents: np.ndarray) -> np.ndarray:
    x2 = (latents**2).sum(axis=1, keepdims=True)
    return np.maximum(x2 + x2.T - 2.0 * latents @ latents.T, 0.0)


def nearest_neighbors(latents: np.ndarray, k: int) -> np.ndarray:
    dist2 = pairwise_sq_dist(latents)
    np.fill_diagonal(dist2, np.inf)
    k = min(int(k), latents.shape[0] - 1)
    idx = np.argpartition(dist2, kth=k, axis=1)[:, :k]
    row = np.arange(latents.shape[0])[:, None]
    order = np.argsort(dist2[row, idx], axis=1)
    return idx[row, order]


def nearest_neighbors_with_distances(latents: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    dist2 = pairwise_sq_dist(latents)
    np.fill_diagonal(dist2, np.inf)
    k = min(int(k), latents.shape[0] - 1)
    idx = np.argpartition(dist2, kth=k, axis=1)[:, :k]
    row = np.arange(latents.shape[0])[:, None]
    order = np.argsort(dist2[row, idx], axis=1)
    sorted_idx = idx[row, order]
    sorted_dist2 = dist2[row, sorted_idx]
    return sorted_idx, sorted_dist2


def macro_knn_purity(
    labels: Sequence[str],
    neighbors: np.ndarray,
    *,
    min_group_size: int = 1,
) -> float:
    labels_arr = np.asarray(labels, dtype=object)
    label_counts: Dict[str, int] = {}
    for raw_label in labels_arr:
        label = str(raw_label)
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
    per_label: Dict[str, List[float]] = {}
    for i in range(labels_arr.shape[0]):
        label = str(labels_arr[i])
        if not label:
            continue
        if label_counts.get(label, 0) < int(min_group_size):
            continue
        purity = float((labels_arr[neighbors[i]] == label).mean())
        per_label.setdefault(label, []).append(purity)
    if not per_label:
        return float("nan")
    return float(np.mean([np.mean(values) for values in per_label.values()]))


def adaptive_group_knn_purity(
    labels: Sequence[str],
    neighbors: np.ndarray,
    neighbor_dist2: np.ndarray,
    *,
    min_group_size: int = 1,
) -> float:
    labels_arr = np.asarray(labels, dtype=object)
    per_label: Dict[str, List[float]] = {}
    label_counts: Dict[str, int] = {}
    for raw_label in labels_arr:
        label = str(raw_label)
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1

    for i in range(labels_arr.shape[0]):
        label = str(labels_arr[i])
        if not label:
            continue
        if label_counts.get(label, 0) < int(min_group_size):
            continue

        same_mask = labels_arr[neighbors[i]] == label
        same_count = int(same_mask.sum())
        available_same = max(label_counts.get(label, 0) - 1, 0)
        purity = float(same_count / max(neighbors.shape[1], 1))

        if 0 < available_same <= neighbors.shape[1] and same_count == available_same:
            furthest_same_dist2 = float(np.max(neighbor_dist2[i][same_mask]))
            keep_mask = same_mask | (neighbor_dist2[i] <= furthest_same_dist2)
            kept = int(keep_mask.sum())
            if kept > 0:
                purity = float(same_count / kept)

        per_label.setdefault(label, []).append(purity)

    if not per_label:
        return float("nan")
    return float(np.mean([np.mean(values) for values in per_label.values()]))


def date_neighbor_metrics(dates: np.ndarray, neighbors: np.ndarray) -> Tuple[float, float]:
    maes: List[float] = []
    neighbor_means: List[float] = []
    source_dates: List[float] = []
    for i, date in enumerate(dates):
        if not np.isfinite(date):
            continue
        neigh_dates = dates[neighbors[i]]
        neigh_dates = neigh_dates[np.isfinite(neigh_dates)]
        if neigh_dates.size == 0:
            continue
        maes.append(float(np.mean(np.abs(neigh_dates - date))))
        neighbor_means.append(float(neigh_dates.mean()))
        source_dates.append(float(date))
    if not maes:
        return float("nan"), float("nan")
    return float(np.mean(maes)), spearman_corr(np.asarray(source_dates), np.asarray(neighbor_means))


def pca_project(latents: np.ndarray, n_components: int) -> np.ndarray:
    centered = latents - latents.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:n_components].T
    return centered @ basis


def maybe_umap_project(latents: np.ndarray) -> Optional[np.ndarray]:
    try:
        import umap
    except ImportError:
        return None
    reducer = umap.UMAP(n_components=2, random_state=42)
    return reducer.fit_transform(latents)


def _write_delimited(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]], delimiter: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def write_latents_csv(path: Path, sample_ids: Sequence[str], latents: np.ndarray) -> None:
    header = ["sample_id"] + [f"z{i+1}" for i in range(latents.shape[1])]
    rows = [[sample_ids[i], *map(float, latents[i])] for i in range(latents.shape[0])]
    _write_delimited(path, header, rows, ",")


def read_latents_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not header or header[0] != "sample_id":
            raise ValueError(f"Unexpected latents CSV header in {path}")
        sample_ids: List[str] = []
        latents: List[List[float]] = []
        for row in reader:
            sample_ids.append(row[0])
            latents.append([float(x) for x in row[1:]])
    return sample_ids, np.asarray(latents, dtype=np.float32)


def compute_coverage_dim_correlations(latents: np.ndarray, coverage: np.ndarray) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for dim in range(latents.shape[1]):
        rows.append(
            {
                "dim": dim + 1,
                "pearson": pearson_corr(latents[:, dim], coverage),
                "spearman": spearman_corr(latents[:, dim], coverage),
            }
        )
    return rows


def write_coverage_dim_corr(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    _write_delimited(
        path,
        ["dim", "pearson", "spearman"],
        [[row["dim"], row["pearson"], row["spearman"]] for row in rows],
        "\t",
    )


def write_knn_report(path: Path, metrics: Dict[str, float]) -> None:
    _write_delimited(path, ["metric", "value"], [[key, value] for key, value in metrics.items()], "\t")


def plot_embedding(
    coords: np.ndarray,
    labels: Sequence[str],
    output_path: Path,
    *,
    title: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    unique_labels = sorted({label for label in labels if label})
    color_lookup = {label: idx for idx, label in enumerate(unique_labels)}
    colors = [color_lookup.get(label, -1) for label in labels]
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, cmap="tab20", alpha=0.8)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    if unique_labels:
        handles, _ = scatter.legend_elements(num=min(len(unique_labels), 12))
        plt.legend(handles, unique_labels[: len(handles)], loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def evaluate_embedding(
    latents: np.ndarray,
    sample_stats_tsv: Path,
    *,
    k: int = 15,
) -> Tuple[Dict[str, float], List[Dict[str, float]], np.ndarray, List[Dict[str, object]]]:
    rows = load_sample_stats_tsv(sample_stats_tsv)
    sample_ids = [str(row["sample_id"]) for row in rows]
    coverage = np.asarray([float(row["observed_fraction"]) for row in rows], dtype=np.float64)
    country = [str(row["country"]) for row in rows]
    original_group = [str(row["original_group_id"]) for row in rows]
    dates = np.asarray([float(row["date_mean_bp"]) for row in rows], dtype=np.float64)

    neighbors, neighbor_dist2 = nearest_neighbors_with_distances(latents, k)
    metrics = {
        "coverage_r2": ridge_probe_r2(latents, coverage),
        "country_macro_knn_purity@15": macro_knn_purity(country, neighbors),
        "original_group_knn_purity@15": macro_knn_purity(original_group, neighbors),
        "original_group_knn_purity@15_non_singleton": macro_knn_purity(
            original_group,
            neighbors,
            min_group_size=2,
        ),
        "original_group_adaptive_knn_purity@15": adaptive_group_knn_purity(
            original_group,
            neighbors,
            neighbor_dist2,
        ),
        "original_group_adaptive_knn_purity@15_non_singleton": adaptive_group_knn_purity(
            original_group,
            neighbors,
            neighbor_dist2,
            min_group_size=2,
        ),
    }
    date_mae, date_spearman = date_neighbor_metrics(dates, neighbors)
    metrics["date_neighbor_mae@15"] = date_mae
    metrics["date_neighbor_spearman"] = date_spearman
    corr_rows = compute_coverage_dim_correlations(latents, coverage)
    merged_rows = [
        {
            "sample_id": sample_ids[i],
            "observed_fraction": coverage[i],
            "country": country[i],
            "original_group_id": original_group[i],
            "date_mean_bp": dates[i],
        }
        for i in range(len(sample_ids))
    ]
    return metrics, corr_rows, neighbors, merged_rows


def export_evaluation_bundle(
    *,
    latents: np.ndarray,
    sample_ids: Sequence[str],
    sample_stats_tsv: Path,
    output_dir: Path,
    prefix: str = "final",
    plot_label: str = "country",
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, corr_rows, _, merged_rows = evaluate_embedding(latents, sample_stats_tsv)
    latents_csv = output_dir / f"{prefix}_latents.csv"
    write_latents_csv(latents_csv, sample_ids, latents)
    write_coverage_dim_corr(output_dir / "coverage_dim_corr.tsv", corr_rows)
    write_knn_report(output_dir / "knn_report.tsv", metrics)

    label_key = "country" if plot_label == "country" else "original_group_id"
    labels = [str(row[label_key]) for row in merged_rows]
    pca_2d = pca_project(latents, 2)
    plot_embedding(
        pca_2d,
        labels,
        output_dir / "final_latent_pca.png",
        title=f"PCA of overlap embedding ({plot_label})",
    )
    umap_2d = maybe_umap_project(latents)
    if umap_2d is not None:
        plot_embedding(
            umap_2d,
            labels,
            output_dir / "final_latent_umap.png",
            title=f"UMAP of overlap embedding ({plot_label})",
        )
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics
