from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssignmentSpec:
    label: str
    path: Path


@dataclass
class PairwiseComparison:
    run_a: str
    run_b: str
    mode: str
    n_samples: int
    n_total_aligned: int
    n_dropped_for_noise: int
    ari: float
    vi: float
    normalized_vi: float
    n_clusters_a: int
    n_clusters_b: int
    n_noise_a: int
    n_noise_b: int
    counts: pd.DataFrame
    row_fraction: pd.DataFrame


def parse_assignment_arg(value: str) -> AssignmentSpec:
    if "=" not in value:
        raise ValueError(f"Assignment must be LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Assignment label is empty: {value}")
    return AssignmentSpec(label=safe_label(label), path=Path(path).expanduser())


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    label = label.strip("_")
    if not label:
        raise ValueError(f"Could not derive a safe label from {value!r}")
    return label


def load_assignment(
    spec: AssignmentSpec,
    *,
    sample_col: str = "sample_id",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    frame = pd.read_csv(spec.path, sep="\t")
    missing = [col for col in (sample_col, cluster_col) if col not in frame.columns]
    if missing:
        raise ValueError(f"{spec.path} is missing required columns: {missing}")
    if frame[sample_col].duplicated().any():
        duplicated = frame.loc[frame[sample_col].duplicated(), sample_col].head(5).tolist()
        raise ValueError(f"{spec.label} has duplicate sample IDs, examples: {duplicated}")

    out = frame[[sample_col, cluster_col]].copy()
    out = out.rename(columns={sample_col: "sample_id", cluster_col: "cluster_id"})
    out["sample_id"] = out["sample_id"].astype(str)
    out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="raise").astype(np.int64)
    if "is_optics_noise" in frame.columns:
        noise = frame["is_optics_noise"]
        if noise.dtype == bool:
            out["is_optics_noise"] = noise.astype(bool)
        else:
            out["is_optics_noise"] = noise.astype(str).str.lower().isin({"1", "true", "yes"})
    else:
        out["is_optics_noise"] = out["cluster_id"].eq(-1)
    return out


def load_assignments(
    specs: Sequence[AssignmentSpec],
    *,
    sample_col: str = "sample_id",
    cluster_col: str = "cluster_id",
) -> Dict[str, pd.DataFrame]:
    if len(specs) < 2:
        raise ValueError("At least two assignments are required")
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Assignment labels must be unique, got: {labels}")
    return {
        spec.label: load_assignment(spec, sample_col=sample_col, cluster_col=cluster_col)
        for spec in specs
    }


def _sort_cluster_values(values: Iterable[int]) -> list[int]:
    unique = sorted({int(v) for v in values})
    return [-1] + [value for value in unique if value != -1] if -1 in unique else unique


def contingency_matrix(labels_a: Sequence[int], labels_b: Sequence[int]) -> pd.DataFrame:
    a = pd.Series(np.asarray(labels_a, dtype=np.int64), name="cluster_a")
    b = pd.Series(np.asarray(labels_b, dtype=np.int64), name="cluster_b")
    counts = pd.crosstab(a, b)
    row_order = _sort_cluster_values(counts.index.tolist())
    col_order = _sort_cluster_values(counts.columns.tolist())
    return counts.reindex(index=row_order, columns=col_order, fill_value=0).astype(np.int64)


def row_normalize(counts: pd.DataFrame) -> pd.DataFrame:
    totals = counts.sum(axis=1).replace(0, np.nan)
    fractions = counts.div(totals, axis=0).fillna(0.0)
    return fractions.astype(float)


def variation_of_information_from_counts(counts: pd.DataFrame) -> tuple[float, float]:
    matrix = counts.to_numpy(dtype=np.float64)
    n = float(matrix.sum())
    if n <= 0:
        return float("nan"), float("nan")

    pxy = matrix / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    px_nonzero = px[px > 0]
    py_nonzero = py[py > 0]
    h_x = float(-(px_nonzero * np.log(px_nonzero)).sum())
    h_y = float(-(py_nonzero * np.log(py_nonzero)).sum())

    nz = pxy > 0
    px_py = np.outer(px, py)
    mi = float((pxy[nz] * np.log(pxy[nz] / px_py[nz])).sum())
    vi = max(h_x + h_y - 2.0 * mi, 0.0)
    normalized = vi / math.log(n) if n > 1 else 0.0
    return vi, normalized


def _ari(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(labels_a, labels_b))


def _aligned_pair(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    return frame_a.merge(
        frame_b,
        on="sample_id",
        how="inner",
        suffixes=(f"_{label_a}", f"_{label_b}"),
        validate="one_to_one",
    )


def _mode_frame(aligned: pd.DataFrame, label_a: str, label_b: str, mode: str) -> pd.DataFrame:
    if mode == "include_noise":
        return aligned
    if mode != "exclude_noise_either":
        raise ValueError(f"Unknown comparison mode: {mode}")
    noise_a = aligned[f"cluster_id_{label_a}"].eq(-1) | aligned[f"is_optics_noise_{label_a}"]
    noise_b = aligned[f"cluster_id_{label_b}"].eq(-1) | aligned[f"is_optics_noise_{label_b}"]
    return aligned.loc[~(noise_a | noise_b)].copy()


def compare_pair(
    label_a: str,
    frame_a: pd.DataFrame,
    label_b: str,
    frame_b: pd.DataFrame,
    *,
    mode: str,
) -> PairwiseComparison:
    aligned = _aligned_pair(frame_a, frame_b, label_a, label_b)
    used = _mode_frame(aligned, label_a, label_b, mode)
    labels_a = used[f"cluster_id_{label_a}"].to_numpy(dtype=np.int64)
    labels_b = used[f"cluster_id_{label_b}"].to_numpy(dtype=np.int64)
    counts = contingency_matrix(labels_a, labels_b)
    row_fraction = row_normalize(counts)
    vi, normalized_vi = variation_of_information_from_counts(counts)

    return PairwiseComparison(
        run_a=label_a,
        run_b=label_b,
        mode=mode,
        n_samples=int(len(used)),
        n_total_aligned=int(len(aligned)),
        n_dropped_for_noise=int(len(aligned) - len(used)),
        ari=_ari(labels_a, labels_b) if len(used) else float("nan"),
        vi=vi,
        normalized_vi=normalized_vi,
        n_clusters_a=int(pd.Series(labels_a[labels_a >= 0]).nunique()) if len(used) else 0,
        n_clusters_b=int(pd.Series(labels_b[labels_b >= 0]).nunique()) if len(used) else 0,
        n_noise_a=int((labels_a == -1).sum()) if len(used) else 0,
        n_noise_b=int((labels_b == -1).sum()) if len(used) else 0,
        counts=counts,
        row_fraction=row_fraction,
    )


def compare_assignments(
    assignments: Mapping[str, pd.DataFrame],
    *,
    modes: Sequence[str] = ("include_noise", "exclude_noise_either"),
) -> list[PairwiseComparison]:
    labels = list(assignments.keys())
    results: list[PairwiseComparison] = []
    for label_a, label_b in combinations(labels, 2):
        for mode in modes:
            results.append(
                compare_pair(
                    label_a,
                    assignments[label_a],
                    label_b,
                    assignments[label_b],
                    mode=mode,
                )
            )
    return results


def metrics_frame(results: Sequence[PairwiseComparison]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = asdict(result)
        row.pop("counts")
        row.pop("row_fraction")
        rows.append(row)
    return pd.DataFrame(rows)


def _format_tick(value: object) -> str:
    return str(int(value)) if isinstance(value, (int, np.integer)) else str(value)


def write_heatmap(matrix: pd.DataFrame, path: Path, *, title: str, fraction: bool = False) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = matrix.shape
    width = min(max(7.0, n_cols * 0.28), 22.0)
    height = min(max(5.5, n_rows * 0.24), 22.0)
    fig, ax = plt.subplots(figsize=(width, height))
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("run B cluster")
    ax.set_ylabel("run A cluster")
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([_format_tick(v) for v in matrix.columns], rotation=90, fontsize=6)
    ax.set_yticklabels([_format_tick(v) for v in matrix.index], fontsize=6)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("row fraction" if fraction else "sample count")

    if n_rows <= 40 and n_cols <= 40:
        for i in range(n_rows):
            for j in range(n_cols):
                value = values[i, j]
                text = f"{value:.2f}" if fraction else str(int(value))
                if value > 0:
                    ax.text(j, i, text, ha="center", va="center", fontsize=5, color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_comparison_outputs(
    *,
    output_dir: Path,
    specs: Sequence[AssignmentSpec],
    assignments: Mapping[str, pd.DataFrame],
    results: Sequence[PairwiseComparison],
    sample_col: str,
    cluster_col: str,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir = output_dir / "overlap_matrices"
    heatmaps_dir = output_dir / "overlap_heatmaps"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    metrics = metrics_frame(results)
    metrics.to_csv(output_dir / "pairwise_cluster_metrics.tsv", sep="\t", index=False)
    with (output_dir / "pairwise_cluster_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(orient="records"), handle, indent=2, allow_nan=True)

    heatmap_paths: list[str] = []
    for result in results:
        stem = f"{result.run_a}_vs_{result.run_b}_{result.mode}"
        counts_path = matrices_dir / f"{stem}_counts.tsv"
        fraction_path = matrices_dir / f"{stem}_row_fraction.tsv"
        result.counts.to_csv(counts_path, sep="\t")
        result.row_fraction.to_csv(fraction_path, sep="\t")
        counts_heatmap = write_heatmap(
            result.counts,
            heatmaps_dir / f"{stem}_counts.png",
            title=f"{result.run_a} vs {result.run_b} ({result.mode}, counts)",
        )
        fraction_heatmap = write_heatmap(
            result.row_fraction,
            heatmaps_dir / f"{stem}_row_fraction.png",
            title=f"{result.run_a} vs {result.run_b} ({result.mode}, row fraction)",
            fraction=True,
        )
        for path in (counts_heatmap, fraction_heatmap):
            if path is not None:
                heatmap_paths.append(str(path))

    sample_sets = {label: set(frame["sample_id"]) for label, frame in assignments.items()}
    all_samples = set.union(*sample_sets.values())
    common_samples = set.intersection(*sample_sets.values())
    missing_by_assignment = {
        label: int(len(all_samples - samples))
        for label, samples in sample_sets.items()
    }
    summary: Dict[str, object] = {
        "assignments": {spec.label: str(spec.path) for spec in specs},
        "sample_col": sample_col,
        "cluster_col": cluster_col,
        "n_assignments": len(specs),
        "n_unique_samples_union": int(len(all_samples)),
        "n_common_samples_all_runs": int(len(common_samples)),
        "missing_samples_by_assignment": missing_by_assignment,
        "metrics_tsv": str(output_dir / "pairwise_cluster_metrics.tsv"),
        "metrics_json": str(output_dir / "pairwise_cluster_metrics.json"),
        "overlap_matrices_dir": str(matrices_dir),
        "overlap_heatmaps_dir": str(heatmaps_dir),
        "n_heatmaps_written": len(heatmap_paths),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    return summary
