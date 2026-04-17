from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

from .eval import pearson_corr, rankdata_average

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MergeReport:
    n_latents: int
    n_metadata: int
    n_joined: int
    n_after_filter: int
    n_dropped_missing: int
    latent_columns: list[str]


@dataclass(frozen=True)
class ProjectionResult:
    xy: np.ndarray
    crs: str
    method: str
    projected: bool


@dataclass(frozen=True)
class ReductionResult:
    coords: np.ndarray
    explained_variance_ratio: list[float]
    selected_components: list[int]
    method: str


@dataclass(frozen=True)
class SimilarityTransform:
    rotation: np.ndarray
    scale: float
    translation: np.ndarray
    source_mean: np.ndarray
    source_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    disparity: float


def _as_bool(value: int | bool) -> bool:
    return bool(int(value)) if not isinstance(value, bool) else value


def _latent_columns(frame: pd.DataFrame, id_col: str) -> list[str]:
    zcols = [col for col in frame.columns if col != id_col and col.startswith("z")]
    if zcols:
        return zcols
    numeric = [col for col in frame.columns if col != id_col and pd.api.types.is_numeric_dtype(frame[col])]
    if not numeric:
        raise ValueError("No latent columns found. Expected columns such as z0...z5 or z1...z6.")
    return numeric


def load_and_merge_latent_geo(
    latents_csv: Path,
    metadata_tsv: Path,
    *,
    id_col: str = "sample_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    label_col: Optional[str] = None,
    time_col: Optional[str] = None,
    min_samples: int = 10,
) -> Tuple[pd.DataFrame, np.ndarray, MergeReport]:
    """Load latent and metadata tables, inner-join by ID, and drop unusable rows."""
    latents = pd.read_csv(latents_csv)
    metadata = pd.read_csv(metadata_tsv, sep="\t", low_memory=False)
    for name, frame in (("latents_csv", latents), ("metadata_tsv", metadata)):
        if id_col not in frame.columns:
            raise ValueError(f"{name} is missing ID column '{id_col}'")
    for col in (lat_col, lon_col):
        if col not in metadata.columns:
            raise ValueError(f"metadata_tsv is missing coordinate column '{col}'")

    latent_cols = _latent_columns(latents, id_col)
    keep_meta_cols = [id_col, lat_col, lon_col]
    for optional in (label_col, time_col):
        if optional and optional not in metadata.columns:
            raise ValueError(f"metadata_tsv is missing requested column '{optional}'")
        if optional:
            keep_meta_cols.append(optional)
    keep_meta_cols = list(dict.fromkeys(keep_meta_cols))

    merged = latents[[id_col, *latent_cols]].merge(
        metadata[keep_meta_cols],
        on=id_col,
        how="inner",
        validate="one_to_one",
    )
    n_joined = int(len(merged))
    for col in [*latent_cols, lat_col, lon_col]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    before = len(merged)
    merged = merged.dropna(subset=[*latent_cols, lat_col, lon_col]).reset_index(drop=True)
    if len(merged) < int(min_samples):
        raise ValueError(
            f"Only {len(merged)} samples remain after merge/filter; min_samples={min_samples}"
        )
    latents_arr = merged[latent_cols].to_numpy(dtype=np.float64)
    report = MergeReport(
        n_latents=int(len(latents)),
        n_metadata=int(len(metadata)),
        n_joined=n_joined,
        n_after_filter=int(len(merged)),
        n_dropped_missing=int(before - len(merged)),
        latent_columns=latent_cols,
    )
    return merged, latents_arr, report


def _looks_like_europe(lat: np.ndarray, lon: np.ndarray) -> bool:
    return bool(
        np.nanmedian(lat) >= 34.0
        and np.nanmedian(lat) <= 72.0
        and np.nanmedian(lon) >= -31.0
        and np.nanmedian(lon) <= 45.0
    )


def _equirectangular_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    radius_m = 6_371_000.0
    lat0 = np.deg2rad(float(np.mean(lat)))
    x = radius_m * np.deg2rad(lon) * math.cos(lat0)
    y = radius_m * np.deg2rad(lat)
    return np.column_stack([x, y]).astype(np.float64)


def project_latlon_to_xy(
    lat: Sequence[float],
    lon: Sequence[float],
    *,
    project_coords: int | bool = True,
    crs: Optional[str] = None,
) -> ProjectionResult:
    """Project latitude/longitude to planar x/y coordinates for metric alignment."""
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    if not _as_bool(project_coords):
        warnings.warn("Using raw lon/lat degrees for alignment; metric projection is recommended.")
        return ProjectionResult(
            xy=np.column_stack([lon_arr, lat_arr]),
            crs="EPSG:4326",
            method="raw_lon_lat",
            projected=False,
        )

    selected_crs = crs or ("EPSG:3035" if _looks_like_europe(lat_arr, lon_arr) else "EPSG:3857")
    try:
        from pyproj import Transformer
    except ImportError:
        warnings.warn("pyproj is unavailable; using local equirectangular projection fallback.")
        return ProjectionResult(
            xy=_equirectangular_xy(lat_arr, lon_arr),
            crs="local_equirectangular",
            method="equirectangular",
            projected=True,
        )

    transformer = Transformer.from_crs("EPSG:4326", selected_crs, always_xy=True)
    x, y = transformer.transform(lon_arr, lat_arr)
    return ProjectionResult(
        xy=np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)]),
        crs=selected_crs,
        method="pyproj",
        projected=True,
    )


def _pca_svd(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt.T
    variances = singular_values**2 / max(x.shape[0] - 1, 1)
    ratios = variances / variances.sum() if variances.sum() > 0 else np.zeros_like(variances)
    return coords, ratios


def _parse_component_indices(component_indices: Optional[Sequence[int] | str]) -> Optional[list[int]]:
    if component_indices is None:
        return None
    if isinstance(component_indices, str):
        parts = [part.strip() for part in component_indices.split(",") if part.strip()]
        return [int(part) for part in parts]
    return [int(part) for part in component_indices]


def reduce_latent_to_2d(
    latents: np.ndarray,
    *,
    reduce_method: str = "pca",
    n_components: int = 2,
    component_indices: Optional[Sequence[int] | str] = None,
    seed: int = 0,
) -> ReductionResult:
    """Reduce latents to 2D. Manual PCA components are 1-based, e.g. '1,3'."""
    x = np.asarray(latents, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("latents must be a 2D array")
    selected = _parse_component_indices(component_indices)
    if selected is not None and len(selected) != 2:
        raise ValueError("Exactly two component indices are required for 2D alignment")
    if reduce_method not in {"pca", "none"}:
        raise ValueError("reduce_method must be one of: pca, none")

    if reduce_method == "pca" and x.shape[1] == 2 and selected is None:
        variances = x.var(axis=0)
        total = float(variances.sum())
        ratios = (variances / total).tolist() if total > 0 else [0.0, 0.0]
        return ReductionResult(
            coords=x,
            explained_variance_ratio=[float(v) for v in ratios],
            selected_components=[1, 2],
            method="none_2d_passthrough",
        )

    if reduce_method == "none":
        if selected is None:
            selected = list(range(1, int(n_components) + 1))
        zero_based = [idx - 1 for idx in selected]
        if min(zero_based) < 0 or max(zero_based) >= x.shape[1]:
            raise ValueError(f"Requested latent dimensions {selected} outside width {x.shape[1]}")
        return ReductionResult(
            coords=x[:, zero_based],
            explained_variance_ratio=[],
            selected_components=selected,
            method="none",
        )

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        coords_all, ratios = _pca_svd(x)
    else:
        pca = PCA(n_components=min(x.shape), random_state=int(seed))
        coords_all = pca.fit_transform(x)
        ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)

    if selected is None:
        selected = list(range(1, int(n_components) + 1))
    zero_based = [idx - 1 for idx in selected]
    if min(zero_based) < 0 or max(zero_based) >= coords_all.shape[1]:
        raise ValueError(f"Requested PCA components {selected} outside available range")
    return ReductionResult(
        coords=coords_all[:, zero_based],
        explained_variance_ratio=[float(v) for v in ratios.tolist()],
        selected_components=selected,
        method="pca",
    )


def _standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    # Use one isotropic scale per point cloud. Per-axis scaling would shear the
    # data before Procrustes and can destroy exact rotation+scale recovery.
    scalar = float(np.sqrt(((x - mean) ** 2).mean()))
    if scalar < 1e-12:
        scalar = 1.0
    std = np.asarray([scalar, scalar], dtype=np.float64)
    return (x - mean) / std, mean, std


def fit_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    allow_reflection: bool = True,
) -> SimilarityTransform:
    """Fit target ~= scale * standardize(source) @ rotation + translation via SVD."""
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("source and target must both have shape (n_samples, 2)")

    src_z, src_mean, src_std = _standardize(src)
    tgt_z, tgt_mean, tgt_std = _standardize(tgt)
    covariance = src_z.T @ tgt_z
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if not allow_reflection and np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = u @ vt
    rotated = src_z @ rotation
    denom = float((rotated**2).sum())
    scale = 1.0 if denom <= 0.0 else float((rotated * tgt_z).sum() / denom)
    translation = tgt_z.mean(axis=0) - scale * rotated.mean(axis=0)
    aligned_z = scale * rotated + translation
    residual = aligned_z - tgt_z
    disparity = float((residual**2).sum())
    return SimilarityTransform(
        rotation=rotation,
        scale=scale,
        translation=translation,
        source_mean=src_mean,
        source_std=src_std,
        target_mean=tgt_mean,
        target_std=tgt_std,
        disparity=disparity,
    )


def apply_similarity_transform(source: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
    src = np.asarray(source, dtype=np.float64)
    src_z = (src - transform.source_mean) / transform.source_std
    aligned_z = transform.scale * (src_z @ transform.rotation) + transform.translation
    return aligned_z * transform.target_std + transform.target_mean


def compute_distortion_vectors(
    frame: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    geo_xy: np.ndarray,
    aligned_xy: np.ndarray,
    label_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    dxdy = aligned_xy - geo_xy
    mag = np.sqrt((dxdy**2).sum(axis=1))
    unit = np.divide(dxdy, mag[:, None], out=np.zeros_like(dxdy), where=mag[:, None] > 0)
    out = pd.DataFrame(
        {
            "sample_id": frame[id_col].astype(str).to_numpy(),
            "latitude": frame[lat_col].to_numpy(dtype=np.float64),
            "longitude": frame[lon_col].to_numpy(dtype=np.float64),
            "projected_x": geo_xy[:, 0],
            "projected_y": geo_xy[:, 1],
            "aligned_x": aligned_xy[:, 0],
            "aligned_y": aligned_xy[:, 1],
            "dx": dxdy[:, 0],
            "dy": dxdy[:, 1],
            "distortion_magnitude": mag,
            "unit_dx": unit[:, 0],
            "unit_dy": unit[:, 1],
        }
    )
    for col in (label_col, time_col):
        if col and col in frame.columns:
            out[col] = frame[col].to_numpy()
    return out


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata_average(np.asarray(x, dtype=np.float64)), rankdata_average(np.asarray(y, dtype=np.float64)))


def compute_distortion_metrics(
    latent_2d: np.ndarray,
    geo_xy: np.ndarray,
    distortion_frame: pd.DataFrame,
    *,
    label_col: Optional[str] = None,
    knn_k: int = 15,
    seed: int = 0,
    max_pairs: int = 200_000,
) -> Dict[str, Any]:
    latent_dist = _pairwise_distances(np.asarray(latent_2d, dtype=np.float64))
    geo_dist = _pairwise_distances(np.asarray(geo_xy, dtype=np.float64))
    tri = np.triu_indices(latent_dist.shape[0], k=1)
    n_pairs = int(tri[0].shape[0])
    if n_pairs > int(max_pairs):
        rng = np.random.default_rng(seed)
        keep = rng.choice(n_pairs, size=int(max_pairs), replace=False)
        pair_latent = latent_dist[tri][keep]
        pair_geo = geo_dist[tri][keep]
    else:
        pair_latent = latent_dist[tri]
        pair_geo = geo_dist[tri]

    magnitudes = distortion_frame["distortion_magnitude"].to_numpy(dtype=np.float64)
    metrics: Dict[str, Any] = {
        "n_samples": int(len(distortion_frame)),
        "pairwise_distance_n_pairs_used": int(pair_latent.shape[0]),
        "pairwise_latent_geo_pearson": pearson_corr(pair_latent, pair_geo),
        "pairwise_latent_geo_spearman": _spearman_corr(pair_latent, pair_geo),
        "rms_alignment_error": float(np.sqrt(np.mean(magnitudes**2))),
        "median_distortion_magnitude": float(np.median(magnitudes)),
        "mean_distortion_magnitude": float(np.mean(magnitudes)),
    }

    if label_col and label_col in distortion_frame.columns:
        grouped = distortion_frame.groupby(label_col, dropna=False)["distortion_magnitude"].mean()
        metrics["per_label_mean_distortion_magnitude"] = {
            str(key): float(value) for key, value in grouped.sort_values(ascending=False).items()
        }

    if knn_k > 0 and len(distortion_frame) > 1:
        k = min(int(knn_k), len(distortion_frame) - 1)
        latent_order = np.argsort(latent_dist, axis=1)[:, 1 : k + 1]
        geo_order = np.argsort(geo_dist, axis=1)[:, 1 : k + 1]
        overlaps = [
            len(set(latent_order[i].tolist()).intersection(geo_order[i].tolist())) / float(k)
            for i in range(len(distortion_frame))
        ]
        metrics[f"knn_geography_overlap@{k}"] = float(np.mean(overlaps))
    return metrics


def _color_values(frame: pd.DataFrame, color_col: Optional[str]) -> tuple[Any, Optional[str], bool]:
    if color_col is None or color_col not in frame.columns:
        return "#1f77b4", None, False
    series = frame[color_col]
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=np.float64), color_col, True
    labels = series.fillna("NA").astype(str)
    top = labels.value_counts().index[:20]
    lookup = {label: idx for idx, label in enumerate(top)}
    colors = labels.map(lambda x: lookup.get(x, len(top))).to_numpy()
    return colors, color_col, False


def plot_distortion_map(
    distortion_frame: pd.DataFrame,
    *,
    output_path: Path,
    color_col: Optional[str] = None,
    plot_arrows: int | bool = True,
    plot_labels: int | bool = False,
    arrow_scale: Optional[float] = None,
    title: str = "Genetic distortion map",
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    colors, color_label, numeric = _color_values(distortion_frame, color_col)
    scatter = ax.scatter(
        distortion_frame["projected_x"],
        distortion_frame["projected_y"],
        c=colors,
        s=18,
        alpha=0.85,
        cmap="viridis" if numeric else "tab20",
        linewidths=0,
    )
    if numeric and color_label:
        fig.colorbar(scatter, ax=ax, label=color_label)
    if _as_bool(plot_arrows):
        dx = distortion_frame["dx"].to_numpy(dtype=np.float64)
        dy = distortion_frame["dy"].to_numpy(dtype=np.float64)
        scale = 1.0 if arrow_scale is None else float(arrow_scale)
        ax.quiver(
            distortion_frame["projected_x"],
            distortion_frame["projected_y"],
            dx * scale,
            dy * scale,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.002,
            alpha=0.55,
            color="black",
        )
    if _as_bool(plot_labels):
        for _, row in distortion_frame.iterrows():
            ax.text(row["projected_x"], row["projected_y"], str(row["sample_id"]), fontsize=5, alpha=0.65)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Projected x")
    ax.set_ylabel("Projected y")
    ax.set_title(title)
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_aligned_vs_true(distortion_frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    ax.scatter(distortion_frame["projected_x"], distortion_frame["projected_y"], s=14, alpha=0.75, label="true geography")
    ax.scatter(distortion_frame["aligned_x"], distortion_frame["aligned_y"], s=14, alpha=0.75, label="aligned latent")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Projected x")
    ax.set_ylabel("Projected y")
    ax.set_title("Aligned latent coordinates vs true geography")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_distortion_histogram(distortion_frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.hist(distortion_frame["distortion_magnitude"], bins=40, color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Distortion magnitude")
    ax.set_ylabel("Sample count")
    ax.set_title("Distribution of genetic-geographic distortion")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_distance_correlation(
    latent_2d: np.ndarray,
    geo_xy: np.ndarray,
    metrics: Dict[str, Any],
    output_path: Path,
    *,
    seed: int,
    max_pairs: int = 50_000,
) -> None:
    latent_dist = _pairwise_distances(latent_2d)
    geo_dist = _pairwise_distances(geo_xy)
    tri = np.triu_indices(latent_dist.shape[0], k=1)
    n_pairs = int(tri[0].shape[0])
    rng = np.random.default_rng(seed)
    keep = np.arange(n_pairs)
    if n_pairs > max_pairs:
        keep = rng.choice(n_pairs, size=max_pairs, replace=False)
    x = latent_dist[tri][keep]
    y = geo_dist[tri][keep]
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.scatter(x, y, s=4, alpha=0.18, linewidths=0)
    ax.set_xlabel("Pairwise latent 2D distance")
    ax.set_ylabel("Pairwise geographic distance")
    ax.set_title("Latent distance vs geographic distance")
    ax.text(
        0.02,
        0.98,
        f"Pearson={metrics['pairwise_latent_geo_pearson']:.3f}\n"
        f"Spearman={metrics['pairwise_latent_geo_spearman']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_distortion_pipeline(
    *,
    latents_csv: Path,
    metadata_tsv: Path,
    out_dir: Path,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col: str = "sample_id",
    label_col: Optional[str] = None,
    time_col: Optional[str] = None,
    reduce_method: str = "pca",
    n_components: int = 2,
    pc_indices: Optional[str] = None,
    project_coords: int | bool = True,
    crs: Optional[str] = None,
    min_samples: int = 10,
    plot_arrows: int | bool = True,
    plot_labels: int | bool = False,
    arrow_scale: Optional[float] = None,
    knn_k: int = 15,
    seed: int = 0,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, latent_values, merge_report = load_and_merge_latent_geo(
        latents_csv,
        metadata_tsv,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        label_col=label_col,
        time_col=time_col,
        min_samples=min_samples,
    )
    projection = project_latlon_to_xy(
        frame[lat_col].to_numpy(dtype=np.float64),
        frame[lon_col].to_numpy(dtype=np.float64),
        project_coords=project_coords,
        crs=crs,
    )
    reduction = reduce_latent_to_2d(
        latent_values,
        reduce_method=reduce_method,
        n_components=n_components,
        component_indices=pc_indices,
        seed=seed,
    )
    transform = fit_similarity_transform(reduction.coords, projection.xy)
    aligned = apply_similarity_transform(reduction.coords, transform)
    distortion = compute_distortion_vectors(
        frame,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        geo_xy=projection.xy,
        aligned_xy=aligned,
        label_col=label_col,
        time_col=time_col,
    )
    metrics = compute_distortion_metrics(
        reduction.coords,
        projection.xy,
        distortion,
        label_col=label_col,
        knn_k=knn_k,
        seed=seed,
    )
    metrics.update(
        {
            "merge_report": merge_report.__dict__,
            "projection": projection.__dict__ | {"xy": None},
            "reduction": reduction.__dict__ | {"coords": None},
            "transform": {
                "rotation": transform.rotation.tolist(),
                "scale": transform.scale,
                "translation": transform.translation.tolist(),
                "source_mean": transform.source_mean.tolist(),
                "source_std": transform.source_std.tolist(),
                "target_mean": transform.target_mean.tolist(),
                "target_std": transform.target_std.tolist(),
                "disparity": transform.disparity,
            },
        }
    )

    distortion.to_csv(out_dir / "distortion_vectors.tsv", sep="\t", index=False)
    with (out_dir / "distortion_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    frame.to_csv(out_dir / "merged_latent_geo.tsv", sep="\t", index=False)

    color_col = time_col or label_col
    plot_distortion_map(
        distortion,
        output_path=out_dir / "distortion_map.png",
        color_col=color_col,
        plot_arrows=plot_arrows,
        plot_labels=plot_labels,
        arrow_scale=arrow_scale,
    )
    _plot_aligned_vs_true(distortion, out_dir / "aligned_vs_true.png")
    _plot_distortion_histogram(distortion, out_dir / "distortion_histogram.png")
    _plot_distance_correlation(reduction.coords, projection.xy, metrics, out_dir / "distance_correlation.png", seed=seed)

    return {
        "out_dir": str(out_dir),
        "n_samples": int(len(distortion)),
        "distortion_vectors": str(out_dir / "distortion_vectors.tsv"),
        "distortion_metrics": str(out_dir / "distortion_metrics.json"),
        "metrics": metrics,
    }
