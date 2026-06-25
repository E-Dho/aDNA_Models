from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import read_anno_rows
from .eval import nearest_neighbors_with_distances


GENETIC_ID_PREFIX = "Genetic ID"


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in {"", "..", "nan", "NaN"} else text


def _resolve_anno_column(columns: Sequence[str], preferred: str, prefix: Optional[str] = None) -> str:
    if preferred in columns:
        return preferred
    if prefix is not None:
        for col in columns:
            if col.startswith(prefix):
                return col
    raise ValueError(f"Could not find required .anno column '{preferred}'")


def load_anno_coordinates(anno_path: Path) -> pd.DataFrame:
    rows = read_anno_rows(anno_path)
    if not rows:
        raise ValueError(f"No rows found in .anno file: {anno_path}")
    columns = list(rows[0].keys())
    sample_col = _resolve_anno_column(columns, "Genetic ID", GENETIC_ID_PREFIX)
    lat_col = _resolve_anno_column(columns, "Lat.")
    lon_col = _resolve_anno_column(columns, "Long.")
    frame = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "sample_id": frame[sample_col].map(_clean_text),
            "lat": pd.to_numeric(frame[lat_col], errors="coerce"),
            "lon": pd.to_numeric(frame[lon_col], errors="coerce"),
        }
    )
    out = out[out["sample_id"] != ""].drop_duplicates("sample_id", keep="first")
    return out


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if not all(np.isfinite([lat1, lon1, lat2, lon2])):
        return float("nan")
    radius_km = 6371.0088
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return float(2.0 * radius_km * math.asin(min(1.0, math.sqrt(a))))


def spherical_mean_latlon(lat: np.ndarray, lon: np.ndarray) -> Tuple[float, float]:
    finite = np.isfinite(lat) & np.isfinite(lon)
    if finite.sum() == 0:
        return float("nan"), float("nan")
    lat_rad = np.deg2rad(lat[finite].astype(np.float64))
    lon_rad = np.deg2rad(lon[finite].astype(np.float64))
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    z_mean = float(z.mean())
    hyp = math.hypot(x_mean, y_mean)
    if hyp == 0.0 and z_mean == 0.0:
        return float("nan"), float("nan")
    mean_lat = math.degrees(math.atan2(z_mean, hyp))
    mean_lon = math.degrees(math.atan2(y_mean, x_mean))
    return float(mean_lat), float(mean_lon)


def per_sample_adaptive_group_knn_purity(
    labels: Sequence[str],
    neighbors: np.ndarray,
    neighbor_dist2: np.ndarray,
) -> np.ndarray:
    labels_arr = np.asarray([_clean_text(label) for label in labels], dtype=object)
    label_counts: Dict[str, int] = {}
    for label in labels_arr:
        if label:
            label_counts[str(label)] = label_counts.get(str(label), 0) + 1

    out = np.full(labels_arr.shape[0], np.nan, dtype=np.float64)
    for i in range(labels_arr.shape[0]):
        label = str(labels_arr[i])
        if not label:
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
        out[i] = purity
    return out


def _latent_columns(frame: pd.DataFrame) -> list[str]:
    zcols = [col for col in frame.columns if col.startswith("z")]
    if not zcols:
        raise ValueError("No latent columns found; expected z1..zN columns")
    return zcols


def load_outlier_inputs(latents_csv: Path, sample_stats_tsv: Path, anno_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    latents = pd.read_csv(latents_csv)
    stats = pd.read_csv(sample_stats_tsv, sep="\t")
    coords = load_anno_coordinates(anno_path)
    if "sample_id" not in latents.columns:
        raise ValueError("latents_csv is missing sample_id")
    if "sample_id" not in stats.columns:
        raise ValueError("sample_stats_tsv is missing sample_id")
    zcols = _latent_columns(latents)
    frame = latents.merge(stats, on="sample_id", how="inner", validate="one_to_one")
    frame = frame.merge(coords, on="sample_id", how="left", validate="one_to_one")
    for col in ["country", "original_group_id"]:
        frame[col] = frame[col].map(_clean_text)
    for col in ["observed_fraction", "date_mean_bp", "lat", "lon", *zcols]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=zcols).reset_index(drop=True)
    latents_arr = frame[zcols].to_numpy(dtype=np.float64)
    return frame, latents_arr


def compute_knn_geo_displacement(frame: pd.DataFrame, neighbors: np.ndarray) -> pd.DataFrame:
    lat = frame["lat"].to_numpy(dtype=np.float64)
    lon = frame["lon"].to_numpy(dtype=np.float64)
    rows = []
    for i in range(len(frame)):
        neigh = neighbors[i]
        mean_lat, mean_lon = spherical_mean_latlon(lat[neigh], lon[neigh])
        distance = haversine_km(lat[i], lon[i], mean_lat, mean_lon)
        valid_neighbor_coords = int((np.isfinite(lat[neigh]) & np.isfinite(lon[neigh])).sum())
        rows.append(
            {
                "knn_mean_lat": mean_lat,
                "knn_mean_lon": mean_lon,
                "knn_geo_distance_km": distance,
                "knn_neighbors_with_coords": valid_neighbor_coords,
            }
        )
    return pd.DataFrame(rows)


def calibrate_epsilon(
    distances_km: np.ndarray,
    adaptive_purity: np.ndarray,
    *,
    adaptive_purity_min: float,
    epsilon_quantile: float,
) -> Tuple[float, pd.DataFrame, int]:
    eligible = (
        np.isfinite(distances_km)
        & np.isfinite(adaptive_purity)
        & (adaptive_purity >= float(adaptive_purity_min))
    )
    calibration_distances = distances_km[eligible]
    quantiles = [0.50, 0.75, 0.90, 0.95, 0.975, 0.99]
    if calibration_distances.size == 0:
        rows = [{"quantile": q, "distance_km": float("nan")} for q in quantiles]
        return float("nan"), pd.DataFrame(rows), 0
    rows = [
        {"quantile": q, "distance_km": float(np.quantile(calibration_distances, q))}
        for q in quantiles
    ]
    epsilon = float(np.quantile(calibration_distances, float(epsilon_quantile)))
    return epsilon, pd.DataFrame(rows), int(calibration_distances.size)


def add_geo_candidate_filters(
    scores: pd.DataFrame,
    *,
    max_rank1_neighbor_latent_distance: Optional[float] = 0.6,
) -> pd.DataFrame:
    out = scores.copy()
    rank1_distances = []
    drop_group_prefix = []
    drop_rank1_distance = []
    for _, row in out.iterrows():
        group = _clean_text(row["original_group_id"])
        group_size = int(row["original_group_size"])
        neighbor_groups = str(row["neighbor_original_group_ids"]).split(";")
        neighbor_distances = [
            float(value) for value in str(row["neighbor_latent_distances"]).split(";") if value
        ]
        required_group_neighbors = max(group_size - 1, 0)
        group_prefix_matches = (
            group_size > 1
            and bool(group)
            and len(neighbor_groups) >= required_group_neighbors
            and all(value == group for value in neighbor_groups[:required_group_neighbors])
        )
        rank1_distance = neighbor_distances[0] if neighbor_distances else float("nan")
        rank1_distance_too_large = (
            max_rank1_neighbor_latent_distance is not None
            and np.isfinite(rank1_distance)
            and rank1_distance > float(max_rank1_neighbor_latent_distance)
        )
        rank1_distances.append(rank1_distance)
        drop_group_prefix.append(bool(group_prefix_matches))
        drop_rank1_distance.append(bool(rank1_distance_too_large))

    out["rank1_neighbor_latent_distance"] = rank1_distances
    out["drop_original_group_prefix"] = drop_group_prefix
    out["drop_rank1_neighbor_distance"] = drop_rank1_distance
    out["flag_geo_candidate_filtered"] = (
        out["flag_geo_displacement"]
        & ~out["drop_original_group_prefix"]
        & ~out["drop_rank1_neighbor_distance"]
    )
    return out


def expand_candidate_knn_rows(scores: pd.DataFrame, *, candidate_flag: str) -> pd.DataFrame:
    rows = []
    for _, row in scores[scores[candidate_flag]].iterrows():
        neighbor_fields = {
            "neighbor_sample_id": str(row["neighbor_sample_ids"]).split(";"),
            "neighbor_original_group_id": str(row["neighbor_original_group_ids"]).split(";"),
            "neighbor_country": str(row["neighbor_countries"]).split(";"),
            "neighbor_latent_distance": str(row["neighbor_latent_distances"]).split(";"),
        }
        for rank, values in enumerate(zip(*neighbor_fields.values()), start=1):
            sample_values = {
                "sample_id": row["sample_id"],
                "country": row["country"],
                "original_group_id": row["original_group_id"],
                "original_group_size": row["original_group_size"],
                "same_original_group_neighbors": row["same_original_group_neighbors"],
                "knn_geo_distance_km": row["knn_geo_distance_km"],
                "adaptive_original_group_purity": row["adaptive_original_group_purity"],
                "rank1_neighbor_latent_distance": row["rank1_neighbor_latent_distance"],
                "flag_original_group_isolation": row["flag_original_group_isolation"],
                "flag_geo_displacement": row["flag_geo_displacement"],
                "drop_original_group_prefix": row["drop_original_group_prefix"],
                "drop_rank1_neighbor_distance": row["drop_rank1_neighbor_distance"],
                "flag_geo_candidate_filtered": row["flag_geo_candidate_filtered"],
                "neighbor_rank": rank,
            }
            sample_values.update(dict(zip(neighbor_fields.keys(), values)))
            sample_values["neighbor_latent_distance"] = float(
                sample_values["neighbor_latent_distance"]
            )
            rows.append(sample_values)
    return pd.DataFrame(rows)


def score_migrant_outliers(
    *,
    latents_csv: Path,
    sample_stats_tsv: Path,
    anno_path: Path,
    k: int = 15,
    epsilon_km: Optional[float] = None,
    epsilon_quantile: float = 0.95,
    adaptive_purity_min: float = 0.80,
    min_original_group_size: int = 2,
    max_rank1_neighbor_latent_distance: Optional[float] = 0.6,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    frame, latents = load_outlier_inputs(latents_csv, sample_stats_tsv, anno_path)
    neighbors, neighbor_dist2 = nearest_neighbors_with_distances(latents, k)
    labels = frame["original_group_id"].astype(str).tolist()
    labels_arr = np.asarray(labels, dtype=object)
    group_sizes = frame["original_group_id"].map(frame["original_group_id"].value_counts()).fillna(0).astype(int)
    same_counts = np.asarray(
        [(labels_arr[neighbors[i]] == labels_arr[i]).sum() if labels_arr[i] else 0 for i in range(len(frame))],
        dtype=np.int64,
    )
    adaptive_purity = per_sample_adaptive_group_knn_purity(labels, neighbors, neighbor_dist2)
    geo = compute_knn_geo_displacement(frame, neighbors)

    scores = frame[
        ["sample_id", "country", "original_group_id", "observed_fraction", "date_mean_bp", "lat", "lon"]
    ].copy()
    scores["original_group_size"] = group_sizes.to_numpy(dtype=np.int64)
    scores["same_original_group_neighbors"] = same_counts
    scores["original_group_knn_fraction"] = same_counts / max(neighbors.shape[1], 1)
    scores["flag_original_group_isolation"] = (
        (scores["original_group_size"] >= int(min_original_group_size))
        & (scores["same_original_group_neighbors"] <= 1)
        & (scores["original_group_id"] != "")
    )
    scores = pd.concat([scores, geo], axis=1)
    scores["adaptive_original_group_purity"] = adaptive_purity

    auto_epsilon, calibration, calibration_n = calibrate_epsilon(
        scores["knn_geo_distance_km"].to_numpy(dtype=np.float64),
        adaptive_purity,
        adaptive_purity_min=adaptive_purity_min,
        epsilon_quantile=epsilon_quantile,
    )
    chosen_epsilon = float(epsilon_km) if epsilon_km is not None else auto_epsilon
    scores["flag_geo_displacement"] = (
        np.isfinite(scores["knn_geo_distance_km"].to_numpy(dtype=np.float64))
        & np.isfinite(chosen_epsilon)
        & (scores["knn_geo_distance_km"] > chosen_epsilon)
    )
    scores["flag_any"] = scores["flag_original_group_isolation"] | scores["flag_geo_displacement"]
    scores["neighbor_sample_ids"] = [";".join(frame.iloc[neighbors[i]]["sample_id"].astype(str)) for i in range(len(frame))]
    scores["neighbor_original_group_ids"] = [
        ";".join(frame.iloc[neighbors[i]]["original_group_id"].astype(str)) for i in range(len(frame))
    ]
    scores["neighbor_countries"] = [";".join(frame.iloc[neighbors[i]]["country"].astype(str)) for i in range(len(frame))]
    scores["neighbor_latent_distances"] = [
        ";".join(f"{math.sqrt(float(x)):.6g}" for x in neighbor_dist2[i]) for i in range(len(frame))
    ]
    scores = add_geo_candidate_filters(
        scores,
        max_rank1_neighbor_latent_distance=max_rank1_neighbor_latent_distance,
    )

    summary: Dict[str, object] = {
        "latents_csv": str(latents_csv),
        "sample_stats_tsv": str(sample_stats_tsv),
        "anno": str(anno_path),
        "n_samples": int(len(scores)),
        "k": int(neighbors.shape[1]),
        "epsilon_mode": "manual" if epsilon_km is not None else "auto_quantile",
        "epsilon_km": chosen_epsilon,
        "epsilon_quantile": float(epsilon_quantile),
        "adaptive_purity_min": float(adaptive_purity_min),
        "calibration_n": calibration_n,
        "min_original_group_size": int(min_original_group_size),
        "max_rank1_neighbor_latent_distance": max_rank1_neighbor_latent_distance,
        "n_missing_sample_coordinates": int((~np.isfinite(scores["lat"]) | ~np.isfinite(scores["lon"])).sum()),
        "n_flag_original_group_isolation": int(scores["flag_original_group_isolation"].sum()),
        "n_flag_geo_displacement": int(scores["flag_geo_displacement"].sum()),
        "n_drop_geo_original_group_prefix": int(
            (scores["flag_geo_displacement"] & scores["drop_original_group_prefix"]).sum()
        ),
        "n_drop_geo_rank1_neighbor_distance": int(
            (scores["flag_geo_displacement"] & scores["drop_rank1_neighbor_distance"]).sum()
        ),
        "n_flag_geo_candidate_filtered": int(scores["flag_geo_candidate_filtered"].sum()),
        "n_flag_any": int(scores["flag_any"].sum()),
    }
    return scores, calibration, summary


def write_outlier_outputs(
    *,
    scores: pd.DataFrame,
    calibration: pd.DataFrame,
    summary: Dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "migrant_outlier_scores.tsv", sep="\t", index=False)
    scores[scores["flag_any"]].to_csv(output_dir / "migrant_outlier_candidates.tsv", sep="\t", index=False)
    scores[scores["flag_geo_candidate_filtered"]].to_csv(
        output_dir / "migrant_outlier_geo_displacement_candidates.tsv",
        sep="\t",
        index=False,
    )
    expand_candidate_knn_rows(
        scores,
        candidate_flag="flag_geo_candidate_filtered",
    ).to_csv(
        output_dir / "migrant_outlier_geo_displacement_candidates_knn_filtered.tsv",
        sep="\t",
        index=False,
    )
    calibration.to_csv(output_dir / "epsilon_calibration.tsv", sep="\t", index=False)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)


def maybe_write_plots(scores: pd.DataFrame, output_dir: Path) -> list[str]:
    produced: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return produced

    distances = scores["knn_geo_distance_km"].to_numpy(dtype=np.float64)
    distances = distances[np.isfinite(distances)]
    epsilon = None
    if "flag_geo_displacement" in scores.columns and scores["flag_geo_displacement"].any():
        flagged_dist = scores.loc[scores["flag_geo_displacement"], "knn_geo_distance_km"].min()
        epsilon = float(flagged_dist) if np.isfinite(flagged_dist) else None
    if distances.size:
        path = output_dir / "knn_geo_distance_hist.png"
        plt.figure(figsize=(8, 5))
        plt.hist(distances, bins=50, color="#456990", alpha=0.85)
        if epsilon is not None:
            plt.axvline(epsilon, color="#e63946", linestyle="--", label="flag threshold lower bound")
            plt.legend()
        plt.xlabel("KNN geographic displacement (km)")
        plt.ylabel("Samples")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        produced.append(str(path))

    path = output_dir / "adaptive_purity_vs_geo_distance.png"
    plt.figure(figsize=(8, 5))
    plt.scatter(
        scores["adaptive_original_group_purity"],
        scores["knn_geo_distance_km"],
        s=8,
        c=scores["flag_any"].map({True: "#e63946", False: "#555555"}),
        alpha=0.7,
    )
    plt.xlabel("Per-sample adaptive original-group KNN purity")
    plt.ylabel("KNN geographic displacement (km)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    produced.append(str(path))

    valid_coords = scores[np.isfinite(scores["lat"]) & np.isfinite(scores["lon"])]
    if len(valid_coords):
        path = output_dir / "candidate_location_scatter.png"
        plt.figure(figsize=(8, 6))
        plt.scatter(valid_coords["lon"], valid_coords["lat"], s=6, color="#999999", alpha=0.25)
        candidates = valid_coords[valid_coords["flag_any"]]
        if len(candidates):
            plt.scatter(candidates["lon"], candidates["lat"], s=18, color="#e63946", alpha=0.85)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        produced.append(str(path))
    return produced
