from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ProjectionResult:
    xy: np.ndarray
    crs: str
    method: str


def _looks_like_europe(lat: np.ndarray, lon: np.ndarray) -> bool:
    return bool(
        np.nanmedian(lat) >= 30.0
        and np.nanmedian(lat) <= 72.0
        and np.nanmedian(lon) >= -31.0
        and np.nanmedian(lon) <= 50.0
    )


def _equirectangular_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    radius_m = 6_371_000.0
    lat0 = np.deg2rad(float(np.nanmean(lat)))
    x = radius_m * np.deg2rad(lon) * math.cos(lat0)
    y = radius_m * np.deg2rad(lat)
    return np.column_stack([x, y]).astype(np.float64)


def project_latlon_to_xy(lat: Sequence[float], lon: Sequence[float], crs: str = "EPSG:3035") -> ProjectionResult:
    """Project lon/lat to metric coordinates, preferring EPSG:3035 for Europe."""
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    if lat_arr.shape != lon_arr.shape:
        raise ValueError("lat and lon must have the same shape")
    if not np.isfinite(lat_arr).all() or not np.isfinite(lon_arr).all():
        raise ValueError("lat/lon contain non-finite values")

    selected = crs or ("EPSG:3035" if _looks_like_europe(lat_arr, lon_arr) else "EPSG:3857")
    try:
        from pyproj import Transformer
    except ImportError:
        warnings.warn("pyproj is unavailable; using local equirectangular fallback", RuntimeWarning)
        return ProjectionResult(_equirectangular_xy(lat_arr, lon_arr), "local_equirectangular", "equirectangular")

    transformer = Transformer.from_crs("EPSG:4326", selected, always_xy=True)
    x, y = transformer.transform(lon_arr, lat_arr)
    xy = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
    if not np.isfinite(xy).all():
        raise ValueError("coordinate projection produced non-finite values")
    return ProjectionResult(xy=xy, crs=selected, method="pyproj")


def euclidean_km(pred_xy: np.ndarray, true_xy: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_xy, dtype=np.float64)
    true = np.asarray(true_xy, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError("pred_xy and true_xy must both have shape (n, 2)")
    return np.linalg.norm(pred - true, axis=1) / 1000.0


def isotropic_radius_for_probability(sigma_m: np.ndarray | float, probability: float) -> np.ndarray:
    """Radius enclosing `probability` mass for a 2D isotropic Gaussian."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in (0, 1)")
    sigma = np.asarray(sigma_m, dtype=np.float64)
    return sigma * math.sqrt(-2.0 * math.log(1.0 - probability))


def calibration_containment(error_m: np.ndarray, sigma_m: np.ndarray, probabilities=(0.50, 0.68, 0.90, 0.95)) -> dict[str, float]:
    err = np.asarray(error_m, dtype=np.float64)
    sig = np.asarray(sigma_m, dtype=np.float64)
    if err.shape != sig.shape:
        raise ValueError("error_m and sigma_m must have the same shape")
    out: dict[str, float] = {}
    for p in probabilities:
        radius = isotropic_radius_for_probability(sig, p)
        out[f"containment_{int(round(p * 100))}"] = float(np.mean(err <= radius))
    return out


def gaussian_jitter_xy(xy: np.ndarray, sigma_m: float, seed: int | None = None) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("xy must have shape (n, 2)")
    rng = np.random.default_rng(seed)
    return arr + rng.normal(0.0, float(sigma_m), size=arr.shape)


def isotropic_gaussian_nll_numpy(pred: np.ndarray, target_xy: np.ndarray, min_log_sigma: float, max_log_sigma: float) -> float:
    """Mean isotropic 2D Gaussian NLL for testability without torch."""
    arr = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("pred must have shape (n, 3)")
    if target.shape != (arr.shape[0], 2):
        raise ValueError("target_xy must have shape (n, 2)")
    log_sigma = np.clip(arr[:, 2], min_log_sigma, max_log_sigma)
    sigma2 = np.exp(2.0 * log_sigma)
    sq = np.sum((arr[:, :2] - target) ** 2, axis=1)
    nll = math.log(2.0 * math.pi) + 2.0 * log_sigma + sq / (2.0 * sigma2)
    return float(np.mean(nll))
