#!/usr/bin/env python3
"""Export 3D latent coordinates (PCA/UMAP/t-SNE) from global_latents.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA as SKPCA
except ImportError:
    SKPCA = None

try:
    from sklearn.manifold import TSNE as SKTSNE
except ImportError:
    SKTSNE = None

try:
    import umap
except ImportError:
    umap = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export 3D coordinates from latent vectors.")
    p.add_argument("--latents_csv", required=True, help="Path to global_latents.csv")
    p.add_argument("--output_dir", required=True, help="Directory for 3D coordinate CSVs")
    p.add_argument("--latent_key", default="sample_id", help="Sample ID column in latents CSV")
    p.add_argument("--umap_neighbors", type=int, default=30)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iterations", type=int, default=1000)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--methods",
        default="pca,umap,tsne",
        help="Comma-separated methods to export: pca,umap,tsne",
    )
    return p.parse_args()


def latent_columns(df: pd.DataFrame) -> list[str]:
    zcols = [c for c in df.columns if c.startswith("z")]
    if not zcols:
        raise ValueError("No latent columns found; expected z* columns.")
    return zcols


def pca_3d(z: np.ndarray, random_state: int) -> np.ndarray:
    if SKPCA is not None:
        return SKPCA(n_components=3, random_state=random_state).fit_transform(z)

    z_centered = z - z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(z_centered, full_matrices=False)
    return z_centered @ vt[:3].T


def umap_3d(z: np.ndarray, neighbors: int, min_dist: float, random_state: int) -> np.ndarray:
    if umap is None:
        raise RuntimeError("UMAP requested but umap-learn is not installed.")
    return umap.UMAP(
        n_components=3,
        n_neighbors=neighbors,
        min_dist=min_dist,
        random_state=random_state,
    ).fit_transform(z)


def tsne_3d(z: np.ndarray, perplexity: float, iterations: int, random_state: int) -> np.ndarray:
    if SKTSNE is None:
        raise RuntimeError("t-SNE requested but scikit-learn is not installed.")

    max_perplexity = max(5.0, min(perplexity, (len(z) - 1) / 3))
    return SKTSNE(
        n_components=3,
        perplexity=max_perplexity,
        max_iter=iterations,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(z)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.latents_csv)
    if args.latent_key not in df.columns:
        raise ValueError(f"Latent key '{args.latent_key}' not found in {args.latents_csv}")

    zcols = latent_columns(df)
    z = df[zcols].to_numpy(dtype=np.float64, copy=False)

    wanted = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    valid = {"pca", "umap", "tsne"}
    unknown = [m for m in wanted if m not in valid]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; valid: {sorted(valid)}")

    sample_ids = df[args.latent_key].astype(str).to_numpy()

    produced: list[str] = []

    if "pca" in wanted:
        c = pca_3d(z, random_state=args.random_state)
        p = out_dir / "pca_3d_coords.csv"
        pd.DataFrame({
            args.latent_key: sample_ids,
            "dim1": c[:, 0],
            "dim2": c[:, 1],
            "dim3": c[:, 2],
        }).to_csv(p, index=False)
        produced.append(str(p))

    if "umap" in wanted:
        c = umap_3d(
            z,
            neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.random_state,
        )
        p = out_dir / "umap_3d_coords.csv"
        pd.DataFrame({
            args.latent_key: sample_ids,
            "dim1": c[:, 0],
            "dim2": c[:, 1],
            "dim3": c[:, 2],
        }).to_csv(p, index=False)
        produced.append(str(p))

    if "tsne" in wanted:
        c = tsne_3d(
            z,
            perplexity=args.tsne_perplexity,
            iterations=args.tsne_iterations,
            random_state=args.random_state,
        )
        p = out_dir / "tsne_3d_coords.csv"
        pd.DataFrame({
            args.latent_key: sample_ids,
            "dim1": c[:, 0],
            "dim2": c[:, 1],
            "dim3": c[:, 2],
        }).to_csv(p, index=False)
        produced.append(str(p))

    print("Exported files:")
    for path in produced:
        print(path)


if __name__ == "__main__":
    main()
