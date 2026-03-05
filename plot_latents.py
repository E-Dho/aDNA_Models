#!/usr/bin/env python3
"""Merge global latents with metadata and write PCA/UMAP/t-SNE plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import numpy as np

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot latent embeddings merged with metadata."
    )
    parser.add_argument("--latents_csv", required=True, help="Path to global_latents.csv")
    parser.add_argument(
        "--metadata_tsv",
        required=True,
        help="Path to AADR metadata TSV/ANNO file",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for CSVs and PNGs")
    parser.add_argument(
        "--model_label",
        default="model",
        help="Label used in plot titles (e.g., HMAE, token-snp).",
    )
    parser.add_argument(
        "--latent_key",
        default="sample_id",
        help="Column name in latents CSV used for merging",
    )
    parser.add_argument(
        "--metadata_key",
        default=None,
        help="Column name in metadata used for merging; defaults to the first metadata column",
    )
    parser.add_argument(
        "--color_by",
        action="append",
        default=[],
        help="Metadata column to color points by; can be passed multiple times",
    )
    parser.add_argument("--point_size", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iterations", type=int, default=1000)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def load_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    latents = pd.read_csv(args.latents_csv)
    metadata = pd.read_csv(args.metadata_tsv, sep="\t", low_memory=False)

    if args.latent_key not in latents.columns:
        raise ValueError(f"Latent key '{args.latent_key}' not found in latents CSV")

    metadata_key = args.metadata_key or metadata.columns[0]
    if metadata_key not in metadata.columns:
        raise ValueError(f"Metadata key '{metadata_key}' not found in metadata table")

    return latents, metadata, metadata_key


def latent_columns(frame: pd.DataFrame) -> list[str]:
    cols = [col for col in frame.columns if col.startswith("z")]
    if not cols:
        raise ValueError("No latent columns found; expected columns like z1, z2, ...")
    return cols


def compute_embeddings(
    latents: pd.DataFrame,
    zcols: list[str],
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    z = latents[zcols].to_numpy(dtype=np.float64, copy=False)

    if SKPCA is not None:
        pca_coords = SKPCA(n_components=2, random_state=args.random_state).fit_transform(z)
    else:
        z_centered = z - z.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(z_centered, full_matrices=False)
        pca_coords = z_centered @ vt[:2].T

    index = latents.index
    out = {
        "pca": pd.DataFrame({"dim1": pca_coords[:, 0], "dim2": pca_coords[:, 1]}, index=index),
    }

    if umap is not None:
        umap_coords = umap.UMAP(
            n_components=2,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.random_state,
        ).fit_transform(z)
        out["umap"] = pd.DataFrame({"dim1": umap_coords[:, 0], "dim2": umap_coords[:, 1]}, index=index)

    if SKTSNE is not None:
        max_perplexity = max(5.0, min(args.tsne_perplexity, (len(latents) - 1) / 3))
        tsne_coords = SKTSNE(
            n_components=2,
            perplexity=max_perplexity,
            max_iter=args.tsne_iterations,
            init="pca",
            learning_rate="auto",
            random_state=args.random_state,
        ).fit_transform(z)
        out["tsne"] = pd.DataFrame({"dim1": tsne_coords[:, 0], "dim2": tsne_coords[:, 1]}, index=index)

    return out


def save_plot(
    frame: pd.DataFrame,
    method: str,
    color_by: str,
    model_label: str,
    output_path: Path,
    point_size: float,
    alpha: float,
) -> None:
    plt.figure(figsize=(9, 7))
    series = frame[color_by]

    if pd.api.types.is_numeric_dtype(series):
        scatter = plt.scatter(
            frame["dim1"],
            frame["dim2"],
            c=series,
            cmap="viridis",
            s=point_size,
            alpha=alpha,
            linewidths=0,
        )
        plt.colorbar(scatter, label=color_by)
    else:
        filled = series.fillna("NA").astype(str)
        counts = filled.value_counts()
        keep = counts.index[:20]
        grouped = filled.where(filled.isin(keep), other="Other")
        for label in grouped.unique():
            mask = grouped == label
            plt.scatter(
                frame.loc[mask, "dim1"],
                frame.loc[mask, "dim2"],
                s=point_size,
                alpha=alpha,
                linewidths=0,
                label=label,
            )
        plt.legend(
            title=color_by,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            markerscale=1.6,
        )

    plt.xlabel(f"{method.upper()} 1")
    plt.ylabel(f"{method.upper()} 2")
    plt.title(f"{method.upper()} of {model_label} latents colored by {color_by}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latents, metadata, metadata_key = load_tables(args)
    zcols = latent_columns(latents)

    merged = latents.merge(
        metadata,
        left_on=args.latent_key,
        right_on=metadata_key,
        how="left",
        validate="one_to_one",
    )
    merged.to_csv(output_dir / "merged_latents_metadata.csv", index=False)

    embeddings = compute_embeddings(latents, zcols, args)
    available = ",".join(sorted(embeddings.keys()))
    print(f"Computed embeddings: {available}")
    default_color_fields = ["Group ID", "Political Entity", "Date mean", "Locality"]
    color_fields = args.color_by or [field for field in default_color_fields if field in merged.columns]

    for method, coords in embeddings.items():
        frame = merged.copy()
        frame["dim1"] = coords["dim1"].to_numpy()
        frame["dim2"] = coords["dim2"].to_numpy()
        frame.to_csv(output_dir / f"{method}_coords.csv", index=False)

        for color_by in color_fields:
            if color_by not in frame.columns:
                continue
            save_plot(
                frame=frame,
                method=method,
                color_by=color_by,
                model_label=args.model_label,
                output_path=output_dir / f"{method}_{sanitize_name(color_by)}.png",
                point_size=args.point_size,
                alpha=args.alpha,
            )


if __name__ == "__main__":
    main()
