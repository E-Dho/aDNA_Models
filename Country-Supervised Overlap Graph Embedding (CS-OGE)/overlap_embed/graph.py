from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import (
    GenotypeMemmapMeta,
    build_memmap_from_eigenstrat,
    build_sample_stats_rows,
    compute_observed_fraction,
    load_meta,
    load_sample_ids_from_meta,
    open_genotype_memmap,
    write_tsv,
)


@dataclass(frozen=True)
class GraphBuildArtifacts:
    graph_npz_path: Path
    sample_stats_tsv_path: Path
    spectral_init_path: Path
    build_summary_path: Path
    meta_json_path: Path


@dataclass(frozen=True)
class GraphCheckpointPaths:
    numerator_memmap_path: Path
    overlap_memmap_path: Path
    checkpoint_json_path: Path


def _log(message: str) -> None:
    print(message, flush=True)


def _checkpoint_paths(out_dir: Path) -> GraphCheckpointPaths:
    return GraphCheckpointPaths(
        numerator_memmap_path=out_dir / "partial_numerator.float32.memmap",
        overlap_memmap_path=out_dir / "partial_overlap.uint32.memmap",
        checkpoint_json_path=out_dir / "graph_checkpoint.json",
    )


def _sample_indices_signature(sample_indices: np.ndarray) -> str:
    arr = np.asarray(sample_indices, dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _atomic_json_write(path: Path, payload: Dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def _select_torch_device(compute_device: str) -> str:
    if compute_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("compute_device must be one of: auto, cpu, cuda")
    if compute_device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if compute_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("compute_device=cuda requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_compute_backend(compute_backend: str, compute_device: str) -> Tuple[str, str]:
    if compute_backend not in {"auto", "numpy", "torch"}:
        raise ValueError("compute_backend must be one of: auto, numpy, torch")
    selected_device = _select_torch_device(compute_device)
    if compute_backend == "numpy":
        return "numpy", "cpu"
    if compute_backend == "torch":
        return "torch", selected_device
    try:
        import torch  # noqa: F401
    except ImportError:
        return "numpy", "cpu"
    return "torch", selected_device


def _init_or_resume_accumulators(
    *,
    out_dir: Path,
    n_samples: int,
    meta_json_path: Path,
    chunk_snps: int,
    max_snps: int,
    latent_dim: int,
    min_overlap: int,
    top_k: int,
    overlap_shrinkage: float,
    ploidy: str,
    max_samples: int,
    sample_indices_signature: str,
    compute_backend: str,
    compute_device: str,
    resume_existing: bool,
) -> Tuple[np.memmap, np.memmap, int, int]:
    paths = _checkpoint_paths(out_dir)
    expected_shape = (n_samples, n_samples)
    if resume_existing and all(
        path.exists()
        for path in (
            paths.numerator_memmap_path,
            paths.overlap_memmap_path,
            paths.checkpoint_json_path,
        )
    ):
        checkpoint = json.loads(paths.checkpoint_json_path.read_text(encoding="utf-8"))
        config_matches = (
            int(checkpoint.get("n_samples", -1)) == int(n_samples)
            and int(checkpoint.get("chunk_snps", -1)) == int(chunk_snps)
            and int(checkpoint.get("max_snps", -1)) == int(max_snps)
            and int(checkpoint.get("latent_dim", -1)) == int(latent_dim)
            and int(checkpoint.get("min_overlap", -1)) == int(min_overlap)
            and int(checkpoint.get("top_k", -1)) == int(top_k)
            and float(checkpoint.get("overlap_shrinkage", -1.0)) == float(overlap_shrinkage)
            and str(checkpoint.get("ploidy", "")) == str(ploidy)
            and int(checkpoint.get("max_samples", -1)) == int(max_samples)
            and str(checkpoint.get("sample_indices_signature", "")) == str(sample_indices_signature)
            and str(checkpoint.get("meta_json_path", "")) == str(meta_json_path)
        )
        if config_matches:
            numerator_mm = np.memmap(
                paths.numerator_memmap_path,
                dtype=np.float32,
                mode="r+",
                shape=expected_shape,
            )
            overlap_mm = np.memmap(
                paths.overlap_memmap_path,
                dtype=np.uint32,
                mode="r+",
                shape=expected_shape,
            )
            last_completed_chunk = int(checkpoint.get("last_completed_chunk", 0))
            informative_snps = int(checkpoint.get("informative_snps", 0))
            _log(
                f"Resuming graph checkpoint from chunk={last_completed_chunk} "
                f"informative_snps={informative_snps} backend={compute_backend} device={compute_device}"
            )
            return numerator_mm, overlap_mm, last_completed_chunk, informative_snps
        _log("Existing partial checkpoint found but config mismatch; restarting from scratch.")

    numerator_mm = np.memmap(
        paths.numerator_memmap_path,
        dtype=np.float32,
        mode="w+",
        shape=expected_shape,
    )
    overlap_mm = np.memmap(
        paths.overlap_memmap_path,
        dtype=np.uint32,
        mode="w+",
        shape=expected_shape,
    )
    numerator_mm[:] = 0.0
    overlap_mm[:] = 0
    numerator_mm.flush()
    overlap_mm.flush()
    _atomic_json_write(
        paths.checkpoint_json_path,
        {
            "n_samples": int(n_samples),
            "chunk_snps": int(chunk_snps),
            "max_snps": int(max_snps),
            "latent_dim": int(latent_dim),
            "min_overlap": int(min_overlap),
            "top_k": int(top_k),
            "overlap_shrinkage": float(overlap_shrinkage),
            "ploidy": ploidy,
            "max_samples": int(max_samples),
            "sample_indices_signature": str(sample_indices_signature),
            "meta_json_path": str(meta_json_path),
            "compute_backend": str(compute_backend),
            "compute_device": str(compute_device),
            "last_completed_chunk": 0,
            "informative_snps": 0,
        },
    )
    return numerator_mm, overlap_mm, 0, 0


def _flush_accumulators_to_disk(
    *,
    numerator_acc,
    overlap_acc,
    numerator_mm: np.memmap,
    overlap_mm: np.memmap,
    checkpoint_paths: GraphCheckpointPaths,
    last_completed_chunk: int,
    informative_snps: int,
) -> None:
    if hasattr(numerator_acc, "detach"):
        numerator_np = numerator_acc.detach().cpu().numpy()
        overlap_np = overlap_acc.detach().cpu().numpy()
    else:
        numerator_np = np.asarray(numerator_acc, dtype=np.float32)
        overlap_np = np.asarray(overlap_acc, dtype=np.float32)
    numerator_mm[:] = numerator_np
    overlap_mm[:] = np.rint(overlap_np).astype(np.uint32)
    numerator_mm.flush()
    overlap_mm.flush()
    checkpoint = json.loads(checkpoint_paths.checkpoint_json_path.read_text(encoding="utf-8"))
    checkpoint["last_completed_chunk"] = int(last_completed_chunk)
    checkpoint["informative_snps"] = int(informative_snps)
    _atomic_json_write(checkpoint_paths.checkpoint_json_path, checkpoint)


def prepare_meta(
    *,
    meta_json: Optional[Path],
    geno_path: Optional[Path],
    ind_path: Optional[Path],
    out_dir: Path,
    ploidy: str,
) -> Tuple[GenotypeMemmapMeta, Path]:
    if meta_json is not None:
        return load_meta(meta_json), meta_json
    if geno_path is None:
        raise ValueError("Either meta_json or geno_path must be provided")
    out_prefix = out_dir / "prepared_genotypes"
    meta = build_memmap_from_eigenstrat(
        geno_path=geno_path,
        ind_path=ind_path,
        out_prefix=out_prefix,
        ploidy=ploidy,
    )
    return meta, out_prefix.with_suffix(".meta.json")


def compute_overlap_numerator_and_counts(
    geno_mm: np.ndarray,
    *,
    out_dir: Path,
    meta_json_path: Path,
    latent_dim: int,
    min_overlap: int,
    top_k: int,
    overlap_shrinkage: float,
    ploidy: str,
    max_samples: int,
    sample_indices: Optional[np.ndarray] = None,
    chunk_snps: int = 4096,
    max_snps: int = -1,
    min_maf: float = 1e-4,
    eps: float = 1e-6,
    progress_every: int = 25,
    checkpoint_every: int = 5,
    compute_backend: str = "auto",
    compute_device: str = "auto",
    resume_existing: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    torch = None
    total_samples, total_snps = map(int, geno_mm.shape)
    selected_sample_indices = (
        np.arange(total_samples, dtype=np.int64)
        if sample_indices is None
        else np.asarray(sample_indices, dtype=np.int64)
    )
    n_samples = int(selected_sample_indices.shape[0])
    n_snps = total_snps if max_snps <= 0 else min(int(max_snps), total_snps)
    selected_backend, selected_device = _select_compute_backend(compute_backend, compute_device)
    sample_indices_signature = _sample_indices_signature(selected_sample_indices)
    numerator_mm, overlap_mm, last_completed_chunk, informative_snps = _init_or_resume_accumulators(
        out_dir=out_dir,
        n_samples=n_samples,
        meta_json_path=meta_json_path,
        chunk_snps=chunk_snps,
        max_snps=n_snps,
        latent_dim=latent_dim,
        min_overlap=min_overlap,
        top_k=top_k,
        overlap_shrinkage=overlap_shrinkage,
        ploidy=ploidy,
        max_samples=max_samples,
        sample_indices_signature=sample_indices_signature,
        compute_backend=selected_backend,
        compute_device=selected_device,
        resume_existing=resume_existing,
    )
    checkpoint_paths = _checkpoint_paths(out_dir)
    if selected_backend == "torch":
        import torch as _torch

        torch = _torch
        numerator = torch.tensor(np.asarray(numerator_mm), dtype=torch.float32, device=selected_device)
        overlap = torch.tensor(np.asarray(overlap_mm, dtype=np.float32), dtype=torch.float32, device=selected_device)
    else:
        numerator = np.asarray(numerator_mm, dtype=np.float32)
        overlap = np.asarray(overlap_mm, dtype=np.uint32)
    started = time.time()
    n_chunks = (n_snps + chunk_snps - 1) // chunk_snps
    _log(
        f"Using compute backend={selected_backend} device={selected_device} "
        f"checkpoint_every={checkpoint_every} resume_existing={resume_existing}"
    )

    for chunk_idx, start in enumerate(range(0, n_snps, chunk_snps), start=1):
        if chunk_idx <= last_completed_chunk:
            continue
        end = min(start + chunk_snps, n_snps)
        block = np.asarray(geno_mm[selected_sample_indices, start:end], dtype=np.float32)
        obs = block >= 0
        obs_f = obs.astype(np.float32)
        observed_per_snp = obs_f.sum(axis=0)
        safe_obs = np.maximum(observed_per_snp, 1.0)
        block = np.where(obs, block, 0.0)
        p = block.sum(axis=0) / safe_obs
        maf = np.minimum(p, 1.0 - p)
        valid = (observed_per_snp >= 2.0) & (maf >= min_maf)
        if np.any(valid):
            informative_snps += int(valid.sum())
            valid_block = block[:, valid]
            valid_obs = obs[:, valid]
            valid_p = p[valid]
            denom = np.sqrt(np.maximum(valid_p * (1.0 - valid_p), eps), dtype=np.float32)
            residual = np.where(valid_obs, (valid_block - valid_p) / denom, 0.0).astype(np.float32)
            if selected_backend == "torch":
                residual_t = torch.from_numpy(residual).to(device=selected_device, dtype=torch.float32)
                valid_obs_t = torch.from_numpy(valid_obs.astype(np.float32)).to(device=selected_device, dtype=torch.float32)
                numerator += residual_t @ residual_t.T
                overlap += valid_obs_t @ valid_obs_t.T
            else:
                numerator += residual @ residual.T
                overlap += (valid_obs.astype(np.uint16) @ valid_obs.T.astype(np.uint16)).astype(np.uint32)
        if checkpoint_every > 0 and (chunk_idx % checkpoint_every == 0 or chunk_idx == n_chunks):
            _flush_accumulators_to_disk(
                numerator_acc=numerator,
                overlap_acc=overlap,
                numerator_mm=numerator_mm,
                overlap_mm=overlap_mm,
                checkpoint_paths=checkpoint_paths,
                last_completed_chunk=chunk_idx,
                informative_snps=informative_snps,
            )
        if progress_every > 0 and (chunk_idx == 1 or chunk_idx % progress_every == 0 or chunk_idx == n_chunks):
            elapsed = time.time() - started
            _log(
                "graph progress "
                f"chunk={chunk_idx}/{n_chunks} snps={end}/{n_snps} "
                f"informative_snps={informative_snps} elapsed_s={elapsed:.1f}"
            )

    _flush_accumulators_to_disk(
        numerator_acc=numerator,
        overlap_acc=overlap,
        numerator_mm=numerator_mm,
        overlap_mm=overlap_mm,
        checkpoint_paths=checkpoint_paths,
        last_completed_chunk=n_chunks,
        informative_snps=informative_snps,
    )
    stats = {
        "n_samples": n_samples,
        "n_samples_total": int(total_samples),
        "n_snps": n_snps,
        "n_snps_total": int(total_snps),
        "informative_snps": informative_snps,
        "chunk_snps": int(chunk_snps),
        "compute_backend": selected_backend,
        "compute_device": selected_device,
        "checkpoint_every": int(checkpoint_every),
        "last_completed_chunk": int(n_chunks),
        "sample_indices_signature": sample_indices_signature,
    }
    return np.asarray(numerator_mm, dtype=np.float32), np.asarray(overlap_mm, dtype=np.uint32), stats


def similarity_from_components(
    numerator: np.ndarray,
    overlap: np.ndarray,
    *,
    overlap_shrinkage: float,
) -> np.ndarray:
    safe_overlap = np.maximum(overlap.astype(np.float32), 1.0)
    overlap_f = overlap.astype(np.float32)
    similarity = (numerator / safe_overlap) * (overlap_f / (overlap_f + float(overlap_shrinkage)))
    np.fill_diagonal(similarity, 0.0)
    return similarity.astype(np.float32)


def build_topk_symmetric_graph(
    similarity: np.ndarray,
    overlap: np.ndarray,
    *,
    min_overlap: int,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(similarity.shape[0])
    if similarity.shape != overlap.shape:
        raise ValueError("similarity and overlap must have the same shape")
    filtered = similarity.copy()
    filtered[overlap < int(min_overlap)] = -np.inf
    np.fill_diagonal(filtered, -np.inf)

    best_weights: Dict[Tuple[int, int], float] = {}
    best_overlap: Dict[Tuple[int, int], int] = {}
    for i in range(n_samples):
        row = filtered[i]
        valid_idx = np.flatnonzero(np.isfinite(row))
        if valid_idx.size == 0:
            continue
        keep = valid_idx
        if valid_idx.size > top_k:
            part = np.argpartition(row[valid_idx], -top_k)[-top_k:]
            keep = valid_idx[part]
        keep = keep[np.argsort(row[keep])[::-1]]
        for j in keep:
            key = (i, int(j)) if i < int(j) else (int(j), i)
            weight = float(similarity[i, j])
            if weight <= 0.0:
                continue
            if weight > best_weights.get(key, float("-inf")):
                best_weights[key] = weight
                best_overlap[key] = int(overlap[i, j])

    src: List[int] = []
    dst: List[int] = []
    weight: List[float] = []
    ov: List[int] = []
    for (i, j), w in sorted(best_weights.items()):
        src.extend([i, j])
        dst.extend([j, i])
        weight.extend([w, w])
        ov_value = best_overlap[(i, j)]
        ov.extend([ov_value, ov_value])
    return (
        np.asarray(src, dtype=np.int32),
        np.asarray(dst, dtype=np.int32),
        np.asarray(weight, dtype=np.float32),
        np.asarray(ov, dtype=np.int32),
    )


def _normalized_adjacency(src: np.ndarray, dst: np.ndarray, weight: np.ndarray, n_nodes: int) -> np.ndarray:
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    adj[src, dst] = weight.astype(np.float64)
    adj = np.maximum(adj, adj.T)
    degree = adj.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    norm_adj = inv_sqrt[:, None] * adj * inv_sqrt[None, :]
    return norm_adj


def spectral_initialization(
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    *,
    n_nodes: int,
    latent_dim: int,
) -> np.ndarray:
    if src.size == 0:
        return np.zeros((n_nodes, latent_dim), dtype=np.float32)

    norm_adj = _normalized_adjacency(src, dst, weight, n_nodes)
    if n_nodes <= 2048:
        eigvals, eigvecs = np.linalg.eigh(norm_adj)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        basis = eigvecs[:, :latent_dim]
    else:
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
        except ImportError as exc:
            try:
                import torch
            except ImportError:
                raise RuntimeError(
                    "Either scipy or torch is required for spectral initialization on large graphs"
                ) from exc
            device = "cuda" if torch.cuda.is_available() else "cpu"
            matrix = torch.tensor(norm_adj, dtype=torch.float32, device=device)
            _, vectors = torch.lobpcg(matrix, k=latent_dim, largest=True, niter=200)
            basis = vectors.detach().cpu().numpy()
        else:
            sparse_adj = sp.csr_matrix(norm_adj)
            _, basis = spla.eigsh(sparse_adj, k=latent_dim, which="LA")
    basis = basis.astype(np.float32)
    basis -= basis.mean(axis=0, keepdims=True)
    std = basis.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return basis / std


def build_overlap_graph(
    *,
    out_dir: Path,
    anno_path: Path,
    meta_json: Optional[Path] = None,
    geno_path: Optional[Path] = None,
    ind_path: Optional[Path] = None,
    ploidy: str = "pseudohaploid",
    chunk_snps: int = 4096,
    min_overlap: int = 50_000,
    top_k: int = 32,
    overlap_shrinkage: float = 100_000.0,
    latent_dim: int = 6,
    max_samples: int = -1,
    max_snps: int = -1,
    min_observed_fraction: float = 0.0,
    progress_every: int = 25,
    reuse_existing: bool = False,
    checkpoint_every: int = 5,
    compute_backend: str = "auto",
    compute_device: str = "auto",
    resume_existing: bool = False,
) -> GraphBuildArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_npz_path = out_dir / "graph_edges.npz"
    sample_stats_tsv_path = out_dir / "sample_stats.tsv"
    spectral_init_path = out_dir / "spectral_init.npy"
    build_summary_path = out_dir / "build_summary.json"
    meta, meta_json_path = prepare_meta(
        meta_json=meta_json,
        geno_path=geno_path,
        ind_path=ind_path,
        out_dir=out_dir,
        ploidy=ploidy,
    )
    if reuse_existing and all(path.exists() for path in (graph_npz_path, sample_stats_tsv_path, spectral_init_path, build_summary_path)):
        try:
            summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        config_matches = (
            int(summary.get("latent_dim", -999)) == int(latent_dim)
            and int(summary.get("min_overlap", -999)) == int(min_overlap)
            and int(summary.get("top_k", -999)) == int(top_k)
            and int(summary.get("max_samples", -999)) == int(max_samples)
            and int(summary.get("max_snps", -999)) == int(max_snps)
            and float(summary.get("min_observed_fraction", -999.0)) == float(min_observed_fraction)
            and str(summary.get("ploidy", "")) == str(ploidy)
            and int(summary.get("chunk_snps", -999)) == int(chunk_snps)
            and int(summary.get("checkpoint_every", -999)) == int(checkpoint_every)
            and (
                compute_backend == "auto"
                or str(summary.get("compute_backend", "")) == str(compute_backend)
            )
        )
        if config_matches:
            _log(f"Reusing existing graph artifacts in {out_dir}")
            return GraphBuildArtifacts(
                graph_npz_path=graph_npz_path,
                sample_stats_tsv_path=sample_stats_tsv_path,
                spectral_init_path=spectral_init_path,
                build_summary_path=build_summary_path,
                meta_json_path=meta_json_path,
            )
        _log(f"Existing graph artifacts found in {out_dir} but config mismatch; rebuilding.")
    geno_mm = open_genotype_memmap(meta)
    all_sample_ids = load_sample_ids_from_meta(meta)
    candidate_indices = np.arange(meta.n_samples, dtype=np.int64)
    if max_samples > 0:
        candidate_indices = candidate_indices[:max_samples]
    snp_limit = meta.n_snps if max_snps <= 0 else min(int(max_snps), int(meta.n_snps))
    observed_fraction_candidates = compute_observed_fraction(
        geno_mm,
        sample_indices=candidate_indices,
        max_snps=snp_limit,
    )
    sample_ids = [all_sample_ids[int(i)] for i in candidate_indices]
    if min_observed_fraction > 0.0:
        keep_mask = observed_fraction_candidates >= float(min_observed_fraction)
        kept_count = int(keep_mask.sum())
        dropped_count = int(keep_mask.shape[0] - kept_count)
        selected_indices = candidate_indices[keep_mask]
        sample_ids = [sample_ids[idx] for idx, keep in enumerate(keep_mask.tolist()) if keep]
        observed_fraction = observed_fraction_candidates[keep_mask]
        _log(
            f"Observed-fraction filter kept {kept_count}/{candidate_indices.shape[0]} samples "
            f"with min_observed_fraction={min_observed_fraction:.4f} dropped={dropped_count}"
        )
    else:
        selected_indices = candidate_indices
        observed_fraction = observed_fraction_candidates
    if selected_indices.shape[0] < 2:
        raise ValueError(
            "Observed-fraction filtering left fewer than 2 samples; "
            "lower min_observed_fraction or increase the sample pool."
        )
    _log(
        f"Starting graph build with n_samples={selected_indices.shape[0]} "
        f"n_snps={snp_limit} ploidy={ploidy} "
        f"chunk_snps={chunk_snps} top_k={top_k} min_overlap={min_overlap} "
        f"compute_backend={compute_backend} compute_device={compute_device}"
    )
    sample_stats_rows = build_sample_stats_rows(sample_ids, observed_fraction, anno_path)
    _log("Sample stats built")

    numerator, overlap, stats = compute_overlap_numerator_and_counts(
        geno_mm,
        out_dir=out_dir,
        meta_json_path=meta_json_path,
        latent_dim=latent_dim,
        min_overlap=min_overlap,
        top_k=top_k,
        overlap_shrinkage=overlap_shrinkage,
        ploidy=ploidy,
        max_samples=max_samples,
        sample_indices=selected_indices,
        chunk_snps=chunk_snps,
        max_snps=snp_limit,
        progress_every=progress_every,
        checkpoint_every=checkpoint_every,
        compute_backend=compute_backend,
        compute_device=compute_device,
        resume_existing=resume_existing,
    )
    _log("Overlap numerator and counts finished")
    similarity = similarity_from_components(
        numerator,
        overlap,
        overlap_shrinkage=overlap_shrinkage,
    )
    _log("Similarity matrix finished")
    src, dst, weight, ov = build_topk_symmetric_graph(
        similarity,
        overlap,
        min_overlap=min_overlap,
        top_k=top_k,
    )
    _log(
        f"Top-k graph finished with directed_edges={weight.size} "
        f"undirected_edges={weight.size // 2}"
    )
    spectral_init = spectral_initialization(
        src,
        dst,
        weight,
        n_nodes=selected_indices.shape[0],
        latent_dim=latent_dim,
    )
    _log("Spectral initialization finished")

    np.savez_compressed(graph_npz_path, src=src, dst=dst, weight=weight, overlap=ov)
    np.save(spectral_init_path, spectral_init)
    write_tsv(
        sample_stats_tsv_path,
        sample_stats_rows,
        fieldnames=["sample_id", "observed_fraction", "country", "original_group_id", "date_mean_bp"],
    )

    density = float(weight.size) / float(max(selected_indices.shape[0] * max(selected_indices.shape[0] - 1, 1), 1))
    summary = {
        **stats,
        "ploidy": meta.ploidy,
        "latent_dim": int(latent_dim),
        "min_overlap": int(min_overlap),
        "top_k": int(top_k),
        "max_samples": int(max_samples),
        "max_snps": int(max_snps),
        "min_observed_fraction": float(min_observed_fraction),
        "overlap_shrinkage": float(overlap_shrinkage),
        "checkpoint_every": int(checkpoint_every),
        "n_edges_directed": int(weight.size),
        "n_edges_undirected": int(weight.size // 2),
        "graph_density": density,
        "n_samples_selected": int(selected_indices.shape[0]),
        "n_samples_dropped_low_observed_fraction": int(candidate_indices.shape[0] - selected_indices.shape[0]),
        "observed_fraction_mean": float(observed_fraction.mean()),
        "observed_fraction_std": float(observed_fraction.std()),
    }
    with build_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    _log(f"Graph build artifacts written to {out_dir}")

    return GraphBuildArtifacts(
        graph_npz_path=graph_npz_path,
        sample_stats_tsv_path=sample_stats_tsv_path,
        spectral_init_path=spectral_init_path,
        build_summary_path=build_summary_path,
        meta_json_path=meta_json_path,
    )
