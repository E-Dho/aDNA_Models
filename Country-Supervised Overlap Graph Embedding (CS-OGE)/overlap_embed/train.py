from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import load_sample_stats_tsv
from .eval import evaluate_embedding, export_evaluation_bundle, nearest_neighbors, ridge_probe_r2, spearman_corr
from .model import (
    NodeEmbeddingModelFactory,
    covariance_regularizers,
    graph_similarity_loss,
    negative_edge_loss,
    nonedge_margin_violation_rate,
    positive_edge_loss,
    require_torch,
    snapshot_state,
)


@dataclass(frozen=True)
class TrainConfig:
    objective_mode: str = "label_supervised"
    latent_dim: int = 6
    epochs: int = 400
    batch_size: int = 4096
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    negatives_per_anchor: int = 16
    lambda_cov: float = 0.5
    lambda_var: float = 1.0
    lambda_covariance: float = 0.1
    lambda_graph_pos: float = 1.0
    lambda_graph_neg: float = 1.0
    lambda_graph_similarity: float = 1.0
    graph_similarity_temperature: float = 1.0
    lambda_country: float = 0.0
    lambda_original_group: float = 0.0
    margin: float = 1.0
    country_margin: float = 1.5
    country_negatives_per_anchor: int = 8
    original_group_negatives_per_anchor: int = 4
    original_group_temperature: float = 1.0
    val_ratio: float = 0.1
    patience: int = 40
    seed: int = 42
    wandb: bool = True
    wandb_mode: str = "offline"
    wandb_project: str = "adna-overlap-embed"
    wandb_entity: str = ""
    wandb_group: str = "europe_drop_premerge_gt90"
    wandb_name: str = "overlap_embed"
    wandb_tags: str = "overlap-embed,europe"
    wandb_log_every: int = 10
    wandb_init_timeout: int = 30
    wandb_watch: int = 0
    wandb_online_fallback: bool = True


class SafeWandbLogger:
    def __init__(self, config: TrainConfig, output_dir: Path) -> None:
        self._enabled = bool(config.wandb)
        self._run = None
        self._mode = config.wandb_mode
        self._note = ""
        if not self._enabled or self._mode == "disabled":
            self._enabled = False
            return
        try:
            import wandb
        except ImportError:
            self._enabled = False
            self._note = "wandb import unavailable; logging disabled"
            return
        settings = wandb.Settings(
            init_timeout=max(int(config.wandb_init_timeout), 1),
            start_method="thread",
            _disable_stats=True,
        )
        kwargs = {
            "project": config.wandb_project,
            "name": config.wandb_name,
            "group": config.wandb_group or None,
            "entity": config.wandb_entity or None,
            "mode": self._mode,
            "tags": [tag.strip() for tag in config.wandb_tags.split(",") if tag.strip()],
            "dir": str(output_dir),
            "settings": settings,
            "config": {
                "latent_dim": config.latent_dim,
                "objective_mode": config.objective_mode,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "negatives_per_anchor": config.negatives_per_anchor,
                "lambda_cov": config.lambda_cov,
                "lambda_var": config.lambda_var,
                "lambda_covariance": config.lambda_covariance,
                "lambda_graph_pos": config.lambda_graph_pos,
                "lambda_graph_neg": config.lambda_graph_neg,
                "lambda_graph_similarity": config.lambda_graph_similarity,
                "graph_similarity_temperature": config.graph_similarity_temperature,
                "lambda_country": config.lambda_country,
                "lambda_original_group": config.lambda_original_group,
                "country_margin": config.country_margin,
                "country_negatives_per_anchor": config.country_negatives_per_anchor,
                "original_group_negatives_per_anchor": config.original_group_negatives_per_anchor,
                "original_group_temperature": config.original_group_temperature,
            },
        }
        try:
            self._run = wandb.init(**kwargs)
        except Exception as exc:
            if self._mode == "online" and config.wandb_online_fallback:
                self._note = f"wandb online init failed, fell back to offline: {exc}"
                kwargs["mode"] = "offline"
                try:
                    self._run = wandb.init(**kwargs)
                    self._mode = "offline"
                except Exception as fallback_exc:
                    self._enabled = False
                    self._note = f"wandb fallback init failed: {fallback_exc}"
            else:
                self._enabled = False
                self._note = f"wandb init failed: {exc}"

    @property
    def note(self) -> str:
        return self._note

    @property
    def mode(self) -> str:
        return self._mode

    def log(self, payload: Dict[str, float]) -> None:
        if self._enabled and self._run is not None:
            self._run.log(payload)

    def summary_update(self, payload: Dict[str, float]) -> None:
        if self._enabled and self._run is not None:
            for key, value in payload.items():
                self._run.summary[key] = value

    def finish(self) -> None:
        if self._enabled and self._run is not None:
            self._run.finish()


class CoverageMatchedNegativeSampler:
    def __init__(self, coverage: np.ndarray, src: np.ndarray, dst: np.ndarray, seed: int = 42) -> None:
        self.coverage = np.asarray(coverage, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.n_nodes = int(self.coverage.shape[0])
        self.sorted_nodes = np.argsort(self.coverage)
        self.rank = np.empty(self.n_nodes, dtype=np.int32)
        self.rank[self.sorted_nodes] = np.arange(self.n_nodes, dtype=np.int32)
        self.neighbors: List[set[int]] = [set() for _ in range(self.n_nodes)]
        for i, j in zip(src.tolist(), dst.tolist()):
            self.neighbors[int(i)].add(int(j))

    def sample(self, anchors: np.ndarray, negatives_per_anchor: int, window: int = 128) -> np.ndarray:
        out = np.empty(len(anchors) * negatives_per_anchor, dtype=np.int64)
        cursor = 0
        for anchor in anchors.tolist():
            rank = int(self.rank[int(anchor)])
            lo = max(0, rank - window)
            hi = min(self.n_nodes, rank + window + 1)
            pool = self.sorted_nodes[lo:hi].tolist()
            pool = [node for node in pool if node != anchor and node not in self.neighbors[int(anchor)]]
            if not pool:
                pool = [node for node in range(self.n_nodes) if node != anchor]
            if len(pool) < negatives_per_anchor:
                extra = self.rng.permutation(self.n_nodes).tolist()
                for node in extra:
                    if node == anchor or node in self.neighbors[int(anchor)]:
                        continue
                    pool.append(int(node))
                    if len(pool) >= negatives_per_anchor:
                        break
            chosen = self.rng.choice(pool, size=negatives_per_anchor, replace=len(pool) < negatives_per_anchor)
            out[cursor : cursor + negatives_per_anchor] = chosen.astype(np.int64)
            cursor += negatives_per_anchor
        return out


class CountryConstraintSampler:
    def __init__(self, labels: Sequence[str], coverage: np.ndarray, seed: int = 42) -> None:
        self.labels = np.asarray([str(label).strip() for label in labels], dtype=object)
        self.coverage = np.asarray(coverage, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.n_nodes = int(self.labels.shape[0])
        self.sorted_nodes = np.argsort(self.coverage)
        self.rank = np.empty(self.n_nodes, dtype=np.int32)
        self.rank[self.sorted_nodes] = np.arange(self.n_nodes, dtype=np.int32)
        self.members_by_label: Dict[str, np.ndarray] = {}
        for label in sorted({label for label in self.labels.tolist() if label}):
            members = np.flatnonzero(self.labels == label).astype(np.int64)
            if members.size >= 2:
                self.members_by_label[label] = members

    def sample(
        self,
        anchors: np.ndarray,
        negatives_per_anchor: int,
        *,
        window: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos_src: List[int] = []
        pos_dst: List[int] = []
        neg_src: List[int] = []
        neg_dst: List[int] = []
        for anchor in np.unique(np.asarray(anchors, dtype=np.int64)):
            label = str(self.labels[int(anchor)])
            members = self.members_by_label.get(label)
            if members is None:
                continue
            pos_pool = members[members != int(anchor)]
            if pos_pool.size == 0:
                continue
            pos_src.append(int(anchor))
            pos_dst.append(int(self.rng.choice(pos_pool)))

            rank = int(self.rank[int(anchor)])
            lo = max(0, rank - window)
            hi = min(self.n_nodes, rank + window + 1)
            local_pool = self.sorted_nodes[lo:hi]
            local_pool = local_pool[self.labels[local_pool] != label]
            local_pool = local_pool[self.labels[local_pool] != ""]
            if local_pool.size == 0:
                local_pool = np.flatnonzero((self.labels != label) & (self.labels != "")).astype(np.int64)
            if local_pool.size == 0:
                continue
            chosen = self.rng.choice(
                local_pool,
                size=int(negatives_per_anchor),
                replace=local_pool.size < int(negatives_per_anchor),
            )
            neg_src.extend([int(anchor)] * int(negatives_per_anchor))
            neg_dst.extend(chosen.astype(np.int64).tolist())
        return (
            np.asarray(pos_src, dtype=np.int64),
            np.asarray(pos_dst, dtype=np.int64),
            np.asarray(neg_src, dtype=np.int64),
            np.asarray(neg_dst, dtype=np.int64),
        )


class OriginalGroupConstraintSampler:
    def __init__(
        self,
        original_groups: Sequence[str],
        coverage: np.ndarray,
        seed: int = 42,
    ) -> None:
        self.original_groups = np.asarray([str(label).strip() for label in original_groups], dtype=object)
        self.coverage = np.asarray(coverage, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.n_nodes = int(self.original_groups.shape[0])
        self.sorted_nodes = np.argsort(self.coverage)
        self.rank = np.empty(self.n_nodes, dtype=np.int32)
        self.rank[self.sorted_nodes] = np.arange(self.n_nodes, dtype=np.int32)
        valid = self.original_groups != ""
        self.members_by_group: Dict[str, np.ndarray] = {}
        for group in sorted({str(label) for label in self.original_groups[valid].tolist() if label}):
            members = np.flatnonzero(valid & (self.original_groups == group)).astype(np.int64)
            if members.size >= 2:
                self.members_by_group[group] = members

    def sample(
        self,
        anchors: np.ndarray,
        negatives_per_anchor: int,
        *,
        window: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos_src: List[int] = []
        pos_dst: List[int] = []
        neg_src: List[int] = []
        neg_dst: List[int] = []
        for anchor in np.unique(np.asarray(anchors, dtype=np.int64)):
            anchor_i = int(anchor)
            group = str(self.original_groups[anchor_i])
            if not group:
                continue
            same_group = self.members_by_group.get(group)
            if same_group is None:
                continue
            pos_pool = same_group[same_group != anchor_i]
            if pos_pool.size == 0:
                continue

            rank = int(self.rank[anchor_i])
            lo = max(0, rank - window)
            hi = min(self.n_nodes, rank + window + 1)
            local_pool = self.sorted_nodes[lo:hi]
            local_pool = local_pool[
                (self.original_groups[local_pool] != group)
                & (self.original_groups[local_pool] != "")
            ]
            if local_pool.size == 0:
                local_pool = np.flatnonzero(
                    (self.original_groups != group)
                    & (self.original_groups != "")
                ).astype(np.int64)
            if local_pool.size == 0:
                continue
            pos_src.append(anchor_i)
            pos_dst.append(int(self.rng.choice(pos_pool)))
            chosen = self.rng.choice(
                local_pool,
                size=int(negatives_per_anchor),
                replace=local_pool.size < int(negatives_per_anchor),
            )
            neg_src.extend([anchor_i] * int(negatives_per_anchor))
            neg_dst.extend(chosen.astype(np.int64).tolist())
        return (
            np.asarray(pos_src, dtype=np.int64),
            np.asarray(pos_dst, dtype=np.int64),
            np.asarray(neg_src, dtype=np.int64),
            np.asarray(neg_dst, dtype=np.int64),
        )


def original_group_constraint_loss(
    latents,
    pos_src,
    pos_dst,
    neg_src,
    neg_dst,
    temperature: float,
):
    same_group_loss = latents.new_zeros(())
    diff_group_loss = latents.new_zeros(())
    if pos_src.numel() > 0:
        diff = latents[pos_src] - latents[pos_dst]
        same_group_loss = diff.pow(2).sum(dim=1).mean()
    if neg_src.numel() > 0:
        temp = max(float(temperature), 1e-6)
        diff = latents[neg_src] - latents[neg_dst]
        dist2 = diff.pow(2).sum(dim=1)
        diff_group_loss = (-dist2 / temp).exp().mean()
    return same_group_loss + diff_group_loss


def _split_edges(n_edges: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_edges)
    n_val = max(int(n_edges * val_ratio), 1)
    return order[n_val:], order[:n_val]


def _load_graph(graph_npz: Path) -> Dict[str, np.ndarray]:
    data = np.load(graph_npz)
    return {
        "src": data["src"].astype(np.int64),
        "dst": data["dst"].astype(np.int64),
        "weight": data["weight"].astype(np.float32),
        "overlap": data["overlap"].astype(np.int64),
    }


def normalize_graph_weights(weights: np.ndarray, reference_idx: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    ref = weights[np.asarray(reference_idx, dtype=np.int64)]
    finite = np.isfinite(ref)
    if finite.sum() == 0:
        return np.ones_like(weights, dtype=np.float32)
    lo = float(np.min(ref[finite]))
    hi = float(np.max(ref[finite]))
    if hi - lo < 1e-8:
        return np.ones_like(weights, dtype=np.float32)
    return np.clip((weights - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def graph_knn_recall(latents: np.ndarray, src: np.ndarray, dst: np.ndarray, *, k: int = 15) -> float:
    if latents.shape[0] < 2:
        return float("nan")
    neighbors = nearest_neighbors(latents, k)
    graph_neighbors: List[set[int]] = [set() for _ in range(latents.shape[0])]
    for i, j in zip(src.tolist(), dst.tolist()):
        graph_neighbors[int(i)].add(int(j))
    recalls: List[float] = []
    for i in range(latents.shape[0]):
        denom = min(int(k), len(graph_neighbors[i]))
        if denom <= 0:
            continue
        hits = sum(1 for node in neighbors[i].tolist() if int(node) in graph_neighbors[i])
        recalls.append(float(hits / denom))
    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def weighted_graph_edge_spearman(
    latents: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    *,
    temperature: float,
) -> float:
    if src.size < 2:
        return float("nan")
    diff = latents[src] - latents[dst]
    dist2 = (diff**2).sum(axis=1)
    latent_similarity = np.exp(-dist2 / max(float(temperature), 1e-6))
    return spearman_corr(np.asarray(weight, dtype=np.float64), latent_similarity)


def _batch_slices(n: int, batch_size: int) -> Sequence[Tuple[int, int]]:
    return [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]


def train_overlap_embedding(
    *,
    graph_npz: Path,
    spectral_init: Path,
    sample_stats_tsv: Path,
    output_dir: Path,
    config: TrainConfig,
) -> Dict[str, object]:
    torch, _ = require_torch()
    if config.objective_mode not in {"label_supervised", "snp_similarity"}:
        raise ValueError(
            "objective_mode must be one of {'label_supervised', 'snp_similarity'}, "
            f"got {config.objective_mode!r}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = _load_graph(graph_npz)
    sample_stats = load_sample_stats_tsv(sample_stats_tsv)
    sample_ids = [str(row["sample_id"]) for row in sample_stats]
    coverage = np.asarray([float(row["observed_fraction"]) for row in sample_stats], dtype=np.float32)
    country_labels = [str(row["country"]) for row in sample_stats]
    original_group_labels = [str(row["original_group_id"]) for row in sample_stats]
    init_latents = np.load(spectral_init).astype(np.float32)
    n_nodes = init_latents.shape[0]

    if config.latent_dim != init_latents.shape[1]:
        raise ValueError(
            f"latent_dim={config.latent_dim} does not match spectral_init width={init_latents.shape[1]}"
        )

    model = NodeEmbeddingModelFactory.build(n_nodes, config.latent_dim, init_latents)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    coverage_t = torch.tensor(coverage, dtype=torch.float32)
    sampler = CoverageMatchedNegativeSampler(coverage, graph["src"], graph["dst"], seed=config.seed)
    use_label_losses = config.objective_mode == "label_supervised"
    effective_lambda_country = float(config.lambda_country) if use_label_losses else 0.0
    effective_lambda_original_group = float(config.lambda_original_group) if use_label_losses else 0.0
    country_sampler = (
        CountryConstraintSampler(country_labels, coverage, seed=config.seed)
        if effective_lambda_country > 0.0
        else None
    )
    original_group_sampler = (
        OriginalGroupConstraintSampler(
            original_group_labels,
            coverage,
            seed=config.seed,
        )
        if effective_lambda_original_group > 0.0
        else None
    )
    train_idx, val_idx = _split_edges(graph["src"].shape[0], config.val_ratio, config.seed)
    graph_similarity_target = normalize_graph_weights(graph["weight"], train_idx)
    logger = SafeWandbLogger(config, output_dir)
    metrics_path = output_dir / "metrics.jsonl"
    best_model_path = output_dir / "best_model.pt"

    best_val = math.inf
    best_epoch = -1
    epochs_without_improve = 0
    last_epoch = 0

    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for epoch in range(1, config.epochs + 1):
            last_epoch = epoch
            epoch_start = time.time()
            order = np.random.default_rng(config.seed + epoch).permutation(train_idx)
            train_losses: List[float] = []
            train_pos_losses: List[float] = []
            train_neg_losses: List[float] = []
            train_graph_similarity_losses: List[float] = []
            train_country_losses: List[float] = []
            train_original_group_losses: List[float] = []

            for batch_lo, batch_hi in _batch_slices(order.shape[0], config.batch_size):
                batch_ids = order[batch_lo:batch_hi]
                src = graph["src"][batch_ids]
                dst = graph["dst"][batch_ids]
                weight = graph["weight"][batch_ids]
                graph_target = graph_similarity_target[batch_ids]
                neg_dst = sampler.sample(src, config.negatives_per_anchor)
                neg_src = np.repeat(src, config.negatives_per_anchor)

                optimizer.zero_grad(set_to_none=True)
                latents = model()
                src_t = torch.tensor(src, dtype=torch.long)
                dst_t = torch.tensor(dst, dtype=torch.long)
                weight_t = torch.tensor(weight, dtype=torch.float32)
                graph_target_t = torch.tensor(graph_target, dtype=torch.float32)
                neg_src_t = torch.tensor(neg_src, dtype=torch.long)
                neg_dst_t = torch.tensor(neg_dst, dtype=torch.long)
                pos_loss = positive_edge_loss(latents, src_t, dst_t, weight_t)
                graph_sim_loss = graph_similarity_loss(
                    latents,
                    src_t,
                    dst_t,
                    graph_target_t,
                    config.graph_similarity_temperature,
                )
                neg_loss = negative_edge_loss(latents, neg_src_t, neg_dst_t, config.margin)
                country_loss = torch.zeros((), dtype=torch.float32)
                if country_sampler is not None:
                    country_pos_src, country_pos_dst, country_neg_src, country_neg_dst = country_sampler.sample(
                        src,
                        config.country_negatives_per_anchor,
                    )
                    if country_pos_src.size > 0:
                        country_weight = torch.ones(country_pos_src.shape[0], dtype=torch.float32)
                        country_loss = country_loss + positive_edge_loss(
                            latents,
                            torch.tensor(country_pos_src, dtype=torch.long),
                            torch.tensor(country_pos_dst, dtype=torch.long),
                            country_weight,
                        )
                    if country_neg_src.size > 0:
                        country_loss = country_loss + negative_edge_loss(
                            latents,
                            torch.tensor(country_neg_src, dtype=torch.long),
                            torch.tensor(country_neg_dst, dtype=torch.long),
                            config.country_margin,
                        )
                original_group_loss = latents.new_zeros(())
                if original_group_sampler is not None:
                    og_pos_src, og_pos_dst, og_neg_src, og_neg_dst = original_group_sampler.sample(
                        src,
                        config.original_group_negatives_per_anchor,
                    )
                    original_group_loss = original_group_constraint_loss(
                        latents,
                        torch.tensor(og_pos_src, dtype=torch.long),
                        torch.tensor(og_pos_dst, dtype=torch.long),
                        torch.tensor(og_neg_src, dtype=torch.long),
                        torch.tensor(og_neg_dst, dtype=torch.long),
                        config.original_group_temperature,
                    )
                regs = covariance_regularizers(latents, coverage_t)
                total_loss = (
                    float(config.lambda_graph_pos) * pos_loss
                    + float(config.lambda_graph_neg) * neg_loss
                    + (
                        float(config.lambda_graph_similarity) * graph_sim_loss
                        if config.objective_mode == "snp_similarity"
                        else latents.new_zeros(())
                    )
                    + effective_lambda_country * country_loss
                    + effective_lambda_original_group * original_group_loss
                    + float(config.lambda_var) * regs["var_penalty"]
                    + float(config.lambda_covariance) * regs["cov_penalty"]
                    + float(config.lambda_cov) * regs["coverage_penalty"]
                )
                total_loss.backward()
                optimizer.step()
                train_losses.append(float(total_loss.detach().cpu()))
                train_pos_losses.append(float(pos_loss.detach().cpu()))
                train_neg_losses.append(float(neg_loss.detach().cpu()))
                train_graph_similarity_losses.append(float(graph_sim_loss.detach().cpu()))
                train_country_losses.append(float(country_loss.detach().cpu()))
                train_original_group_losses.append(float(original_group_loss.detach().cpu()))

            with torch.no_grad():
                latents = model()
                val_src = torch.tensor(graph["src"][val_idx], dtype=torch.long)
                val_dst = torch.tensor(graph["dst"][val_idx], dtype=torch.long)
                val_weight = torch.tensor(graph["weight"][val_idx], dtype=torch.float32)
                val_graph_target = torch.tensor(graph_similarity_target[val_idx], dtype=torch.float32)
                val_pos = positive_edge_loss(latents, val_src, val_dst, val_weight)
                val_graph_sim = graph_similarity_loss(
                    latents,
                    val_src,
                    val_dst,
                    val_graph_target,
                    config.graph_similarity_temperature,
                )
                neg_dst = sampler.sample(graph["src"][val_idx], config.negatives_per_anchor)
                neg_src = np.repeat(graph["src"][val_idx], config.negatives_per_anchor)
                val_neg = negative_edge_loss(
                    latents,
                    torch.tensor(neg_src, dtype=torch.long),
                    torch.tensor(neg_dst, dtype=torch.long),
                    config.margin,
                )
                val_country = torch.zeros((), dtype=torch.float32)
                if country_sampler is not None:
                    country_pos_src, country_pos_dst, country_neg_src, country_neg_dst = country_sampler.sample(
                        graph["src"][val_idx],
                        config.country_negatives_per_anchor,
                    )
                    if country_pos_src.size > 0:
                        val_country = val_country + positive_edge_loss(
                            latents,
                            torch.tensor(country_pos_src, dtype=torch.long),
                            torch.tensor(country_pos_dst, dtype=torch.long),
                            torch.ones(country_pos_src.shape[0], dtype=torch.float32),
                        )
                    if country_neg_src.size > 0:
                        val_country = val_country + negative_edge_loss(
                            latents,
                            torch.tensor(country_neg_src, dtype=torch.long),
                            torch.tensor(country_neg_dst, dtype=torch.long),
                            config.country_margin,
                        )
                val_original_group = latents.new_zeros(())
                if original_group_sampler is not None:
                    og_pos_src, og_pos_dst, og_neg_src, og_neg_dst = original_group_sampler.sample(
                        graph["src"][val_idx],
                        config.original_group_negatives_per_anchor,
                    )
                    val_original_group = original_group_constraint_loss(
                        latents,
                        torch.tensor(og_pos_src, dtype=torch.long),
                        torch.tensor(og_pos_dst, dtype=torch.long),
                        torch.tensor(og_neg_src, dtype=torch.long),
                        torch.tensor(og_neg_dst, dtype=torch.long),
                        config.original_group_temperature,
                    )
                regs = covariance_regularizers(latents, coverage_t)
                val_total = (
                    float(config.lambda_graph_pos) * val_pos
                    + float(config.lambda_graph_neg) * val_neg
                    + (
                        float(config.lambda_graph_similarity) * val_graph_sim
                        if config.objective_mode == "snp_similarity"
                        else latents.new_zeros(())
                    )
                    + effective_lambda_country * val_country
                    + effective_lambda_original_group * val_original_group
                    + float(config.lambda_var) * regs["var_penalty"]
                    + float(config.lambda_covariance) * regs["cov_penalty"]
                    + float(config.lambda_cov) * regs["coverage_penalty"]
                )
                latents_np = latents.detach().cpu().numpy()
                coverage_r2 = ridge_probe_r2(latents_np, coverage)
                periodic_eval = {}
                if epoch == 1 or epoch % max(config.wandb_log_every, 1) == 0:
                    periodic_eval, _, _, _ = evaluate_embedding(latents_np, sample_stats_tsv)
                    periodic_eval["graph_knn_recall@15"] = graph_knn_recall(
                        latents_np,
                        graph["src"],
                        graph["dst"],
                        k=15,
                    )
                    periodic_eval["weighted_graph_edge_spearman"] = weighted_graph_edge_spearman(
                        latents_np,
                        graph["src"][val_idx],
                        graph["dst"][val_idx],
                        graph["weight"][val_idx],
                        temperature=config.graph_similarity_temperature,
                    )
                    periodic_eval["nonedge_margin_violation_rate"] = nonedge_margin_violation_rate(
                        latents,
                        torch.tensor(neg_src, dtype=torch.long),
                        torch.tensor(neg_dst, dtype=torch.long),
                        config.margin,
                    )
                log_payload = {
                    "epoch": epoch,
                    "objective_mode_snp_similarity": 1.0 if config.objective_mode == "snp_similarity" else 0.0,
                    "train_loss": float(np.mean(train_losses)),
                    "train_positive_loss": float(np.mean(train_pos_losses)),
                    "train_negative_loss": float(np.mean(train_neg_losses)),
                    "train_graph_similarity_loss": float(np.mean(train_graph_similarity_losses)),
                    "train_country_loss": float(np.mean(train_country_losses)) if train_country_losses else 0.0,
                    "train_original_group_loss": (
                        float(np.mean(train_original_group_losses)) if train_original_group_losses else 0.0
                    ),
                    "val_graph_loss": float(val_total.detach().cpu()),
                    "val_positive_loss": float(val_pos.detach().cpu()),
                    "val_negative_loss": float(val_neg.detach().cpu()),
                    "val_graph_similarity_loss": float(val_graph_sim.detach().cpu()),
                    "val_country_loss": float(val_country.detach().cpu()),
                    "val_original_group_loss": float(val_original_group.detach().cpu()),
                    "lambda_graph_pos": float(config.lambda_graph_pos),
                    "lambda_graph_neg": float(config.lambda_graph_neg),
                    "lambda_graph_similarity": float(config.lambda_graph_similarity),
                    "graph_similarity_temperature": float(config.graph_similarity_temperature),
                    "effective_lambda_country": float(effective_lambda_country),
                    "effective_lambda_original_group": float(effective_lambda_original_group),
                    "lambda_original_group": float(config.lambda_original_group),
                    "original_group_temperature": float(config.original_group_temperature),
                    "latent_var_mean": float(regs["var_mean"].detach().cpu()),
                    "latent_cov_penalty": float(regs["cov_penalty"].detach().cpu()),
                    "coverage_penalty": float(regs["coverage_penalty"].detach().cpu()),
                    "coverage_r2_probe": float(coverage_r2),
                    "best_epoch": best_epoch if best_epoch >= 0 else epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "epoch_seconds": float(time.time() - epoch_start),
                }
                if periodic_eval:
                    log_payload["country_knn_purity@15"] = float(periodic_eval["country_macro_knn_purity@15"])
                    log_payload["date_neighbor_spearman"] = float(periodic_eval["date_neighbor_spearman"])
                    log_payload["graph_knn_recall@15"] = float(periodic_eval["graph_knn_recall@15"])
                    log_payload["weighted_graph_edge_spearman"] = float(periodic_eval["weighted_graph_edge_spearman"])
                    log_payload["nonedge_margin_violation_rate"] = float(periodic_eval["nonedge_margin_violation_rate"])

            metrics_handle.write(json.dumps(log_payload) + "\n")
            metrics_handle.flush()
            print(
                "epoch "
                f"{epoch}/{config.epochs} "
                f"train_loss={log_payload['train_loss']:.6f} "
                f"val_graph_loss={log_payload['val_graph_loss']:.6f} "
                f"coverage_r2={log_payload['coverage_r2_probe']:.6f}",
                flush=True,
            )

            if epoch == 1 or epoch % max(config.wandb_log_every, 1) == 0:
                logger.log(log_payload)

            current_val = log_payload["val_graph_loss"]
            if current_val < best_val:
                best_val = current_val
                best_epoch = epoch
                epochs_without_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config.__dict__,
                    },
                    best_model_path,
                )
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= config.patience:
                break

    checkpoint = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    latents_np = model().detach().cpu().numpy()
    np.save(output_dir / "latents.npy", latents_np)
    evaluation_metrics = export_evaluation_bundle(
        latents=latents_np,
        sample_ids=sample_ids,
        sample_stats_tsv=sample_stats_tsv,
        output_dir=output_dir,
    )
    evaluation_metrics["graph_knn_recall@15"] = graph_knn_recall(
        latents_np,
        graph["src"],
        graph["dst"],
        k=15,
    )
    evaluation_metrics["weighted_graph_edge_spearman"] = weighted_graph_edge_spearman(
        latents_np,
        graph["src"][val_idx],
        graph["dst"][val_idx],
        graph["weight"][val_idx],
        temperature=config.graph_similarity_temperature,
    )
    final_neg_dst = sampler.sample(graph["src"][val_idx], config.negatives_per_anchor)
    final_neg_src = np.repeat(graph["src"][val_idx], config.negatives_per_anchor)
    evaluation_metrics["nonedge_margin_violation_rate"] = nonedge_margin_violation_rate(
        torch.tensor(latents_np, dtype=torch.float32),
        torch.tensor(final_neg_src, dtype=torch.long),
        torch.tensor(final_neg_dst, dtype=torch.long),
        config.margin,
    )
    np.save(output_dir / "spectral_init.npy", init_latents)
    baseline_output_dir = output_dir / "spectral_init_eval"
    baseline_metrics = export_evaluation_bundle(
        latents=init_latents,
        sample_ids=sample_ids,
        sample_stats_tsv=sample_stats_tsv,
        output_dir=baseline_output_dir,
        prefix="spectral_init",
    )
    summary = {
        "graph_npz": str(graph_npz),
        "spectral_init": str(spectral_init),
        "sample_stats_tsv": str(sample_stats_tsv),
        "output_dir": str(output_dir),
        "config": config.__dict__,
        "wandb_mode_requested": config.wandb_mode,
        "wandb_mode_effective": logger.mode,
        "wandb_note": logger.note,
        "training": snapshot_state(
            best_epoch=best_epoch,
            best_val_loss=best_val,
            n_epochs_completed=last_epoch,
        ),
        "evaluation": evaluation_metrics,
        "spectral_init_evaluation": baseline_metrics,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.summary_update({**evaluation_metrics, "best_epoch": best_epoch, "best_val_graph_loss": best_val})
    logger.finish()
    return summary
