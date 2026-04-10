from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class TrainingStateSnapshot:
    best_epoch: int
    best_val_loss: float
    n_epochs_completed: int


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch is required for overlap embedding training") from exc
    return torch, nn


class NodeEmbeddingModelFactory:
    @staticmethod
    def build(n_nodes: int, latent_dim: int, init_latents: np.ndarray):
        torch, nn = require_torch()
        if init_latents.shape != (n_nodes, latent_dim):
            raise ValueError(
                f"Expected init_latents shape {(n_nodes, latent_dim)}, got {init_latents.shape}"
            )

        class NodeEmbeddingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(n_nodes, latent_dim)
                self.embedding.weight.data.copy_(torch.tensor(init_latents, dtype=torch.float32))

            def forward(self):
                return self.embedding.weight

        return NodeEmbeddingModel()


def covariance_regularizers(latents, coverage):
    torch, _ = require_torch()
    centered = latents - latents.mean(dim=0, keepdim=True)
    var = centered.pow(2).mean(dim=0)
    var_penalty = (var - 1.0).pow(2).mean()
    if latents.shape[1] > 1:
        cov = centered.t().matmul(centered) / max(int(latents.shape[0]) - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_penalty = off_diag.pow(2).mean()
    else:
        cov_penalty = torch.zeros((), device=latents.device)
    coverage_centered = coverage - coverage.mean()
    coverage_cov = (centered * coverage_centered[:, None]).mean(dim=0)
    coverage_penalty = coverage_cov.pow(2).mean()
    return {
        "var_mean": var.mean(),
        "var_penalty": var_penalty,
        "cov_penalty": cov_penalty,
        "coverage_penalty": coverage_penalty,
    }


def positive_edge_loss(latents, src, dst, weight):
    diff = latents[src] - latents[dst]
    dist2 = diff.pow(2).sum(dim=1)
    weighted = weight * dist2
    return weighted.mean()


def negative_edge_loss(latents, src, neg_dst, margin: float):
    torch, _ = require_torch()
    diff = latents[src] - latents[neg_dst]
    dist = torch.sqrt(diff.pow(2).sum(dim=1) + 1e-8)
    return torch.relu(float(margin) - dist).pow(2).mean()


def snapshot_state(best_epoch: int, best_val_loss: float, n_epochs_completed: int) -> Dict[str, float]:
    snap = TrainingStateSnapshot(
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val_loss),
        n_epochs_completed=int(n_epochs_completed),
    )
    return {
        "best_epoch": snap.best_epoch,
        "best_val_loss": snap.best_val_loss,
        "n_epochs_completed": snap.n_epochs_completed,
    }
