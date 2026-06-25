from __future__ import annotations

import torch
from torch import nn


class GeoMLP(nn.Module):
    def __init__(self, input_dim: int = 257, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def isotropic_gaussian_nll(pred: torch.Tensor, target_xy: torch.Tensor, min_log_sigma: float, max_log_sigma: float) -> torch.Tensor:
    log_sigma = pred[:, 2].clamp(min=min_log_sigma, max=max_log_sigma)
    sigma2 = torch.exp(2.0 * log_sigma)
    sq = torch.sum((pred[:, :2] - target_xy) ** 2, dim=1)
    return (torch.log(torch.tensor(2.0 * torch.pi, device=pred.device)) + 2.0 * log_sigma + sq / (2.0 * sigma2)).mean()
