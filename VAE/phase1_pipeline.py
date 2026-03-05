#!/usr/bin/env python3
"""Phase 1 pipeline for masked VAE training from existing X sparse matrix.

This module intentionally does NOT create X. It expects an existing Datamatrix X
in sparse COO format with companion .info/.ind/.snp files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - handled at runtime when training
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def parse_info_file(info_path: Path) -> Dict[str, str]:
    info: Dict[str, str] = {}
    with info_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info


def write_info_file(info_path: Path, fields: Dict[str, object]) -> None:
    with info_path.open("w", encoding="utf-8") as handle:
        for key, value in fields.items():
            handle.write(f"{key}: {value}\n")


def read_matrix_shape(info_path: Path) -> Tuple[int, int]:
    info = parse_info_file(info_path)
    if "nrows" not in info or "ncols" not in info:
        raise ValueError(f"Missing nrows/ncols in info file: {info_path}")
    return int(info["nrows"]), int(info["ncols"])


def iter_sparse_entries(sparse_path: Path) -> Iterator[Tuple[int, int, float]]:
    with sparse_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid sparse COO line {line_no} in {sparse_path}: {raw_line!r}"
                )
            row = int(parts[0])
            col = int(parts[1])
            val = float(parts[2])
            yield row, col, val


def build_mask_from_x(
    x_sparse_path: Path,
    x_info_path: Path,
    m_sparse_path: Path,
    m_info_path: Path,
) -> Dict[str, int]:
    """Build binary support mask M from existing sparse matrix X.

    M is defined as support(X): M[row, col] = 1 for every stored X COO entry.
    """

    nrows, ncols = read_matrix_shape(x_info_path)

    x_nnz = 0
    m_nnz = 0
    deduped = 0
    last_coord: Optional[Tuple[int, int]] = None

    m_sparse_path.parent.mkdir(parents=True, exist_ok=True)
    with m_sparse_path.open("w", encoding="utf-8") as out_handle:
        for row, col, _ in iter_sparse_entries(x_sparse_path):
            x_nnz += 1
            if row < 0 or col < 0 or row >= nrows or col >= ncols:
                raise ValueError(
                    f"Out-of-bounds coordinate in X: ({row}, {col}) for shape ({nrows}, {ncols})"
                )

            coord = (row, col)
            if coord == last_coord:
                deduped += 1
                continue

            out_handle.write(f"{row} {col} 1\n")
            m_nnz += 1
            last_coord = coord

    write_info_file(
        m_info_path,
        {
            "format": "COO",
            "nrows": nrows,
            "ncols": ncols,
            "nnz": m_nnz,
            "source": str(x_sparse_path),
            "derived_from": "support(X)",
            "value_meaning": "1 => coordinate present in X.sparse",
        },
    )

    return {
        "nrows": nrows,
        "ncols": ncols,
        "x_nnz_input": x_nnz,
        "m_nnz_written": m_nnz,
        "deduped_consecutive_entries": deduped,
    }


def verify_x_covered_by_m(x_sparse_path: Path, m_sparse_path: Path) -> Dict[str, object]:
    """Verify every coordinate in X appears in M.

    Since M is derived from X, this check assumes both files are sorted row-major
    and compares coordinate streams.
    """

    x_iter = iter_sparse_entries(x_sparse_path)
    m_iter = iter_sparse_entries(m_sparse_path)

    x_count = 0
    m_count = 0

    try:
        m_current = next(m_iter)
        m_count += 1
    except StopIteration:
        return {
            "covered": False,
            "reason": "M is empty while X has coordinates or requires checking",
            "x_count": 0,
            "m_count": 0,
        }

    for x_row, x_col, _ in x_iter:
        x_count += 1
        x_coord = (x_row, x_col)

        while (m_current[0], m_current[1]) < x_coord:
            try:
                m_current = next(m_iter)
                m_count += 1
            except StopIteration:
                return {
                    "covered": False,
                    "reason": "M exhausted before covering all X coordinates",
                    "x_count": x_count,
                    "m_count": m_count,
                    "missing_coordinate": [x_row, x_col],
                }

        if (m_current[0], m_current[1]) != x_coord:
            return {
                "covered": False,
                "reason": "Coordinate in X missing from M",
                "x_count": x_count,
                "m_count": m_count,
                "missing_coordinate": [x_row, x_col],
            }

    for _ in m_iter:
        m_count += 1

    return {
        "covered": True,
        "reason": "All X coordinates are present in M",
        "x_count": x_count,
        "m_count": m_count,
    }


def compute_column_stats(
    x_sparse_path: Path,
    nrows: int,
    ncols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(ncols, dtype=np.int64)
    sums = np.zeros(ncols, dtype=np.float64)
    sq_sums = np.zeros(ncols, dtype=np.float64)

    for row, col, val in iter_sparse_entries(x_sparse_path):
        if row < 0 or col < 0 or row >= nrows or col >= ncols:
            raise ValueError(
                f"Out-of-bounds coordinate in X during stats pass: ({row}, {col})"
            )
        counts[col] += 1
        sums[col] += val
        sq_sums[col] += val * val

    return counts, sums, sq_sums


def select_features(
    counts: np.ndarray,
    sums: np.ndarray,
    sq_sums: np.ndarray,
    nrows: int,
    min_call_rate: float,
    min_variance: float,
    max_features: int,
) -> Dict[str, object]:
    if not (0.0 <= min_call_rate <= 1.0):
        raise ValueError("min_call_rate must be in [0, 1]")

    call_rate = counts.astype(np.float64) / float(nrows)
    means = sums / float(nrows)
    variances = (sq_sums / float(nrows)) - np.square(means)

    eligible = (call_rate >= min_call_rate) & (variances >= min_variance)
    eligible_indices = np.flatnonzero(eligible)

    if eligible_indices.size == 0:
        raise ValueError(
            "No eligible features after call-rate/variance filtering. "
            "Lower thresholds or verify X input."
        )

    if eligible_indices.size > max_features:
        candidate_vars = variances[eligible_indices]
        top_local = np.argpartition(candidate_vars, -max_features)[-max_features:]
        selected = eligible_indices[top_local]
        selected = selected[np.argsort(variances[selected])[::-1]]
    else:
        selected = eligible_indices[np.argsort(variances[eligible_indices])[::-1]]

    return {
        "selected_indices": selected.astype(np.int64),
        "eligible_count": int(eligible_indices.size),
        "selected_count": int(selected.size),
        "mean_call_rate_selected": float(call_rate[selected].mean()),
        "mean_variance_selected": float(variances[selected].mean()),
        "min_call_rate": float(min_call_rate),
        "min_variance": float(min_variance),
        "max_features": int(max_features),
    }


def build_selected_dense_memmaps(
    x_sparse_path: Path,
    m_sparse_path: Path,
    nrows: int,
    selected_indices: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = int(selected_indices.size)
    col_to_new = {int(col): i for i, col in enumerate(selected_indices.tolist())}

    x_memmap_path = output_dir / f"{prefix}_X_selected.float32.memmap"
    m_memmap_path = output_dir / f"{prefix}_M_selected.uint8.memmap"

    x_mm = np.memmap(x_memmap_path, dtype=np.float32, mode="w+", shape=(nrows, n_features))
    m_mm = np.memmap(m_memmap_path, dtype=np.uint8, mode="w+", shape=(nrows, n_features))

    x_mm[:, :] = 0.0
    m_mm[:, :] = 0

    for row, col, val in iter_sparse_entries(x_sparse_path):
        mapped = col_to_new.get(col)
        if mapped is None:
            continue
        x_mm[row, mapped] = np.float32(val)

    for row, col, val in iter_sparse_entries(m_sparse_path):
        if val <= 0:
            continue
        mapped = col_to_new.get(col)
        if mapped is None:
            continue
        m_mm[row, mapped] = 1

    x_mm.flush()
    m_mm.flush()

    return {
        "x_memmap_path": str(x_memmap_path),
        "m_memmap_path": str(m_memmap_path),
        "nrows": nrows,
        "n_features": n_features,
    }


def load_individual_ids(ind_path: Path) -> List[str]:
    ids: List[str] = []
    with ind_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            ids.append(line.split()[0])
    return ids


def split_indices(
    nrows: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, np.ndarray]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios. Need train>0, val>0, train+val<1")

    indices = list(range(nrows))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(nrows * train_ratio)
    val_end = train_end + int(nrows * val_ratio)

    train = np.array(indices[:train_end], dtype=np.int64)
    val = np.array(indices[train_end:val_end], dtype=np.int64)
    test = np.array(indices[val_end:], dtype=np.int64)

    return {"train": train, "val": val, "test": test}


class MemmapDataset(Dataset):
    def __init__(
        self,
        x_memmap_path: Path,
        m_memmap_path: Path,
        nrows: int,
        n_features: int,
        indices: Sequence[int],
    ) -> None:
        self.x = np.memmap(x_memmap_path, dtype=np.float32, mode="r", shape=(nrows, n_features))
        self.m = np.memmap(m_memmap_path, dtype=np.uint8, mode="r", shape=(nrows, n_features))
        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int):
        row_idx = int(self.indices[item])
        x_row = np.array(self.x[row_idx], copy=False)
        m_row = np.array(self.m[row_idx], copy=False)
        return (
            torch.from_numpy(x_row.astype(np.float32)),
            torch.from_numpy(m_row.astype(np.float32)),
            row_idx,
        )


if nn is not None:
    class MaskedVAE(nn.Module):
        def __init__(self, n_features: int, hidden_1: int, hidden_2: int, latent_dim: int) -> None:
            super().__init__()
            input_dim = n_features * 2

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_1),
                nn.ReLU(),
                nn.Linear(hidden_1, hidden_2),
                nn.ReLU(),
            )
            self.mu_layer = nn.Linear(hidden_2, latent_dim)
            self.logvar_layer = nn.Linear(hidden_2, latent_dim)

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_2),
                nn.ReLU(),
                nn.Linear(hidden_2, hidden_1),
                nn.ReLU(),
                nn.Linear(hidden_1, n_features),
            )

        def encode(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.encoder(torch.cat([x, m], dim=1))
            return self.mu_layer(h), self.logvar_layer(h)

        @staticmethod
        def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        def forward(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(x, m)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar
else:  # pragma: no cover
    class MaskedVAE:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required to use MaskedVAE.")


def masked_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    m: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    observed = torch.clamp(m.sum(), min=1.0)
    recon_loss = (((recon_x - x) ** 2) * m).sum() / observed
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total = recon_loss + beta * kl_loss

    return total, {
        "recon": float(recon_loss.detach().cpu().item()),
        "kl": float(kl_loss.detach().cpu().item()),
        "total": float(total.detach().cpu().item()),
    }


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    latent_dim: int = 32
    hidden_1: int = 512
    hidden_2: int = 128
    beta_max: float = 1.0
    beta_warmup_fraction: float = 0.2
    patience: int = 5
    seed: int = 42


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "vae-phase1"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    mode: str = "online"


@dataclass
class GateConfig:
    min_recon_improvement: float = 0.01
    min_active_latent_dims: int = 4
    active_latent_std_threshold: float = 0.05


def parse_wandb_tags(raw_tags: Optional[str]) -> Optional[List[str]]:
    if raw_tags is None:
        return None
    tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    return tags if tags else None


def maybe_init_wandb_run(
    wandb_cfg: WandbConfig,
    output_dir: Path,
    run_name: str,
    train_cfg: TrainConfig,
    nrows: int,
    n_features: int,
    feature_data: Dict[str, object],
    split_counts: Dict[str, int],
) -> Optional[Any]:
    if not wandb_cfg.enabled:
        return None
    if wandb is None:
        raise RuntimeError(
            "W&B logging enabled but wandb is not installed. Install with `pip install wandb`."
        )

    config_payload = {
        "run_name": run_name,
        "nrows": nrows,
        "n_features": n_features,
        "train_cfg": {
            "epochs": train_cfg.epochs,
            "batch_size": train_cfg.batch_size,
            "learning_rate": train_cfg.learning_rate,
            "latent_dim": train_cfg.latent_dim,
            "hidden_1": train_cfg.hidden_1,
            "hidden_2": train_cfg.hidden_2,
            "beta_max": train_cfg.beta_max,
            "beta_warmup_fraction": train_cfg.beta_warmup_fraction,
            "patience": train_cfg.patience,
            "seed": train_cfg.seed,
        },
        "feature_selection": {
            "selected_count": feature_data.get("selected_count"),
            "eligible_count": feature_data.get("eligible_count"),
            "mean_call_rate_selected": feature_data.get("mean_call_rate_selected"),
            "mean_variance_selected": feature_data.get("mean_variance_selected"),
            "min_call_rate": feature_data.get("min_call_rate"),
            "min_variance": feature_data.get("min_variance"),
            "max_features": feature_data.get("max_features"),
        },
        "splits": split_counts,
    }

    return wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=wandb_cfg.name or run_name,
        group=wandb_cfg.group,
        tags=wandb_cfg.tags,
        mode=wandb_cfg.mode,
        dir=str(output_dir),
        config=config_payload,
    )


def evaluate_epoch(
    model: MaskedVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> Dict[str, float]:
    model.eval()
    recon_vals: List[float] = []
    kl_vals: List[float] = []
    total_vals: List[float] = []

    with torch.no_grad():
        for x_batch, m_batch, _ in loader:
            x_batch = x_batch.to(device)
            m_batch = m_batch.to(device)
            recon, mu, logvar = model(x_batch, m_batch)
            _, pieces = masked_vae_loss(recon, x_batch, m_batch, mu, logvar, beta)
            recon_vals.append(pieces["recon"])
            kl_vals.append(pieces["kl"])
            total_vals.append(pieces["total"])

    return {
        "recon": float(np.mean(recon_vals)) if recon_vals else float("nan"),
        "kl": float(np.mean(kl_vals)) if kl_vals else float("nan"),
        "total": float(np.mean(total_vals)) if total_vals else float("nan"),
    }


def collect_latents(
    model: MaskedVAE,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    all_mu: List[np.ndarray] = []
    all_rows: List[int] = []

    with torch.no_grad():
        for x_batch, m_batch, row_idx in loader:
            x_batch = x_batch.to(device)
            m_batch = m_batch.to(device)
            mu, _ = model.encode(x_batch, m_batch)
            all_mu.append(mu.detach().cpu().numpy())
            all_rows.extend([int(v) for v in row_idx])

    if all_mu:
        mu_matrix = np.concatenate(all_mu, axis=0)
    else:
        mu_matrix = np.zeros((0, 0), dtype=np.float32)

    return {"mu": mu_matrix, "rows": all_rows}


def train_masked_vae(
    x_memmap_path: Path,
    m_memmap_path: Path,
    nrows: int,
    n_features: int,
    splits: Dict[str, np.ndarray],
    train_cfg: TrainConfig,
    output_dir: Path,
    wandb_run: Optional[Any] = None,
) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed. Install dependencies first, e.g. pip install torch."
        )

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MemmapDataset(x_memmap_path, m_memmap_path, nrows, n_features, splits["train"])
    val_ds = MemmapDataset(x_memmap_path, m_memmap_path, nrows, n_features, splits["val"])
    test_ds = MemmapDataset(x_memmap_path, m_memmap_path, nrows, n_features, splits["test"])

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False)

    model = MaskedVAE(
        n_features=n_features,
        hidden_1=train_cfg.hidden_1,
        hidden_2=train_cfg.hidden_2,
        latent_dim=train_cfg.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    warmup_epochs = max(1, int(math.ceil(train_cfg.epochs * train_cfg.beta_warmup_fraction)))

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        beta = train_cfg.beta_max * min(1.0, epoch / float(warmup_epochs))

        train_recon: List[float] = []
        train_kl: List[float] = []
        train_total: List[float] = []

        for x_batch, m_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            m_batch = m_batch.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(x_batch, m_batch)
            loss, parts = masked_vae_loss(recon, x_batch, m_batch, mu, logvar, beta)

            if torch.isnan(loss):
                raise RuntimeError("Encountered NaN loss during training.")

            loss.backward()
            optimizer.step()

            train_recon.append(parts["recon"])
            train_kl.append(parts["kl"])
            train_total.append(parts["total"])

        val_parts = evaluate_epoch(model, val_loader, device, beta)

        epoch_row = {
            "epoch": epoch,
            "beta": beta,
            "train_recon": float(np.mean(train_recon)),
            "train_kl": float(np.mean(train_kl)),
            "train_total": float(np.mean(train_total)),
            "val_recon": val_parts["recon"],
            "val_kl": val_parts["kl"],
            "val_total": val_parts["total"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_row)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "beta": epoch_row["beta"],
                    "train/recon": epoch_row["train_recon"],
                    "train/kl": epoch_row["train_kl"],
                    "train/total": epoch_row["train_total"],
                    "val/recon": epoch_row["val_recon"],
                    "val/kl": epoch_row["val_kl"],
                    "val/total": epoch_row["val_total"],
                    "train/learning_rate": epoch_row["learning_rate"],
                },
                step=epoch,
            )

        if val_parts["recon"] < best_val:
            best_val = val_parts["recon"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= train_cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid best checkpoint.")

    model.load_state_dict(best_state)

    model_path = output_dir / "best_model.pt"
    torch.save(best_state, model_path)

    val_final = evaluate_epoch(model, val_loader, device, train_cfg.beta_max)
    test_final = evaluate_epoch(model, test_loader, device, train_cfg.beta_max)
    val_latents = collect_latents(model, val_loader, device)

    if wandb_run is not None:
        final_step = len(history) + 1
        wandb_run.log(
            {
                "final/best_epoch": best_epoch,
                "final/best_val_recon": best_val,
                "final/val_recon": val_final["recon"],
                "final/val_kl": val_final["kl"],
                "final/test_recon": test_final["recon"],
                "final/test_kl": test_final["kl"],
            },
            step=final_step,
        )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_recon": best_val,
        "val_metrics": val_final,
        "test_metrics": test_final,
        "val_latents": val_latents,
        "model_path": str(model_path),
        "device": str(device),
    }


def evaluate_gate(history: List[Dict[str, float]], val_latents: np.ndarray, gate_cfg: GateConfig) -> Dict[str, object]:
    if not history:
        raise ValueError("Empty training history.")

    stable = True
    reason_stable = "Training metrics are finite."
    for row in history:
        for key in ("train_recon", "train_total", "val_recon", "val_total"):
            if not math.isfinite(float(row[key])):
                stable = False
                reason_stable = f"Non-finite metric detected ({key}) at epoch {row['epoch']}"
                break
        if not stable:
            break

    first_val_recon = float(history[0]["val_recon"])
    best_val_recon = float(min(row["val_recon"] for row in history))
    if first_val_recon <= 0:
        recon_improvement = 0.0
    else:
        recon_improvement = (first_val_recon - best_val_recon) / first_val_recon

    recon_ok = recon_improvement >= gate_cfg.min_recon_improvement

    if val_latents.size == 0:
        active_dims = 0
    else:
        latent_std = np.std(val_latents, axis=0)
        active_dims = int(np.sum(latent_std >= gate_cfg.active_latent_std_threshold))

    latent_ok = active_dims >= gate_cfg.min_active_latent_dims

    gate_pass = bool(stable and recon_ok and latent_ok)

    return {
        "pass": gate_pass,
        "checks": {
            "stable_training": {"pass": stable, "detail": reason_stable},
            "reconstruction": {
                "pass": recon_ok,
                "first_val_recon": first_val_recon,
                "best_val_recon": best_val_recon,
                "improvement": recon_improvement,
                "min_required": gate_cfg.min_recon_improvement,
            },
            "latent_structure": {
                "pass": latent_ok,
                "active_dims": active_dims,
                "min_required": gate_cfg.min_active_latent_dims,
                "active_std_threshold": gate_cfg.active_latent_std_threshold,
            },
        },
    }


def write_review_bundle(
    output_dir: Path,
    metrics: Dict[str, object],
    gate_decision: Dict[str, object],
) -> None:
    markdown_path = output_dir / "review_bundle.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Phase 1 VAE Review Bundle\n\n")
        handle.write("## Core Metrics\n")
        handle.write(f"- Best epoch: {metrics['best_epoch']}\n")
        handle.write(f"- Best validation reconstruction: {metrics['best_val_recon']:.6f}\n")
        handle.write(f"- Validation reconstruction (final model): {metrics['val_metrics']['recon']:.6f}\n")
        handle.write(f"- Test reconstruction (final model): {metrics['test_metrics']['recon']:.6f}\n")
        handle.write("\n## Gate Decision\n")
        handle.write(f"- Pass: {gate_decision['pass']}\n")
        for name, check in gate_decision["checks"].items():
            handle.write(f"- {name}: pass={check['pass']}\n")


def run_phase1_vae_training(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_sparse = Path(args.x_sparse)
    x_info = Path(args.x_info)
    x_ind = Path(args.x_ind)

    if not x_sparse.exists() or not x_info.exists() or not x_ind.exists():
        raise FileNotFoundError("Missing required X input files (x_sparse/x_info/x_ind).")

    nrows, ncols = read_matrix_shape(x_info)

    m_sparse = Path(args.m_sparse) if args.m_sparse else output_dir / "M.sparse"
    m_info = Path(args.m_info) if args.m_info else output_dir / "M.info"

    mask_rebuilt = args.rebuild_mask or (not m_sparse.exists()) or (not m_info.exists())

    if mask_rebuilt:
        mask_stats = build_mask_from_x(x_sparse, x_info, m_sparse, m_info)
    else:
        mask_stats = {
            "nrows": nrows,
            "ncols": ncols,
            "x_nnz_input": -1,
            "m_nnz_written": int(parse_info_file(m_info).get("nnz", "-1")),
            "deduped_consecutive_entries": -1,
        }

    m_shape = read_matrix_shape(m_info)
    if m_shape != (nrows, ncols):
        raise ValueError(
            f"M shape {m_shape} does not match X shape {(nrows, ncols)}"
        )

    if mask_rebuilt:
        # M was generated directly from the current X stream in this run.
        # Skip the separate stream-merge checker, which assumes row-major order.
        coverage = {
            "covered": True,
            "reason": "Skipped explicit coverage check because M was rebuilt from X in this run.",
        }
    else:
        coverage = verify_x_covered_by_m(x_sparse, m_sparse)
        if not coverage.get("covered", False):
            raise ValueError(f"X->M coverage check failed: {coverage}")

    counts, sums, sq_sums = compute_column_stats(x_sparse, nrows, ncols)

    feature_data = select_features(
        counts=counts,
        sums=sums,
        sq_sums=sq_sums,
        nrows=nrows,
        min_call_rate=args.min_call_rate,
        min_variance=args.min_variance,
        max_features=args.max_features,
    )
    selected_indices = feature_data["selected_indices"]

    dense_info = build_selected_dense_memmaps(
        x_sparse_path=x_sparse,
        m_sparse_path=m_sparse,
        nrows=nrows,
        selected_indices=selected_indices,
        output_dir=output_dir,
        prefix=args.run_name,
    )

    sample_ids = load_individual_ids(x_ind)
    if len(sample_ids) != nrows:
        raise ValueError(
            f"X row count ({nrows}) and .ind sample count ({len(sample_ids)}) differ"
        )

    splits = split_indices(
        nrows=nrows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    np.savez(
        output_dir / "splits.npz",
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        beta_max=args.beta_max,
        beta_warmup_fraction=args.beta_warmup_fraction,
        patience=args.patience,
        seed=args.seed,
    )

    split_counts = {
        "train": int(len(splits["train"])),
        "val": int(len(splits["val"])),
        "test": int(len(splits["test"])),
    }

    wandb_cfg = WandbConfig(
        enabled=bool(getattr(args, "wandb", False)),
        project=str(getattr(args, "wandb_project", "vae-phase1")),
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_name", None),
        group=getattr(args, "wandb_group", None),
        tags=parse_wandb_tags(getattr(args, "wandb_tags", None)),
        mode=str(getattr(args, "wandb_mode", "online")),
    )
    wandb_run = maybe_init_wandb_run(
        wandb_cfg=wandb_cfg,
        output_dir=output_dir,
        run_name=args.run_name,
        train_cfg=train_cfg,
        nrows=nrows,
        n_features=int(dense_info["n_features"]),
        feature_data=feature_data,
        split_counts=split_counts,
    )
    wandb_run_url = getattr(wandb_run, "url", None) if wandb_run is not None else None

    try:
        metrics = train_masked_vae(
            x_memmap_path=Path(dense_info["x_memmap_path"]),
            m_memmap_path=Path(dense_info["m_memmap_path"]),
            nrows=nrows,
            n_features=int(dense_info["n_features"]),
            splits=splits,
            train_cfg=train_cfg,
            output_dir=output_dir,
            wandb_run=wandb_run,
        )
    except Exception:
        if wandb_run is not None:
            wandb_run.finish()
        raise

    val_latents = metrics["val_latents"]["mu"]
    gate_cfg = GateConfig(
        min_recon_improvement=args.min_recon_improvement,
        min_active_latent_dims=args.min_active_latent_dims,
        active_latent_std_threshold=args.active_latent_std_threshold,
    )
    gate = evaluate_gate(metrics["history"], val_latents, gate_cfg)
    if wandb_run is not None:
        wandb_run.log(
            {
                "gate/pass": int(gate["pass"]),
                "gate/reconstruction_pass": int(gate["checks"]["reconstruction"]["pass"]),
                "gate/latent_pass": int(gate["checks"]["latent_structure"]["pass"]),
                "gate/stable_training_pass": int(gate["checks"]["stable_training"]["pass"]),
                "gate/recon_improvement": float(gate["checks"]["reconstruction"]["improvement"]),
                "gate/active_dims": int(gate["checks"]["latent_structure"]["active_dims"]),
            },
            step=len(metrics["history"]) + 2,
        )

    latent_csv = output_dir / "val_latents.csv"
    with latent_csv.open("w", encoding="utf-8") as handle:
        header = ["row_index", "sample_id"] + [f"z{i+1}" for i in range(val_latents.shape[1])]
        handle.write(",".join(header) + "\n")
        for idx, row_id in enumerate(metrics["val_latents"]["rows"]):
            sample_id = sample_ids[row_id]
            z_vals = ",".join(f"{v:.8f}" for v in val_latents[idx])
            handle.write(f"{row_id},{sample_id},{z_vals}\n")

    feature_json = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in feature_data.items()
    }

    output_payload = {
        "run_name": args.run_name,
        "inputs": {
            "x_sparse": str(x_sparse),
            "x_info": str(x_info),
            "x_ind": str(x_ind),
            "m_sparse": str(m_sparse),
            "m_info": str(m_info),
        },
        "mask_stats": mask_stats,
        "coverage_check": coverage,
        "feature_selection": feature_json,
        "dense_data": dense_info,
        "splits": split_counts,
        "metrics": {
            "best_epoch": metrics["best_epoch"],
            "best_val_recon": metrics["best_val_recon"],
            "val_metrics": metrics["val_metrics"],
            "test_metrics": metrics["test_metrics"],
            "device": metrics["device"],
            "history": metrics["history"],
        },
        "gate": gate,
        "wandb": {
            "enabled": bool(wandb_cfg.enabled),
            "project": wandb_cfg.project if wandb_cfg.enabled else None,
            "entity": wandb_cfg.entity if wandb_cfg.enabled else None,
            "run_name": getattr(wandb_run, "name", None) if wandb_run is not None else None,
            "run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
            "run_url": wandb_run_url,
        },
    }

    with (output_dir / "phase1_results.json").open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    with (output_dir / "gate_decision.json").open("w", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2)

    with (output_dir / "CAN_START_VQVAE.txt").open("w", encoding="utf-8") as handle:
        handle.write("YES\n" if gate["pass"] else "NO\n")

    write_review_bundle(output_dir, output_payload["metrics"], gate)
    if wandb_run is not None:
        wandb_run.summary["gate_pass"] = bool(gate["pass"])
        wandb_run.summary["best_epoch"] = int(metrics["best_epoch"])
        wandb_run.summary["best_val_recon"] = float(metrics["best_val_recon"])
        wandb_run.finish()

    return output_payload


def build_mask_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build support mask M from existing X sparse COO matrix.")
    parser.add_argument("--x_sparse", required=True, help="Path to X.sparse")
    parser.add_argument("--x_info", required=True, help="Path to X.info")
    parser.add_argument("--m_sparse", required=True, help="Path to output M.sparse")
    parser.add_argument("--m_info", required=True, help="Path to output M.info")
    return parser


def build_train_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 masked VAE pipeline: build M from existing X, train/evaluate VAE, "
            "and emit review gate artifacts."
        )
    )
    parser.add_argument("--x_sparse", required=True, help="Path to X.sparse")
    parser.add_argument("--x_info", required=True, help="Path to X.info")
    parser.add_argument("--x_ind", required=True, help="Path to X.ind")

    parser.add_argument("--m_sparse", default=None, help="Optional path to M.sparse")
    parser.add_argument("--m_info", default=None, help="Optional path to M.info")
    parser.add_argument("--rebuild_mask", action="store_true", help="Force rebuild of M from X")

    parser.add_argument("--output_dir", required=True, help="Output directory for artifacts")
    parser.add_argument("--run_name", default="phase1", help="Prefix for generated files")

    parser.add_argument("--min_call_rate", type=float, default=0.05)
    parser.add_argument("--min_variance", type=float, default=1e-8)
    parser.add_argument("--max_features", type=int, default=20000)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_1", type=int, default=512)
    parser.add_argument("--hidden_2", type=int, default=128)
    parser.add_argument("--beta_max", type=float, default=1.0)
    parser.add_argument("--beta_warmup_fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--min_recon_improvement", type=float, default=0.01)
    parser.add_argument("--min_active_latent_dims", type=int, default=4)
    parser.add_argument("--active_latent_std_threshold", type=float, default=0.05)

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="vae-phase1", help="W&B project name")
    parser.add_argument("--wandb_entity", default=None, help="W&B entity/team (optional)")
    parser.add_argument("--wandb_name", default=None, help="W&B run name (optional)")
    parser.add_argument("--wandb_group", default=None, help="W&B run group (optional)")
    parser.add_argument("--wandb_tags", default=None, help="Comma-separated W&B tags")
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline"),
        default="online",
        help="W&B mode",
    )

    return parser


def main_build_mask(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_mask_cli_parser()
    args = parser.parse_args(argv)

    stats = build_mask_from_x(
        x_sparse_path=Path(args.x_sparse),
        x_info_path=Path(args.x_info),
        m_sparse_path=Path(args.m_sparse),
        m_info_path=Path(args.m_info),
    )
    print(json.dumps(stats, indent=2))


def main_train(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_train_cli_parser()
    args = parser.parse_args(argv)
    payload = run_phase1_vae_training(args)
    print(json.dumps({"gate_pass": payload["gate"]["pass"], "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main_train()
