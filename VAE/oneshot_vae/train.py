from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .data import (
    MISSING_VALUE,
    GenotypeMemmapMeta,
    OneShotMemmapDataset,
    build_stratified_splits,
    compute_observed_fraction,
    load_meta,
    load_sample_ids,
    open_genotype_memmap,
)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None
    F = None
    nn = None
    DataLoader = None

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

if torch is not None:
    from .model import OneShotMaskedVAE, OneShotMaskedVAEConfig
else:  # pragma: no cover
    OneShotMaskedVAE = Any  # type: ignore[misc,assignment]
    OneShotMaskedVAEConfig = Any  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    mask_prob: float = 0.2
    eval_mask_prob: float = 0.2
    epochs: int = 15
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    patience: int = 4
    grad_clip_norm: float = 1.0
    latent_dim: int = 64
    chunk_size: int = 512
    token_embed_dim: int = 8
    chunk_dim: int = 256
    encoder_layers: int = 2
    encoder_heads: int = 8
    encoder_ff_mult: int = 4
    decoder_hidden_dim: int = 512
    dropout: float = 0.1
    beta_max: float = 0.05
    beta_warmup_epochs: int = 10
    num_workers: int = 0
    monitor_every: int = 1
    coverage_monitor_subset: int = 512
    max_train_samples: int = -1
    max_val_samples: int = -1
    max_test_samples: int = -1
    max_train_batches_per_epoch: int = -1
    max_eval_batches: int = -1
    max_export_samples_per_split: int = -1


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "vae-oneshot"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    mode: str = "offline"


def parse_wandb_tags(raw_tags: Optional[str]) -> Optional[List[str]]:
    if raw_tags is None:
        return None
    tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    return tags if tags else None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _build_model_config(meta: GenotypeMemmapMeta, cfg: TrainConfig) -> OneShotMaskedVAEConfig:
    return OneShotMaskedVAEConfig(
        n_snps=meta.n_snps,
        latent_dim=cfg.latent_dim,
        chunk_size=cfg.chunk_size,
        token_embed_dim=cfg.token_embed_dim,
        chunk_dim=cfg.chunk_dim,
        encoder_layers=cfg.encoder_layers,
        encoder_heads=cfg.encoder_heads,
        encoder_ff_mult=cfg.encoder_ff_mult,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        dropout=cfg.dropout,
        n_genotype_classes=meta.n_classes,
    )


def _beta_for_epoch(epoch: int, beta_max: float, beta_warmup_epochs: int) -> float:
    if beta_max <= 0:
        return 0.0
    if beta_warmup_epochs <= 0:
        return float(beta_max)
    return float(beta_max * min(1.0, epoch / float(beta_warmup_epochs)))


def _map_raw_to_visible_tokens(raw: torch.Tensor, missing_token_id: int) -> torch.Tensor:
    tokens = raw.to(dtype=torch.long)
    tokens = tokens.clone()
    tokens[tokens < 0] = missing_token_id
    return tokens


def create_training_masked_batch(
    raw: torch.Tensor,
    mask_prob: float,
    missing_token_id: int,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = raw.device
    observed = raw >= 0
    rand = torch.rand(raw.shape, device=device)
    target_mask = observed & (rand < mask_prob)

    for row in range(raw.shape[0]):
        if bool(observed[row].any()) and not bool(target_mask[row].any()):
            obs_idx = torch.nonzero(observed[row], as_tuple=False).squeeze(1)
            choice = obs_idx[torch.randint(len(obs_idx), (1,), device=device)]
            target_mask[row, choice] = True

    tokens = _map_raw_to_visible_tokens(raw, missing_token_id)
    tokens[target_mask] = mask_token_id
    targets = raw.to(dtype=torch.long).clamp_min(0)
    return tokens, targets, target_mask


def create_eval_masked_batch(
    raw: torch.Tensor,
    row_indices: Sequence[int],
    mask_prob: float,
    missing_token_id: int,
    mask_token_id: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_np = raw.detach().cpu().numpy()
    target_mask = np.zeros_like(raw_np, dtype=bool)

    for batch_row, row_idx in enumerate(row_indices):
        observed_idx = np.flatnonzero(raw_np[batch_row] >= 0)
        if observed_idx.size == 0:
            continue
        n_select = int(round(mask_prob * float(observed_idx.size)))
        if mask_prob > 0.0:
            n_select = max(1, n_select)
        n_select = min(observed_idx.size, n_select)
        if n_select == 0:
            continue
        rng = np.random.default_rng(seed + (1009 * int(row_idx)))
        picked = rng.choice(observed_idx, size=n_select, replace=False)
        target_mask[batch_row, picked] = True

    target_mask_t = torch.from_numpy(target_mask).to(device=raw.device)
    tokens = _map_raw_to_visible_tokens(raw, missing_token_id)
    tokens[target_mask_t] = mask_token_id
    targets = raw.to(dtype=torch.long).clamp_min(0)
    return tokens, targets, target_mask_t


def _masked_ce_and_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, float, int]:
    n_valid = int(valid_mask.sum().detach().cpu().item())
    if n_valid == 0:
        zero = logits.sum() * 0.0
        return zero, float("nan"), 0

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_valid = valid_mask.reshape(-1)
    loss = F.cross_entropy(flat_logits[flat_valid], flat_targets[flat_valid], reduction="mean")
    preds = flat_logits[flat_valid].argmax(dim=-1)
    acc = float((preds == flat_targets[flat_valid]).float().mean().detach().cpu().item())
    return loss, acc, n_valid


def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def _corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    x_std = math.sqrt(float(np.mean(x ** 2)))
    y_std = math.sqrt(float(np.mean(y ** 2)))
    if x_std <= 0 or y_std <= 0:
        return float("nan")
    return float(np.mean(x * y) / (x_std * y_std))


def _linear_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    if x.shape[0] == 0:
        return float("nan"), np.zeros(0, dtype=np.float64)
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    beta, _, _, _ = np.linalg.lstsq(x_aug, y.astype(np.float64), rcond=None)
    pred = x_aug @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan"), pred
    return float(1.0 - (ss_res / ss_tot)), pred


def compute_coverage_audit(latents: np.ndarray, observed_fraction: np.ndarray) -> Dict[str, float]:
    if latents.shape[0] == 0:
        return {
            "n_samples": 0,
            "latent_dim": 0,
            "r2_cov_from_z": float("nan"),
            "pearson_cov_from_z": float("nan"),
            "pearson_cov_from_z_unitnorm": float("nan"),
            "pearson_latent_norm_vs_coverage": float("nan"),
        }

    coverage = np.asarray(observed_fraction, dtype=np.float64)
    r2, pred = _linear_r2(latents, coverage)
    norms = np.linalg.norm(latents, axis=1)
    unit = latents / np.clip(norms[:, None], a_min=1e-8, a_max=None)
    _, pred_unit = _linear_r2(unit, coverage)
    return {
        "n_samples": int(latents.shape[0]),
        "latent_dim": int(latents.shape[1]),
        "r2_cov_from_z": r2,
        "pearson_cov_from_z": _corrcoef(pred, coverage),
        "pearson_cov_from_z_unitnorm": _corrcoef(pred_unit, coverage),
        "pearson_latent_norm_vs_coverage": _corrcoef(norms, coverage),
    }


def write_pc_coverage_corr(latents: np.ndarray, observed_fraction: np.ndarray, out_path: Path) -> None:
    centered = latents.astype(np.float64) - latents.astype(np.float64).mean(axis=0, keepdims=True)
    max_pcs = min(10, centered.shape[0], centered.shape[1]) if centered.size else 0
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("pc_index\tpc_coverage_r\n")
        if max_pcs == 0:
            return
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        scores = centered @ vh[:max_pcs].T
        for idx in range(max_pcs):
            _ = singular_values[idx]
            corr = _corrcoef(scores[:, idx], observed_fraction)
            handle.write(f"{idx + 1}\t{corr}\n")


def maybe_init_wandb_run(
    wandb_cfg: WandbConfig,
    output_dir: Path,
    run_name: str,
    train_cfg: TrainConfig,
    meta_path: Path,
    meta: GenotypeMemmapMeta,
    split_counts: Dict[str, int],
    excluded_count: int,
) -> Optional[Any]:
    if not wandb_cfg.enabled:
        return None
    if wandb is None:
        raise RuntimeError("W&B logging enabled but wandb is not installed.")

    return wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        name=wandb_cfg.name or run_name,
        group=wandb_cfg.group,
        tags=wandb_cfg.tags,
        mode=wandb_cfg.mode,
        dir=str(output_dir),
        config={
            "run_name": run_name,
            "meta_json": str(meta_path),
            "memmap_path": meta.memmap_path,
            "n_samples": meta.n_samples,
            "n_snps": meta.n_snps,
            "n_classes": meta.n_classes,
            "excluded_zero_observed_samples": excluded_count,
            "splits": split_counts,
            "train_cfg": asdict(train_cfg),
        },
    )


def _build_loader(
    dataset: OneShotMemmapDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _device_and_amp() -> Tuple[torch.device, bool, torch.dtype, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    return device, use_amp, amp_dtype, use_scaler


def evaluate_model(
    model: OneShotMaskedVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    eval_mask_prob: float,
    seed: int,
    max_batches: int = -1,
) -> Dict[str, float]:
    model.eval()
    ce_vals: List[float] = []
    acc_vals: List[float] = []
    kl_vals: List[float] = []
    total_vals: List[float] = []
    counts: List[int] = []

    use_amp, amp_dtype = device.type == "cuda", (torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16)
    with torch.no_grad():
        for batch_idx, (raw_batch, row_idx, _, _) in enumerate(loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            raw_batch = raw_batch.to(device=device, dtype=torch.long)
            row_idx_list = [int(v) for v in row_idx]
            tokens, targets, valid_mask = create_eval_masked_batch(
                raw=raw_batch,
                row_indices=row_idx_list,
                mask_prob=eval_mask_prob,
                missing_token_id=model.missing_token_id,
                mask_token_id=model.mask_token_id,
                seed=seed,
            )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(tokens)
                ce_loss, acc, n_valid = _masked_ce_and_accuracy(out["logits"], targets, valid_mask)
                kl = _kl_loss(out["mu"], out["logvar"])
                total = ce_loss + (beta * kl)
            if n_valid == 0:
                continue
            ce_vals.append(float(ce_loss.detach().cpu().item()))
            acc_vals.append(acc)
            kl_vals.append(float(kl.detach().cpu().item()))
            total_vals.append(float(total.detach().cpu().item()))
            counts.append(n_valid)

    return {
        "masked_ce": float(np.average(ce_vals, weights=counts)) if counts else float("nan"),
        "masked_acc": float(np.average(acc_vals, weights=counts)) if counts else float("nan"),
        "kl": float(np.mean(kl_vals)) if kl_vals else float("nan"),
        "total": float(np.mean(total_vals)) if total_vals else float("nan"),
        "n_masked": int(np.sum(counts)) if counts else 0,
    }


def collect_latents(
    model: OneShotMaskedVAE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = -1,
) -> Dict[str, object]:
    model.eval()
    latents: List[np.ndarray] = []
    rows: List[int] = []
    sample_ids: List[str] = []
    coverage: List[float] = []
    use_amp, amp_dtype = device.type == "cuda", (torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16)

    with torch.no_grad():
        for batch_idx, (raw_batch, row_idx, batch_sample_ids, batch_coverage) in enumerate(loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            raw_batch = raw_batch.to(device=device, dtype=torch.long)
            tokens = _map_raw_to_visible_tokens(raw_batch, model.missing_token_id)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                mu, _ = model.encode(tokens)
            latents.append(mu.detach().float().cpu().numpy())
            rows.extend(int(v) for v in row_idx)
            sample_ids.extend(list(batch_sample_ids))
            coverage.extend(float(v) for v in batch_coverage)

    mu_all = np.concatenate(latents, axis=0) if latents else np.zeros((0, model.cfg.latent_dim), dtype=np.float32)
    return {
        "mu": mu_all,
        "rows": rows,
        "sample_ids": sample_ids,
        "observed_fraction": np.asarray(coverage, dtype=np.float32),
    }


def _sample_monitor_indices(indices: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return np.asarray(indices, dtype=np.int64)
    return np.asarray(indices[:limit], dtype=np.int64)


def _cap_indices(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if limit <= 0 or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    picked = rng.choice(indices, size=limit, replace=False)
    return np.asarray(np.sort(picked), dtype=np.int64)


def train_oneshot_masked_vae(
    meta: GenotypeMemmapMeta,
    sample_ids: Sequence[str],
    observed_fraction: np.ndarray,
    splits: Dict[str, np.ndarray],
    train_cfg: TrainConfig,
    output_dir: Path,
    run_name: str,
    wandb_run: Optional[Any] = None,
) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError("PyTorch is required for one-shot VAE training.")

    _set_seed(train_cfg.seed)
    device, use_amp, amp_dtype, use_scaler = _device_and_amp()

    train_ds = OneShotMemmapDataset(meta, splits["train"], observed_fraction, sample_ids)
    val_ds = OneShotMemmapDataset(meta, splits["val"], observed_fraction, sample_ids)
    test_ds = OneShotMemmapDataset(meta, splits["test"], observed_fraction, sample_ids)

    train_loader = _build_loader(train_ds, train_cfg.batch_size, True, train_cfg.num_workers)
    val_loader = _build_loader(val_ds, train_cfg.batch_size, False, train_cfg.num_workers)
    test_loader = _build_loader(test_ds, train_cfg.batch_size, False, train_cfg.num_workers)

    model_cfg = _build_model_config(meta, train_cfg)
    model = OneShotMaskedVAE(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    metrics_path = output_dir / "metrics.jsonl"
    history: List[Dict[str, float]] = []
    best_val_ce = float("inf")
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    monitor_indices = _sample_monitor_indices(splits["val"], train_cfg.coverage_monitor_subset)
    monitor_ds = OneShotMemmapDataset(meta, monitor_indices, observed_fraction, sample_ids)
    monitor_loader = _build_loader(monitor_ds, train_cfg.batch_size, False, train_cfg.num_workers)

    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for epoch in range(1, train_cfg.epochs + 1):
            model.train()
            beta = _beta_for_epoch(epoch, train_cfg.beta_max, train_cfg.beta_warmup_epochs)

            ce_vals: List[float] = []
            acc_vals: List[float] = []
            kl_vals: List[float] = []
            total_vals: List[float] = []
            counts: List[int] = []

            for batch_idx, (raw_batch, _, _, _) in enumerate(train_loader, start=1):
                if train_cfg.max_train_batches_per_epoch > 0 and batch_idx > train_cfg.max_train_batches_per_epoch:
                    break
                raw_batch = raw_batch.to(device=device, dtype=torch.long, non_blocking=True)
                tokens, targets, valid_mask = create_training_masked_batch(
                    raw=raw_batch,
                    mask_prob=train_cfg.mask_prob,
                    missing_token_id=model.missing_token_id,
                    mask_token_id=model.mask_token_id,
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(tokens)
                    ce_loss, acc, n_valid = _masked_ce_and_accuracy(out["logits"], targets, valid_mask)
                    kl = _kl_loss(out["mu"], out["logvar"])
                    total = ce_loss + (beta * kl)

                if torch.isnan(total):
                    raise RuntimeError(f"Encountered NaN loss at epoch {epoch}.")

                if scaler.is_enabled():
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                    optimizer.step()

                if n_valid > 0:
                    ce_vals.append(float(ce_loss.detach().cpu().item()))
                    acc_vals.append(acc)
                    kl_vals.append(float(kl.detach().cpu().item()))
                    total_vals.append(float(total.detach().cpu().item()))
                    counts.append(n_valid)

            val_metrics = evaluate_model(
                model=model,
                loader=val_loader,
                device=device,
                beta=beta,
                eval_mask_prob=train_cfg.eval_mask_prob,
                seed=train_cfg.seed + 17,
                max_batches=train_cfg.max_eval_batches,
            )
            epoch_row = {
                "epoch": epoch,
                "beta": beta,
                "train_masked_ce": float(np.average(ce_vals, weights=counts)) if counts else float("nan"),
                "train_masked_acc": float(np.average(acc_vals, weights=counts)) if counts else float("nan"),
                "train_kl": float(np.mean(kl_vals)) if kl_vals else float("nan"),
                "train_total": float(np.mean(total_vals)) if total_vals else float("nan"),
                "val_masked_ce": val_metrics["masked_ce"],
                "val_masked_acc": val_metrics["masked_acc"],
                "val_kl": val_metrics["kl"],
                "val_total": val_metrics["total"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }

            if train_cfg.monitor_every > 0 and (epoch % train_cfg.monitor_every == 0):
                monitor_latents = collect_latents(
                    model,
                    monitor_loader,
                    device,
                    max_batches=train_cfg.max_eval_batches,
                )
                monitor_audit = compute_coverage_audit(
                    monitor_latents["mu"],
                    np.asarray(monitor_latents["observed_fraction"], dtype=np.float32),
                )
                epoch_row["monitor_r2_cov_from_z"] = monitor_audit["r2_cov_from_z"]
                epoch_row["monitor_pearson_latent_norm_vs_coverage"] = monitor_audit[
                    "pearson_latent_norm_vs_coverage"
                ]

            history.append(epoch_row)
            metrics_handle.write(json.dumps(epoch_row) + "\n")
            metrics_handle.flush()

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "beta": epoch_row["beta"],
                        "train/masked_ce": epoch_row["train_masked_ce"],
                        "train/masked_acc": epoch_row["train_masked_acc"],
                        "train/kl": epoch_row["train_kl"],
                        "train/total": epoch_row["train_total"],
                        "val/masked_ce": epoch_row["val_masked_ce"],
                        "val/masked_acc": epoch_row["val_masked_acc"],
                        "val/kl": epoch_row["val_kl"],
                        "val/total": epoch_row["val_total"],
                        "train/learning_rate": epoch_row["learning_rate"],
                        **(
                            {
                                "monitor/r2_cov_from_z": epoch_row.get("monitor_r2_cov_from_z"),
                                "monitor/pearson_latent_norm_vs_coverage": epoch_row.get(
                                    "monitor_pearson_latent_norm_vs_coverage"
                                ),
                            }
                            if "monitor_r2_cov_from_z" in epoch_row
                            else {}
                        ),
                    },
                    step=epoch,
                )

            if val_metrics["masked_ce"] < best_val_ce:
                best_val_ce = float(val_metrics["masked_ce"])
                best_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= train_cfg.patience:
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_state)
    best_model_path = output_dir / "best_model.pt"
    torch.save({"state_dict": best_state, "model_config": asdict(model_cfg)}, best_model_path)

    val_final = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
        beta=train_cfg.beta_max,
        eval_mask_prob=train_cfg.eval_mask_prob,
        seed=train_cfg.seed + 17,
        max_batches=train_cfg.max_eval_batches,
    )
    test_final = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        beta=train_cfg.beta_max,
        eval_mask_prob=train_cfg.eval_mask_prob,
        seed=train_cfg.seed + 29,
        max_batches=train_cfg.max_eval_batches,
    )

    return {
        "model": model,
        "device": str(device),
        "amp_dtype": str(amp_dtype) if use_amp else None,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_masked_ce": best_val_ce,
        "val_metrics": val_final,
        "test_metrics": test_final,
        "best_model_path": str(best_model_path),
        "model_config": asdict(model_cfg),
    }


def export_all_latents(
    model: OneShotMaskedVAE,
    meta: GenotypeMemmapMeta,
    sample_ids: Sequence[str],
    observed_fraction: np.ndarray,
    splits: Dict[str, np.ndarray],
    train_cfg: TrainConfig,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    rows_all: List[int] = []
    sample_ids_all: List[str] = []
    splits_all: List[str] = []
    coverage_all: List[float] = []
    latents_all: List[np.ndarray] = []

    split_order = ("train", "val", "test")
    split_seed_offset = {"train": 101, "val": 202, "test": 303}
    for split_name in split_order:
        export_indices = _cap_indices(
            splits[split_name],
            train_cfg.max_export_samples_per_split,
            seed=train_cfg.seed + split_seed_offset[split_name],
        )
        ds = OneShotMemmapDataset(meta, export_indices, observed_fraction, sample_ids)
        loader = _build_loader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        split_latents = collect_latents(
            model,
            loader,
            device,
            max_batches=train_cfg.max_eval_batches,
        )
        rows_all.extend(split_latents["rows"])
        sample_ids_all.extend(split_latents["sample_ids"])
        coverage_all.extend(split_latents["observed_fraction"].tolist())
        splits_all.extend([split_name] * len(split_latents["rows"]))
        latents_all.append(split_latents["mu"])

    mu_all = np.concatenate(latents_all, axis=0) if latents_all else np.zeros((0, model.cfg.latent_dim), dtype=np.float32)
    out_csv = output_dir / "all_latents.csv"
    with out_csv.open("w", encoding="utf-8") as handle:
        header = ["row_index", "sample_id", "split", "observed_fraction"] + [f"z{i + 1}" for i in range(mu_all.shape[1])]
        handle.write(",".join(header) + "\n")
        for idx, row_idx in enumerate(rows_all):
            z_vals = ",".join(f"{float(v):.8f}" for v in mu_all[idx])
            handle.write(
                f"{row_idx},{sample_ids_all[idx]},{splits_all[idx]},{float(coverage_all[idx]):.8f},{z_vals}\n"
            )

    return {
        "mu": mu_all,
        "rows": rows_all,
        "sample_ids": sample_ids_all,
        "splits": splits_all,
        "observed_fraction": np.asarray(coverage_all, dtype=np.float32),
        "path": str(out_csv),
    }


def run_oneshot_vae_training(args: argparse.Namespace) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError("PyTorch is required for one-shot VAE training.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.meta_json)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta_json: {meta_path}")

    meta = load_meta(meta_path)
    sample_ids = load_sample_ids(Path(meta.sample_ids_path) if meta.sample_ids_path else None, meta.n_samples)
    geno_mm = open_genotype_memmap(meta, mode="r")
    observed_fraction = compute_observed_fraction(geno_mm)
    eligible_indices = np.flatnonzero(observed_fraction > 0.0).astype(np.int64)
    excluded_zero_observed = int(meta.n_samples - eligible_indices.size)

    train_cfg = TrainConfig(
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        mask_prob=args.mask_prob,
        eval_mask_prob=args.eval_mask_prob,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        latent_dim=args.latent_dim,
        chunk_size=args.chunk_size,
        token_embed_dim=args.token_embed_dim,
        chunk_dim=args.chunk_dim,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        encoder_ff_mult=args.encoder_ff_mult,
        decoder_hidden_dim=args.decoder_hidden_dim,
        dropout=args.dropout,
        beta_max=args.beta_max,
        beta_warmup_epochs=args.beta_warmup_epochs,
        num_workers=args.num_workers,
        monitor_every=args.monitor_every,
        coverage_monitor_subset=args.coverage_monitor_subset,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        max_train_batches_per_epoch=args.max_train_batches_per_epoch,
        max_eval_batches=args.max_eval_batches,
        max_export_samples_per_split=args.max_export_samples_per_split,
    )
    wandb_cfg = WandbConfig(
        enabled=bool(args.wandb),
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=parse_wandb_tags(args.wandb_tags),
        mode=args.wandb_mode,
    )

    splits = build_stratified_splits(
        eligible_indices=eligible_indices,
        observed_fraction=observed_fraction,
        seed=train_cfg.seed,
        train_ratio=train_cfg.train_ratio,
        val_ratio=train_cfg.val_ratio,
    )
    splits = {
        "train": _cap_indices(splits["train"], train_cfg.max_train_samples, seed=train_cfg.seed + 11),
        "val": _cap_indices(splits["val"], train_cfg.max_val_samples, seed=train_cfg.seed + 22),
        "test": _cap_indices(splits["test"], train_cfg.max_test_samples, seed=train_cfg.seed + 33),
    }
    split_counts = {name: int(len(values)) for name, values in splits.items()}
    np.savez(output_dir / "splits.npz", train=splits["train"], val=splits["val"], test=splits["test"])

    config_payload = vars(args).copy()
    config_payload["meta_summary"] = {
        "n_samples": meta.n_samples,
        "n_snps": meta.n_snps,
        "n_classes": meta.n_classes,
        "ploidy": meta.ploidy,
        "missing_value": meta.missing_value,
    }
    config_payload["excluded_zero_observed_samples"] = excluded_zero_observed
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    run_name = getattr(args, "run_name", None) or output_dir.name
    wandb_run = maybe_init_wandb_run(
        wandb_cfg=wandb_cfg,
        output_dir=output_dir,
        run_name=run_name,
        train_cfg=train_cfg,
        meta_path=meta_path,
        meta=meta,
        split_counts=split_counts,
        excluded_count=excluded_zero_observed,
    )

    try:
        train_results = train_oneshot_masked_vae(
            meta=meta,
            sample_ids=sample_ids,
            observed_fraction=observed_fraction,
            splits=splits,
            train_cfg=train_cfg,
            output_dir=output_dir,
            run_name=run_name,
            wandb_run=wandb_run,
        )
        model: OneShotMaskedVAE = train_results["model"]
        device = torch.device(train_results["device"])
        latent_export = export_all_latents(
            model=model,
            meta=meta,
            sample_ids=sample_ids,
            observed_fraction=observed_fraction,
            splits=splits,
            train_cfg=train_cfg,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            device=device,
            output_dir=output_dir,
        )
        coverage_audit = compute_coverage_audit(latent_export["mu"], latent_export["observed_fraction"])
        coverage_audit.update(
            {
                "split_counts": split_counts,
                "excluded_zero_observed_samples": excluded_zero_observed,
            }
        )
        with (output_dir / "coverage_audit.json").open("w", encoding="utf-8") as handle:
            json.dump(coverage_audit, handle, indent=2)
        write_pc_coverage_corr(
            latent_export["mu"],
            latent_export["observed_fraction"],
            output_dir / "pc_coverage_corr.tsv",
        )

        run_summary = {
            "run_name": run_name,
            "meta_json": str(meta_path),
            "n_samples_total": meta.n_samples,
            "n_snps": meta.n_snps,
            "n_classes": meta.n_classes,
            "ploidy": meta.ploidy,
            "excluded_zero_observed_samples": excluded_zero_observed,
            "split_counts": split_counts,
            "best_epoch": train_results["best_epoch"],
            "best_val_masked_ce": train_results["best_val_masked_ce"],
            "val_metrics": train_results["val_metrics"],
            "test_metrics": train_results["test_metrics"],
            "coverage_audit": coverage_audit,
            "device": train_results["device"],
            "amp_dtype": train_results["amp_dtype"],
            "best_model_path": train_results["best_model_path"],
            "all_latents_csv": latent_export["path"],
        }
        with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(run_summary, handle, indent=2)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "final/best_epoch": train_results["best_epoch"],
                    "final/best_val_masked_ce": train_results["best_val_masked_ce"],
                    "final/val_masked_ce": train_results["val_metrics"]["masked_ce"],
                    "final/val_masked_acc": train_results["val_metrics"]["masked_acc"],
                    "final/test_masked_ce": train_results["test_metrics"]["masked_ce"],
                    "final/test_masked_acc": train_results["test_metrics"]["masked_acc"],
                    "audit/r2_cov_from_z": coverage_audit["r2_cov_from_z"],
                    "audit/pearson_cov_from_z": coverage_audit["pearson_cov_from_z"],
                    "audit/pearson_cov_from_z_unitnorm": coverage_audit["pearson_cov_from_z_unitnorm"],
                    "audit/pearson_latent_norm_vs_coverage": coverage_audit[
                        "pearson_latent_norm_vs_coverage"
                    ],
                },
                step=len(train_results["history"]) + 1,
            )
            wandb_run.summary["best_epoch"] = int(train_results["best_epoch"])
            wandb_run.summary["best_val_masked_ce"] = float(train_results["best_val_masked_ce"])
            wandb_run.summary["excluded_zero_observed_samples"] = excluded_zero_observed
            wandb_run.finish()
    except Exception:
        if wandb_run is not None:
            wandb_run.finish()
        raise

    return {
        "config": config_payload,
        "train_results": train_results,
        "coverage_audit": coverage_audit,
        "run_summary_path": str(output_dir / "run_summary.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a one-shot full-SNP masked VAE from an existing genotype memmap meta.json."
    )
    parser.add_argument("--meta_json", required=True, help="Path to prepared genotype memmap meta.json")
    parser.add_argument("--output_dir", required=True, help="Directory for run artifacts")
    parser.add_argument("--run_name", default=None, help="Optional run name; defaults to output directory name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--mask_prob", type=float, default=0.2)
    parser.add_argument("--eval_mask_prob", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--token_embed_dim", type=int, default=8)
    parser.add_argument("--chunk_dim", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--encoder_heads", type=int, default=8)
    parser.add_argument("--encoder_ff_mult", type=int, default=4)
    parser.add_argument("--decoder_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta_max", type=float, default=0.05)
    parser.add_argument("--beta_warmup_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--monitor_every", type=int, default=1)
    parser.add_argument("--coverage_monitor_subset", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_val_samples", type=int, default=-1)
    parser.add_argument("--max_test_samples", type=int, default=-1)
    parser.add_argument("--max_train_batches_per_epoch", type=int, default=-1)
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--max_export_samples_per_split", type=int, default=-1)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="vae-oneshot", help="W&B project name")
    parser.add_argument("--wandb_entity", default=None, help="W&B entity/team")
    parser.add_argument("--wandb_name", default=None, help="W&B run name")
    parser.add_argument("--wandb_group", default=None, help="W&B run group")
    parser.add_argument("--wandb_tags", default=None, help="Comma-separated W&B tags")
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline"),
        default="offline",
        help="W&B mode",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_oneshot_vae_training(args)
    print(json.dumps({"run_summary_path": payload["run_summary_path"]}, indent=2))


if __name__ == "__main__":
    main()
