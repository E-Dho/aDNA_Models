from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from .data import (
    MISSING_VALUE,
    Window,
    build_window_batch,
    build_windows,
    compute_observed_fraction,
    load_meta,
    make_splits,
    open_genotype_memmap,
)
from .model import HMAEConfig, HierarchicalMaskedAutoencoder


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str
    meta_json: str
    seed: int = 42
    window_size: int = 8192
    windows_per_step: int = 8
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    epochs: int = 20
    steps_per_epoch: int = 250
    val_steps: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    coverage_corr_weight: float = 0.0
    mask_prob: float = 0.2
    observed_dropout: float = 0.1
    patience: int = 5
    variational: bool = False
    kl_beta_max: float = 1.0
    kl_warmup_fraction: float = 0.3
    window_latent_dim: int = 32
    global_latent_dim: int = 32
    embed_dim: int = 16
    local_hidden_dim: int = 128
    local_encoder_type: str = "conv_attn"
    local_conv_layers: int = 4
    local_conv_kernel: int = 7
    local_attn_heads: int = 4
    local_dropout: float = 0.1
    global_model_dim: int = 128
    global_heads: int = 4
    global_layers: int = 2
    decoder_hidden_dim: int = 128
    include_window_coverage: bool = True
    require_cuda: bool = False
    amp: Optional[bool] = None
    embedding_batch_size: int = 16
    embedding_window_chunk: int = 4
    coverage_monitor_subset: int = 128
    monitor_every: int = 1
    batch_labels_tsv: Optional[str] = None
    wandb_enable: bool = False
    wandb_project: str = "hmae-phase1"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[str] = None
    wandb_mode: str = "offline"


def _sample_rows(rng: np.random.Generator, pool: np.ndarray, size: int) -> np.ndarray:
    picks = rng.integers(0, len(pool), size=size)
    return pool[picks]


def _sample_window_ids(rng: np.random.Generator, n_windows: int, k: int) -> np.ndarray:
    k_eff = min(k, n_windows)
    return rng.choice(n_windows, size=k_eff, replace=False).astype(np.int64)


def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def _beta_at_epoch(epoch: int, epochs: int, beta_max: float, warmup_fraction: float) -> float:
    warmup_epochs = max(1, int(math.ceil(epochs * warmup_fraction)))
    return float(beta_max * min(1.0, epoch / float(warmup_epochs)))


def _autocast_context(device: torch.device, amp_enabled: bool):
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _is_cuda_oom(exc: RuntimeError, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def _squared_pearson_corr_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if x.numel() < 2 or y.numel() < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt(
        torch.sum(x_centered * x_centered) * torch.sum(y_centered * y_centered) + eps
    )
    corr = torch.sum(x_centered * y_centered) / denom
    return corr * corr


def _masked_recon_ce(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, int]:
    # logits: [B, K, W, C], targets: [B, K, W] with -100 for ignore
    n_classes = logits.shape[-1]
    loss = F.cross_entropy(
        logits.reshape(-1, n_classes),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    n_masked = int((targets != -100).sum().detach().cpu().item())
    denom = max(1, n_masked)
    return loss / float(denom), n_masked


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom <= 0:
        return 0.0
    return float((x * y).sum() / denom)


def _write_jsonl(path: Path, row: Dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _parse_wandb_tags(raw_tags: Optional[str]) -> Optional[List[str]]:
    if raw_tags is None:
        return None
    tags = [v.strip() for v in raw_tags.split(",") if v.strip()]
    return tags if tags else None


def _load_batch_labels(path: Path, sample_ids: Sequence[str]) -> np.ndarray:
    sample_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    raw_labels = np.full(len(sample_ids), -1, dtype=np.int64)
    batch_to_id: Dict[str, int] = {}
    next_id = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            sid, batch = parts[0], parts[1]
            idx = sample_to_idx.get(sid)
            if idx is None:
                continue
            if batch not in batch_to_id:
                batch_to_id[batch] = next_id
                next_id += 1
            raw_labels[idx] = batch_to_id[batch]
    return raw_labels


def _batch_r2(latent_norm: np.ndarray, batch_ids: np.ndarray) -> float:
    mask = batch_ids >= 0
    if mask.sum() < 3:
        return float("nan")
    y = latent_norm[mask].astype(np.float64)
    b = batch_ids[mask]
    y_mean = y.mean()
    total = ((y - y_mean) ** 2).sum()
    if total <= 0:
        return 0.0
    between = 0.0
    for bid in np.unique(b):
        grp = y[b == bid]
        if grp.size == 0:
            continue
        between += grp.size * (grp.mean() - y_mean) ** 2
    return float(between / total)


def _load_sample_ids(meta_path: Path, n_samples: int) -> List[str]:
    meta = load_meta(meta_path)
    if meta.sample_ids_path is None:
        return [f"sample_{i}" for i in range(n_samples)]
    p = Path(meta.sample_ids_path)
    if not p.exists():
        return [f"sample_{i}" for i in range(n_samples)]
    ids: List[str] = []
    with p.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                ids.append(line)
    if len(ids) != n_samples:
        raise ValueError(
            f"Sample ID count mismatch in {p}: {len(ids)} != expected {n_samples}"
        )
    return ids


@torch.no_grad()
def _encode_global_for_indices(
    model: HierarchicalMaskedAutoencoder,
    geno_mm: np.memmap,
    windows: Sequence[Window],
    indices: np.ndarray,
    cfg: TrainConfig,
    n_classes: int,
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    model.eval()

    all_window_ids = np.arange(len(windows), dtype=np.int64)
    all_window_ids_t = torch.from_numpy(all_window_ids).to(device=device, dtype=torch.long)

    result = np.zeros((len(indices), model.cfg.global_latent_dim), dtype=np.float32)

    current_bs = max(1, int(cfg.embedding_batch_size))
    current_wc = max(1, int(cfg.embedding_window_chunk))
    write_ptr = 0
    start = 0
    dummy_rng = np.random.default_rng(cfg.seed)

    while start < len(indices):
        batch_success = False
        while not batch_success:
            bs = min(current_bs, len(indices) - start)
            end = start + bs
            batch_idx = indices[start:end]
            lat_chunks: List[torch.Tensor] = []
            try:
                for w_start in range(0, len(windows), current_wc):
                    w_end = min(w_start + current_wc, len(windows))
                    wids = all_window_ids[w_start:w_end]
                    batch = build_window_batch(
                        geno_mm=geno_mm,
                        sample_indices=batch_idx,
                        windows=windows,
                        window_ids=wids,
                        window_size=cfg.window_size,
                        n_classes=n_classes,
                        rng=dummy_rng,
                        mask_prob=0.0,
                        observed_dropout=0.0,
                        training=False,
                        device=device,
                    )
                    with _autocast_context(device=device, amp_enabled=amp_enabled):
                        win_lat = model.encode_windows(
                            input_idx=batch["input_idx"],
                            obs_mask=batch["obs_mask"],
                            coverage=batch["coverage"] if cfg.include_window_coverage else None,
                        )
                    lat_chunks.append(win_lat)

                with _autocast_context(device=device, amp_enabled=amp_enabled):
                    all_win_lat = torch.cat(lat_chunks, dim=1)
                    z, _, _ = model.aggregate(all_win_lat, all_window_ids_t)
                result[write_ptr : write_ptr + bs] = z.detach().cpu().numpy()
                write_ptr += bs
                start += bs
                batch_success = True
            except RuntimeError as exc:
                if not _is_cuda_oom(exc, device):
                    raise
                # Back off window chunk first, then sample batch size.
                if current_wc > 1:
                    current_wc = max(1, current_wc // 2)
                elif current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                else:
                    raise RuntimeError(
                        "CUDA OOM during latent encoding even at "
                        "embedding_batch_size=1 and embedding_window_chunk=1."
                    ) from exc
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"[hmae] CUDA OOM in embedding path; retrying with "
                    f"embedding_batch_size={current_bs}, "
                    f"embedding_window_chunk={current_wc}"
                )
            finally:
                for t in lat_chunks:
                    del t
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    return result


def run_training(cfg: TrainConfig) -> Dict[str, object]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    meta = load_meta(Path(cfg.meta_json))
    if meta.missing_value != MISSING_VALUE:
        raise ValueError(
            f"Unsupported missing value {meta.missing_value}. Expected {MISSING_VALUE}."
        )

    geno_mm = open_genotype_memmap(meta, mode="r")
    windows = build_windows(meta.n_snps, cfg.window_size)
    splits = make_splits(
        n_samples=meta.n_samples,
        seed=cfg.seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    if len(splits["val"]) == 0:
        fallback_n = max(1, min(32, len(splits["train"])))
        splits["val"] = splits["train"][:fallback_n].copy()
    if len(splits["test"]) == 0:
        fallback_n = max(1, min(32, len(splits["train"])))
        splits["test"] = splits["train"][:fallback_n].copy()
    np.savez(
        out_dir / "splits.npz",
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )

    observed_fraction = compute_observed_fraction(geno_mm)
    sample_ids = _load_sample_ids(Path(cfg.meta_json), meta.n_samples)
    batch_ids = None
    if cfg.batch_labels_tsv:
        batch_ids = _load_batch_labels(Path(cfg.batch_labels_tsv), sample_ids)

    cuda_available = torch.cuda.is_available()
    if cfg.require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is required (--require_cuda) but no CUDA device is available."
        )
    device = torch.device("cuda" if cuda_available else "cpu")
    amp_enabled = bool(cfg.amp) if cfg.amp is not None else (device.type == "cuda")
    if amp_enabled and device.type != "cuda":
        amp_enabled = False

    model_cfg = HMAEConfig(
        n_classes=meta.n_classes,
        n_windows=len(windows),
        window_size=cfg.window_size,
        window_latent_dim=cfg.window_latent_dim,
        global_latent_dim=cfg.global_latent_dim,
        embed_dim=cfg.embed_dim,
        local_hidden_dim=cfg.local_hidden_dim,
        local_encoder_type=cfg.local_encoder_type,
        local_conv_layers=cfg.local_conv_layers,
        local_conv_kernel=cfg.local_conv_kernel,
        local_attn_heads=cfg.local_attn_heads,
        local_dropout=cfg.local_dropout,
        global_model_dim=cfg.global_model_dim,
        global_heads=cfg.global_heads,
        global_layers=cfg.global_layers,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        include_window_coverage=cfg.include_window_coverage,
        variational=cfg.variational,
    )
    model = HierarchicalMaskedAutoencoder(model_cfg).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    wandb_run = None
    if cfg.wandb_enable:
        if wandb is None:
            raise RuntimeError(
                "W&B requested but wandb package is not installed. "
                "Install with: pip install wandb"
            )
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name or Path(cfg.output_dir).name,
            group=cfg.wandb_group,
            tags=_parse_wandb_tags(cfg.wandb_tags),
            mode=cfg.wandb_mode,
            dir=str(out_dir),
            config={
                "train_cfg": asdict(cfg),
                "model_cfg": asdict(model_cfg),
                "n_samples": meta.n_samples,
                "n_snps": meta.n_snps,
                "n_windows": len(windows),
                "device": str(device),
                "amp_enabled": bool(amp_enabled),
            },
        )

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        beta = _beta_at_epoch(
            epoch=epoch,
            epochs=cfg.epochs,
            beta_max=cfg.kl_beta_max,
            warmup_fraction=cfg.kl_warmup_fraction,
        )

        train_recon_vals: List[float] = []
        train_kl_vals: List[float] = []
        train_cov_pen_vals: List[float] = []
        train_total_vals: List[float] = []

        for _step in range(cfg.steps_per_epoch):
            batch_idx = _sample_rows(rng, splits["train"], cfg.batch_size)
            window_ids = _sample_window_ids(rng, len(windows), cfg.windows_per_step)
            batch = build_window_batch(
                geno_mm=geno_mm,
                sample_indices=batch_idx,
                windows=windows,
                window_ids=window_ids,
                window_size=cfg.window_size,
                n_classes=meta.n_classes,
                rng=rng,
                mask_prob=cfg.mask_prob,
                observed_dropout=cfg.observed_dropout,
                training=True,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device=device, amp_enabled=amp_enabled):
                out = model(
                    input_idx=batch["input_idx"],
                    obs_mask=batch["obs_mask"],
                    window_ids=batch["window_ids"],
                    coverage=batch["coverage"] if cfg.include_window_coverage else None,
                )

                recon_loss, n_masked = _masked_recon_ce(out["logits"], batch["targets"])
                if n_masked == 0:
                    continue

                if cfg.variational:
                    if out["mu"] is None or out["logvar"] is None:
                        raise RuntimeError("Variational mode enabled but mu/logvar missing.")
                    kl = _kl_loss(out["mu"], out["logvar"])
                else:
                    kl = torch.zeros((), device=device)
                cov_pen = torch.zeros((), device=device)
                if cfg.coverage_corr_weight > 0:
                    z_norm = torch.linalg.vector_norm(out["global_latent"], ord=2, dim=1)
                    cov_signal = batch["coverage"].squeeze(-1).mean(dim=1)
                    cov_pen = _squared_pearson_corr_torch(z_norm, cov_signal)
                total = recon_loss + beta * kl + cfg.coverage_corr_weight * cov_pen
            if torch.isnan(total):
                raise RuntimeError("NaN loss encountered.")

            if scaler.is_enabled():
                scaler.scale(total).backward()
                if cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            train_recon_vals.append(float(recon_loss.detach().cpu().item()))
            train_kl_vals.append(float(kl.detach().cpu().item()))
            train_cov_pen_vals.append(float(cov_pen.detach().cpu().item()))
            train_total_vals.append(float(total.detach().cpu().item()))

        model.eval()
        val_recon_vals: List[float] = []
        val_kl_vals: List[float] = []
        val_total_vals: List[float] = []
        with torch.no_grad():
            for _ in range(cfg.val_steps):
                batch_idx = _sample_rows(rng, splits["val"], cfg.batch_size)
                window_ids = _sample_window_ids(rng, len(windows), cfg.windows_per_step)
                batch = build_window_batch(
                    geno_mm=geno_mm,
                    sample_indices=batch_idx,
                    windows=windows,
                    window_ids=window_ids,
                    window_size=cfg.window_size,
                    n_classes=meta.n_classes,
                    rng=rng,
                    mask_prob=cfg.mask_prob,
                    observed_dropout=0.0,
                    training=True,
                    device=device,
                )
                with _autocast_context(device=device, amp_enabled=amp_enabled):
                    out = model(
                        input_idx=batch["input_idx"],
                        obs_mask=batch["obs_mask"],
                        window_ids=batch["window_ids"],
                        coverage=batch["coverage"] if cfg.include_window_coverage else None,
                    )
                    recon_loss, n_masked = _masked_recon_ce(out["logits"], batch["targets"])
                if n_masked == 0:
                    continue
                if cfg.variational:
                    if out["mu"] is None or out["logvar"] is None:
                        raise RuntimeError("Variational mode enabled but mu/logvar missing.")
                    kl = _kl_loss(out["mu"], out["logvar"])
                else:
                    kl = torch.zeros((), device=device)
                total = recon_loss + beta * kl
                val_recon_vals.append(float(recon_loss.detach().cpu().item()))
                val_kl_vals.append(float(kl.detach().cpu().item()))
                val_total_vals.append(float(total.detach().cpu().item()))

        epoch_metrics = {
            "epoch": epoch,
            "beta": beta,
            "train_recon": float(np.mean(train_recon_vals)) if train_recon_vals else float("nan"),
            "train_kl": float(np.mean(train_kl_vals)) if train_kl_vals else float("nan"),
            "train_cov_corr_penalty": (
                float(np.mean(train_cov_pen_vals)) if train_cov_pen_vals else float("nan")
            ),
            "train_total": float(np.mean(train_total_vals)) if train_total_vals else float("nan"),
            "val_recon": float(np.mean(val_recon_vals)) if val_recon_vals else float("nan"),
            "val_kl": float(np.mean(val_kl_vals)) if val_kl_vals else float("nan"),
            "val_total": float(np.mean(val_total_vals)) if val_total_vals else float("nan"),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if (epoch % cfg.monitor_every) == 0 and len(splits["val"]) > 0:
            subset_n = min(cfg.coverage_monitor_subset, len(splits["val"]))
            subset = splits["val"][:subset_n]
            z_subset = _encode_global_for_indices(
                model=model,
                geno_mm=geno_mm,
                windows=windows,
                indices=subset,
                cfg=cfg,
                n_classes=meta.n_classes,
                device=device,
                amp_enabled=amp_enabled,
            )
            z_norm = np.linalg.norm(z_subset, axis=1)
            cov_corr = _pearson_corr(z_norm, observed_fraction[subset])
            epoch_metrics["coverage_latent_norm_corr"] = cov_corr
            if batch_ids is not None:
                epoch_metrics["batch_latent_norm_r2"] = _batch_r2(z_norm, batch_ids[subset])

        _write_jsonl(metrics_path, epoch_metrics)
        if wandb_run is not None:
            wandb_run.log(epoch_metrics, step=epoch)

        val_key = epoch_metrics["val_recon"]
        if math.isfinite(val_key) and val_key < best_val:
            best_val = val_key
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("Training ended without a valid best checkpoint.")

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(cfg),
            "best_epoch": best_epoch,
            "best_val_recon": best_val,
        },
        out_dir / "best_model.pt",
    )

    all_indices = np.arange(meta.n_samples, dtype=np.int64)
    global_latents = _encode_global_for_indices(
        model=model,
        geno_mm=geno_mm,
        windows=windows,
        indices=all_indices,
        cfg=cfg,
        n_classes=meta.n_classes,
        device=device,
        amp_enabled=amp_enabled,
    )
    np.save(out_dir / "global_latents.npy", global_latents)

    with (out_dir / "global_latents.csv").open("w", encoding="utf-8") as handle:
        header = ["sample_id"] + [f"z{i+1}" for i in range(global_latents.shape[1])]
        handle.write(",".join(header) + "\n")
        for i, sid in enumerate(sample_ids):
            row = ",".join(f"{v:.8f}" for v in global_latents[i])
            handle.write(f"{sid},{row}\n")

    summary = {
        "best_epoch": best_epoch,
        "best_val_recon": best_val,
        "n_samples": meta.n_samples,
        "n_snps": meta.n_snps,
        "n_windows": len(windows),
        "n_classes": meta.n_classes,
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(cfg),
        "wandb": {
            "enabled": bool(cfg.wandb_enable),
            "run_name": getattr(wandb_run, "name", None) if wandb_run is not None else None,
            "run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
            "run_url": getattr(wandb_run, "url", None) if wandb_run is not None else None,
            "mode": cfg.wandb_mode if cfg.wandb_enable else None,
        },
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = int(best_epoch)
        wandb_run.summary["best_val_recon"] = float(best_val)
        wandb_run.finish()

    return summary
