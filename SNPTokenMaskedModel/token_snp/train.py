from __future__ import annotations

import json
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
    build_random_window_batch,
    compute_observed_fraction,
    load_meta,
    load_sample_ids,
    make_splits,
    open_genotype_memmap,
)
from .model import TokenSNPConfig, TokenSNPMaskedModel


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str
    meta_json: str
    seed: int = 42
    window_size: int = 8192
    windows_per_sample: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    epochs: int = 20
    steps_per_epoch: int = 250
    val_steps: int = 40
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    patience: int = 5
    mask_prob: float = 0.2
    missingness_dropout: float = 0.1
    d_model: int = 128
    n_heads: int = 4
    local_layers: int = 2
    chunk_layers: int = 2
    chunk_size: int = 256
    ff_mult: int = 4
    dropout: float = 0.1
    latent_dim: int = 64
    use_obs_embedding: bool = True
    use_snp_id_embedding: bool = False
    embedding_batch_size: int = 128
    embedding_windows_per_sample: int = 32
    coverage_monitor_subset: int = 512
    monitor_every: int = 1
    batch_labels_tsv: Optional[str] = None
    wandb_enable: bool = False
    wandb_project: str = "token-snp-mask"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[str] = None
    wandb_mode: str = "offline"


def _sample_rows(rng: np.random.Generator, pool: np.ndarray, size: int) -> np.ndarray:
    picks = rng.integers(0, len(pool), size=size)
    return pool[picks]


def _masked_recon_ce(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, int]:
    n_classes = logits.shape[-1]
    n_masked = int((targets != -100).sum().detach().cpu().item())
    if n_masked == 0:
        return logits.sum() * 0.0, 0

    loss = F.cross_entropy(
        logits.reshape(-1, n_classes),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss / float(n_masked), n_masked


def _masked_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    valid = targets != -100
    n = int(valid.sum().detach().cpu().item())
    if n == 0:
        return float("nan")

    preds = logits.argmax(dim=-1)
    correct = (preds[valid] == targets[valid]).float().mean()
    return float(correct.detach().cpu().item())


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


def _write_jsonl(path: Path, row: Dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _parse_wandb_tags(raw_tags: Optional[str]) -> Optional[List[str]]:
    if raw_tags is None:
        return None
    tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    return tags if tags else None


def _load_batch_labels(path: Path, sample_ids: Sequence[str]) -> np.ndarray:
    sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    labels = np.full(len(sample_ids), -1, dtype=np.int64)

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
            idx = sid_to_idx.get(sid)
            if idx is None:
                continue
            if batch not in batch_to_id:
                batch_to_id[batch] = next_id
                next_id += 1
            labels[idx] = batch_to_id[batch]
    return labels


@torch.no_grad()
def _encode_sample_embeddings(
    model: TokenSNPMaskedModel,
    geno_mm: np.memmap,
    indices: np.ndarray,
    cfg: TrainConfig,
    n_classes: int,
    n_snps: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    model.eval()
    rng = np.random.default_rng(seed)

    latents = np.zeros((len(indices), model.cfg.latent_dim), dtype=np.float32)
    max_start = max(1, n_snps - cfg.window_size + 1)

    write_ptr = 0
    for start in range(0, len(indices), cfg.embedding_batch_size):
        end = min(start + cfg.embedding_batch_size, len(indices))
        batch_idx = indices[start:end]

        accum = np.zeros((len(batch_idx), model.cfg.latent_dim), dtype=np.float32)
        counts = np.zeros(len(batch_idx), dtype=np.float32)

        for _ in range(cfg.embedding_windows_per_sample):
            starts = rng.integers(
                low=0,
                high=max_start,
                size=(len(batch_idx), 1),
                dtype=np.int64,
            )
            batch = build_random_window_batch(
                geno_mm=geno_mm,
                sample_indices=batch_idx,
                window_size=cfg.window_size,
                windows_per_sample=1,
                n_classes=n_classes,
                rng=rng,
                mask_prob=0.0,
                missingness_dropout=0.0,
                apply_mask=False,
                apply_missingness_dropout=False,
                device=device,
                window_starts=starts,
            )
            out = model(
                tokens=batch["tokens"],
                obs_mask=batch["obs_mask"],
                snp_idx=batch["snp_idx"] if cfg.use_snp_id_embedding else None,
            )
            emb = out["window_embedding"].detach().cpu().numpy()
            accum += emb
            counts += 1.0

        lat = accum / counts[:, None].clip(min=1.0)
        latents[write_ptr : write_ptr + len(batch_idx)] = lat
        write_ptr += len(batch_idx)

    return latents


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
    splits = make_splits(
        n_samples=meta.n_samples,
        seed=cfg.seed,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    if len(splits["val"]) == 0:
        fallback_n = max(1, min(32, len(splits["train"])))
        splits["val"] = splits["train"][:fallback_n].copy()

    np.savez(
        out_dir / "splits.npz",
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = TokenSNPConfig(
        n_classes=meta.n_classes,
        n_snps=meta.n_snps,
        window_size=cfg.window_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        local_layers=cfg.local_layers,
        chunk_layers=cfg.chunk_layers,
        chunk_size=cfg.chunk_size,
        ff_mult=cfg.ff_mult,
        dropout=cfg.dropout,
        latent_dim=cfg.latent_dim,
        use_obs_embedding=cfg.use_obs_embedding,
        use_snp_id_embedding=cfg.use_snp_id_embedding,
    )
    model = TokenSNPMaskedModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
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
                "n_classes": meta.n_classes,
                "device": str(device),
            },
        )

    coverage_fraction = compute_observed_fraction(geno_mm)
    sample_ids = load_sample_ids(meta.sample_ids_path, meta.n_samples)
    batch_labels = None
    if cfg.batch_labels_tsv is not None:
        batch_labels = _load_batch_labels(Path(cfg.batch_labels_tsv), sample_ids)

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_masked = 0
        train_acc_sum = 0.0
        train_acc_count = 0

        for _ in range(cfg.steps_per_epoch):
            batch_idx = _sample_rows(rng, splits["train"], cfg.batch_size)
            batch = build_random_window_batch(
                geno_mm=geno_mm,
                sample_indices=batch_idx,
                window_size=cfg.window_size,
                windows_per_sample=cfg.windows_per_sample,
                n_classes=meta.n_classes,
                rng=rng,
                mask_prob=cfg.mask_prob,
                missingness_dropout=cfg.missingness_dropout,
                apply_mask=True,
                apply_missingness_dropout=True,
                device=device,
            )

            out = model(
                tokens=batch["tokens"],
                obs_mask=batch["obs_mask"],
                snp_idx=batch["snp_idx"] if cfg.use_snp_id_embedding else None,
            )
            loss, n_masked = _masked_recon_ce(out["logits"], batch["targets"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu().item()) * n_masked
            train_masked += n_masked

            acc = _masked_accuracy(out["logits"], batch["targets"])
            if not np.isnan(acc):
                train_acc_sum += acc
                train_acc_count += 1

        train_ce = train_loss_sum / float(max(1, train_masked))
        train_acc = train_acc_sum / float(max(1, train_acc_count))

        model.eval()
        val_loss_sum = 0.0
        val_masked = 0
        val_acc_sum = 0.0
        val_acc_count = 0

        with torch.no_grad():
            for _ in range(cfg.val_steps):
                batch_idx = _sample_rows(rng, splits["val"], cfg.batch_size)
                batch = build_random_window_batch(
                    geno_mm=geno_mm,
                    sample_indices=batch_idx,
                    window_size=cfg.window_size,
                    windows_per_sample=cfg.windows_per_sample,
                    n_classes=meta.n_classes,
                    rng=rng,
                    mask_prob=cfg.mask_prob,
                    missingness_dropout=0.0,
                    apply_mask=True,
                    apply_missingness_dropout=False,
                    device=device,
                )

                out = model(
                    tokens=batch["tokens"],
                    obs_mask=batch["obs_mask"],
                    snp_idx=batch["snp_idx"] if cfg.use_snp_id_embedding else None,
                )
                loss, n_masked = _masked_recon_ce(out["logits"], batch["targets"])

                val_loss_sum += float(loss.detach().cpu().item()) * n_masked
                val_masked += n_masked

                acc = _masked_accuracy(out["logits"], batch["targets"])
                if not np.isnan(acc):
                    val_acc_sum += acc
                    val_acc_count += 1

        val_ce = val_loss_sum / float(max(1, val_masked))
        val_acc = val_acc_sum / float(max(1, val_acc_count))

        cov_corr = float("nan")
        batch_r2 = float("nan")
        if cfg.monitor_every > 0 and (epoch % cfg.monitor_every == 0):
            monitor_pool = np.concatenate([splits["train"], splits["val"], splits["test"]])
            monitor_k = min(cfg.coverage_monitor_subset, len(monitor_pool))
            monitor_ids = rng.choice(monitor_pool, size=monitor_k, replace=False)

            lat = _encode_sample_embeddings(
                model=model,
                geno_mm=geno_mm,
                indices=monitor_ids,
                cfg=cfg,
                n_classes=meta.n_classes,
                n_snps=meta.n_snps,
                device=device,
                seed=cfg.seed + epoch,
            )
            lat_norm = np.linalg.norm(lat, axis=1)
            cov_corr = _pearson_corr(lat_norm, coverage_fraction[monitor_ids])
            if batch_labels is not None:
                batch_r2 = _batch_r2(lat_norm, batch_labels[monitor_ids])

        row = {
            "epoch": float(epoch),
            "train_ce": float(train_ce),
            "val_ce": float(val_ce),
            "train_masked_acc": float(train_acc),
            "val_masked_acc": float(val_acc),
            "train_masked_tokens": float(train_masked),
            "val_masked_tokens": float(val_masked),
            "coverage_latent_norm_corr": float(cov_corr),
            "batch_latent_norm_r2": float(batch_r2),
        }
        _write_jsonl(metrics_path, row)
        history.append(row)

        if wandb_run is not None:
            wandb_run.log(row, step=epoch)

        if val_ce < best_val:
            best_val = val_ce
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "model_config": asdict(model_cfg),
                    "train_config": asdict(cfg),
                    "best_val_ce": best_val,
                },
                out_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.patience:
            break

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    all_ids = np.arange(meta.n_samples, dtype=np.int64)
    global_latents = _encode_sample_embeddings(
        model=model,
        geno_mm=geno_mm,
        indices=all_ids,
        cfg=cfg,
        n_classes=meta.n_classes,
        n_snps=meta.n_snps,
        device=device,
        seed=cfg.seed + 10_000,
    )

    np.save(out_dir / "global_latents.npy", global_latents)

    with (out_dir / "global_latents.csv").open("w", encoding="utf-8") as handle:
        header = ["sample_id"] + [f"z{i}" for i in range(global_latents.shape[1])]
        handle.write(",".join(header) + "\n")
        for sid, vec in zip(sample_ids, global_latents):
            row = [sid] + [f"{float(v):.7g}" for v in vec]
            handle.write(",".join(row) + "\n")

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_ce": float(best_val),
        "epochs_completed": len(history),
        "n_samples": meta.n_samples,
        "n_snps": meta.n_snps,
        "n_classes": meta.n_classes,
        "device": str(device),
        "meta_json": str(cfg.meta_json),
        "output_dir": str(out_dir),
        "model_config": asdict(model_cfg),
        "train_config": asdict(cfg),
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
        wandb_run.summary.update(
            {
                "best_epoch": best_epoch,
                "best_val_ce": best_val,
            }
        )
        wandb_run.finish()

    return summary
