from __future__ import annotations

import json
import socket
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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
    latent_tokens: int = 8
    latent_feedback_mode: str = "cross_attn"
    use_obs_embedding: bool = True
    use_snp_id_embedding: bool = False
    embedding_batch_size: int = 128
    embedding_windows_per_sample: int = 32
    coverage_monitor_subset: int = 512
    monitor_every: int = 1
    adv_coverage_enable: bool = False
    lambda_adv_target: float = 0.0
    lambda_adv_warmup_epochs: int = 10
    lambda_cov_target: float = 0.0
    adv_steps_per_batch: int = 1
    lambda_ramp_start_epoch: int = 6
    lambda_ramp_end_epoch: int = 15
    adv_mlp_hidden_dim: int = 128
    adv_mlp_dropout: float = 0.1
    coverage_conditioning_mode: str = "none"
    coverage_embed_dim: int = 0
    batch_labels_tsv: Optional[str] = None
    probe_eval_enable: bool = False
    probe_metadata_tsv: Optional[str] = None
    probe_target_col: str = "Political Entity"
    probe_seed: int = 42
    wandb_enable: bool = False
    wandb_project: str = "token-snp-mask"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[str] = None
    wandb_mode: str = "offline"
    debug_mode: bool = False
    debug_examples_per_batch: int = 20
    debug_max_batches_per_phase: int = -1


def _sample_rows(rng: np.random.Generator, pool: np.ndarray, size: int) -> np.ndarray:
    picks = rng.integers(0, len(pool), size=size)
    return pool[picks]


def _build_valid_mask(targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    return (targets >= 0) & (targets < n_classes)


def _masked_recon_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    n_used = int(valid_mask.sum().detach().cpu().item())
    if n_used == 0:
        return logits.sum() * 0.0, 0

    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    valid_flat = valid_mask.reshape(-1)
    loss = F.cross_entropy(logits_flat[valid_flat], targets_flat[valid_flat], reduction="sum")
    return loss / float(n_used), n_used


def _masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> float:
    n = int(valid_mask.sum().detach().cpu().item())
    if n == 0:
        return float("nan")

    preds = logits.argmax(dim=-1)
    correct = (preds[valid_mask] == targets[valid_mask]).float().mean()
    return float(correct.detach().cpu().item())


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
        ctx.lambda_adv = float(lambda_adv)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_adv * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
    return GradientReversalFunction.apply(x, float(lambda_adv))


class CoverageAdversary(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        _ = hidden_dim  # Keep CLI/config surface stable while using fixed stronger architecture.
        _ = dropout
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def _scheduled_lambda(
    epoch: int,
    target: float,
    start_epoch: int,
    end_epoch: int,
) -> float:
    if target <= 0.0:
        return 0.0
    if end_epoch < start_epoch:
        return float(target if epoch >= start_epoch else 0.0)
    if epoch < start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return float(target)
    span = max(1, end_epoch - start_epoch + 1)
    progress = float(epoch - start_epoch + 1) / float(span)
    return float(target * progress)


def _corr_penalty_per_dim(
    z: torch.Tensor,
    coverage_std: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    zc = z - z.mean(dim=0, keepdim=True)
    cc = coverage_std - coverage_std.mean()
    z_std = torch.sqrt((zc ** 2).mean(dim=0) + eps)
    c_std = torch.sqrt((cc ** 2).mean() + eps)
    corr = (zc * cc.unsqueeze(1)).mean(dim=0) / (z_std * c_std)
    return (corr ** 2).mean()


def _in_sample_r2_with_intercept(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    beta, _, _, _ = np.linalg.lstsq(x_aug, y.astype(np.float64), rcond=None)
    y_pred = x_aug @ beta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def _debug_print_token_config(meta_n_classes: int, model: TokenSNPMaskedModel) -> None:
    special = {
        "pad_token_id": None,  # No explicit PAD token in input vocabulary.
        "mask_token_id": model.mask_token,
        "missing_token_id": model.missing_token,
        "cls_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
    }
    print(
        "[DEBUG] token_config "
        f"n_classes={meta_n_classes} vocab_size={model.vocab_size} specials={special}"
    )


def _debug_top_hist(counter: Counter, total: int, top_k: int = 20) -> str:
    if total <= 0:
        return "[]"
    rows: List[str] = []
    for token, count in counter.most_common(top_k):
        pct = 100.0 * float(count) / float(total)
        rows.append(f"{token}:{count} ({pct:.2f}%)")
    return "[" + ", ".join(rows) + "]"


def _counter_from_tensor(values: torch.Tensor) -> Counter:
    c: Counter = Counter()
    if values.numel() == 0:
        return c
    vals = values.detach().cpu().numpy().astype(np.int64, copy=False)
    for v in vals.tolist():
        c[int(v)] += 1
    return c


def _debug_print_masked_examples(
    phase: str,
    epoch: int,
    step: int,
    tokens: torch.Tensor,
    train_mask: torch.Tensor,
    targets: torch.Tensor,
    logits: torch.Tensor,
    mask_token_id: int,
    missing_token_id: int,
    n_examples: int,
) -> None:
    masked_positions = (train_mask > 0.5).nonzero(as_tuple=False)
    if masked_positions.numel() == 0:
        print(f"[DEBUG] {phase} epoch={epoch} step={step} no masked positions found")
        return

    probs = torch.softmax(logits, dim=-1)
    n_take = min(n_examples, masked_positions.shape[0])
    print(
        f"[DEBUG] {phase} epoch={epoch} step={step} masked_examples="
        f"{n_take}/{masked_positions.shape[0]}"
    )
    for i in range(n_take):
        b = int(masked_positions[i, 0].item())
        t = int(masked_positions[i, 1].item())
        in_tok = int(tokens[b, t].item())
        tgt_tok = int(targets[b, t].item())
        pred_tok = int(logits[b, t].argmax().item())
        tgt_prob = float(probs[b, t, tgt_tok].item()) if tgt_tok >= 0 else float("nan")
        print(
            "[DEBUG] masked_position "
            f"phase={phase} epoch={epoch} step={step} b={b} t={t} "
            f"input_token={in_tok} target_token={tgt_tok} predicted_token={pred_tok} "
            f"target_prob={tgt_prob:.6f}"
        )

    masked_targets = targets[train_mask > 0.5]
    invalid_special = (
        (masked_targets == mask_token_id)
        | (masked_targets == missing_token_id)
        | (masked_targets < 0)
    )
    n_invalid = int(invalid_special.sum().detach().cpu().item())
    n_masked = int(masked_targets.numel())
    # Expect zero; tolerate <=1% to avoid stopping if data is partially corrupted.
    max_allowed = max(0, int(0.01 * n_masked))
    assert n_invalid <= max_allowed, (
        f"Masked targets contain invalid specials: {n_invalid}/{n_masked}, "
        f"mask_token={mask_token_id}, missing_token={missing_token_id}"
    )


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


@dataclass(frozen=True)
class EncodedEmbeddings:
    mean: np.ndarray
    tokens: np.ndarray
    concat: np.ndarray


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
    coverage_z_t: torch.Tensor,
) -> EncodedEmbeddings:
    model.eval()
    rng = np.random.default_rng(seed)

    latents_mean = np.zeros((len(indices), model.cfg.latent_dim), dtype=np.float32)
    latents_tokens = np.zeros(
        (len(indices), model.cfg.latent_tokens, model.cfg.latent_dim),
        dtype=np.float32,
    )
    latents_concat = np.zeros(
        (len(indices), model.cfg.latent_tokens * model.cfg.latent_dim),
        dtype=np.float32,
    )
    max_start = max(1, n_snps - cfg.window_size + 1)

    write_ptr = 0
    for start in range(0, len(indices), cfg.embedding_batch_size):
        end = min(start + cfg.embedding_batch_size, len(indices))
        batch_idx = indices[start:end]

        accum_tokens = np.zeros(
            (len(batch_idx), model.cfg.latent_tokens, model.cfg.latent_dim),
            dtype=np.float32,
        )
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
                coverage_std=coverage_z_t[batch["sample_idx"]],
            )
            emb_tokens = out["window_latent_tokens"].detach().cpu().numpy()
            accum_tokens += emb_tokens
            counts += 1.0

        tokens = accum_tokens / counts[:, None, None].clip(min=1.0)
        mean = tokens.mean(axis=1)
        concat = tokens.reshape(tokens.shape[0], -1)
        latents_mean[write_ptr : write_ptr + len(batch_idx)] = mean
        latents_tokens[write_ptr : write_ptr + len(batch_idx)] = tokens
        latents_concat[write_ptr : write_ptr + len(batch_idx)] = concat
        write_ptr += len(batch_idx)

    return EncodedEmbeddings(
        mean=latents_mean,
        tokens=latents_tokens,
        concat=latents_concat,
    )


def _run_probe_eval(cfg: TrainConfig, out_dir: Path) -> Dict[str, object]:
    if not cfg.probe_eval_enable:
        return {}
    if not cfg.probe_metadata_tsv:
        raise ValueError("probe_eval_enable=True requires probe_metadata_tsv to be set")

    script_path = Path(__file__).resolve().parents[1] / "evaluate_latent_probe.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Probe evaluation script not found: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--run_dir",
        str(out_dir),
        "--metadata_tsv",
        str(cfg.probe_metadata_tsv),
        "--target_col",
        str(cfg.probe_target_col),
        "--seed",
        str(cfg.probe_seed),
    ]
    subprocess.run(cmd, check=True)

    probe_summary_path = out_dir / "probe_summary.json"
    if not probe_summary_path.exists():
        raise FileNotFoundError(f"Probe summary not produced: {probe_summary_path}")
    with probe_summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_single_run_coverage_diagnostics(
    out_dir: Path,
    run_name: str,
    sample_ids: Sequence[str],
    z_mean: np.ndarray,
    coverage_fraction: np.ndarray,
    pc_count: int = 5,
) -> Dict[str, float]:
    if z_mean.shape[0] != len(sample_ids) or z_mean.shape[0] != coverage_fraction.shape[0]:
        raise ValueError(
            "Coverage diagnostics shape mismatch: "
            f"z={z_mean.shape[0]} sample_ids={len(sample_ids)} coverage={coverage_fraction.shape[0]}"
        )

    norm = np.linalg.norm(z_mean, axis=1)
    z_unit = z_mean / np.maximum(norm[:, None], 1e-12)
    pred_cov_from_z = np.zeros_like(coverage_fraction, dtype=np.float64)
    pred_cov_from_z_unit = np.zeros_like(coverage_fraction, dtype=np.float64)

    if z_mean.shape[0] > 0:
        x_aug = np.concatenate(
            [np.ones((z_mean.shape[0], 1), dtype=np.float64), z_mean.astype(np.float64, copy=False)],
            axis=1,
        )
        beta, _, _, _ = np.linalg.lstsq(x_aug, coverage_fraction.astype(np.float64, copy=False), rcond=None)
        pred_cov_from_z = x_aug @ beta

        x_u_aug = np.concatenate(
            [np.ones((z_unit.shape[0], 1), dtype=np.float64), z_unit.astype(np.float64, copy=False)],
            axis=1,
        )
        beta_u, _, _, _ = np.linalg.lstsq(
            x_u_aug,
            coverage_fraction.astype(np.float64, copy=False),
            rcond=None,
        )
        pred_cov_from_z_unit = x_u_aug @ beta_u

    r2_cov_from_z = _in_sample_r2_with_intercept(z_mean, coverage_fraction.astype(np.float64, copy=False))
    r2_cov_from_z_unit = _in_sample_r2_with_intercept(
        z_unit,
        coverage_fraction.astype(np.float64, copy=False),
    )
    pearson_pred_from_z = _pearson_corr(pred_cov_from_z, coverage_fraction)
    pearson_pred_from_z_unit = _pearson_corr(pred_cov_from_z_unit, coverage_fraction)
    pearson_norm_vs_cov = _pearson_corr(norm, coverage_fraction)

    joined_path = out_dir / f"joined_data_{run_name}.csv"
    with joined_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "sample_id,coverage_observed_fraction,norm,pred_cov_from_z,pred_cov_from_z_unit\n"
        )
        for sid, cov, nrm, pz, pzu in zip(
            sample_ids,
            coverage_fraction,
            norm,
            pred_cov_from_z,
            pred_cov_from_z_unit,
        ):
            handle.write(f"{sid},{float(cov):.7g},{float(nrm):.7g},{float(pz):.7g},{float(pzu):.7g}\n")

    probe_tsv = out_dir / "probe_summary.tsv"
    with probe_tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "\t".join(
                [
                    "run_name",
                    "n_samples",
                    "latent_dim",
                    "r2_cov_from_z",
                    "r2_cov_from_z_unit",
                    "pearson_pred_from_z",
                    "pearson_pred_from_z_unit",
                    "pearson_norm_vs_cov",
                ]
            )
            + "\n"
        )
        handle.write(
            "\t".join(
                [
                    run_name,
                    str(z_mean.shape[0]),
                    str(z_mean.shape[1]),
                    str(float(r2_cov_from_z)),
                    str(float(r2_cov_from_z_unit)),
                    str(float(pearson_pred_from_z)),
                    str(float(pearson_pred_from_z_unit)),
                    str(float(pearson_norm_vs_cov)),
                ]
            )
            + "\n"
        )

    corr_pcs: Dict[str, float] = {}
    pc_rows: List[Tuple[int, float, float]] = []
    z_center = z_mean - z_mean.mean(axis=0, keepdims=True)
    if z_center.shape[0] > 0 and z_center.shape[1] > 0:
        _, _, vt = np.linalg.svd(z_center, full_matrices=False)
        scores = z_center @ vt.T
        n_pc = min(pc_count, scores.shape[1])
        for i in range(n_pc):
            r = _pearson_corr(scores[:, i], coverage_fraction)
            corr_pcs[f"corr_pc{i+1}_coverage"] = float(r)
            # p-value is optional here; keep nan for compatibility with previous table schema.
            pc_rows.append((i + 1, float(r), float("nan")))
    for i in range(1, pc_count + 1):
        corr_pcs.setdefault(f"corr_pc{i}_coverage", float("nan"))

    pc_tsv = out_dir / "pc_coverage_corr.tsv"
    with pc_tsv.open("w", encoding="utf-8") as handle:
        handle.write("run_name\tpc_index\tpc_coverage_r\tpc_coverage_p\n")
        for pc_idx, r, p in pc_rows:
            handle.write(f"{run_name}\t{pc_idx}\t{r}\t{p}\n")

    scatter_path = out_dir / f"norm_vs_coverage_scatter_{run_name}.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.scatter(coverage_fraction, norm, s=8, alpha=0.5, linewidths=0)
        plt.xlabel("coverage_observed_fraction")
        plt.ylabel("||z||")
        plt.title(f"||z|| vs coverage ({run_name})")
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=220)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Could not render scatter plot at {scatter_path}: {exc!r}")
        scatter_path.touch()

    out = {
        "r2_coverage_from_z": float(r2_cov_from_z),
        "r2_coverage_from_z_unit": float(r2_cov_from_z_unit),
        "pearson_pred_from_z": float(pearson_pred_from_z),
        "pearson_pred_from_z_unit": float(pearson_pred_from_z_unit),
        "pearson_norm_vs_cov": float(pearson_norm_vs_cov),
    }
    out.update(corr_pcs)
    return out


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
        # Keep train/val disjoint even for tiny smoke datasets.
        if len(splits["train"]) > 1:
            fallback_n = max(1, min(32, len(splits["train"]) - 1))
            splits["val"] = splits["train"][:fallback_n].copy()
            splits["train"] = splits["train"][fallback_n:].copy()
        elif len(splits["test"]) > 0:
            fallback_n = 1
            splits["val"] = splits["test"][:fallback_n].copy()
            splits["test"] = splits["test"][fallback_n:].copy()
        else:
            # Degenerate case with only one sample available.
            splits["val"] = splits["train"].copy()

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
        latent_tokens=cfg.latent_tokens,
        latent_feedback_mode=cfg.latent_feedback_mode,
        coverage_conditioning_mode=cfg.coverage_conditioning_mode,
        coverage_embed_dim=cfg.coverage_embed_dim,
        use_obs_embedding=cfg.use_obs_embedding,
        use_snp_id_embedding=cfg.use_snp_id_embedding,
    )
    model = TokenSNPMaskedModel(model_cfg).to(device)
    adv_enabled = bool(cfg.adv_coverage_enable and cfg.lambda_adv_target > 0.0)
    cov_penalty_enabled = bool(cfg.lambda_cov_target > 0.0)
    coverage_adversary: Optional[CoverageAdversary] = None
    trainable_params: List[torch.nn.Parameter] = list(model.parameters())
    if adv_enabled:
        coverage_adversary = CoverageAdversary(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.adv_mlp_hidden_dim,
            dropout=cfg.adv_mlp_dropout,
        ).to(device)
    optimizer_main = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    optimizer_adv = (
        torch.optim.AdamW(
            coverage_adversary.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        if coverage_adversary is not None
        else None
    )

    wandb_run = None
    wandb_mode_active = None
    if cfg.wandb_enable:
        if wandb is None:
            raise RuntimeError(
                "W&B requested but wandb package is not installed. "
                "Install with: pip install wandb"
            )
        requested_mode = cfg.wandb_mode
        init_mode = requested_mode
        if requested_mode == "online":
            try:
                with socket.create_connection(("api.wandb.ai", 443), timeout=5):
                    pass
            except OSError as exc:
                print(
                    "[WARN] W&B connectivity check failed on this node; "
                    f"switching to offline mode. reason={exc!r}"
                )
                init_mode = "offline"
        wandb_mode_active = init_mode

        init_kwargs = {
            "project": cfg.wandb_project,
            "entity": cfg.wandb_entity,
            "name": cfg.wandb_name or Path(cfg.output_dir).name,
            "group": cfg.wandb_group,
            "tags": _parse_wandb_tags(cfg.wandb_tags),
            "mode": init_mode,
            "dir": str(out_dir),
            "settings": wandb.Settings(init_timeout=180),
            "config": {
                "train_cfg": asdict(cfg),
                "model_cfg": asdict(model_cfg),
                "n_samples": meta.n_samples,
                "n_snps": meta.n_snps,
                "n_classes": meta.n_classes,
                "device": str(device),
            },
        }
        try:
            wandb_run = wandb.init(**init_kwargs)
        except Exception as exc:
            if init_mode == "online":
                print(
                    "[WARN] W&B online init failed; falling back to offline mode. "
                    f"reason={exc!r}"
                )
                init_kwargs["mode"] = "offline"
                wandb_mode_active = "offline"
                wandb_run = wandb.init(**init_kwargs)
            else:
                raise

    coverage_fraction = compute_observed_fraction(geno_mm)
    train_cov = coverage_fraction[splits["train"]]
    cov_mu = float(np.mean(train_cov))
    cov_sigma = float(np.std(train_cov))
    cov_sigma = max(cov_sigma, 1e-6)
    coverage_z = (coverage_fraction - cov_mu) / cov_sigma
    coverage_z_t = torch.from_numpy(coverage_z.astype(np.float32, copy=False)).to(device=device)
    print(
        "[INFO] coverage_control "
        f"adv_enabled={int(adv_enabled)} lambda_adv_target={cfg.lambda_adv_target:.6g} "
        f"cov_penalty_enabled={int(cov_penalty_enabled)} lambda_cov_target={cfg.lambda_cov_target:.6g} "
        f"ramp_start={cfg.lambda_ramp_start_epoch} ramp_end={cfg.lambda_ramp_end_epoch} "
        f"adv_steps_per_batch={cfg.adv_steps_per_batch} cov_mu={cov_mu:.6f} cov_sigma={cov_sigma:.6f}"
    )
    sample_ids = load_sample_ids(meta.sample_ids_path, meta.n_samples)

    # Step 5: split leakage diagnostics.
    train_set = set(splits["train"].tolist())
    val_set = set(splits["val"].tolist())
    test_set = set(splits["test"].tolist())
    overlap_train_val = len(train_set & val_set)
    overlap_train_test = len(train_set & test_set)
    overlap_val_test = len(val_set & test_set)
    if overlap_train_val != 0 or overlap_train_test != 0 or overlap_val_test != 0:
        raise AssertionError(
            "Split leakage detected: "
            f"train/val={overlap_train_val}, train/test={overlap_train_test}, "
            f"val/test={overlap_val_test}"
        )

    train_first100 = [sample_ids[int(i)] for i in splits["train"][:100]]
    val_first100 = [sample_ids[int(i)] for i in splits["val"][:100]]
    sample_overlap = len(set(train_first100) & set(val_first100))

    print(
        "[DEBUG] split_info "
        f"seed={cfg.seed} n_samples={meta.n_samples} "
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} "
        f"overlap_train_val={overlap_train_val} overlap_train_test={overlap_train_test} "
        f"overlap_val_test={overlap_val_test}"
    )
    print(
        "[DEBUG] data_sources "
        f"meta_json={cfg.meta_json} memmap_path={meta.memmap_path} "
        f"sample_ids_path={meta.sample_ids_path} dtype={meta.dtype}"
    )
    print(f"[DEBUG] train_first100_sample_ids={train_first100}")
    print(f"[DEBUG] val_first100_sample_ids={val_first100}")
    print(f"[DEBUG] first100_sample_id_overlap={sample_overlap}")
    assert sample_overlap == 0, f"Unexpected overlap among first100 train/val IDs: {sample_overlap}"

    _debug_print_token_config(meta_n_classes=meta.n_classes, model=model)
    print(
        "[DEBUG] ce_scaling "
        "CrossEntropy is computed with reduction='sum' over valid masked targets "
        "then divided by n_used_for_loss; train and val use the same code path."
    )

    batch_labels = None
    if cfg.batch_labels_tsv is not None:
        batch_labels = _load_batch_labels(Path(cfg.batch_labels_tsv), sample_ids)

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history: List[Dict[str, float]] = []
    fixed_monitor_ids: Optional[np.ndarray] = None
    if cfg.monitor_every > 0:
        monitor_pool = np.concatenate([splits["train"], splits["val"], splits["test"]])
        monitor_k = min(cfg.coverage_monitor_subset, len(monitor_pool))
        rng_monitor = np.random.default_rng(cfg.seed + 7_777)
        fixed_monitor_ids = rng_monitor.choice(monitor_pool, size=monitor_k, replace=False)
        print(
            "[DEBUG] monitor_subset "
            f"mode=fixed seed={cfg.seed + 7777} size={monitor_k} "
            f"preview={fixed_monitor_ids[:20].tolist()}"
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if coverage_adversary is not None:
            coverage_adversary.train()
        lambda_adv_now = _scheduled_lambda(
            epoch=epoch,
            target=cfg.lambda_adv_target if adv_enabled else 0.0,
            start_epoch=cfg.lambda_ramp_start_epoch,
            end_epoch=cfg.lambda_ramp_end_epoch,
        )
        lambda_cov_now = _scheduled_lambda(
            epoch=epoch,
            target=cfg.lambda_cov_target if cov_penalty_enabled else 0.0,
            start_epoch=cfg.lambda_ramp_start_epoch,
            end_epoch=cfg.lambda_ramp_end_epoch,
        )
        train_loss_sum = 0.0
        train_masked = 0
        train_acc_sum = 0.0
        train_acc_count = 0
        train_adv_sum = 0.0
        train_adv_count = 0
        train_cov_pen_sum = 0.0
        train_cov_pen_count = 0
        train_eligible_total = 0
        train_masked_total = 0
        train_hist: Counter = Counter()
        printed_train_examples = False
        epoch_lambda_adv = float(lambda_adv_now)
        epoch_lambda_cov = float(lambda_cov_now)

        for step_idx in range(cfg.steps_per_epoch):
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

            cov_true = coverage_z_t[batch["sample_idx"]]
            out = model(
                tokens=batch["tokens"],
                obs_mask=batch["obs_mask"],
                snp_idx=batch["snp_idx"] if cfg.use_snp_id_embedding else None,
                coverage_std=cov_true,
            )

            valid_mask = _build_valid_mask(batch["targets"], meta.n_classes)
            n_total_tokens = int(batch["tokens"].numel())
            n_eligible_maskable = int((batch["eligible_mask"] > 0.5).sum().detach().cpu().item())
            n_masked_positions = int((batch["train_mask"] > 0.5).sum().detach().cpu().item())
            recon_loss, n_used_for_loss = _masked_recon_ce(out["logits"], batch["targets"], valid_mask)
            epoch_lambda_adv = float(lambda_adv_now)

            if coverage_adversary is not None and optimizer_adv is not None:
                z_detached = out["window_embedding"].detach()
                adv_steps = max(1, int(cfg.adv_steps_per_batch))
                for _ in range(adv_steps):
                    optimizer_adv.zero_grad(set_to_none=True)
                    cov_pred_adv = coverage_adversary(z_detached)
                    adv_detached_loss = F.mse_loss(cov_pred_adv, cov_true, reduction="mean")
                    adv_detached_loss.backward()
                    optimizer_adv.step()

            out = model(
                tokens=batch["tokens"],
                obs_mask=batch["obs_mask"],
                snp_idx=batch["snp_idx"] if cfg.use_snp_id_embedding else None,
                coverage_std=cov_true,
            )
            recon_loss, n_used_for_loss = _masked_recon_ce(out["logits"], batch["targets"], valid_mask)

            z = out["window_embedding"]
            cov_penalty = _corr_penalty_per_dim(z, cov_true) if lambda_cov_now > 0.0 else (recon_loss * 0.0)
            if coverage_adversary is not None:
                coverage_adversary.requires_grad_(False)
                z_rev = grad_reverse(z, lambda_adv_now)
                cov_pred = coverage_adversary(z_rev)
                adv_loss = F.mse_loss(cov_pred, cov_true, reduction="mean")
                coverage_adversary.requires_grad_(True)
            else:
                adv_loss = recon_loss * 0.0

            total_loss = recon_loss + adv_loss + (lambda_cov_now * cov_penalty)

            if cfg.debug_mode and (
                cfg.debug_max_batches_per_phase < 0 or step_idx < cfg.debug_max_batches_per_phase
            ):
                print(
                    "[DEBUG] batch_counts "
                    f"phase=train epoch={epoch} step={step_idx+1}/{cfg.steps_per_epoch} "
                    f"n_total_tokens={n_total_tokens} n_eligible_maskable={n_eligible_maskable} "
                    f"n_masked_positions={n_masked_positions} n_used_for_loss={n_used_for_loss}"
                )

            assert n_used_for_loss == n_masked_positions, (
                "n_used_for_loss != n_masked_positions after filtering; "
                f"used={n_used_for_loss}, masked={n_masked_positions}. "
                "If this fires, masked targets may contain invalid/special token IDs."
            )

            optimizer_main.zero_grad(set_to_none=True)
            total_loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
            optimizer_main.step()

            train_loss_sum += float(recon_loss.detach().cpu().item()) * n_used_for_loss
            train_masked += n_used_for_loss
            train_adv_sum += float(adv_loss.detach().cpu().item())
            train_adv_count += 1
            train_cov_pen_sum += float(cov_penalty.detach().cpu().item())
            train_cov_pen_count += 1
            train_eligible_total += n_eligible_maskable
            train_masked_total += n_masked_positions
            train_hist.update(_counter_from_tensor(batch["targets"][batch["train_mask"] > 0.5]))

            acc = _masked_accuracy(out["logits"], batch["targets"], valid_mask)
            if not np.isnan(acc):
                train_acc_sum += acc
                train_acc_count += 1

            if cfg.debug_mode and not printed_train_examples:
                _debug_print_masked_examples(
                    phase="train",
                    epoch=epoch,
                    step=step_idx + 1,
                    tokens=batch["tokens"],
                    train_mask=batch["train_mask"],
                    targets=batch["targets"],
                    logits=out["logits"],
                    mask_token_id=model.mask_token,
                    missing_token_id=model.missing_token,
                    n_examples=cfg.debug_examples_per_batch,
                )
                printed_train_examples = True

        train_ce = train_loss_sum / float(max(1, train_masked))
        train_acc = train_acc_sum / float(max(1, train_acc_count))
        train_adv_mse = train_adv_sum / float(max(1, train_adv_count))
        train_cov_penalty = train_cov_pen_sum / float(max(1, train_cov_pen_count))
        train_masked_rate = float(train_masked_total) / float(max(1, train_eligible_total))

        model.eval()
        if coverage_adversary is not None:
            coverage_adversary.eval()
        val_loss_sum = 0.0
        val_masked = 0
        val_acc_sum = 0.0
        val_acc_count = 0
        val_adv_sum = 0.0
        val_adv_count = 0
        val_cov_pen_sum = 0.0
        val_cov_pen_count = 0
        val_eligible_total = 0
        val_masked_total = 0
        val_hist: Counter = Counter()
        printed_val_examples = False

        with torch.no_grad():
            for step_idx in range(cfg.val_steps):
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

                cov_true = coverage_z_t[batch["sample_idx"]]
                out = model(
                    tokens=batch["tokens"],
                    obs_mask=batch["obs_mask"],
                    snp_idx=batch["snp_idx"] if cfg.use_snp_id_embedding else None,
                    coverage_std=cov_true,
                )
                valid_mask = _build_valid_mask(batch["targets"], meta.n_classes)
                n_total_tokens = int(batch["tokens"].numel())
                n_eligible_maskable = int((batch["eligible_mask"] > 0.5).sum().detach().cpu().item())
                n_masked_positions = int((batch["train_mask"] > 0.5).sum().detach().cpu().item())
                recon_loss, n_used_for_loss = _masked_recon_ce(out["logits"], batch["targets"], valid_mask)
                if coverage_adversary is not None:
                    cov_pred = coverage_adversary(out["window_embedding"])
                    adv_loss = F.mse_loss(cov_pred, cov_true, reduction="mean")
                else:
                    adv_loss = recon_loss * 0.0
                cov_penalty = _corr_penalty_per_dim(out["window_embedding"], cov_true)

                if cfg.debug_mode and (
                    cfg.debug_max_batches_per_phase < 0 or step_idx < cfg.debug_max_batches_per_phase
                ):
                    print(
                        "[DEBUG] batch_counts "
                        f"phase=val epoch={epoch} step={step_idx+1}/{cfg.val_steps} "
                        f"n_total_tokens={n_total_tokens} n_eligible_maskable={n_eligible_maskable} "
                        f"n_masked_positions={n_masked_positions} n_used_for_loss={n_used_for_loss}"
                    )

                assert n_used_for_loss == n_masked_positions, (
                    "n_used_for_loss != n_masked_positions after filtering (val); "
                    f"used={n_used_for_loss}, masked={n_masked_positions}"
                )

                val_loss_sum += float(recon_loss.detach().cpu().item()) * n_used_for_loss
                val_masked += n_used_for_loss
                val_adv_sum += float(adv_loss.detach().cpu().item())
                val_adv_count += 1
                val_cov_pen_sum += float(cov_penalty.detach().cpu().item())
                val_cov_pen_count += 1
                val_eligible_total += n_eligible_maskable
                val_masked_total += n_masked_positions
                val_hist.update(_counter_from_tensor(batch["targets"][batch["train_mask"] > 0.5]))

                acc = _masked_accuracy(out["logits"], batch["targets"], valid_mask)
                if not np.isnan(acc):
                    val_acc_sum += acc
                    val_acc_count += 1

                if cfg.debug_mode and not printed_val_examples:
                    _debug_print_masked_examples(
                        phase="val",
                        epoch=epoch,
                        step=step_idx + 1,
                        tokens=batch["tokens"],
                        train_mask=batch["train_mask"],
                        targets=batch["targets"],
                        logits=out["logits"],
                        mask_token_id=model.mask_token,
                        missing_token_id=model.missing_token,
                        n_examples=cfg.debug_examples_per_batch,
                    )
                    printed_val_examples = True

        val_ce = val_loss_sum / float(max(1, val_masked))
        val_acc = val_acc_sum / float(max(1, val_acc_count))
        val_adv_mse = val_adv_sum / float(max(1, val_adv_count))
        val_cov_penalty = val_cov_pen_sum / float(max(1, val_cov_pen_count))
        val_masked_rate = float(val_masked_total) / float(max(1, val_eligible_total))

        if cfg.debug_mode:
            train_hist_total = sum(train_hist.values())
            val_hist_total = sum(val_hist.values())
            print(
                "[DEBUG] target_hist "
                f"phase=train epoch={epoch} total_masked_targets={train_hist_total} "
                f"top20={_debug_top_hist(train_hist, train_hist_total, top_k=20)}"
            )
            print(
                "[DEBUG] target_hist "
                f"phase=val epoch={epoch} total_masked_targets={val_hist_total} "
                f"top20={_debug_top_hist(val_hist, val_hist_total, top_k=20)}"
            )
            print(
                "[DEBUG] masked_rate "
                f"epoch={epoch} train_masked_rate={train_masked_rate:.6f} "
                f"val_masked_rate={val_masked_rate:.6f} "
                f"train_masked={train_masked_total}/{max(1, train_eligible_total)} "
                f"val_masked={val_masked_total}/{max(1, val_eligible_total)}"
            )

        cov_corr = float("nan")
        batch_r2 = float("nan")
        if cfg.monitor_every > 0 and (epoch % cfg.monitor_every == 0):
            assert fixed_monitor_ids is not None
            monitor_ids = fixed_monitor_ids

            lat = _encode_sample_embeddings(
                model=model,
                geno_mm=geno_mm,
                indices=monitor_ids,
                cfg=cfg,
                n_classes=meta.n_classes,
                n_snps=meta.n_snps,
                device=device,
                # Keep embedding windows fixed across epochs for comparable confound monitoring.
                seed=cfg.seed + 8_888,
                coverage_z_t=coverage_z_t,
            )
            lat_norm = np.linalg.norm(lat.mean, axis=1)
            cov_corr = _pearson_corr(lat_norm, coverage_fraction[monitor_ids])
            if batch_labels is not None:
                batch_r2 = _batch_r2(lat_norm, batch_labels[monitor_ids])

        row = {
            "epoch": float(epoch),
            "train_ce": float(train_ce),
            "val_ce": float(val_ce),
            "train_masked_acc": float(train_acc),
            "val_masked_acc": float(val_acc),
            "train_adv_mse": float(train_adv_mse),
            "val_adv_mse": float(val_adv_mse),
            "lambda_adv": float(epoch_lambda_adv if adv_enabled else 0.0),
            "lambda_cov": float(epoch_lambda_cov if cov_penalty_enabled else 0.0),
            "cov_penalty": float(train_cov_penalty),
            "val_cov_penalty": float(val_cov_penalty),
            "adv_steps_per_batch": float(max(1, int(cfg.adv_steps_per_batch))),
            "train_masked_tokens": float(train_masked),
            "val_masked_tokens": float(val_masked),
            "train_masked_rate": float(train_masked_rate),
            "val_masked_rate": float(val_masked_rate),
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
                    "coverage_adversary_state": (
                        coverage_adversary.state_dict() if coverage_adversary is not None else None
                    ),
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
    if coverage_adversary is not None and ckpt.get("coverage_adversary_state") is not None:
        coverage_adversary.load_state_dict(ckpt["coverage_adversary_state"])

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
        coverage_z_t=coverage_z_t,
    )

    np.save(out_dir / "global_latents.npy", global_latents.mean)
    np.save(out_dir / "global_latent_tokens.npy", global_latents.tokens)
    np.save(out_dir / "global_latents_concat.npy", global_latents.concat)

    with (out_dir / "global_latents.csv").open("w", encoding="utf-8") as handle:
        header = ["sample_id"] + [f"z{i}" for i in range(global_latents.mean.shape[1])]
        handle.write(",".join(header) + "\n")
        for sid, vec in zip(sample_ids, global_latents.mean):
            row = [sid] + [f"{float(v):.7g}" for v in vec]
            handle.write(",".join(row) + "\n")

    with (out_dir / "coverage_observed_fraction.csv").open("w", encoding="utf-8") as handle:
        handle.write("sample_id,coverage_observed_fraction\n")
        for sid, cov in zip(sample_ids, coverage_fraction):
            handle.write(f"{sid},{float(cov):.7g}\n")

    coverage_corr_mean = _pearson_corr(np.linalg.norm(global_latents.mean, axis=1), coverage_fraction)
    coverage_corr_concat = _pearson_corr(
        np.linalg.norm(global_latents.concat, axis=1),
        coverage_fraction,
    )
    coverage_corr_tokens: Dict[str, float] = {}
    for tok_i in range(global_latents.tokens.shape[1]):
        tok_norm = np.linalg.norm(global_latents.tokens[:, tok_i, :], axis=1)
        coverage_corr_tokens[f"token_{tok_i}"] = _pearson_corr(tok_norm, coverage_fraction)

    z_mean = global_latents.mean.astype(np.float64, copy=False)
    run_name = Path(cfg.output_dir).name
    coverage_diag = _write_single_run_coverage_diagnostics(
        out_dir=out_dir,
        run_name=run_name,
        sample_ids=sample_ids,
        z_mean=z_mean,
        coverage_fraction=coverage_fraction.astype(np.float64, copy=False),
        pc_count=5,
    )
    r2_coverage_from_z = float(coverage_diag.get("r2_coverage_from_z", float("nan")))
    r2_coverage_from_z_unit = float(coverage_diag.get("r2_coverage_from_z_unit", float("nan")))
    corr_pc_metrics = {
        f"corr_pc{i}_coverage": float(coverage_diag.get(f"corr_pc{i}_coverage", float("nan")))
        for i in range(1, 6)
    }

    probe_summary: Dict[str, object] = {}
    if cfg.probe_eval_enable:
        probe_summary = _run_probe_eval(cfg=cfg, out_dir=out_dir)

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
        "final_representation_metrics": {
            "coverage_corr_mean": float(coverage_corr_mean),
            "coverage_corr_concat": float(coverage_corr_concat),
            "coverage_corr_tokens": coverage_corr_tokens,
            "r2_coverage_from_z": float(r2_coverage_from_z),
            "r2_coverage_from_z_unit": float(r2_coverage_from_z_unit),
            **corr_pc_metrics,
        },
        "probe": probe_summary,
        "wandb": {
            "enabled": bool(cfg.wandb_enable),
            "run_name": getattr(wandb_run, "name", None) if wandb_run is not None else None,
            "run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
            "run_url": getattr(wandb_run, "url", None) if wandb_run is not None else None,
            "mode": wandb_mode_active,
        },
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if wandb_run is not None:
        probe_metrics = probe_summary.get("metrics", {}) if isinstance(probe_summary, dict) else {}
        probe_log = {}
        if isinstance(probe_metrics, dict):
            for key, values in probe_metrics.items():
                if not isinstance(values, dict):
                    continue
                if "macro_f1" in values:
                    probe_log[f"probe_macro_f1_{key}"] = float(values["macro_f1"])
                if "accuracy" in values:
                    probe_log[f"probe_accuracy_{key}"] = float(values["accuracy"])
        wandb_run.summary.update(
            {
                "best_epoch": best_epoch,
                "best_val_ce": best_val,
                "coverage_corr_mean": float(coverage_corr_mean),
                "coverage_corr_concat": float(coverage_corr_concat),
                "r2_coverage_from_z": float(r2_coverage_from_z),
                "r2_coverage_from_z_unit": float(r2_coverage_from_z_unit),
                **corr_pc_metrics,
                **{f"coverage_corr_token_{k.split('_')[-1]}": float(v) for k, v in coverage_corr_tokens.items()},
                **probe_log,
            }
        )
        wandb_run.finish()

    return summary
