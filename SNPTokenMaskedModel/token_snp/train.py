from __future__ import annotations

import json
from collections import Counter
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
        train_loss_sum = 0.0
        train_masked = 0
        train_acc_sum = 0.0
        train_acc_count = 0
        train_eligible_total = 0
        train_masked_total = 0
        train_hist: Counter = Counter()
        printed_train_examples = False

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

            out = model(
                tokens=batch["tokens"],
                obs_mask=batch["obs_mask"],
                snp_idx=batch["snp_idx"] if cfg.use_snp_id_embedding else None,
            )

            valid_mask = _build_valid_mask(batch["targets"], meta.n_classes)
            n_total_tokens = int(batch["tokens"].numel())
            n_eligible_maskable = int((batch["eligible_mask"] > 0.5).sum().detach().cpu().item())
            n_masked_positions = int((batch["train_mask"] > 0.5).sum().detach().cpu().item())
            loss, n_used_for_loss = _masked_recon_ce(out["logits"], batch["targets"], valid_mask)

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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            train_loss_sum += float(loss.detach().cpu().item()) * n_used_for_loss
            train_masked += n_used_for_loss
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
        train_masked_rate = float(train_masked_total) / float(max(1, train_eligible_total))

        model.eval()
        val_loss_sum = 0.0
        val_masked = 0
        val_acc_sum = 0.0
        val_acc_count = 0
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

                out = model(
                    tokens=batch["tokens"],
                    obs_mask=batch["obs_mask"],
                    snp_idx=batch["snp_idx"] if cfg.use_snp_id_embedding else None,
                )
                valid_mask = _build_valid_mask(batch["targets"], meta.n_classes)
                n_total_tokens = int(batch["tokens"].numel())
                n_eligible_maskable = int((batch["eligible_mask"] > 0.5).sum().detach().cpu().item())
                n_masked_positions = int((batch["train_mask"] > 0.5).sum().detach().cpu().item())
                loss, n_used_for_loss = _masked_recon_ce(out["logits"], batch["targets"], valid_mask)

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

                val_loss_sum += float(loss.detach().cpu().item()) * n_used_for_loss
                val_masked += n_used_for_loss
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
