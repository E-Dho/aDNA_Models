from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .data import (
    MISSING_VALUE,
    GenotypeMemmapMeta,
    compute_observed_fraction,
    load_meta,
    load_sample_ids,
    load_sample_ids_from_meta,
    load_sample_stats_tsv,
    open_genotype_memmap,
    save_meta,
    write_tsv,
)
from .eval import (
    evaluate_embedding,
    nearest_neighbors,
    read_latents_csv,
    ridge_probe_r2,
    write_latents_csv,
)
from .model import require_torch


@dataclass(frozen=True)
class SnpRecord:
    teacher_index: int
    snp_id: str
    chrom: str
    genetic_pos: str
    physical_pos: str
    allele1: str
    allele2: str


@dataclass(frozen=True)
class AlignmentReport:
    n_teacher_snps: int
    n_source_snps: int
    n_aligned_snps: int
    n_missing_snps: int
    n_extra_source_snps: int
    n_allele_order_mismatch: int
    n_allele_set_mismatch: int
    min_overlap_snps: int
    min_observed_fraction: float


@dataclass(frozen=True)
class EncoderConfig:
    n_snps: int
    latent_dim: int
    hidden_dim: int
    ploidy: str
    snp_dropout: float
    low_overlap_fraction: float
    low_overlap_snps: int
    model_type: str = "sparse_residual_encoder"
    deep_hidden_dim: int = 100000
    deep_bottleneck_dim: int = 1024
    sparse_fanout: int = 2
    sparse_init: str = "fan_in"
    dst_enable: bool = False
    dst_update_interval_steps: int = 100
    dst_start_epoch: int = 5
    dst_end_epoch: int = 80
    dst_initial_drop_fraction: float = 0.30
    dst_final_drop_fraction: float = 0.05
    dst_candidate_multiplier: int = 16
    dst_grow_init: str = "zero"
    dst_score: str = "genotype_plus_mask_grad"


def prefixed_path(prefix: Path, extension: str) -> Path:
    """Return EIGENSTRAT-style prefix + extension without treating dots as suffixes."""
    if not extension.startswith("."):
        extension = f".{extension}"
    return Path(f"{prefix}{extension}")


def _parse_snp_line(raw: str, index: int) -> SnpRecord:
    parts = raw.strip().split()
    if len(parts) < 6:
        raise ValueError(f"Expected at least 6 columns in .snp line {index + 1}, got: {raw!r}")
    return SnpRecord(
        teacher_index=index,
        snp_id=parts[0],
        chrom=parts[1],
        genetic_pos=parts[2],
        physical_pos=parts[3],
        allele1=parts[4],
        allele2=parts[5],
    )


def read_snp_records(snp_path: Path) -> List[SnpRecord]:
    records: List[SnpRecord] = []
    with snp_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.strip():
                records.append(_parse_snp_line(raw, len(records)))
    if not records:
        raise ValueError(f"No SNP records found in {snp_path}")
    return records


def write_snp_manifest(path: Path, records: Sequence[SnpRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "teacher_index",
                "snp_id",
                "chrom",
                "genetic_pos",
                "physical_pos",
                "allele1",
                "allele2",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def load_snp_manifest(path: Path) -> List[SnpRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: List[SnpRecord] = []
        for row in reader:
            rows.append(
                SnpRecord(
                    teacher_index=int(row["teacher_index"]),
                    snp_id=row["snp_id"],
                    chrom=row["chrom"],
                    genetic_pos=row["genetic_pos"],
                    physical_pos=row["physical_pos"],
                    allele1=row["allele1"],
                    allele2=row["allele2"],
                )
            )
    if not rows:
        raise ValueError(f"No SNP records found in manifest {path}")
    rows = sorted(rows, key=lambda item: item.teacher_index)
    expected = list(range(len(rows)))
    found = [row.teacher_index for row in rows]
    if found != expected:
        raise ValueError("SNP manifest teacher_index must be contiguous and zero-based")
    return rows


def _map_genotype_line(line: str, ploidy: str) -> np.ndarray:
    vals = np.frombuffer(line.encode("ascii"), dtype=np.uint8).astype(np.int16) - ord("0")
    out = np.full(vals.shape, MISSING_VALUE, dtype=np.int8)
    valid = (vals != 9) & ((vals == 0) | (vals == 1) | (vals == 2))
    if ploidy == "diploid":
        out[valid] = vals[valid].astype(np.int8)
    elif ploidy == "pseudohaploid":
        out[valid] = (vals[valid] > 0).astype(np.int8)
    else:
        raise ValueError("ploidy must be one of: diploid, pseudohaploid")
    return out


def align_eigenstrat_to_manifest(
    *,
    geno_path: Path,
    snp_path: Path,
    ind_path: Path,
    manifest_path: Path,
    out_prefix: Path,
    ploidy: str = "pseudohaploid",
    low_overlap_fraction: float = 0.05,
    low_overlap_snps: int = 0,
) -> Tuple[GenotypeMemmapMeta, AlignmentReport]:
    manifest = load_snp_manifest(manifest_path)
    source_records = read_snp_records(snp_path)
    source_by_id: Dict[str, Tuple[int, SnpRecord]] = {
        record.snp_id: (idx, record) for idx, record in enumerate(source_records)
    }
    first = ""
    with geno_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.strip():
                first = raw.strip()
                break
    if not first:
        raise ValueError(f"Empty genotype file: {geno_path}")
    n_samples = len(first)
    sample_ids = load_sample_ids(ind_path, expected_n=n_samples)
    n_teacher_snps = len(manifest)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    memmap_path = out_prefix.with_suffix(".genotypes.int8.memmap")
    aligned = np.memmap(memmap_path, dtype=np.int8, mode="w+", shape=(n_samples, n_teacher_snps))
    aligned[:, :] = MISSING_VALUE

    target_for_source: Dict[int, int] = {}
    allele_order_mismatch = 0
    allele_set_mismatch = 0
    for target_idx, teacher in enumerate(manifest):
        found = source_by_id.get(teacher.snp_id)
        if found is None:
            continue
        source_idx, source = found
        teacher_pair = (teacher.allele1, teacher.allele2)
        source_pair = (source.allele1, source.allele2)
        if teacher_pair != source_pair:
            if set(teacher_pair) == set(source_pair):
                allele_order_mismatch += 1
            else:
                allele_set_mismatch += 1
                continue
        target_for_source[source_idx] = target_idx

    with geno_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for source_idx, raw in enumerate(handle):
            line = raw.strip()
            if not line:
                continue
            if len(line) != n_samples:
                raise ValueError(
                    f"Inconsistent .geno width at source SNP {source_idx}: {len(line)} vs {n_samples}"
                )
            target_idx = target_for_source.get(source_idx)
            if target_idx is None:
                continue
            aligned[:, target_idx] = _map_genotype_line(line, ploidy)
    aligned.flush()

    sample_ids_path = out_prefix.with_suffix(".sample_ids.txt")
    with sample_ids_path.open("w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            handle.write(f"{sample_id}\n")
    meta = GenotypeMemmapMeta(
        memmap_path=str(memmap_path),
        n_samples=n_samples,
        n_snps=n_teacher_snps,
        dtype="int8",
        ploidy=ploidy,
        n_classes=3 if ploidy == "diploid" else 2,
        missing_value=MISSING_VALUE,
        sample_ids_path=str(sample_ids_path),
    )
    save_meta(out_prefix.with_suffix(".meta.json"), meta)
    min_overlap_snps = int(low_overlap_snps)
    if min_overlap_snps <= 0:
        min_overlap_snps = int(math.ceil(float(low_overlap_fraction) * n_teacher_snps))
    report = AlignmentReport(
        n_teacher_snps=n_teacher_snps,
        n_source_snps=len(source_records),
        n_aligned_snps=len(target_for_source),
        n_missing_snps=n_teacher_snps - len(target_for_source),
        n_extra_source_snps=max(len(source_records) - len(target_for_source), 0),
        n_allele_order_mismatch=allele_order_mismatch,
        n_allele_set_mismatch=allele_set_mismatch,
        min_overlap_snps=min_overlap_snps,
        min_observed_fraction=float(low_overlap_fraction),
    )
    with out_prefix.with_suffix(".alignment_report.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2)
    return meta, report


def prepare_encoder_reference(
    *,
    data_prefix: Path,
    teacher_run_dir: Path,
    graph_dir: Path,
    out_dir: Path,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "snp_manifest.tsv"
    records = read_snp_records(prefixed_path(data_prefix, ".snp"))
    write_snp_manifest(manifest_path, records)
    for src_name, dst_name in (
        ("final_latents.csv", "teacher_latents.csv"),
        ("run_summary.json", "teacher_run_summary.json"),
    ):
        src = teacher_run_dir / src_name
        if src.exists():
            shutil.copy2(src, out_dir / dst_name)
    graph_stats = graph_dir / "sample_stats.tsv"
    if graph_stats.exists():
        shutil.copy2(graph_stats, out_dir / "teacher_sample_stats.tsv")
    config = {
        "data_prefix": str(data_prefix),
        "teacher_run_dir": str(teacher_run_dir),
        "graph_dir": str(graph_dir),
        "snp_manifest": str(manifest_path),
        "teacher_latents_csv": str(out_dir / "teacher_latents.csv"),
        "teacher_sample_stats_tsv": str(out_dir / "teacher_sample_stats.tsv"),
    }
    with (out_dir / "encoder_reference_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return config


class SafeWandbRun:
    def __init__(
        self,
        *,
        enabled: bool,
        mode: str,
        output_dir: Path,
        project: str,
        name: str,
        group: str,
        tags: str,
        init_timeout: int,
        config: Dict[str, object],
    ) -> None:
        self.run = None
        self.note = ""
        self.mode = mode
        if not enabled or mode == "disabled":
            return
        try:
            import wandb
        except ImportError:
            self.note = "wandb import unavailable"
            return
        try:
            settings = wandb.Settings(init_timeout=max(int(init_timeout), 1), _disable_stats=True)
            self.run = wandb.init(
                project=project,
                name=name,
                group=group or None,
                tags=[tag.strip() for tag in tags.split(",") if tag.strip()],
                mode=mode,
                dir=str(output_dir),
                settings=settings,
                config=config,
            )
        except Exception as exc:
            self.note = f"wandb init failed: {exc}"
            if mode == "online":
                try:
                    self.run = wandb.init(
                        project=project,
                        name=name,
                        group=group or None,
                        tags=[tag.strip() for tag in tags.split(",") if tag.strip()],
                        mode="offline",
                        dir=str(output_dir),
                        settings=wandb.Settings(init_timeout=5, _disable_stats=True),
                        config={**config, "wandb_online_fallback": True},
                    )
                    self.mode = "offline"
                    self.note = f"wandb online init failed; continued offline: {exc}"
                except Exception as fallback_exc:
                    self.note = f"wandb init failed; offline fallback failed: {fallback_exc}"

    def log(self, payload: Dict[str, float]) -> None:
        if self.run is not None:
            try:
                self.run.log(payload)
            except Exception as exc:
                self.note = f"wandb log failed: {exc}"

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def build_encoder_model(
    n_snps: int,
    latent_dim: int,
    hidden_dim: int,
    *,
    model_type: str = "sparse_residual_encoder",
    deep_hidden_dim: int = 100000,
    deep_bottleneck_dim: int = 1024,
    sparse_fanout: int = 2,
    sparse_init: str = "fan_in",
    dst_enable: bool = False,
    dst_update_interval_steps: int = 100,
    dst_start_epoch: int = 5,
    dst_end_epoch: int = 80,
    dst_initial_drop_fraction: float = 0.30,
    dst_final_drop_fraction: float = 0.05,
    dst_candidate_multiplier: int = 16,
    dst_grow_init: str = "zero",
    dst_score: str = "genotype_plus_mask_grad",
):
    torch, nn = require_torch()

    class SparseResidualEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scale = 1.0 / math.sqrt(max(n_snps, 1))
            self.genotype_weight = nn.Parameter(torch.randn(n_snps, latent_dim) * scale)
            self.mask_weight = nn.Parameter(torch.randn(n_snps, latent_dim) * scale)
            self.bias = nn.Parameter(torch.zeros(latent_dim))
            self.head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )

        def forward(self, values, mask):
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0).sqrt()
            base = (values.matmul(self.genotype_weight) + mask.matmul(self.mask_weight)) / denom
            base = base + self.bias
            return base + self.head(base)

    class SparseDeepEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if int(deep_hidden_dim) < int(deep_bottleneck_dim):
                raise ValueError("deep_hidden_dim should be >= deep_bottleneck_dim")
            if int(sparse_fanout) < 1:
                raise ValueError("sparse_fanout must be >= 1")
            if sparse_init not in {"legacy", "fan_in"}:
                raise ValueError("sparse_init must be one of: legacy, fan_in")
            if dst_grow_init != "zero":
                raise ValueError("dst_grow_init must be 'zero'")
            if dst_score != "genotype_plus_mask_grad":
                raise ValueError("dst_score must be 'genotype_plus_mask_grad'")
            fanout = int(sparse_fanout)
            hidden = int(deep_hidden_dim)
            self.n_snps = int(n_snps)
            self.latent_dim = int(latent_dim)
            self.hidden_dim = hidden
            self.fanout = fanout
            self.sparse_init = sparse_init
            self.dst_enable = bool(dst_enable)
            self.dst_update_interval_steps = max(int(dst_update_interval_steps), 1)
            self.dst_start_epoch = int(dst_start_epoch)
            self.dst_end_epoch = int(dst_end_epoch)
            self.dst_initial_drop_fraction = float(dst_initial_drop_fraction)
            self.dst_final_drop_fraction = float(dst_final_drop_fraction)
            self.dst_candidate_multiplier = max(int(dst_candidate_multiplier), 1)
            self.dst_grow_init = dst_grow_init
            self.dst_score = dst_score
            self.dst_updates = 0
            self.dst_last_drop_fraction = 0.0
            self.dst_last_pruned = 0
            self.dst_last_grown = 0
            self._last_hidden_grad = None

            # Each SNP connects to a few deterministic hidden buckets. This keeps the
            # 584k -> 100k layer sparse instead of creating an impossible dense matrix.
            snp_idx = torch.arange(int(n_snps), dtype=torch.long)
            bucket_rows = []
            for offset in range(fanout):
                bucket_rows.append((snp_idx * (2_654_435_761 + 97 * offset) + 1_013_904_223 * offset) % hidden)
            bucket_idx = torch.stack(bucket_rows, dim=0)
            if fanout > 1 and hidden > 1:
                for row in range(1, fanout):
                    duplicate = (bucket_idx[row : row + 1] == bucket_idx[:row]).any(dim=0)
                    attempts = 0
                    while bool(duplicate.any()) and attempts < hidden:
                        bucket_idx[row, duplicate] = (bucket_idx[row, duplicate] + 1) % hidden
                        duplicate = (bucket_idx[row : row + 1] == bucket_idx[:row]).any(dim=0)
                        attempts += 1
            self.register_buffer("bucket_idx", bucket_idx, persistent=True)

            connection_scale = self._connection_init_scale(bucket_idx, sparse_init)
            self.register_buffer("connection_init_scale", connection_scale, persistent=False)
            self.genotype_bucket_weight = nn.Parameter(torch.randn(fanout, n_snps) * connection_scale)
            self.mask_bucket_weight = nn.Parameter(torch.randn(fanout, n_snps) * connection_scale)
            self.hidden_bias = nn.Parameter(torch.zeros(hidden))
            self.net = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, int(deep_bottleneck_dim)),
                nn.GELU(),
                nn.LayerNorm(int(deep_bottleneck_dim)),
                nn.Linear(int(deep_bottleneck_dim), int(latent_dim)),
            )

        def _connection_init_scale(self, bucket_idx, init_mode: str):
            if init_mode == "legacy":
                scale = 1.0 / math.sqrt(max(int(n_snps) * int(sparse_fanout) / max(int(deep_hidden_dim), 1), 1.0))
                return torch.full(bucket_idx.shape, float(scale), dtype=torch.float32)
            counts = torch.bincount(bucket_idx.reshape(-1), minlength=int(deep_hidden_dim)).to(torch.float32)
            effective_fan_in = (2.0 * counts).clamp_min(1.0)
            return torch.sqrt(2.0 / effective_fan_in[bucket_idx])

        def sparse_fan_in_stats(self) -> Dict[str, float]:
            counts = torch.bincount(self.bucket_idx.detach().reshape(-1).cpu(), minlength=self.hidden_dim).to(torch.float32)
            effective = counts * 2.0
            return {
                "sparse_fan_in_min": float(effective.min().item()) if effective.numel() else 0.0,
                "sparse_fan_in_mean": float(effective.mean().item()) if effective.numel() else 0.0,
                "sparse_fan_in_max": float(effective.max().item()) if effective.numel() else 0.0,
                "sparse_fan_in_std": float(effective.std(unbiased=False).item()) if effective.numel() else 0.0,
            }

        def dst_metrics(self) -> Dict[str, float]:
            return {
                "dst_enabled": float(bool(self.dst_enable)),
                "dst_updates": float(self.dst_updates),
                "dst_last_drop_fraction": float(self.dst_last_drop_fraction),
                "dst_last_pruned": float(self.dst_last_pruned),
                "dst_last_grown": float(self.dst_last_grown),
                "sparse_init": 1.0 if self.sparse_init == "fan_in" else 0.0,
                **self.sparse_fan_in_stats(),
            }

        def _drop_fraction_for_epoch(self, epoch: int) -> float:
            if self.dst_end_epoch <= self.dst_start_epoch:
                return self.dst_final_drop_fraction
            progress = (float(epoch) - float(self.dst_start_epoch)) / float(self.dst_end_epoch - self.dst_start_epoch)
            progress = min(max(progress, 0.0), 1.0)
            return self.dst_initial_drop_fraction + progress * (self.dst_final_drop_fraction - self.dst_initial_drop_fraction)

        def _score_growth_candidates(self, values, mask, snp_indices, candidate_buckets):
            if self._last_hidden_grad is None:
                return None
            grad = self._last_hidden_grad.to(values.device)
            batch_size = values.shape[0]
            chunk_size = 4096
            scores = []
            for lo in range(0, int(snp_indices.numel()), chunk_size):
                hi = min(lo + chunk_size, int(snp_indices.numel()))
                snp_chunk = snp_indices[lo:hi]
                cand_chunk = candidate_buckets[lo:hi]
                grad_chunk = grad[:, cand_chunk.reshape(-1)].reshape(batch_size, hi - lo, cand_chunk.shape[1])
                value_chunk = values[:, snp_chunk].unsqueeze(2)
                mask_chunk = mask[:, snp_chunk].unsqueeze(2)
                score = (grad_chunk * value_chunk).sum(dim=0).abs() + (grad_chunk * mask_chunk).sum(dim=0).abs()
                scores.append(score)
            return torch.cat(scores, dim=0) if scores else values.new_zeros((0, candidate_buckets.shape[1]))

        def _reset_optimizer_slots(self, optimizer, flat_slots):
            if optimizer is None or flat_slots.numel() == 0:
                return
            for param in (self.genotype_bucket_weight, self.mask_bucket_weight):
                state = optimizer.state.get(param, {})
                for value in state.values():
                    if torch.is_tensor(value) and value.numel() == param.numel():
                        value.reshape(-1)[flat_slots] = 0

        @torch.no_grad()
        def dynamic_sparse_update(self, *, values, mask, optimizer, drop_fraction: float, candidate_multiplier: int):
            total_slots = int(self.genotype_bucket_weight.numel())
            n_prune = int(round(total_slots * float(drop_fraction)))
            if n_prune <= 0 or self.hidden_dim <= self.fanout:
                return {"pruned": 0, "grown": 0}
            n_prune = min(n_prune, total_slots)
            salience = (self.genotype_bucket_weight.detach().abs() + self.mask_bucket_weight.detach().abs()).reshape(-1)
            pruned_flat = torch.topk(salience, k=n_prune, largest=False).indices
            fan_idx = torch.div(pruned_flat, self.n_snps, rounding_mode="floor")
            snp_idx = pruned_flat % self.n_snps
            cand_mult = max(int(candidate_multiplier), 1)
            candidate_buckets = torch.randint(0, self.hidden_dim, (n_prune, cand_mult), device=values.device)
            existing = self.bucket_idx[:, snp_idx].transpose(0, 1).to(values.device)
            invalid = (candidate_buckets.unsqueeze(2) == existing.unsqueeze(1)).any(dim=2)
            attempts = 0
            while bool(invalid.any()) and attempts < 8:
                candidate_buckets[invalid] = torch.randint(0, self.hidden_dim, (int(invalid.sum().item()),), device=values.device)
                invalid = (candidate_buckets.unsqueeze(2) == existing.unsqueeze(1)).any(dim=2)
                attempts += 1
            if bool(invalid.any()):
                candidate_buckets[invalid] = (candidate_buckets[invalid] + self.fanout + 1) % self.hidden_dim
            scores = self._score_growth_candidates(values, mask, snp_idx, candidate_buckets)
            if scores is None:
                return {"pruned": 0, "grown": 0}
            best = scores.argmax(dim=1)
            new_bucket = candidate_buckets[torch.arange(n_prune, device=values.device), best].to(self.bucket_idx.device)
            fan_idx_cpu = fan_idx.to(self.bucket_idx.device)
            snp_idx_cpu = snp_idx.to(self.bucket_idx.device)
            existing_cpu = self.bucket_idx[:, snp_idx_cpu].transpose(0, 1)
            duplicate_new = (new_bucket[:, None] == existing_cpu).any(dim=1)
            if bool(duplicate_new.any()):
                bad = torch.flatnonzero(duplicate_new).tolist()
                for local_idx in bad:
                    occupied = set(int(x) for x in existing_cpu[local_idx].tolist())
                    for candidate in range(self.hidden_dim):
                        if candidate not in occupied:
                            new_bucket[local_idx] = candidate
                            break
            self.bucket_idx[fan_idx_cpu, snp_idx_cpu] = new_bucket
            self.genotype_bucket_weight.reshape(-1)[pruned_flat] = 0.0
            self.mask_bucket_weight.reshape(-1)[pruned_flat] = 0.0
            self._reset_optimizer_slots(optimizer, pruned_flat)
            self.dst_updates += 1
            self.dst_last_drop_fraction = float(drop_fraction)
            self.dst_last_pruned = int(n_prune)
            self.dst_last_grown = int(n_prune)
            return {"pruned": int(n_prune), "grown": int(n_prune)}

        def maybe_dynamic_sparse_update(self, *, global_step: int, epoch: int, optimizer, values, mask):
            if not self.dst_enable:
                return {"pruned": 0, "grown": 0}
            if epoch < self.dst_start_epoch or epoch > self.dst_end_epoch:
                return {"pruned": 0, "grown": 0}
            if int(global_step) % self.dst_update_interval_steps != 0:
                return {"pruned": 0, "grown": 0}
            drop_fraction = self._drop_fraction_for_epoch(epoch)
            return self.dynamic_sparse_update(
                values=values,
                mask=mask,
                optimizer=optimizer,
                drop_fraction=drop_fraction,
                candidate_multiplier=self.dst_candidate_multiplier,
            )

        def forward(self, values, mask):
            batch = values.shape[0]
            hidden = values.new_zeros((batch, int(deep_hidden_dim)))
            for fanout_idx in range(int(sparse_fanout)):
                idx = self.bucket_idx[fanout_idx].unsqueeze(0).expand(batch, -1)
                contrib = (
                    values * self.genotype_bucket_weight[fanout_idx].unsqueeze(0)
                    + mask * self.mask_bucket_weight[fanout_idx].unsqueeze(0)
                )
                hidden.scatter_add_(dim=1, index=idx, src=contrib)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0).sqrt()
            hidden = hidden / denom + self.hidden_bias.unsqueeze(0)
            if hidden.requires_grad:
                hidden.register_hook(lambda grad: setattr(self, "_last_hidden_grad", grad.detach()))
            return self.net(hidden)

    if model_type == "sparse_residual_encoder":
        return SparseResidualEncoder()
    if model_type == "sparse_deep_encoder":
        return SparseDeepEncoder()
    raise ValueError("model_type must be one of: sparse_residual_encoder, sparse_deep_encoder")


def _batch_to_tensors(block: np.ndarray, device: str, snp_dropout: float, training: bool):
    torch, _ = require_torch()
    observed = block >= 0
    values = np.where(observed, block, 0).astype(np.float32)
    mask = observed.astype(np.float32)
    values_t = torch.tensor(values, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
    if training and snp_dropout > 0.0:
        keep = torch.rand_like(mask_t) >= float(snp_dropout)
        mask_t = mask_t * keep
        values_t = values_t * keep
    return values_t, mask_t


def _align_targets_to_meta(meta: GenotypeMemmapMeta, teacher_latents_csv: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    all_sample_ids = load_sample_ids_from_meta(meta)
    latent_ids, latents = read_latents_csv(teacher_latents_csv)
    latent_by_id = {sample_id: i for i, sample_id in enumerate(latent_ids)}
    selected_meta_idx: List[int] = []
    selected_ids: List[str] = []
    selected_latents: List[np.ndarray] = []
    for idx, sample_id in enumerate(all_sample_ids):
        latent_idx = latent_by_id.get(sample_id)
        if latent_idx is None:
            continue
        selected_meta_idx.append(idx)
        selected_ids.append(sample_id)
        selected_latents.append(latents[latent_idx])
    if not selected_ids:
        raise ValueError("No overlapping sample IDs between memmap metadata and teacher latents")
    return (
        np.asarray(selected_meta_idx, dtype=np.int64),
        selected_ids,
        np.asarray(selected_latents, dtype=np.float32),
    )


def _split_indices(
    sample_ids: Sequence[str],
    sample_stats_tsv: Path,
    *,
    strategy: str,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(sample_ids)
    if n < 2:
        raise ValueError("Need at least two samples for train/validation split")
    sample_to_pos = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    rows = [row for row in load_sample_stats_tsv(sample_stats_tsv) if str(row["sample_id"]) in sample_to_pos]
    val_size = max(1, int(round(float(val_ratio) * n)))
    if strategy == "random":
        order = rng.permutation(n)
        return order[val_size:], order[:val_size]
    if strategy == "low_coverage":
        coverage = np.zeros(n, dtype=np.float64)
        for row in rows:
            coverage[sample_to_pos[str(row["sample_id"])]] = float(row["observed_fraction"])
        order = np.argsort(coverage)
        val = order[:val_size]
        train = np.setdiff1d(np.arange(n), val, assume_unique=False)
        return train, val
    if strategy == "date":
        dates = np.full(n, np.nan, dtype=np.float64)
        for row in rows:
            dates[sample_to_pos[str(row["sample_id"])]] = float(row["date_mean_bp"])
        finite = np.flatnonzero(np.isfinite(dates))
        if finite.size >= val_size:
            val = finite[np.argsort(dates[finite])[-val_size:]]
            train = np.setdiff1d(np.arange(n), val, assume_unique=False)
            return train, val
    label_key = "country" if strategy == "country" else "original_group_id"
    labels: Dict[str, List[int]] = {}
    for row in rows:
        label = str(row.get(label_key, ""))
        if label:
            labels.setdefault(label, []).append(sample_to_pos[str(row["sample_id"])])
    if labels:
        label_names = np.asarray(sorted(labels))
        rng.shuffle(label_names)
        val_labels: set[str] = set()
        val_list: List[int] = []
        for label in label_names.tolist():
            val_labels.add(str(label))
            val_list.extend(labels[str(label)])
            if len(val_list) >= val_size:
                break
        val = np.asarray(sorted(set(val_list)), dtype=np.int64)
        if 0 < val.size < n:
            train = np.setdiff1d(np.arange(n), val, assume_unique=False)
            return train, val
    order = rng.permutation(n)
    return order[val_size:], order[:val_size]


def _pairwise_distance_loss(pred, target):
    torch, _ = require_torch()
    if pred.shape[0] < 3:
        return pred.new_zeros(())
    pred_dist = torch.cdist(pred, pred, p=2)
    target_dist = torch.cdist(target, target, p=2)
    pred_dist = pred_dist / pred_dist.detach().mean().clamp_min(1e-6)
    target_dist = target_dist / target_dist.detach().mean().clamp_min(1e-6)
    return (pred_dist - target_dist).pow(2).mean()


def _coverage_penalty(pred, coverage_t):
    centered = pred - pred.mean(dim=0, keepdim=True)
    cov_centered = coverage_t - coverage_t.mean()
    return (centered * cov_centered[:, None]).mean(dim=0).pow(2).mean()


def _stats_arrays_for_sample_ids(
    sample_ids: Sequence[str],
    sample_stats_tsv: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_by_id = {str(row["sample_id"]): row for row in load_sample_stats_tsv(sample_stats_tsv)}
    country_lookup: Dict[str, int] = {}
    original_lookup: Dict[str, int] = {}
    countries = np.full(len(sample_ids), -1, dtype=np.int64)
    originals = np.full(len(sample_ids), -1, dtype=np.int64)
    dates = np.full(len(sample_ids), np.nan, dtype=np.float32)
    for idx, sample_id in enumerate(sample_ids):
        row = rows_by_id.get(str(sample_id))
        if row is None:
            continue
        country = str(row.get("country", ""))
        original = str(row.get("original_group_id", ""))
        if country:
            countries[idx] = country_lookup.setdefault(country, len(country_lookup))
        if original:
            originals[idx] = original_lookup.setdefault(original, len(original_lookup))
        dates[idx] = float(row.get("date_mean_bp", float("nan")))
    return countries, originals, dates


def _metadata_structure_loss(pred, country_t, original_t, date_t):
    torch, _ = require_torch()
    if pred.shape[0] < 3:
        return pred.new_zeros(())
    dist2 = torch.cdist(pred, pred, p=2).pow(2)
    eye = torch.eye(pred.shape[0], dtype=torch.bool, device=pred.device)
    valid_country = country_t >= 0
    same_country = (country_t[:, None] == country_t[None, :]) & valid_country[:, None] & valid_country[None, :] & ~eye
    diff_country = (country_t[:, None] != country_t[None, :]) & valid_country[:, None] & valid_country[None, :]
    valid_original = original_t >= 0
    same_original = same_country & (original_t[:, None] == original_t[None, :]) & valid_original[:, None] & valid_original[None, :]
    diff_original_within_country = (
        same_country & (original_t[:, None] != original_t[None, :]) & valid_original[:, None] & valid_original[None, :]
    )
    pieces = []
    if same_country.any():
        pieces.append(dist2[same_country].mean())
    if diff_country.any():
        pieces.append(torch.exp(-dist2[diff_country]).mean())
    if same_original.any():
        pieces.append(0.5 * dist2[same_original].mean())
    if diff_original_within_country.any():
        pieces.append(0.5 * torch.exp(-dist2[diff_original_within_country]).mean())
    finite_date = torch.isfinite(date_t)
    if int(finite_date.sum().detach().cpu()) >= 3:
        date_diff = torch.abs(date_t[:, None] - date_t[None, :])
        finite_pair = finite_date[:, None] & finite_date[None, :] & ~eye
        if finite_pair.any():
            pred_dist = torch.sqrt(dist2.clamp_min(1e-8))
            pred_norm = pred_dist / pred_dist[finite_pair].detach().mean().clamp_min(1e-6)
            date_norm = date_diff / date_diff[finite_pair].detach().mean().clamp_min(1e-6)
            pieces.append(0.1 * (pred_norm[finite_pair] - date_norm[finite_pair]).pow(2).mean())
    if not pieces:
        return pred.new_zeros(())
    total = pieces[0]
    for piece in pieces[1:]:
        total = total + piece
    return total


def _pretrain_encoder_on_source(
    *,
    model,
    optimizer,
    source_meta_json: Path,
    source_sample_stats_tsv: Path,
    expected_n_snps: int,
    device: str,
    epochs: int,
    batch_size: int,
    snp_dropout: float,
    lambda_consistency: float,
    lambda_structure: float,
    seed: int,
) -> Dict[str, float]:
    torch, _ = require_torch()
    if int(epochs) <= 0:
        return {"pretrain_epochs_completed": 0}
    source_meta = load_meta(source_meta_json)
    if int(source_meta.n_snps) != int(expected_n_snps):
        raise ValueError(
            f"pretrain_source n_snps={source_meta.n_snps} does not match teacher/encoder n_snps={expected_n_snps}"
        )
    source_mm = open_genotype_memmap(source_meta)
    source_ids = load_sample_ids_from_meta(source_meta)
    country, original, dates = _stats_arrays_for_sample_ids(source_ids, source_sample_stats_tsv)
    country_t = torch.tensor(country, dtype=torch.long, device=device)
    original_t = torch.tensor(original, dtype=torch.long, device=device)
    date_t = torch.tensor(dates, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed + 10_003)
    indices = np.arange(source_meta.n_samples, dtype=np.int64)
    last_metrics: Dict[str, float] = {"pretrain_epochs_completed": 0}
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(indices)
        consistency_losses: List[float] = []
        structure_losses: List[float] = []
        total_losses: List[float] = []
        for lo in range(0, order.shape[0], int(batch_size)):
            idx = order[lo : lo + int(batch_size)]
            block = np.asarray(source_mm[idx], dtype=np.int8)
            values_a, mask_a = _batch_to_tensors(block, device, snp_dropout, training=True)
            values_b, mask_b = _batch_to_tensors(block, device, snp_dropout, training=True)
            pred_a = model(values_a, mask_a)
            pred_b = model(values_b, mask_b)
            batch_idx_t = torch.tensor(idx, dtype=torch.long, device=device)
            consistency = (pred_a - pred_b).pow(2).mean()
            structure = _metadata_structure_loss(pred_a, country_t[batch_idx_t], original_t[batch_idx_t], date_t[batch_idx_t])
            loss = float(lambda_consistency) * consistency + float(lambda_structure) * structure
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            consistency_losses.append(float(consistency.detach().cpu()))
            structure_losses.append(float(structure.detach().cpu()))
            total_losses.append(float(loss.detach().cpu()))
        last_metrics = {
            "pretrain_epochs_completed": int(epoch),
            "pretrain_loss": float(np.mean(total_losses)) if total_losses else 0.0,
            "pretrain_consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else 0.0,
            "pretrain_structure_loss": float(np.mean(structure_losses)) if structure_losses else 0.0,
        }
        print(
            f"pretrain epoch {epoch}/{epochs} loss={last_metrics['pretrain_loss']:.6f} "
            f"consistency={last_metrics['pretrain_consistency_loss']:.6f} "
            f"structure={last_metrics['pretrain_structure_loss']:.6f}",
            flush=True,
        )
    return last_metrics


def train_oos_encoder(
    *,
    meta_json: Path,
    snp_manifest: Path,
    teacher_latents_csv: Path,
    sample_stats_tsv: Path,
    output_dir: Path,
    latent_dim: int = 6,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    model_type: str = "sparse_residual_encoder",
    deep_hidden_dim: int = 100000,
    deep_bottleneck_dim: int = 1024,
    sparse_fanout: int = 2,
    sparse_init: str = "fan_in",
    dst_enable: bool = False,
    dst_update_interval_steps: int = 100,
    dst_start_epoch: int = 5,
    dst_end_epoch: int = 80,
    dst_initial_drop_fraction: float = 0.30,
    dst_final_drop_fraction: float = 0.05,
    dst_candidate_multiplier: int = 16,
    dst_grow_init: str = "zero",
    dst_score: str = "genotype_plus_mask_grad",
    lambda_latent: float = 1.0,
    lambda_neighbor: float = 0.1,
    lambda_coverage: float = 0.5,
    snp_dropout: float = 0.15,
    val_ratio: float = 0.2,
    split_strategy: str = "random",
    max_samples: int = 0,
    seed: int = 42,
    wandb_mode: str = "disabled",
    wandb_project: str = "adna-overlap-embed",
    wandb_group: str = "oos-encoder",
    wandb_name: str = "oos_encoder",
    wandb_tags: str = "oos-encoder",
    wandb_log_every: int = 10,
    wandb_init_timeout: int = 30,
    pretrain_source_meta_json: Optional[Path] = None,
    pretrain_source_sample_stats_tsv: Optional[Path] = None,
    pretrain_epochs: int = 20,
    lambda_pretrain_consistency: float = 1.0,
    lambda_pretrain_structure: float = 0.25,
) -> Dict[str, object]:
    torch, _ = require_torch()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(meta_json)
    manifest = load_snp_manifest(snp_manifest)
    if len(manifest) != int(meta.n_snps):
        raise ValueError(f"SNP manifest length {len(manifest)} does not match meta n_snps {meta.n_snps}")
    selected_meta_idx, sample_ids, targets = _align_targets_to_meta(meta, teacher_latents_csv)
    if int(targets.shape[1]) != int(latent_dim):
        raise ValueError(f"latent_dim={latent_dim} does not match teacher latents width={targets.shape[1]}")
    if int(max_samples) > 0 and len(sample_ids) > int(max_samples):
        subset_rng = np.random.default_rng(seed)
        keep = np.sort(subset_rng.choice(len(sample_ids), size=int(max_samples), replace=False))
        selected_meta_idx = selected_meta_idx[keep]
        sample_ids = [sample_ids[int(i)] for i in keep]
        targets = targets[keep]
    geno_mm = open_genotype_memmap(meta)
    coverage_all = compute_observed_fraction(geno_mm, sample_indices=selected_meta_idx)
    train_rel, val_rel = _split_indices(
        sample_ids,
        sample_stats_tsv,
        strategy=split_strategy,
        val_ratio=val_ratio,
        seed=seed,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_encoder_model(
        meta.n_snps,
        latent_dim,
        hidden_dim,
        model_type=model_type,
        deep_hidden_dim=deep_hidden_dim,
        deep_bottleneck_dim=deep_bottleneck_dim,
        sparse_fanout=sparse_fanout,
        sparse_init=sparse_init,
        dst_enable=dst_enable,
        dst_update_interval_steps=dst_update_interval_steps,
        dst_start_epoch=dst_start_epoch,
        dst_end_epoch=dst_end_epoch,
        dst_initial_drop_fraction=dst_initial_drop_fraction,
        dst_final_drop_fraction=dst_final_drop_fraction,
        dst_candidate_multiplier=dst_candidate_multiplier,
        dst_grow_init=dst_grow_init,
        dst_score=dst_score,
    )
    try:
        model = model.to(device)
    except Exception as exc:
        if device != "cuda":
            raise
        print(f"CUDA initialization failed, falling back to CPU: {exc}", flush=True)
        device = "cpu"
        model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    target_t = torch.tensor(targets, dtype=torch.float32, device=device)
    coverage_t = torch.tensor(coverage_all, dtype=torch.float32, device=device)
    config = EncoderConfig(
        n_snps=meta.n_snps,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        ploidy=meta.ploidy,
        snp_dropout=snp_dropout,
        low_overlap_fraction=0.05,
        low_overlap_snps=max(1, int(math.ceil(0.05 * meta.n_snps))),
        model_type=model_type,
        deep_hidden_dim=deep_hidden_dim,
        deep_bottleneck_dim=deep_bottleneck_dim,
        sparse_fanout=sparse_fanout,
        sparse_init=sparse_init,
        dst_enable=dst_enable,
        dst_update_interval_steps=dst_update_interval_steps,
        dst_start_epoch=dst_start_epoch,
        dst_end_epoch=dst_end_epoch,
        dst_initial_drop_fraction=dst_initial_drop_fraction,
        dst_final_drop_fraction=dst_final_drop_fraction,
        dst_candidate_multiplier=dst_candidate_multiplier,
        dst_grow_init=dst_grow_init,
        dst_score=dst_score,
    )
    shutil.copy2(snp_manifest, output_dir / "snp_manifest.tsv")
    with (output_dir / "encoder_config.json").open("w", encoding="utf-8") as handle:
        payload = {
            **asdict(config),
            "teacher_latents_csv": str(teacher_latents_csv),
            "sample_stats_tsv": str(sample_stats_tsv),
            "split_strategy": split_strategy,
            "max_samples": int(max_samples),
            "pretrain_source_meta_json": str(pretrain_source_meta_json) if pretrain_source_meta_json else "",
            "pretrain_source_sample_stats_tsv": (
                str(pretrain_source_sample_stats_tsv) if pretrain_source_sample_stats_tsv else ""
            ),
            "pretrain_epochs": int(pretrain_epochs),
            "lambda_pretrain_consistency": float(lambda_pretrain_consistency),
            "lambda_pretrain_structure": float(lambda_pretrain_structure),
            "pretrain_note": "optional source pretraining uses masked-view consistency plus metadata structure losses before teacher fine-tuning",
        }
        json.dump(payload, handle, indent=2)
    logger = SafeWandbRun(
        enabled=wandb_mode != "disabled",
        mode=wandb_mode,
        output_dir=output_dir,
        project=wandb_project,
        name=wandb_name,
        group=wandb_group,
        tags=wandb_tags,
        init_timeout=wandb_init_timeout,
        config=payload,
    )
    metrics_path = output_dir / "metrics.jsonl"
    best_val = float("inf")
    best_epoch = -1
    best_path = output_dir / "encoder.pt"
    global_step = 0
    pretrain_metrics: Dict[str, float] = {"pretrain_epochs_completed": 0}
    if pretrain_source_meta_json is not None:
        if pretrain_source_sample_stats_tsv is None:
            raise ValueError("--pretrain_source_sample_stats_tsv is required with --pretrain_source_meta_json")
        pretrain_metrics = _pretrain_encoder_on_source(
            model=model,
            optimizer=optimizer,
            source_meta_json=pretrain_source_meta_json,
            source_sample_stats_tsv=pretrain_source_sample_stats_tsv,
            expected_n_snps=int(meta.n_snps),
            device=device,
            epochs=int(pretrain_epochs),
            batch_size=int(batch_size),
            snp_dropout=float(snp_dropout),
            lambda_consistency=float(lambda_pretrain_consistency),
            lambda_structure=float(lambda_pretrain_structure),
            seed=int(seed),
        )
        logger.log(pretrain_metrics)

    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for epoch in range(1, int(epochs) + 1):
            start_time = time.time()
            model.train()
            order = rng.permutation(train_rel)
            losses: List[float] = []
            latent_losses: List[float] = []
            neighbor_losses: List[float] = []
            coverage_losses: List[float] = []
            for lo in range(0, order.shape[0], int(batch_size)):
                rel = order[lo : lo + int(batch_size)]
                meta_idx = selected_meta_idx[rel]
                block = np.asarray(geno_mm[meta_idx], dtype=np.int8)
                values_t, mask_t = _batch_to_tensors(block, device, snp_dropout, training=True)
                batch_target = target_t[rel]
                batch_cov = coverage_t[rel]
                pred = model(values_t, mask_t)
                mse = (pred - batch_target).pow(2).mean()
                cos = 1.0 - torch.nn.functional.cosine_similarity(pred, batch_target, dim=1).mean()
                neighbor = _pairwise_distance_loss(pred, batch_target)
                coverage_loss = _coverage_penalty(pred, batch_cov)
                loss = (
                    float(lambda_latent) * (mse + cos)
                    + float(lambda_neighbor) * neighbor
                    + float(lambda_coverage) * coverage_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1
                if hasattr(model, "maybe_dynamic_sparse_update"):
                    model.maybe_dynamic_sparse_update(
                        global_step=global_step,
                        epoch=epoch,
                        optimizer=optimizer,
                        values=values_t.detach(),
                        mask=mask_t.detach(),
                    )
                losses.append(float(loss.detach().cpu()))
                latent_losses.append(float((mse + cos).detach().cpu()))
                neighbor_losses.append(float(neighbor.detach().cpu()))
                coverage_losses.append(float(coverage_loss.detach().cpu()))
            val_metrics = predict_encoder_subset(
                model=model,
                geno_mm=geno_mm,
                selected_meta_idx=selected_meta_idx,
                rel_indices=val_rel,
                targets=targets,
                coverage=coverage_all,
                device=device,
                batch_size=batch_size,
            )
            sparse_metrics = model.dst_metrics() if hasattr(model, "dst_metrics") else {}
            log_payload = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "train_latent_loss": float(np.mean(latent_losses)) if latent_losses else 0.0,
                "train_neighbor_loss": float(np.mean(neighbor_losses)) if neighbor_losses else 0.0,
                "train_coverage_loss": float(np.mean(coverage_losses)) if coverage_losses else 0.0,
                "val_latent_mse": float(val_metrics["latent_mse"]),
                "val_cosine_similarity": float(val_metrics["mean_cosine_similarity"]),
                "val_coverage_r2": float(val_metrics["coverage_r2"]),
                "epoch_seconds": float(time.time() - start_time),
                **sparse_metrics,
            }
            metrics_handle.write(json.dumps(log_payload) + "\n")
            metrics_handle.flush()
            if epoch == 1 or epoch % max(int(wandb_log_every), 1) == 0:
                logger.log(log_payload)
            if float(val_metrics["latent_mse"]) < best_val:
                best_val = float(val_metrics["latent_mse"])
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "epoch": epoch,
                        "teacher_latent_dim": latent_dim,
                        "dst_updates": int(getattr(model, "dst_updates", 0)),
                    },
                    best_path,
                )
            print(
                f"epoch {epoch}/{epochs} train_loss={log_payload['train_loss']:.6f} "
                f"val_mse={log_payload['val_latent_mse']:.6f} "
                f"val_cos={log_payload['val_cosine_similarity']:.6f}",
                flush=True,
            )
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    heldout_pred = predict_latents_for_indices(
        model=model,
        geno_mm=geno_mm,
        meta_indices=selected_meta_idx[val_rel],
        device=device,
        batch_size=batch_size,
    )
    write_latents_csv(output_dir / "heldout_predictions.csv", [sample_ids[int(i)] for i in val_rel], heldout_pred)
    heldout_metrics = evaluate_predictions(
        predicted=heldout_pred,
        target=targets[val_rel],
        sample_ids=[sample_ids[int(i)] for i in val_rel],
        sample_stats_tsv=sample_stats_tsv,
        coverage=coverage_all[val_rel],
        k=15,
    )
    summary = {
        "output_dir": str(output_dir),
        "meta_json": str(meta_json),
        "snp_manifest": str(output_dir / "snp_manifest.tsv"),
        "teacher_latents_csv": str(teacher_latents_csv),
        "sample_stats_tsv": str(sample_stats_tsv),
        "config": payload,
        "training": {
            "best_epoch": best_epoch,
            "best_val_latent_mse": best_val,
            "n_epochs_completed": int(epochs),
            "n_train": int(train_rel.shape[0]),
            "n_val": int(val_rel.shape[0]),
            "global_steps": int(global_step),
        },
        "sparse": model.dst_metrics() if hasattr(model, "dst_metrics") else {},
        "pretraining": pretrain_metrics,
        "heldout": heldout_metrics,
        "wandb_note": logger.note,
        "wandb_mode": wandb_mode,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.finish()
    return summary


def predict_latents_for_indices(
    *,
    model,
    geno_mm: np.memmap,
    meta_indices: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    torch, _ = require_torch()
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for lo in range(0, meta_indices.shape[0], int(batch_size)):
            idx = meta_indices[lo : lo + int(batch_size)]
            block = np.asarray(geno_mm[idx], dtype=np.int8)
            values_t, mask_t = _batch_to_tensors(block, device, 0.0, training=False)
            pred = model(values_t, mask_t).detach().cpu().numpy()
            preds.append(pred.astype(np.float32))
    return np.vstack(preds) if preds else np.zeros((0, 0), dtype=np.float32)


def predict_encoder_subset(
    *,
    model,
    geno_mm: np.memmap,
    selected_meta_idx: np.ndarray,
    rel_indices: np.ndarray,
    targets: np.ndarray,
    coverage: np.ndarray,
    device: str,
    batch_size: int,
) -> Dict[str, float]:
    pred = predict_latents_for_indices(
        model=model,
        geno_mm=geno_mm,
        meta_indices=selected_meta_idx[rel_indices],
        device=device,
        batch_size=batch_size,
    )
    return _basic_prediction_metrics(pred, targets[rel_indices], coverage[rel_indices])


def _basic_prediction_metrics(predicted: np.ndarray, target: np.ndarray, coverage: np.ndarray) -> Dict[str, float]:
    if predicted.shape != target.shape:
        raise ValueError(f"Predicted shape {predicted.shape} does not match target shape {target.shape}")
    mse = float(np.mean((predicted - target) ** 2))
    denom = np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1)
    valid = denom > 1e-8
    cosine = float(np.mean((predicted[valid] * target[valid]).sum(axis=1) / denom[valid])) if valid.any() else float("nan")
    return {
        "latent_mse": mse,
        "mean_cosine_similarity": cosine,
        "coverage_r2": ridge_probe_r2(predicted, coverage),
    }


def knn_overlap(predicted: np.ndarray, target: np.ndarray, k: int = 15) -> float:
    if predicted.shape[0] < 3:
        return float("nan")
    k_eff = min(int(k), predicted.shape[0] - 1)
    pred_nn = nearest_neighbors(predicted, k_eff)
    target_nn = nearest_neighbors(target, k_eff)
    overlaps = []
    for i in range(predicted.shape[0]):
        overlaps.append(len(set(pred_nn[i].tolist()) & set(target_nn[i].tolist())) / float(k_eff))
    return float(np.mean(overlaps))


def evaluate_predictions(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    sample_ids: Sequence[str],
    sample_stats_tsv: Path,
    coverage: np.ndarray,
    k: int = 15,
) -> Dict[str, float]:
    metrics = _basic_prediction_metrics(predicted, target, coverage)
    metrics[f"knn_overlap@{k}"] = knn_overlap(predicted, target, k)
    tmp_stats = _subset_sample_stats(sample_stats_tsv, sample_ids)
    pred_eval, _, _, _ = evaluate_embedding(predicted, tmp_stats, k=min(k, max(predicted.shape[0] - 1, 1)))
    for key, value in pred_eval.items():
        metrics[f"pred_{key}"] = float(value)
    return metrics


def _subset_sample_stats(sample_stats_tsv: Path, sample_ids: Sequence[str]) -> Path:
    rows = load_sample_stats_tsv(sample_stats_tsv)
    rows_by_id = {str(row["sample_id"]): row for row in rows}
    subset = []
    for sample_id in map(str, sample_ids):
        row = rows_by_id.get(sample_id)
        if row is not None:
            subset.append(row)
    if len(subset) != len(sample_ids):
        missing = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in rows_by_id]
        raise ValueError(f"Missing sample_stats rows for {len(missing)} OOS samples; first missing: {missing[:5]}")
    temp_path = sample_stats_tsv.parent / f".tmp_oos_subset_{abs(hash(tuple(sample_ids))) % 10_000_000}.tsv"
    write_tsv(temp_path, subset, ["sample_id", "observed_fraction", "country", "original_group_id", "date_mean_bp"])
    return temp_path


def load_encoder(encoder_dir: Path, device: Optional[str] = None):
    torch, _ = require_torch()
    config = json.loads((encoder_dir / "encoder_config.json").read_text(encoding="utf-8"))
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_encoder_model(
        n_snps=int(config["n_snps"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        model_type=str(config.get("model_type", "sparse_residual_encoder")),
        deep_hidden_dim=int(config.get("deep_hidden_dim", 100000)),
        deep_bottleneck_dim=int(config.get("deep_bottleneck_dim", 1024)),
        sparse_fanout=int(config.get("sparse_fanout", 2)),
        sparse_init=str(config.get("sparse_init", "fan_in")),
        dst_enable=bool(config.get("dst_enable", False)),
        dst_update_interval_steps=int(config.get("dst_update_interval_steps", 100)),
        dst_start_epoch=int(config.get("dst_start_epoch", 5)),
        dst_end_epoch=int(config.get("dst_end_epoch", 80)),
        dst_initial_drop_fraction=float(config.get("dst_initial_drop_fraction", 0.30)),
        dst_final_drop_fraction=float(config.get("dst_final_drop_fraction", 0.05)),
        dst_candidate_multiplier=int(config.get("dst_candidate_multiplier", 16)),
        dst_grow_init=str(config.get("dst_grow_init", "zero")),
        dst_score=str(config.get("dst_score", "genotype_plus_mask_grad")),
    )
    try:
        model = model.to(selected_device)
    except Exception as exc:
        if selected_device != "cuda":
            raise
        print(f"CUDA initialization failed during encoder load, falling back to CPU: {exc}", flush=True)
        selected_device = "cpu"
        model = model.to(selected_device)
    checkpoint = torch.load(encoder_dir / "encoder.pt", map_location=selected_device)
    ckpt_dim = int(checkpoint.get("teacher_latent_dim", config["latent_dim"]))
    if ckpt_dim != int(config["latent_dim"]):
        raise ValueError("Encoder checkpoint latent dimension does not match encoder_config.json")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, selected_device


def encode_aligned_memmap(
    *,
    encoder_dir: Path,
    meta_json: Path,
    out_dir: Path,
    batch_size: int = 64,
    teacher_latents_csv: Optional[Path] = None,
    teacher_sample_stats_tsv: Optional[Path] = None,
    low_overlap_fraction: Optional[float] = None,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, config, device = load_encoder(encoder_dir)
    meta = load_meta(meta_json)
    if int(meta.n_snps) != int(config["n_snps"]):
        raise ValueError(f"Input n_snps={meta.n_snps} does not match encoder n_snps={config['n_snps']}")
    geno_mm = open_genotype_memmap(meta)
    sample_ids = load_sample_ids_from_meta(meta)
    meta_indices = np.arange(meta.n_samples, dtype=np.int64)
    latents = predict_latents_for_indices(
        model=model,
        geno_mm=geno_mm,
        meta_indices=meta_indices,
        device=device,
        batch_size=batch_size,
    )
    write_latents_csv(out_dir / "encoded_latents.csv", sample_ids, latents)
    observed_fraction = compute_observed_fraction(geno_mm)
    observed_snps = (observed_fraction * float(meta.n_snps)).astype(int)
    threshold_fraction = float(low_overlap_fraction if low_overlap_fraction is not None else config.get("low_overlap_fraction", 0.05))
    threshold_snps = int(config.get("low_overlap_snps", math.ceil(threshold_fraction * meta.n_snps)))
    stats_rows = []
    for idx, sample_id in enumerate(sample_ids):
        warning = observed_snps[idx] < threshold_snps
        stats_rows.append(
            {
                "sample_id": sample_id,
                "observed_fraction": float(observed_fraction[idx]),
                "overlap_snps": int(observed_snps[idx]),
                "low_overlap_warning": int(bool(warning)),
            }
        )
    write_tsv(
        out_dir / "encoded_sample_stats.tsv",
        stats_rows,
        ["sample_id", "observed_fraction", "overlap_snps", "low_overlap_warning"],
    )
    neighbor_path = ""
    if teacher_latents_csv is not None:
        neighbor_path = str(out_dir / "nearest_neighbors.tsv")
        write_nearest_teacher_neighbors(
            path=Path(neighbor_path),
            query_sample_ids=sample_ids,
            query_latents=latents,
            teacher_latents_csv=teacher_latents_csv,
            teacher_sample_stats_tsv=teacher_sample_stats_tsv,
        )
    report = {
        "encoder_dir": str(encoder_dir),
        "meta_json": str(meta_json),
        "n_samples": int(meta.n_samples),
        "n_snps": int(meta.n_snps),
        "latent_dim": int(latents.shape[1]),
        "low_overlap_fraction": threshold_fraction,
        "low_overlap_snps": threshold_snps,
        "n_low_overlap_samples": int(sum(row["low_overlap_warning"] for row in stats_rows)),
        "nearest_neighbors_tsv": neighbor_path,
    }
    with (out_dir / "encoding_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def write_nearest_teacher_neighbors(
    *,
    path: Path,
    query_sample_ids: Sequence[str],
    query_latents: np.ndarray,
    teacher_latents_csv: Path,
    teacher_sample_stats_tsv: Optional[Path],
    k: int = 15,
) -> None:
    teacher_ids, teacher_latents = read_latents_csv(teacher_latents_csv)
    stats_by_id: Dict[str, Dict[str, object]] = {}
    if teacher_sample_stats_tsv is not None and teacher_sample_stats_tsv.exists():
        stats_by_id = {str(row["sample_id"]): row for row in load_sample_stats_tsv(teacher_sample_stats_tsv)}
    dist2 = (
        (query_latents**2).sum(axis=1, keepdims=True)
        + (teacher_latents**2).sum(axis=1, keepdims=True).T
        - 2.0 * query_latents @ teacher_latents.T
    )
    k_eff = min(int(k), teacher_latents.shape[0])
    idx = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows: List[Dict[str, object]] = []
    for q_idx, sample_id in enumerate(query_sample_ids):
        order = idx[q_idx][np.argsort(dist2[q_idx, idx[q_idx]])]
        for rank, teacher_idx in enumerate(order.tolist(), start=1):
            teacher_id = teacher_ids[teacher_idx]
            meta = stats_by_id.get(teacher_id, {})
            rows.append(
                {
                    "query_sample_id": sample_id,
                    "neighbor_rank": rank,
                    "teacher_sample_id": teacher_id,
                    "distance": float(np.sqrt(max(float(dist2[q_idx, teacher_idx]), 0.0))),
                    "country": meta.get("country", ""),
                    "original_group_id": meta.get("original_group_id", ""),
                    "date_mean_bp": meta.get("date_mean_bp", ""),
                }
            )
    write_tsv(
        path,
        rows,
        [
            "query_sample_id",
            "neighbor_rank",
            "teacher_sample_id",
            "distance",
            "country",
            "original_group_id",
            "date_mean_bp",
        ],
    )
