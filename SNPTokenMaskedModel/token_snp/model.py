from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class TokenSNPConfig:
    n_classes: int
    n_snps: int
    window_size: int
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


class TokenSNPMaskedModel(nn.Module):
    def __init__(self, cfg: TokenSNPConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.missing_token = cfg.n_classes
        self.mask_token = cfg.n_classes + 1
        self.vocab_size = cfg.n_classes + 2

        self.token_embed = nn.Embedding(self.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.window_size, cfg.d_model)

        self.use_obs_embedding = cfg.use_obs_embedding
        self.obs_embed = nn.Embedding(2, cfg.d_model) if cfg.use_obs_embedding else None

        self.use_snp_id_embedding = cfg.use_snp_id_embedding
        self.snp_embed = nn.Embedding(cfg.n_snps, cfg.d_model) if cfg.use_snp_id_embedding else None

        local_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, num_layers=cfg.local_layers)

        chunk_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.chunk_encoder = nn.TransformerEncoder(chunk_layer, num_layers=cfg.chunk_layers)

        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        self.latent_proj = nn.Linear(cfg.d_model, cfg.latent_dim)

    @staticmethod
    def _sanitize_padding_mask(mask: torch.Tensor) -> torch.Tensor:
        # Transformer encoder requires at least one unmasked token per sequence.
        out = mask.clone()
        all_masked = out.all(dim=1)
        if all_masked.any():
            out[all_masked, 0] = False
        return out

    def forward(
        self,
        tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        snp_idx: Optional[torch.Tensor] = None,
    ) -> dict:
        # tokens: [B, L], obs_mask: [B, L], snp_idx: [B, L]
        bsz, seq_len = tokens.shape
        if seq_len > self.cfg.window_size:
            raise ValueError(
                f"Input length {seq_len} exceeds configured window size {self.cfg.window_size}"
            )

        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embed(tokens) + self.pos_embed(pos)

        if self.use_obs_embedding and self.obs_embed is not None:
            obs_ids = (obs_mask > 0.5).to(dtype=torch.long)
            x = x + self.obs_embed(obs_ids)

        if self.use_snp_id_embedding and self.snp_embed is not None and snp_idx is not None:
            clipped = snp_idx.clamp(min=0, max=self.cfg.n_snps - 1)
            x = x + self.snp_embed(clipped)

        chunk = self.cfg.chunk_size
        pad_len = (-seq_len) % chunk
        if pad_len > 0:
            x = torch.cat(
                [x, torch.zeros((bsz, pad_len, self.cfg.d_model), device=x.device, dtype=x.dtype)],
                dim=1,
            )
            obs_mask = torch.cat(
                [obs_mask, torch.zeros((bsz, pad_len), device=obs_mask.device, dtype=obs_mask.dtype)],
                dim=1,
            )

        padded_len = x.shape[1]
        n_chunks = padded_len // chunk

        local_in = x.reshape(bsz, n_chunks, chunk, self.cfg.d_model)
        local_in = local_in.reshape(bsz * n_chunks, chunk, self.cfg.d_model)

        chunk_obs = obs_mask.reshape(bsz, n_chunks, chunk)
        token_padding_mask = self._sanitize_padding_mask((chunk_obs <= 0.5).reshape(bsz * n_chunks, chunk))

        local_h = self.local_encoder(local_in, src_key_padding_mask=token_padding_mask)
        local_h = local_h.reshape(bsz, n_chunks, chunk, self.cfg.d_model)

        chunk_weights = chunk_obs.unsqueeze(-1)
        chunk_sum = (local_h * chunk_weights).sum(dim=2)
        chunk_denom = chunk_weights.sum(dim=2).clamp_min(1.0)
        chunk_repr = chunk_sum / chunk_denom

        chunk_padding_mask = self._sanitize_padding_mask((chunk_obs.sum(dim=2) <= 0.5))
        chunk_ctx = self.chunk_encoder(chunk_repr, src_key_padding_mask=chunk_padding_mask)

        fused = local_h + chunk_ctx.unsqueeze(2)
        fused = fused.reshape(bsz, padded_len, self.cfg.d_model)
        fused = fused[:, :seq_len, :]

        h = self.norm(fused)
        logits = self.head(h)

        pooled_weights = obs_mask[:, :seq_len].unsqueeze(-1)
        pooled = (h * pooled_weights).sum(dim=1) / pooled_weights.sum(dim=1).clamp_min(1.0)
        window_embedding = self.latent_proj(pooled)

        return {
            "logits": logits,
            "hidden": h,
            "window_embedding": window_embedding,
        }
