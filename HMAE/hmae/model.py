from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class HMAEConfig:
    n_classes: int
    n_windows: int
    window_size: int
    window_latent_dim: int = 32
    global_latent_dim: int = 32
    embed_dim: int = 16
    local_hidden_dim: int = 128
    local_encoder_type: str = "conv_attn"
    local_conv_layers: int = 4
    local_conv_kernel: int = 7
    local_attn_heads: int = 4
    local_dropout: float = 0.1
    window_latent_slots: int = 4
    global_model_dim: int = 128
    global_heads: int = 4
    global_layers: int = 2
    decoder_hidden_dim: int = 128
    decoder_attn_heads: int = 4
    include_window_coverage: bool = True
    variational: bool = False


def _sanitize_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    # MultiheadAttention needs at least one unmasked token per row.
    out = mask.clone()
    all_masked = out.all(dim=1)
    if all_masked.any():
        out[all_masked, 0] = False
    return out


class MeanPoolWindowEncoder(nn.Module):
    def __init__(self, cfg: HMAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.missing_token = cfg.n_classes
        self.geno_embed = nn.Embedding(cfg.n_classes + 1, cfg.embed_dim)

        in_dim = cfg.embed_dim + (2 if cfg.include_window_coverage else 1)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, cfg.local_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.local_hidden_dim, cfg.window_latent_dim),
        )

    def forward(
        self,
        input_idx: torch.Tensor,
        obs_mask: torch.Tensor,
        coverage: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # input_idx: [N, W]
        # obs_mask:  [N, W]
        embed = self.geno_embed(input_idx)  # [N, W, E]
        mask = obs_mask.unsqueeze(-1)
        pooled = (embed * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        observed_frac = obs_mask.mean(dim=1, keepdim=True)

        if self.cfg.include_window_coverage and coverage is not None:
            features = torch.cat([pooled, observed_frac, coverage], dim=1)
        else:
            features = torch.cat([pooled, observed_frac], dim=1)
        return self.proj(features)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        return x + y


class ConvAttnWindowEncoder(nn.Module):
    def __init__(self, cfg: HMAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.local_hidden_dim % cfg.local_attn_heads != 0:
            raise ValueError(
                "local_hidden_dim must be divisible by local_attn_heads "
                f"({cfg.local_hidden_dim} vs {cfg.local_attn_heads})"
            )
        if cfg.local_conv_layers < 1:
            raise ValueError("local_conv_layers must be >= 1")
        if cfg.local_conv_kernel < 3 or cfg.local_conv_kernel % 2 == 0:
            raise ValueError("local_conv_kernel must be an odd integer >= 3")

        self.missing_token = cfg.n_classes
        self.geno_embed = nn.Embedding(cfg.n_classes + 1, cfg.embed_dim)
        self.token_proj = nn.Linear(cfg.embed_dim + 1, cfg.local_hidden_dim)
        self.conv_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    channels=cfg.local_hidden_dim,
                    kernel_size=cfg.local_conv_kernel,
                    dropout=cfg.local_dropout,
                )
                for _ in range(cfg.local_conv_layers)
            ]
        )
        self.token_norm = nn.LayerNorm(cfg.local_hidden_dim)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, cfg.local_hidden_dim))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=cfg.local_hidden_dim,
            num_heads=cfg.local_attn_heads,
            dropout=cfg.local_dropout,
            batch_first=True,
        )
        context_dim = cfg.local_hidden_dim + (2 if cfg.include_window_coverage else 1)
        self.context_proj = nn.Linear(context_dim, cfg.local_hidden_dim)
        self.dropout = nn.Dropout(cfg.local_dropout)
        self.out_proj = nn.Linear(cfg.local_hidden_dim, cfg.window_latent_dim)

    def forward(
        self,
        input_idx: torch.Tensor,
        obs_mask: torch.Tensor,
        coverage: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # input_idx: [N, W], obs_mask: [N, W]
        embed = self.geno_embed(input_idx)  # [N, W, E]
        x = torch.cat([embed, obs_mask.unsqueeze(-1)], dim=-1)  # [N, W, E+1]
        x = self.token_proj(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # [N, H, W]
        for block in self.conv_blocks:
            x = block(x)
        tokens = x.transpose(1, 2)  # [N, W, H]
        tokens = self.token_norm(tokens)

        key_padding_mask = _sanitize_padding_mask(obs_mask <= 0.5)
        query = self.pool_query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.pool_attn(
            query=query,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = pooled.squeeze(1)  # [N, H]
        observed_frac = obs_mask.mean(dim=1, keepdim=True)
        if self.cfg.include_window_coverage and coverage is not None:
            features = torch.cat([pooled, observed_frac, coverage], dim=1)
        else:
            features = torch.cat([pooled, observed_frac], dim=1)
        h = self.context_proj(features)
        h = F.silu(h)
        h = self.dropout(h)
        return self.out_proj(h)


class ConvAttnMultiSlotWindowEncoder(nn.Module):
    def __init__(self, cfg: HMAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.local_hidden_dim % cfg.local_attn_heads != 0:
            raise ValueError(
                "local_hidden_dim must be divisible by local_attn_heads "
                f"({cfg.local_hidden_dim} vs {cfg.local_attn_heads})"
            )
        if cfg.local_conv_layers < 1:
            raise ValueError("local_conv_layers must be >= 1")
        if cfg.local_conv_kernel < 3 or cfg.local_conv_kernel % 2 == 0:
            raise ValueError("local_conv_kernel must be an odd integer >= 3")
        if cfg.window_latent_slots < 1:
            raise ValueError("window_latent_slots must be >= 1")

        self.missing_token = cfg.n_classes
        self.geno_embed = nn.Embedding(cfg.n_classes + 1, cfg.embed_dim)
        self.token_proj = nn.Linear(cfg.embed_dim + 1, cfg.local_hidden_dim)
        self.conv_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    channels=cfg.local_hidden_dim,
                    kernel_size=cfg.local_conv_kernel,
                    dropout=cfg.local_dropout,
                )
                for _ in range(cfg.local_conv_layers)
            ]
        )
        self.token_norm = nn.LayerNorm(cfg.local_hidden_dim)
        self.window_latents = nn.Parameter(
            torch.randn(1, cfg.window_latent_slots, cfg.local_hidden_dim)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.local_hidden_dim,
            num_heads=cfg.local_attn_heads,
            dropout=cfg.local_dropout,
            batch_first=True,
        )
        context_dim = cfg.local_hidden_dim + (2 if cfg.include_window_coverage else 1)
        self.context_proj = nn.Linear(context_dim, cfg.local_hidden_dim)
        self.dropout = nn.Dropout(cfg.local_dropout)
        self.out_proj = nn.Linear(cfg.local_hidden_dim, cfg.window_latent_dim)

    def forward(
        self,
        input_idx: torch.Tensor,
        obs_mask: torch.Tensor,
        coverage: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_idx: [N, W], obs_mask: [N, W]
        embed = self.geno_embed(input_idx)  # [N, W, E]
        x = torch.cat([embed, obs_mask.unsqueeze(-1)], dim=-1)  # [N, W, E+1]
        x = self.token_proj(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # [N, H, W]
        for block in self.conv_blocks:
            x = block(x)
        tokens = x.transpose(1, 2)  # [N, W, H]
        tokens = self.token_norm(tokens)

        key_padding_mask = _sanitize_padding_mask(obs_mask <= 0.5)
        query = self.window_latents.expand(tokens.shape[0], -1, -1)  # [N, S, H]
        slot_tokens, _ = self.cross_attn(
            query=query,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        observed_frac = obs_mask.mean(dim=1, keepdim=True).unsqueeze(1)  # [N, 1, 1]
        observed_frac = observed_frac.expand(-1, self.cfg.window_latent_slots, -1)
        if self.cfg.include_window_coverage and coverage is not None:
            cov = coverage.unsqueeze(1).expand(-1, self.cfg.window_latent_slots, -1)
            features = torch.cat([slot_tokens, observed_frac, cov], dim=2)
        else:
            features = torch.cat([slot_tokens, observed_frac], dim=2)

        h = self.context_proj(features)
        h = F.silu(h)
        h = self.dropout(h)
        slot_latents = self.out_proj(h)  # [N, S, Dw]
        mean_window_latent = slot_latents.mean(dim=1)  # [N, Dw]
        return slot_latents, mean_window_latent


class GlobalAggregator(nn.Module):
    def __init__(self, cfg: HMAEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.window_proj = nn.Linear(cfg.window_latent_dim, cfg.global_model_dim)
        self.window_pos = nn.Embedding(cfg.n_windows, cfg.global_model_dim)
        self.slot_pos = nn.Embedding(cfg.window_latent_slots, cfg.global_model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.global_model_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.global_model_dim,
            nhead=cfg.global_heads,
            dim_feedforward=cfg.global_model_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.global_layers)

        if cfg.variational:
            self.mu_head = nn.Linear(cfg.global_model_dim, cfg.global_latent_dim)
            self.logvar_head = nn.Linear(cfg.global_model_dim, cfg.global_latent_dim)
        else:
            self.out_head = nn.Linear(cfg.global_model_dim, cfg.global_latent_dim)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        window_latents: torch.Tensor,
        window_ids: torch.Tensor,
        slot_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # window_latents: [B, T, D_w]
        # window_ids: [T]
        # slot_ids: [T] or None
        bsz, _, _ = window_latents.shape
        tokens = self.window_proj(window_latents)
        pos = self.window_pos(window_ids).unsqueeze(0).expand(bsz, -1, -1)
        tokens = tokens + pos
        if slot_ids is not None:
            slot_pos = self.slot_pos(slot_ids).unsqueeze(0).expand(bsz, -1, -1)
            tokens = tokens + slot_pos

        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        h = self.encoder(x)
        cls_out = h[:, 0, :]

        if self.cfg.variational:
            mu = self.mu_head(cls_out)
            logvar = self.logvar_head(cls_out)
            z = self._reparameterize(mu, logvar)
            return z, mu, logvar

        z = self.out_head(cls_out)
        return z, None, None


class WindowDecoder(nn.Module):
    def __init__(self, cfg: HMAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.window_latent_dim + cfg.global_latent_dim
        self.context = nn.Sequential(
            nn.Linear(in_dim, cfg.decoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.decoder_hidden_dim, cfg.decoder_hidden_dim),
            nn.SiLU(),
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1, cfg.window_size, cfg.decoder_hidden_dim)
        )
        self.out = nn.Linear(cfg.decoder_hidden_dim, cfg.n_classes)

    def forward(self, window_latents: torch.Tensor, global_latent: torch.Tensor) -> torch.Tensor:
        # window_latents: [B, K, D_w]
        # global_latent:  [B, D_g]
        _, n_tokens, _ = window_latents.shape
        g = global_latent.unsqueeze(1).expand(-1, n_tokens, -1)
        ctx = self.context(torch.cat([window_latents, g], dim=-1))  # [B, K, H]
        h = ctx.unsqueeze(2) + self.pos_embed  # [B, K, W, H]
        return self.out(h)  # [B, K, W, C]


class WindowDecoderMultiSlot(nn.Module):
    def __init__(self, cfg: HMAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.decoder_hidden_dim % cfg.decoder_attn_heads != 0:
            raise ValueError(
                "decoder_hidden_dim must be divisible by decoder_attn_heads "
                f"({cfg.decoder_hidden_dim} vs {cfg.decoder_attn_heads})"
            )
        self.slot_proj = nn.Linear(cfg.window_latent_dim, cfg.decoder_hidden_dim)
        self.window_ctx_proj = nn.Linear(cfg.window_latent_dim, cfg.decoder_hidden_dim)
        self.token_pos = nn.Parameter(
            torch.zeros(1, 1, cfg.window_size, cfg.decoder_hidden_dim)
        )
        self.slot_attn = nn.MultiheadAttention(
            embed_dim=cfg.decoder_hidden_dim,
            num_heads=cfg.decoder_attn_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.global_proj = nn.Linear(cfg.global_latent_dim, cfg.decoder_hidden_dim)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=cfg.decoder_hidden_dim,
            num_heads=cfg.decoder_attn_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.decoder_hidden_dim, cfg.decoder_hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(cfg.decoder_hidden_dim * 2, cfg.decoder_hidden_dim),
        )
        self.norm1 = nn.LayerNorm(cfg.decoder_hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.decoder_hidden_dim)
        self.norm3 = nn.LayerNorm(cfg.decoder_hidden_dim)
        self.out = nn.Linear(cfg.decoder_hidden_dim, cfg.n_classes)

    def forward(
        self,
        window_slot_latents: torch.Tensor,
        global_latent: torch.Tensor,
    ) -> torch.Tensor:
        # window_slot_latents: [B, K, S, D_w]
        # global_latent: [B, D_g]
        bsz, n_windows, n_slots, _ = window_slot_latents.shape

        slot_memory = self.slot_proj(window_slot_latents)  # [B, K, S, H]
        window_ctx = self.window_ctx_proj(window_slot_latents.mean(dim=2)).unsqueeze(2)  # [B,K,1,H]
        token_states = self.token_pos.expand(bsz, n_windows, -1, -1) + window_ctx  # [B,K,W,H]

        bk = bsz * n_windows
        wlen = token_states.shape[2]
        hidden = token_states.shape[3]

        q = token_states.reshape(bk, wlen, hidden)
        kv_slots = slot_memory.reshape(bk, n_slots, hidden)
        attn_slots, _ = self.slot_attn(
            query=q,
            key=kv_slots,
            value=kv_slots,
            need_weights=False,
        )
        q = self.norm1(q + attn_slots)

        g = self.global_proj(global_latent).unsqueeze(1)  # [B,1,H]
        g = g.unsqueeze(1).expand(-1, n_windows, -1, -1).reshape(bk, 1, hidden)
        attn_global, _ = self.global_attn(
            query=q,
            key=g,
            value=g,
            need_weights=False,
        )
        q = self.norm2(q + attn_global)

        q = self.norm3(q + self.ffn(q))
        logits = self.out(q).reshape(bsz, n_windows, wlen, self.cfg.n_classes)
        return logits


class HierarchicalMaskedAutoencoder(nn.Module):
    def __init__(self, cfg: HMAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.local_encoder_type == "conv_attn":
            self.window_encoder = ConvAttnWindowEncoder(cfg)
            self.decoder = WindowDecoder(cfg)
        elif cfg.local_encoder_type == "conv_attn_multislot":
            self.window_encoder = ConvAttnMultiSlotWindowEncoder(cfg)
            self.decoder = WindowDecoderMultiSlot(cfg)
        elif cfg.local_encoder_type == "meanpool":
            self.window_encoder = MeanPoolWindowEncoder(cfg)
            self.decoder = WindowDecoder(cfg)
        else:
            raise ValueError(
                "Unsupported local_encoder_type. Use one of: conv_attn, "
                "conv_attn_multislot, meanpool"
            )
        self.global_agg = GlobalAggregator(cfg)

    def encode_windows(
        self,
        input_idx: torch.Tensor,
        obs_mask: torch.Tensor,
        coverage: Optional[torch.Tensor],
        window_ids: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # input_idx: [B, K, W]
        bsz, n_windows, wlen = input_idx.shape
        x = input_idx.reshape(bsz * n_windows, wlen)
        m = obs_mask.reshape(bsz * n_windows, wlen)
        c = None
        if coverage is not None:
            c = coverage.reshape(bsz * n_windows, 1)

        if self.cfg.local_encoder_type == "conv_attn_multislot":
            slot_latents_flat, mean_latents_flat = self.window_encoder(x, m, c)
            window_slot_latents = slot_latents_flat.reshape(
                bsz, n_windows, self.cfg.window_latent_slots, -1
            )
            window_latents = mean_latents_flat.reshape(bsz, n_windows, -1)
            agg_window_latents = window_slot_latents.reshape(
                bsz, n_windows * self.cfg.window_latent_slots, -1
            )
            agg_window_ids = window_ids.unsqueeze(1).expand(
                -1, self.cfg.window_latent_slots
            ).reshape(-1)
            slot_ids = torch.arange(
                self.cfg.window_latent_slots,
                device=window_ids.device,
                dtype=window_ids.dtype,
            )
            agg_slot_ids = slot_ids.unsqueeze(0).expand(n_windows, -1).reshape(-1)
            return {
                "window_latents": window_latents,
                "window_slot_latents": window_slot_latents,
                "agg_window_latents": agg_window_latents,
                "agg_window_ids": agg_window_ids,
                "agg_slot_ids": agg_slot_ids,
            }

        window_latents_flat = self.window_encoder(x, m, c)
        window_latents = window_latents_flat.reshape(bsz, n_windows, -1)
        return {
            "window_latents": window_latents,
            "window_slot_latents": None,
            "agg_window_latents": window_latents,
            "agg_window_ids": window_ids,
            "agg_slot_ids": None,
        }

    def aggregate(
        self,
        window_latents: torch.Tensor,
        window_ids: torch.Tensor,
        slot_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.global_agg(window_latents, window_ids, slot_ids=slot_ids)

    def forward(
        self,
        input_idx: torch.Tensor,
        obs_mask: torch.Tensor,
        window_ids: torch.Tensor,
        coverage: Optional[torch.Tensor],
    ) -> dict:
        encoded = self.encode_windows(
            input_idx=input_idx,
            obs_mask=obs_mask,
            coverage=coverage,
            window_ids=window_ids,
        )
        global_latent, mu, logvar = self.aggregate(
            window_latents=encoded["agg_window_latents"],
            window_ids=encoded["agg_window_ids"],
            slot_ids=encoded["agg_slot_ids"],
        )
        if encoded["window_slot_latents"] is not None:
            logits = self.decoder(encoded["window_slot_latents"], global_latent)
        else:
            logits = self.decoder(encoded["window_latents"], global_latent)
        return {
            "logits": logits,
            "window_latents": encoded["window_latents"],
            "window_slot_latents": encoded["window_slot_latents"],
            "global_latent": global_latent,
            "mu": mu,
            "logvar": logvar,
        }
