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
    latent_tokens: int = 8
    latent_feedback_mode: str = "cross_attn"
    use_strong_latent_reconstruction: bool = False
    latent_recon_mode: str = "film_plus_crossattn"
    latent_recon_hidden_dim: Optional[int] = None
    latent_recon_num_heads: int = 4
    latent_recon_dropout: float = 0.05
    latent_recon_num_tokens: int = 4
    latent_cross_attn_residual_scale: float = 0.1
    coverage_conditioning_mode: str = "none"
    coverage_embed_dim: int = 0
    use_obs_embedding: bool = True
    use_snp_id_embedding: bool = False


class LatentFiLM(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * out_dim),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_beta = self.net(z)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma, beta


class TokenSNPMaskedModel(nn.Module):
    def __init__(self, cfg: TokenSNPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.latent_feedback_mode != "cross_attn":
            raise ValueError(
                f"Unsupported latent_feedback_mode={cfg.latent_feedback_mode!r}. "
                "Only 'cross_attn' is currently implemented."
            )
        if cfg.coverage_conditioning_mode not in {"none", "concat", "film", "film_concat"}:
            raise ValueError(
                f"Unsupported coverage_conditioning_mode={cfg.coverage_conditioning_mode!r}. "
                "Use one of: none, concat, film, film_concat"
            )
        if cfg.latent_recon_mode not in {
            "film_only",
            "crossattn_only",
            "film_plus_crossattn",
            "latent_only",
        }:
            raise ValueError(
                f"Unsupported latent_recon_mode={cfg.latent_recon_mode!r}. "
                "Use one of: film_only, crossattn_only, film_plus_crossattn, latent_only"
            )
        if cfg.coverage_embed_dim < 0:
            raise ValueError("coverage_embed_dim must be >= 0")
        if cfg.latent_recon_num_heads < 1:
            raise ValueError("latent_recon_num_heads must be >= 1")
        if cfg.latent_recon_num_tokens < 1:
            raise ValueError("latent_recon_num_tokens must be >= 1")
        if not (0.0 <= cfg.latent_cross_attn_residual_scale <= 1.0):
            raise ValueError("latent_cross_attn_residual_scale must be between 0 and 1")

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
        self.latent_queries = nn.Parameter(torch.randn(cfg.latent_tokens, cfg.d_model) * 0.02)
        self.token_to_latent_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.latent_from_tokens_norm = nn.LayerNorm(cfg.d_model)
        self.token_from_latent_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.token_feedback_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        self.latent_proj = nn.Linear(cfg.d_model, cfg.latent_dim)
        latent_recon_hidden_dim = cfg.latent_recon_hidden_dim or cfg.d_model
        self.use_strong_latent_reconstruction = cfg.use_strong_latent_reconstruction
        self.latent_recon_mode = cfg.latent_recon_mode
        self.latent_recon_num_tokens = cfg.latent_recon_num_tokens
        self.latent_cross_attn_residual_scale = cfg.latent_cross_attn_residual_scale
        self.z_to_dec = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.latent_film = LatentFiLM(
            latent_dim=cfg.latent_dim,
            hidden_dim=latent_recon_hidden_dim,
            out_dim=cfg.d_model,
        )
        self.z_to_latent_tokens = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_recon_num_tokens * cfg.d_model),
            nn.GELU(),
            nn.Linear(
                cfg.latent_recon_num_tokens * cfg.d_model,
                cfg.latent_recon_num_tokens * cfg.d_model,
            ),
        )
        self.latent_cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.latent_recon_num_heads,
            dropout=cfg.latent_recon_dropout,
            batch_first=True,
        )
        self.latent_cross_attn_norm = nn.LayerNorm(cfg.d_model)
        self.latent_only_input_norm = nn.LayerNorm(cfg.d_model)

        self.coverage_conditioning_mode = cfg.coverage_conditioning_mode
        self.coverage_embed = None
        self.coverage_concat_proj = None
        self.coverage_film = None

        if self.coverage_conditioning_mode in {"concat", "film_concat"}:
            cov_feat_dim = 1
            if cfg.coverage_embed_dim > 0:
                self.coverage_embed = nn.Sequential(
                    nn.Linear(1, cfg.coverage_embed_dim),
                    nn.GELU(),
                    nn.Linear(cfg.coverage_embed_dim, cfg.coverage_embed_dim),
                    nn.GELU(),
                )
                cov_feat_dim = cfg.coverage_embed_dim
            self.coverage_concat_proj = nn.Linear(cfg.latent_dim + cov_feat_dim, cfg.d_model)

        if self.coverage_conditioning_mode in {"film", "film_concat"}:
            self.coverage_film = nn.Sequential(
                nn.Linear(1, 64),
                nn.GELU(),
                nn.Linear(64, 2 * cfg.d_model),
            )

    @staticmethod
    def _sanitize_padding_mask(mask: torch.Tensor) -> torch.Tensor:
        # Transformer encoder requires at least one unmasked token per sequence.
        out = mask.clone()
        all_masked = out.all(dim=1)
        if all_masked.any():
            out[all_masked, 0] = False
        return out

    def _apply_coverage_conditioning(
        self,
        h_recon: torch.Tensor,
        latent_for_reconstruction: torch.Tensor,
        coverage_std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.coverage_conditioning_mode == "none":
            return h_recon
        if coverage_std is None:
            raise ValueError("coverage_std is required when coverage_conditioning_mode != 'none'")

        cov_scalar = coverage_std.unsqueeze(-1).to(dtype=h_recon.dtype)
        out = h_recon
        if self.coverage_conditioning_mode in {"concat", "film_concat"}:
            cov_feat = cov_scalar
            if self.coverage_embed is not None:
                cov_feat = self.coverage_embed(cov_scalar)
            z_cov = torch.cat([latent_for_reconstruction, cov_feat], dim=-1)
            assert self.coverage_concat_proj is not None
            out = out + self.coverage_concat_proj(z_cov).unsqueeze(1)

        if self.coverage_conditioning_mode in {"film", "film_concat"}:
            assert self.coverage_film is not None
            gamma_beta = self.coverage_film(cov_scalar)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            out = out * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return out

    def _latent_only_decoder_input(
        self,
        pos: torch.Tensor,
        obs_mask: torch.Tensor,
        snp_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h_recon = self.pos_embed(pos)
        if self.use_obs_embedding and self.obs_embed is not None:
            obs_ids = (obs_mask > 0.5).to(dtype=torch.long)
            h_recon = h_recon + self.obs_embed(obs_ids)
        if self.use_snp_id_embedding and self.snp_embed is not None and snp_idx is not None:
            clipped = snp_idx.clamp(min=0, max=self.cfg.n_snps - 1)
            h_recon = h_recon + self.snp_embed(clipped)
        return self.latent_only_input_norm(h_recon)

    def _latent_tokens_from_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.z_to_latent_tokens(z).view(
            z.shape[0], self.latent_recon_num_tokens, self.cfg.d_model
        )

    def _apply_decoder_latent_cross_attention(
        self,
        query_states: torch.Tensor,
        latent_tokens: torch.Tensor,
        residual_scale: float,
    ) -> torch.Tensor:
        attn_out, _ = self.latent_cross_attn(
            query=query_states,
            key=latent_tokens,
            value=latent_tokens,
            need_weights=False,
        )
        mixed = (residual_scale * query_states) + attn_out
        return self.latent_cross_attn_norm(mixed)

    def _build_latent_only_states(
        self,
        pos: torch.Tensor,
        obs_mask: torch.Tensor,
        snp_idx: Optional[torch.Tensor],
        z: torch.Tensor,
        z_dec: torch.Tensor,
        coverage_std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        latent_only_states = self._latent_only_decoder_input(
            pos=pos,
            obs_mask=obs_mask,
            snp_idx=snp_idx,
        )
        latent_only_states = self._apply_coverage_conditioning(latent_only_states, z, coverage_std)
        latent_only_states = latent_only_states + z_dec.unsqueeze(1)
        latent_tokens = self._latent_tokens_from_z(z)
        return self._apply_decoder_latent_cross_attention(
            query_states=latent_only_states,
            latent_tokens=latent_tokens,
            residual_scale=self.latent_cross_attn_residual_scale,
        )

    def _apply_latent_reconstruction_path(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        obs_mask: torch.Tensor,
        snp_idx: Optional[torch.Tensor],
        window_embedding: torch.Tensor,
        coverage_std: Optional[torch.Tensor],
        zero_latent: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = torch.zeros_like(window_embedding) if zero_latent else window_embedding
        if not self.use_strong_latent_reconstruction:
            return self._apply_coverage_conditioning(h, z, coverage_std), None
        z_dec = self.z_to_dec(z)
        latent_tokens = self._latent_tokens_from_z(z)
        if self.latent_recon_mode == "latent_only":
            h_recon = self._build_latent_only_states(
                pos=pos,
                obs_mask=obs_mask,
                snp_idx=snp_idx,
                z=z,
                z_dec=z_dec,
                coverage_std=coverage_std,
            )
        else:
            h_recon = h

        aux_latent_only_logits: Optional[torch.Tensor] = None
        if self.latent_recon_mode != "latent_only":
            h_recon = self._apply_coverage_conditioning(h_recon, z, coverage_std)
            h_recon = h_recon + z_dec.unsqueeze(1)

            if self.latent_recon_mode in {"film_only", "film_plus_crossattn"}:
                gamma, beta = self.latent_film(z)
                h_recon = h_recon * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

            if self.latent_recon_mode in {"crossattn_only", "film_plus_crossattn"}:
                h_recon = self._apply_decoder_latent_cross_attention(
                    query_states=h_recon,
                    latent_tokens=latent_tokens,
                    residual_scale=self.latent_cross_attn_residual_scale,
                )

            latent_only_states = self._build_latent_only_states(
                pos=pos,
                obs_mask=obs_mask,
                snp_idx=snp_idx,
                z=z,
                z_dec=z_dec,
                coverage_std=coverage_std,
            )
            aux_latent_only_logits = self.head(latent_only_states)

        return h_recon, aux_latent_only_logits

    def forward(
        self,
        tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        snp_idx: Optional[torch.Tensor] = None,
        coverage_std: Optional[torch.Tensor] = None,
        zero_pooled_latent_for_reconstruction: bool = False,
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
        token_padding_mask = self._sanitize_padding_mask(obs_mask[:, :seq_len] <= 0.5)

        latent_q = self.latent_queries.unsqueeze(0).expand(bsz, -1, -1)
        latent_from_tokens, _ = self.token_to_latent_attn(
            query=latent_q,
            key=h,
            value=h,
            key_padding_mask=token_padding_mask,
            need_weights=False,
        )
        latent_states = self.latent_from_tokens_norm(latent_q + latent_from_tokens)

        token_from_latents, _ = self.token_from_latent_attn(
            query=h,
            key=latent_states,
            value=latent_states,
            need_weights=False,
        )
        h = self.token_feedback_norm(h + token_from_latents)

        window_latent_tokens = self.latent_proj(latent_states)
        window_embedding = window_latent_tokens.mean(dim=1)
        h_recon, aux_latent_only_logits = self._apply_latent_reconstruction_path(
            h=h,
            pos=pos,
            obs_mask=obs_mask[:, :seq_len],
            snp_idx=snp_idx if snp_idx is None else snp_idx[:, :seq_len],
            window_embedding=window_embedding,
            coverage_std=coverage_std,
            zero_latent=zero_pooled_latent_for_reconstruction,
        )

        logits = self.head(h_recon)

        return {
            "logits": logits,
            "aux_latent_only_logits": aux_latent_only_logits,
            "hidden": h_recon,
            "window_embedding": window_embedding,
            "window_latent_tokens": window_latent_tokens,
        }
