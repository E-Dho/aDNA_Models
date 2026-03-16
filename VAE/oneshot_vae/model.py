from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OneShotMaskedVAEConfig:
    n_snps: int
    latent_dim: int = 64
    chunk_size: int = 512
    token_embed_dim: int = 8
    chunk_dim: int = 256
    encoder_layers: int = 2
    encoder_heads: int = 8
    encoder_ff_mult: int = 4
    decoder_hidden_dim: int = 512
    dropout: float = 0.1
    n_genotype_classes: int = 3


class OneShotMaskedVAE(nn.Module):
    def __init__(self, cfg: OneShotMaskedVAEConfig) -> None:
        super().__init__()
        if cfg.n_snps <= 0:
            raise ValueError("n_snps must be > 0")
        if cfg.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if cfg.chunk_dim % cfg.encoder_heads != 0:
            raise ValueError("chunk_dim must be divisible by encoder_heads")

        self.cfg = cfg
        self.missing_token_id = cfg.n_genotype_classes
        self.mask_token_id = cfg.n_genotype_classes + 1
        self.vocab_size = cfg.n_genotype_classes + 2
        self.n_chunks = int(math.ceil(cfg.n_snps / cfg.chunk_size))
        self.padded_n_snps = self.n_chunks * cfg.chunk_size

        self.token_embed = nn.Embedding(self.vocab_size, cfg.token_embed_dim)
        self.chunk_encoder = nn.Sequential(
            nn.Linear(cfg.chunk_size * cfg.token_embed_dim, cfg.chunk_dim),
            nn.LayerNorm(cfg.chunk_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.chunk_pos_embed = nn.Embedding(self.n_chunks, cfg.chunk_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.chunk_dim,
            nhead=cfg.encoder_heads,
            dim_feedforward=cfg.chunk_dim * cfg.encoder_ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
        self.transformer_norm = nn.LayerNorm(cfg.chunk_dim)
        self.pool_score = nn.Linear(cfg.chunk_dim, 1)
        self.mu_layer = nn.Linear(cfg.chunk_dim, cfg.latent_dim)
        self.logvar_layer = nn.Linear(cfg.chunk_dim, cfg.latent_dim)

        self.latent_to_chunk = nn.Linear(cfg.latent_dim, cfg.chunk_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.chunk_dim, cfg.decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.decoder_hidden_dim, cfg.chunk_size * cfg.n_genotype_classes),
        )

    def _pad_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == self.padded_n_snps:
            return tokens
        pad_len = self.padded_n_snps - tokens.shape[1]
        pad = torch.full(
            (tokens.shape[0], pad_len),
            fill_value=self.missing_token_id,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return torch.cat([tokens, pad], dim=1)

    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padded = self._pad_tokens(tokens)
        emb = self.token_embed(padded)
        bsz = emb.shape[0]
        chunked = emb.reshape(bsz, self.n_chunks, self.cfg.chunk_size * self.cfg.token_embed_dim)
        chunk_repr = self.chunk_encoder(chunked)
        pos_ids = torch.arange(self.n_chunks, device=tokens.device)
        chunk_repr = chunk_repr + self.chunk_pos_embed(pos_ids).unsqueeze(0)
        encoded = self.transformer(chunk_repr)
        encoded = self.transformer_norm(encoded)
        scores = self.pool_score(encoded).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        mu = self.mu_layer(pooled)
        logvar = self.logvar_layer(pooled)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        chunk_base = self.latent_to_chunk(z).unsqueeze(1)
        pos_ids = torch.arange(self.n_chunks, device=z.device)
        chunk_states = chunk_base + self.chunk_pos_embed(pos_ids).unsqueeze(0)
        chunk_logits = self.decoder(chunk_states)
        logits = chunk_logits.reshape(
            z.shape[0],
            self.n_chunks,
            self.cfg.chunk_size,
            self.cfg.n_genotype_classes,
        )
        logits = logits.reshape(z.shape[0], self.padded_n_snps, self.cfg.n_genotype_classes)
        return logits[:, : self.cfg.n_snps, :]

    def forward(self, tokens: torch.Tensor) -> dict:
        mu, logvar = self.encode(tokens)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return {"logits": logits, "mu": mu, "logvar": logvar, "z": z}
