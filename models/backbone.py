"""
CategoryExpertModel — the shared Transformer backbone.

One instance per category (6 total). Processes the 688-dim unified state
and returns a 512-dim hidden vector used by the policy and value heads.

Architecture:
  InputProjection  : Linear(688, 512) + LayerNorm
  ContextEmbedding : category_embed(6→512) + game_embed(20→512), additive
  HistoryEncoding  : optional 16-step sequence via learned positional embed
  TransformerEncoder: 8 layers, 8 heads, d_ff=2048, pre-norm, GELU
  Output           : [B, 512] hidden state

Total params: ~26.1M (backbone only, excluding LoRA adapters and heads).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from core.types import NUM_CATEGORIES, NUM_GAMES, OBS_DIM

D_MODEL = 512
N_LAYERS = 8
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1
HISTORY_LEN = 16   # steps of history the model can attend over


class _PreNormTransformerLayer(nn.Module):
    """Single Transformer layer with pre-layer-norm (more stable than post-norm)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        # Pre-norm feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class CategoryExpertModel(nn.Module):
    """
    Shared Transformer backbone for one game category.

    Forward modes:
      Single-step (default): obs shape [B, 688]   → hidden [B, 512]
      History mode:          obs shape [B, T, 688] → hidden [B, 512]
        T must be ≤ HISTORY_LEN. Shorter sequences are left-padded with zeros.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(OBS_DIM, d_model, bias=False)
        self.input_norm = nn.LayerNorm(d_model)

        # Context embeddings (additive, not concatenated)
        self.category_embed = nn.Embedding(NUM_CATEGORIES, d_model)
        self.game_embed = nn.Embedding(NUM_GAMES, d_model)

        # Learned positional encoding for history window
        self.pos_embed = nn.Embedding(HISTORY_LEN, d_model)

        # Transformer layers (LoRA adapters are injected into these later)
        self.layers = nn.ModuleList([
            _PreNormTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.normal_(self.category_embed.weight, std=0.02)
        nn.init.normal_(self.game_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.attn.in_proj_weight)
            nn.init.zeros_(layer.attn.in_proj_bias)
            nn.init.xavier_uniform_(layer.attn.out_proj.weight)
            nn.init.zeros_(layer.attn.out_proj.bias)

    def forward(
        self,
        obs: torch.Tensor,
        category_ids: torch.Tensor,
        game_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs:          [B, OBS_DIM] or [B, T, OBS_DIM]
            category_ids: [B] int64
            game_ids:     [B] int64

        Returns:
            hidden: [B, D_MODEL]
        """
        single_step = obs.dim() == 2
        if single_step:
            obs = obs.unsqueeze(1)   # [B, 1, OBS_DIM]

        B, T, _ = obs.shape

        # Project each step to d_model
        x = self.input_norm(self.input_proj(obs))   # [B, T, d_model]

        # Add context embeddings (broadcast over T)
        cat_emb = self.category_embed(category_ids).unsqueeze(1)   # [B, 1, d]
        game_emb = self.game_embed(game_ids).unsqueeze(1)           # [B, 1, d]
        x = x + cat_emb + game_emb

        # Add positional encoding (last T positions in the window)
        positions = torch.arange(HISTORY_LEN - T, HISTORY_LEN, device=obs.device)
        x = x + self.pos_embed(positions).unsqueeze(0)   # [1, T, d]

        # Causal mask so future steps can't attend to past (not needed for T=1,
        # but correct for history mode)
        causal_mask = _causal_mask(T, device=obs.device) if T > 1 else None

        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask)

        x = self.final_norm(x)

        # Pool: take the last token (most recent state)
        hidden = x[:, -1, :]   # [B, d_model]
        return hidden

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask for causal (autoregressive) attention."""
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask.masked_fill(mask, float("-inf"))
