"""
Policy and value heads attached to the backbone's 512-dim hidden state.

PolicyHead : [B, 512] → [B, 512] log-probabilities (masked)
ValueHead  : [B, 512] → [B, 1]  value estimate in (-1, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.types import ACTION_DIM, D_MODEL as _D

# Use the constant from backbone if available, fallback to 512
try:
    from models.backbone import D_MODEL
except ImportError:
    D_MODEL = 512

_NEG_INF = -1e9   # value assigned to illegal actions before softmax


class PolicyHead(nn.Module):
    """
    Maps backbone hidden state to a masked log-probability distribution
    over the 512 unified action slots.

    Illegal actions (action_mask=False) are set to -inf before softmax,
    so the model can never sample them.
    """

    def __init__(self, d_model: int = D_MODEL, n_actions: int = ACTION_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, n_actions),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden:      [B, d_model]
            action_mask: [B, n_actions] bool — True = legal

        Returns:
            log_probs: [B, n_actions] — log-softmax, illegal actions = -inf
        """
        logits = self.net(hidden)                          # [B, n_actions]
        logits = logits.masked_fill(~action_mask, _NEG_INF)
        return F.log_softmax(logits, dim=-1)

    def get_dist(
        self,
        hidden: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.distributions.Categorical:
        """Return a Categorical distribution for sampling and entropy computation."""
        logits = self.net(hidden)
        logits = logits.masked_fill(~action_mask, _NEG_INF)
        return torch.distributions.Categorical(logits=logits)


class ValueHead(nn.Module):
    """
    Maps backbone hidden state to a scalar value estimate.
    Output is tanh-squashed to (-1, 1) matching reward normalization.
    """

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Last linear before tanh: smaller init so values start near 0
        nn.init.orthogonal_(self.net[-2].weight, gain=0.01)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, d_model]
        Returns:
            value: [B, 1]
        """
        return self.net(hidden)
