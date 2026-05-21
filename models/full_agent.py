"""
CardGameAgent — combines backbone + adapter + heads into one inference object.

Usage:
    agent = CardGameAgent(backbone, adapter_registry, policy_head, value_head)
    agent.set_game(GameID.BLACKJACK)

    # During rollout (single step)
    action, log_prob, value = agent.act(obs_tensor, mask_tensor)

    # During PPO update (batch)
    log_probs, entropy, values = agent.evaluate(obs_batch, mask_batch, action_batch)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from core.types import GameID, GAME_TO_CATEGORY, OBS_DIM, ACTION_DIM
from models.backbone import CategoryExpertModel
from models.adapter import AdapterRegistry
from models.policy_value_head import PolicyHead, ValueHead


class CardGameAgent(nn.Module):
    """
    The full inference stack: backbone → adapter → policy + value.

    The backbone and heads are shared across games within a category.
    The adapter is swapped per game via set_game().
    """

    def __init__(
        self,
        backbone: CategoryExpertModel,
        adapter_registry: AdapterRegistry,
        policy_head: PolicyHead,
        value_head: ValueHead,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter_registry = adapter_registry
        self.policy_head = policy_head
        self.value_head = value_head
        self.device = device or torch.device("cpu")

        self._current_game: GameID | None = None

    def set_game(self, game_id: GameID) -> None:
        """Swap to the adapter for this game. Call before each episode."""
        if game_id != self._current_game:
            self.adapter_registry.load(game_id)
            self._current_game = game_id

    # ---- Inference -------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        action_mask: np.ndarray | torch.Tensor,
    ) -> tuple[int, float, float]:
        """
        Sample one action for a single environment step.

        Args:
            obs:         (OBS_DIM,) numpy array or tensor
            action_mask: (ACTION_DIM,) bool array or tensor

        Returns:
            action:   int — unified action slot chosen
            log_prob: float
            value:    float
        """
        obs_t, mask_t = self._to_tensors_single(obs, action_mask)

        hidden = self._forward_backbone(obs_t)
        dist = self.policy_head.get_dist(hidden, mask_t)
        action_t = dist.sample()

        log_prob = dist.log_prob(action_t).item()
        value = self.value_head(hidden).squeeze(-1).item()
        return action_t.item(), log_prob, value

    @torch.no_grad()
    def act_deterministic(
        self,
        obs: np.ndarray | torch.Tensor,
        action_mask: np.ndarray | torch.Tensor,
    ) -> int:
        """Greedy action (argmax), used during evaluation."""
        obs_t, mask_t = self._to_tensors_single(obs, action_mask)
        hidden = self._forward_backbone(obs_t)
        log_probs = self.policy_head(hidden, mask_t)
        return log_probs.argmax(dim=-1).item()

    def evaluate(
        self,
        obs_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate a batch of (obs, action) pairs for the PPO update.

        Args:
            obs_batch:    [B, OBS_DIM]
            mask_batch:   [B, ACTION_DIM] bool
            action_batch: [B] int64

        Returns:
            log_probs: [B]   log prob of action_batch under current policy
            entropy:   [B]   entropy of the distribution
            values:    [B]   value estimates
        """
        assert self._current_game is not None, "Call set_game() before evaluate()"

        B = obs_batch.shape[0]
        category_ids, game_ids = self._id_tensors(B, obs_batch.device)

        hidden = self.backbone(obs_batch, category_ids, game_ids)  # [B, 512]
        dist = self.policy_head.get_dist(hidden, mask_batch)

        log_probs = dist.log_prob(action_batch)   # [B]
        entropy = dist.entropy()                  # [B]
        values = self.value_head(hidden).squeeze(-1)  # [B]

        return log_probs, entropy, values

    # ---- Internal helpers -----------------------------------------------

    def _forward_backbone(self, obs_t: torch.Tensor) -> torch.Tensor:
        """obs_t: [1, OBS_DIM] → hidden [1, 512]"""
        assert self._current_game is not None, "Call set_game() before act()"
        category_ids, game_ids = self._id_tensors(1, obs_t.device)
        return self.backbone(obs_t, category_ids, game_ids)

    def _id_tensors(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cat_id = int(GAME_TO_CATEGORY[self._current_game])
        game_id = int(self._current_game)
        category_ids = torch.full((batch_size,), cat_id, dtype=torch.long, device=device)
        game_ids = torch.full((batch_size,), game_id, dtype=torch.long, device=device)
        return category_ids, game_ids

    def _to_tensors_single(
        self,
        obs: np.ndarray | torch.Tensor,
        action_mask: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.from_numpy(action_mask).bool()
        obs = obs.unsqueeze(0).to(self.device)          # [1, OBS_DIM]
        action_mask = action_mask.unsqueeze(0).to(self.device)  # [1, ACTION_DIM]
        return obs, action_mask

    def trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters that should be updated by the optimizer."""
        params = []
        if self._current_game is not None:
            params += self.adapter_registry.trainable_parameters(self._current_game)
        params += list(self.policy_head.parameters())
        params += list(self.value_head.parameters())
        return params

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_agent(device: torch.device | None = None) -> CardGameAgent:
    """
    Convenience factory: build a fresh agent with default hyperparameters.
    Call once per category at the start of training.
    """
    from models.backbone import CategoryExpertModel
    from models.adapter import AdapterRegistry
    from models.policy_value_head import PolicyHead, ValueHead

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = CategoryExpertModel().to(device)
    adapter_registry = AdapterRegistry(backbone)
    policy_head = PolicyHead().to(device)
    value_head = ValueHead().to(device)

    return CardGameAgent(backbone, adapter_registry, policy_head, value_head, device)
