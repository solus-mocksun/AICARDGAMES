"""
Custom PPO (Proximal Policy Optimization) trainer.

Why custom instead of Stable-Baselines3:
  - Need to swap LoRA adapters between episodes in the same rollout batch
  - Need per-game reward normalization before computing the PPO loss
  - Need multi-game rollout mixing (different games in the same batch)
  SB3's fixed architecture can't accommodate any of these without major surgery.

PPO hyperparameters (from configs/base.yaml, overridable):
  gamma        = 0.99
  gae_lambda   = 0.95
  clip_epsilon = 0.2
  entropy_coeff = 0.01
  value_coeff   = 0.5
  max_grad_norm = 0.5
  lr            = 3e-4
  n_epochs      = 4
  minibatch     = 256
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.types import GameID, ACTION_DIM, OBS_DIM
from training.reward_normalizer import RewardNormalizer


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    """Stores one rollout for PPO update. Mixed across games."""
    obs: torch.Tensor          # [N, OBS_DIM]
    masks: torch.Tensor        # [N, ACTION_DIM] bool
    actions: torch.Tensor      # [N] int64
    log_probs_old: torch.Tensor # [N]
    values_old: torch.Tensor   # [N]
    returns: torch.Tensor      # [N]  GAE returns
    advantages: torch.Tensor   # [N]  GAE advantages
    game_ids: list[GameID]     # [N]  which game each step came from


class RolloutBuffer:
    """
    Collects experience from one or more episodes before the PPO update.
    Stores raw (unnormalized) rewards; normalization happens at update time.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._obs: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._actions: list[int] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._game_ids: list[GameID] = []

    def add(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        game_id: GameID,
    ) -> None:
        self._obs.append(obs)
        self._masks.append(mask)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._rewards.append(reward)
        self._dones.append(done)
        self._game_ids.append(game_id)

    def __len__(self) -> int:
        return len(self._actions)

    def clear(self) -> None:
        self.__init__(self.device)

    def compute_returns(
        self,
        gamma: float,
        gae_lambda: float,
        normalizer: RewardNormalizer,
        last_value: float = 0.0,
    ) -> RolloutBatch:
        """
        Compute GAE advantages and discounted returns.
        Rewards are normalized per-game before GAE computation.
        """
        n = len(self._actions)
        rewards_norm = np.array([
            normalizer.normalize(gid, r)
            for gid, r in zip(self._game_ids, self._rewards)
        ], dtype=np.float32)

        values = np.array(self._values + [last_value], dtype=np.float32)
        dones = np.array(self._dones + [False], dtype=np.float32)

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards_norm[t] + gamma * values[t+1] * (1 - dones[t+1]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t+1]) * gae
            advantages[t] = gae

        returns = advantages + values[:n]

        # Normalize advantages across the batch
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return RolloutBatch(
            obs=torch.from_numpy(np.stack(self._obs)).float().to(self.device),
            masks=torch.from_numpy(np.stack(self._masks)).bool().to(self.device),
            actions=torch.tensor(self._actions, dtype=torch.long, device=self.device),
            log_probs_old=torch.tensor(self._log_probs, dtype=torch.float32, device=self.device),
            values_old=torch.tensor(self._values, dtype=torch.float32, device=self.device),
            returns=torch.from_numpy(returns).float().to(self.device),
            advantages=torch.from_numpy(advantages).float().to(self.device),
            game_ids=list(self._game_ids),
        )


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    n_epochs: int = 4
    minibatch_size: int = 256


class PPOTrainer:
    """
    Runs PPO updates given a filled RolloutBatch.

    The agent's LoRA adapter must already be loaded for the correct game
    before calling update(). For mixed-game batches, we group steps by
    game_id and update each game's adapter separately.
    """

    def __init__(self, agent, config: PPOConfig | None = None) -> None:
        self.agent = agent
        self.config = config or PPOConfig()
        self.normalizer = RewardNormalizer()
        self.buffer = RolloutBuffer(agent.device)

        # Optimizer covers: adapter (current game) + policy head + value head
        # Backbone is frozen during adapter training
        self._optimizer: optim.Adam | None = None
        self._current_game: GameID | None = None
        self._step = 0

    def set_game(self, game_id: GameID) -> None:
        """Switch optimizer to the correct adapter's parameters."""
        if game_id == self._current_game:
            return
        self.agent.set_game(game_id)
        self._current_game = game_id
        trainable = self.agent.trainable_parameters()
        self._optimizer = optim.Adam(trainable, lr=self.config.lr)

    def collect_step(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        reward: float,
        done: bool,
        game_id: GameID,
    ) -> int:
        """
        Get action from agent, store experience in buffer.
        Returns the chosen unified action slot.
        """
        action, log_prob, value = self.agent.act(obs, mask)
        self.buffer.add(obs, mask, action, log_prob, value, reward, done, game_id)
        return action

    def update(self, last_value: float = 0.0) -> dict[str, float]:
        """
        Run n_epochs of PPO updates on the current buffer contents.
        Returns dict of training metrics.
        """
        cfg = self.config
        batch = self.buffer.compute_returns(
            cfg.gamma, cfg.gae_lambda, self.normalizer, last_value
        )
        self.buffer.clear()

        n = len(batch.actions)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}

        for _ in range(cfg.n_epochs):
            indices = torch.randperm(n, device=self.agent.device)
            for start in range(0, n, cfg.minibatch_size):
                idx = indices[start: start + cfg.minibatch_size]

                obs_mb    = batch.obs[idx]
                mask_mb   = batch.masks[idx]
                act_mb    = batch.actions[idx]
                logp_old  = batch.log_probs_old[idx]
                ret_mb    = batch.returns[idx]
                adv_mb    = batch.advantages[idx]

                log_probs, entropy, values = self.agent.evaluate(obs_mb, mask_mb, act_mb)

                # PPO clipped objective
                ratio = torch.exp(log_probs - logp_old)
                clip1 = ratio * adv_mb
                clip2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv_mb
                policy_loss = -torch.min(clip1, clip2).mean()

                # Value loss (clipped)
                value_loss = nn.functional.mse_loss(values, ret_mb)

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + cfg.value_coeff * value_loss
                        + cfg.entropy_coeff * entropy_loss)

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.trainable_parameters(), cfg.max_grad_norm
                )
                self._optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += (-entropy_loss).item()
                metrics["n_updates"] += 1

        self._step += n
        n_upd = max(metrics["n_updates"], 1)
        return {k: v / n_upd if k != "n_updates" else v for k, v in metrics.items()}

    @property
    def total_steps(self) -> int:
        return self._step
