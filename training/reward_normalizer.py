"""
Per-game reward normalization using Welford's online algorithm.

Different games have wildly different reward scales:
  Blackjack: {-1, 0, +1}
  Poker NLH: {-10000, ..., +10000} (chip units)
  Baccarat:  {-1, 0, +0.95, +8}

Without normalization, mixing these in one PPO batch destroys training
because the poker gradient completely overwhelms the blackjack gradient.

Each game gets its own RunningMeanStd that tracks reward statistics.
All rewards are normalized to approximately N(0, 1) before PPO loss.
"""

from __future__ import annotations

import numpy as np

from core.types import GameID


class RunningMeanStd:
    """
    Welford's online algorithm for computing running mean and variance.
    Numerically stable for streaming data.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon   # avoid division by zero on first update

    def update(self, x: float | np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).flatten()
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.size

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


class RewardNormalizer:
    """
    Manages one RunningMeanStd per game.
    Call normalize(game_id, reward) before storing in the rollout buffer.
    """

    def __init__(self) -> None:
        self._stats: dict[GameID, RunningMeanStd] = {}

    def _get(self, game_id: GameID) -> RunningMeanStd:
        if game_id not in self._stats:
            self._stats[game_id] = RunningMeanStd()
        return self._stats[game_id]

    def update(self, game_id: GameID, reward: float) -> None:
        self._get(game_id).update(reward)

    def normalize(self, game_id: GameID, reward: float) -> float:
        rms = self._get(game_id)
        rms.update(reward)
        return rms.normalize(reward)

    def normalize_batch(
        self, game_id: GameID, rewards: np.ndarray
    ) -> np.ndarray:
        rms = self._get(game_id)
        rms.update(rewards)
        return (rewards - rms.mean) / rms.std

    def state_dict(self) -> dict:
        return {str(int(gid)): rms.state_dict() for gid, rms in self._stats.items()}

    def load_state_dict(self, d: dict) -> None:
        for key, val in d.items():
            gid = GameID(int(key))
            self._get(gid).load_state_dict(val)
