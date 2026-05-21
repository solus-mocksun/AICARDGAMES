"""
Self-play opponent pool.

Stores up to MAX_POOL_SIZE snapshots of past model weights.
When sampling an opponent for an episode, draws from this pool using
recency-weighted probabilities — recent checkpoints sampled more often.

Two permanent anchor agents always in the pool:
  - RandomAgent    (always legal random action)
  - HeuristicAgent (game-specific heuristic, e.g. basic strategy for BJ)

This prevents strategy collapse: even if the main agent learns to exploit
a degenerate pool, the anchors ensure it still faces diverse opponents.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from core.types import GameID, ACTION_DIM


MAX_POOL_SIZE = 16
ANCHOR_WEIGHT = 0.20   # anchors get at least 20% of sampling weight


# ---------------------------------------------------------------------------
# Opponent interface (duck-typed — any object with an act() method)
# ---------------------------------------------------------------------------

class RandomAgent:
    """Samples uniformly from legal actions. ELO anchor at 800."""
    name = "RandomAgent"

    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        game_state: dict | None = None,
    ) -> int:
        legal = np.where(action_mask)[0]
        return int(np.random.choice(legal))


# ---------------------------------------------------------------------------
# Model snapshot (frozen copy of agent weights)
# ---------------------------------------------------------------------------

@dataclass
class ModelSnapshot:
    step: int
    game_id: GameID
    backbone_state: dict = field(repr=False)
    adapter_state: dict = field(repr=False)
    policy_state: dict = field(repr=False)
    elo: float = 1000.0


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

class OpponentPool:
    """
    Manages a pool of past model snapshots for self-play.

    Usage:
        pool = OpponentPool()
        pool.add_snapshot(step, game_id, agent)       # after every 100k steps
        opponent = pool.sample_agent(game_id, agent)  # before each episode
    """

    def __init__(self, max_size: int = MAX_POOL_SIZE) -> None:
        self._max_size = max_size
        self._snapshots: list[ModelSnapshot] = []
        self._anchors: dict[str, object] = {
            "random": RandomAgent(),
        }
        self._rng = random.Random()

    def add_anchor(self, name: str, agent: object) -> None:
        """Register a heuristic anchor (e.g. BasicStrategyAgent)."""
        self._anchors[name] = agent

    def add_snapshot(
        self,
        step: int,
        game_id: GameID,
        agent,   # CardGameAgent
    ) -> None:
        """Save a frozen copy of the current agent weights."""
        from models.backbone import CategoryExpertModel
        from models.adapter import GameAdapter

        snapshot = ModelSnapshot(
            step=step,
            game_id=game_id,
            backbone_state=copy.deepcopy(agent.backbone.state_dict()),
            adapter_state=copy.deepcopy(
                agent.adapter_registry.get_or_create(game_id).state_dict()
            ),
            policy_state=copy.deepcopy(agent.policy_head.state_dict()),
        )

        self._snapshots.append(snapshot)

        # Keep only the most recent max_size snapshots
        if len(self._snapshots) > self._max_size:
            self._snapshots.pop(0)

    def sample_opponent(self, game_id: GameID) -> object:
        """
        Sample an opponent for a new episode.
        Returns either an anchor agent or a SnapshotAgent wrapping a past model.

        Sampling weights:
          - Anchors collectively get ANCHOR_WEIGHT probability
          - Remaining pool snapshots get recency-weighted share
        """
        n_anchors = len(self._anchors)
        n_snapshots = len(self._snapshots)

        if n_snapshots == 0:
            # No snapshots yet — always return random
            return self._rng.choice(list(self._anchors.values()))

        # Build weight vector
        # Anchor slots
        anchor_agents = list(self._anchors.values())
        anchor_weights = [ANCHOR_WEIGHT / n_anchors] * n_anchors

        # Snapshot slots: recency-weighted (most recent = highest weight)
        raw = np.array([0.7 ** (n_snapshots - 1 - i) for i in range(n_snapshots)])
        raw = raw / raw.sum() * (1.0 - ANCHOR_WEIGHT)
        snapshot_weights = raw.tolist()

        all_agents = anchor_agents + self._snapshots
        all_weights = anchor_weights + snapshot_weights

        chosen = self._rng.choices(all_agents, weights=all_weights, k=1)[0]

        if isinstance(chosen, ModelSnapshot):
            return SnapshotAgent(chosen)
        return chosen

    def pool_size(self) -> int:
        return len(self._snapshots)


# ---------------------------------------------------------------------------
# Snapshot agent — wraps a frozen model checkpoint for inference
# ---------------------------------------------------------------------------

class SnapshotAgent:
    """
    Wraps a ModelSnapshot and exposes an act() interface.
    Loads weights lazily on first call to act().
    """

    def __init__(self, snapshot: ModelSnapshot) -> None:
        self._snapshot = snapshot
        self._agent = None   # lazy init

    def _ensure_loaded(self) -> None:
        if self._agent is not None:
            return
        from models.full_agent import build_agent
        agent = build_agent(device=torch.device("cpu"))
        agent.backbone.load_state_dict(self._snapshot.backbone_state)
        agent.set_game(self._snapshot.game_id)
        adapter = agent.adapter_registry.get_or_create(self._snapshot.game_id)
        adapter.load_state_dict(self._snapshot.adapter_state)
        agent.policy_head.load_state_dict(self._snapshot.policy_state)
        agent.eval()
        self._agent = agent

    def act(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        game_state: dict | None = None,
    ) -> int:
        self._ensure_loaded()
        return self._agent.act_deterministic(obs, action_mask)

    @property
    def name(self) -> str:
        return f"Snapshot(step={self._snapshot.step})"
