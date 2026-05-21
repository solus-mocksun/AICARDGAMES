"""
BaseCardGameEnv — abstract base for all 18 game environments.

Contract:
  - Follows the PettingZoo AEC (Agent-Environment-Cycle) API.
  - observe() always returns UnifiedObs: {"observation": (688,), "action_mask": (512,)}.
  - step() accepts a unified action slot (int 0–511).
  - Subclasses translate between the unified interface and their native engine
    (RLCard, OpenSpiel, or custom logic).

Single-player games (solitaire):
  - agents = ["player_0"]
  - No opponent logic needed.

Multi-player games:
  - agents = ["player_0", ..., "player_N"]
  - agent_selection cycles through active agents each step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from gymnasium import spaces

from core.action_space import ActionSpaceRegistry, ActionMapper
from core.state_encoder import UnifiedStateEncoder
from core.types import (
    ACTION_DIM, OBS_DIM, EnvMetadata, GameID, UnifiedObs,
)


class BaseCardGameEnv(ABC):
    """
    Base class for all card game environments.

    PettingZoo AEC-compatible interface without requiring the full pettingzoo
    dependency during Phase 0 testing. Subclasses that wrap PettingZoo games
    can inherit from both this class and pettingzoo.utils.env.AECEnv.

    Key invariants:
      - observe(agent) always returns a dict with keys "observation" and "action_mask"
      - action_mask has exactly one True entry per legal action
      - observation is a (688,) float32 vector
      - terminations[agent] is True when the game is over for that agent
    """

    metadata: EnvMetadata           # set on subclass definition
    render_mode: str | None = None

    def __init__(
        self,
        strict_partial_info: bool = True,
        render_mode: str | None = None,
    ) -> None:
        self.render_mode = render_mode
        self._strict_partial_info = strict_partial_info

        self._encoder: UnifiedStateEncoder = self._make_encoder()
        self._mapper: ActionMapper = ActionSpaceRegistry.get(self.metadata.game_id)

        # AEC state (populated by reset())
        self.agents: list[str] = []
        self.possible_agents: list[str] = []
        self.agent_selection: str = ""
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict] = {}

        # Gym spaces (same for every agent)
        self.observation_spaces: dict[str, spaces.Dict] = {}
        self.action_spaces: dict[str, spaces.Discrete] = {}

    # ---- Subclass must implement these ----------------------------------

    @abstractmethod
    def _make_encoder(self) -> UnifiedStateEncoder:
        """Return the encoder instance for this game."""

    @abstractmethod
    def _native_reset(self) -> None:
        """Reset the underlying game engine. Populate self._native_state."""

    @abstractmethod
    def _native_step(self, native_action: Any) -> None:
        """Apply a native action to the underlying game engine."""

    @abstractmethod
    def _get_native_state(self, agent: str) -> Any:
        """Return the native game state object for the given agent."""

    @abstractmethod
    def _get_legal_native_actions(self, agent: str) -> list[Any]:
        """Return all legal native actions for the given agent."""

    @abstractmethod
    def _compute_rewards(self) -> dict[str, float]:
        """Return the reward for each agent after the last step."""

    @abstractmethod
    def _is_done(self) -> bool:
        """Return True when the game has ended."""

    # ---- AEC API --------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        n = self.metadata.max_players if not self.metadata.is_single_player else 1
        self.possible_agents = [f"player_{i}" for i in range(n)]
        self.agents = list(self.possible_agents)
        self.agent_selection = self.agents[0]

        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._cumulative_rewards: dict[str, float] = {a: 0.0 for a in self.agents}

        # Build spaces once (idempotent)
        obs_space = spaces.Dict({
            "observation": spaces.Box(
                low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
            ),
            "action_mask": spaces.MultiBinary(ACTION_DIM),
        })
        act_space = spaces.Discrete(ACTION_DIM)
        for a in self.agents:
            self.observation_spaces[a] = obs_space
            self.action_spaces[a] = act_space

        self._native_reset()

    def observe(self, agent: str) -> UnifiedObs:
        native_state = self._get_native_state(agent)
        obs_vec = self._encoder.encode(native_state)
        legal_native = self._get_legal_native_actions(agent)
        mask = self._build_mask(legal_native)
        return {"observation": obs_vec, "action_mask": mask}

    def step(self, action: int) -> None:
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        native_action = self._mapper.unified_to_native(action)
        self._native_step(native_action)

        if self._is_done():
            rewards = self._compute_rewards()
            for a in self.agents:
                self.rewards[a] = rewards.get(a, 0.0)
                self._cumulative_rewards[a] += self.rewards[a]
                self.terminations[a] = True
        else:
            for a in self.agents:
                self.rewards[a] = 0.0
            self._advance_agent_selection()

    def last(self) -> tuple[UnifiedObs | None, float, bool, bool, dict]:
        """Returns (obs, reward, terminated, truncated, info) for the current agent."""
        agent = self.agent_selection
        obs = None if self.terminations[agent] else self.observe(agent)
        return (
            obs,
            self._cumulative_rewards.get(agent, 0.0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {}),
        )

    def render(self) -> str | None:
        return None  # subclasses may override

    def close(self) -> None:
        pass

    # ---- Internal helpers -----------------------------------------------

    def _build_mask(self, legal_native_actions: list[Any]) -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=bool)
        for na in legal_native_actions:
            slot = self._mapper.native_to_unified(na)
            if 0 <= slot < ACTION_DIM:
                mask[slot] = True
        if not mask.any():
            # Safety: if no legal actions computed, mark the first available
            # to prevent the policy from crashing. Env should never reach here.
            mask[0] = True
        return mask

    def _advance_agent_selection(self) -> None:
        """Cycle to the next live agent."""
        active = [a for a in self.agents if not self.terminations[a]]
        if not active:
            return
        try:
            idx = active.index(self.agent_selection)
            self.agent_selection = active[(idx + 1) % len(active)]
        except ValueError:
            self.agent_selection = active[0]

    def _was_dead_step(self, action: int) -> None:
        """Handle a step called on a terminated agent (AEC protocol requirement)."""
        self.rewards[self.agent_selection] = 0.0
        self._advance_agent_selection()

    # ---- Validation helpers (call in tests / RuleValidator) ------------

    def validate_obs_shape(self, agent: str) -> None:
        obs = self.observe(agent)
        assert obs["observation"].shape == (OBS_DIM,), (
            f"observation shape mismatch: {obs['observation'].shape}"
        )
        assert obs["action_mask"].shape == (ACTION_DIM,), (
            f"action_mask shape mismatch: {obs['action_mask'].shape}"
        )
        assert obs["action_mask"].any(), "action_mask has no legal actions"

    def validate_card_conservation(self) -> None:
        """
        Check that no card appears in more than one observable plane at once.
        Only valid when strict_partial_info=False (all cards visible).
        Override if your game uses special card representations.
        """
        if self._strict_partial_info:
            return
        obs = self.observe(self.agent_selection)
        vec = obs["observation"]
        planes = vec[:self.metadata.deck_size * 11].reshape(11, -1)
        # Each card index should appear in at most one of planes 0–10
        totals = planes.sum(axis=0)
        assert (totals <= 1).all(), (
            f"Card conservation violated: cards {np.where(totals > 1)[0]} appear in multiple planes"
        )
