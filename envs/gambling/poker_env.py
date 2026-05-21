"""
Poker (No-Limit Texas Hold'em) environment — wraps RLCard.

State encoding (688-dim):
  Plane 0  : my hole cards (2 cards)
  Plane 1  : flop card 1
  Plane 2  : flop card 2
  Plane 3  : flop card 3
  Plane 4  : turn card
  Plane 5  : river card
  Planes 6–10: zeros (opponent hole cards hidden)
  Plane 11 : unknown (auto-computed)

  Scalars:
    player_idx      = player position (0–5)
    num_players     = 2–6
    my_stack        = stack / initial_chips
    pot_size        = pot / initial_chips
    current_bet     = current bet to call / initial_chips
    game_phase      = 0=preflop, 1=flop, 2=turn, 3=river
    my_score        = hand strength estimate (0–1)

Actions (unified slots 0–5):
  0=fold, 1=call/check, 2=raise_small(2BB), 3=raise_medium(pot),
  4=raise_large(2x pot), 5=all_in
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.state_encoder import UnifiedStateEncoder
from core.types import EnvMetadata, GameID, CategoryID
from envs.base_env import BaseCardGameEnv

try:
    import rlcard
    _RLCARD_AVAILABLE = True
except ImportError:
    _RLCARD_AVAILABLE = False

_INITIAL_CHIPS = 10_000   # RLCard NLH default


class PokerStateEncoder(UnifiedStateEncoder):
    def __init__(self) -> None:
        super().__init__(game_id=GameID.POKER_NLH, strict_partial_info=True)

    def _fill_card_planes(self, state: dict, planes: np.ndarray) -> None:
        for idx in state.get("hole_indices", []):
            if 0 <= idx < 52:
                planes[0, idx] = 1.0

        community = state.get("community_indices", [])
        for plane_offset, idx in enumerate(community[:5]):
            if 0 <= idx < 52:
                planes[1 + plane_offset, idx] = 1.0

    def _fill_scalars(self, state: dict, scalars: np.ndarray) -> None:
        self.set_player_idx(scalars, state.get("player_idx", 0))
        self.set_num_players(scalars, state.get("num_players", 2))
        self.set_stack(scalars, state.get("my_stack", _INITIAL_CHIPS), _INITIAL_CHIPS)
        self.set_pot(scalars, state.get("pot", 0), _INITIAL_CHIPS)
        self.set_bet(scalars, state.get("current_bet", 0), _INITIAL_CHIPS)
        self.set_game_phase(scalars, state.get("phase", 0))
        self.set_my_score(scalars, state.get("hand_strength", 0.0), max_score=1.0)


class PokerEnv(BaseCardGameEnv):
    metadata = EnvMetadata(
        game_id=GameID.POKER_NLH,
        category_id=CategoryID.GAMBLING,
        min_players=2,
        max_players=6,
        has_partial_info=True,
        is_single_player=False,
        deck_size=52,
        max_hand_size=7,
        max_actions=6,
        engine="rlcard",
    )

    def __init__(
        self,
        num_players: int = 2,
        mock_mode: bool = False,
        **kwargs,
    ) -> None:
        self._num_players = num_players
        self._mock_mode = mock_mode
        if not mock_mode and not _RLCARD_AVAILABLE:
            raise ImportError("rlcard is required. Install with: pip install rlcard")

        super().__init__(**kwargs)

        if not mock_mode:
            self._rlcard_env = rlcard.make(
                "no-limit-holdem",
                config={"num_players": num_players, "seed": None},
            )
        else:
            self._rlcard_env = None

        self._native_state: dict = {}

    def _make_encoder(self) -> PokerStateEncoder:
        return PokerStateEncoder()

    def _native_reset(self) -> None:
        if self._mock_mode:
            self._native_state = _mock_poker_state()
            return
        obs, _ = self._rlcard_env.reset()
        player_id = self._rlcard_env.get_player_id()
        self._native_state = self._parse_rlcard_state(
            self._rlcard_env.get_state(player_id), player_id
        )

    def _native_step(self, native_action: int) -> None:
        if self._mock_mode:
            if native_action == 0:   # fold ends the game
                self._native_state["done"] = True
            return
        obs, _ = self._rlcard_env.step(native_action)
        player_id = self._rlcard_env.get_player_id()
        self._native_state = self._parse_rlcard_state(
            self._rlcard_env.get_state(player_id), player_id
        )

    def _get_native_state(self, agent: str) -> dict:
        return self._native_state

    def _get_legal_native_actions(self, agent: str) -> list[int]:
        return self._native_state.get("legal_actions", [0, 1])

    def _compute_rewards(self) -> dict[str, float]:
        if self._mock_mode:
            return {"player_0": 0.0}
        payoffs = self._rlcard_env.get_payoffs()
        return {f"player_{i}": float(p) / _INITIAL_CHIPS
                for i, p in enumerate(payoffs)}

    def _is_done(self) -> bool:
        if self._mock_mode:
            return self._native_state.get("done", False)
        return self._rlcard_env.is_over()

    def _parse_rlcard_state(self, rlcard_state: dict, player_id: int) -> dict:
        raw = rlcard_state.get("raw_obs", {})

        hole = raw.get("hand", [])
        community = raw.get("public_cards", [])

        from envs.gambling.blackjack_env import _rlcard_card_to_index
        hole_indices = [_rlcard_card_to_index(c) for c in hole]
        community_indices = [_rlcard_card_to_index(c) for c in community]

        phase_map = {0: 0, 1: 1, 2: 2, 3: 3}   # preflop/flop/turn/river
        phase = phase_map.get(len(community_indices) // max(1, 1), 0)
        if len(community_indices) == 0: phase = 0
        elif len(community_indices) == 3: phase = 1
        elif len(community_indices) == 4: phase = 2
        else: phase = 3

        stakes = raw.get("stakes", [_INITIAL_CHIPS])
        my_stack = stakes[player_id] if player_id < len(stakes) else _INITIAL_CHIPS
        pot = raw.get("pot", 0)
        current_bet = raw.get("call", 0)

        legal_actions = list(rlcard_state.get("legal_actions", {}).keys())

        return {
            "hole_indices": hole_indices,
            "community_indices": community_indices,
            "player_idx": player_id,
            "num_players": self._num_players,
            "my_stack": my_stack,
            "pot": pot,
            "current_bet": current_bet,
            "phase": phase,
            "hand_strength": 0.5,   # placeholder; real hand eval added in Phase 2
            "legal_actions": legal_actions,
            "done": False,
        }


def _mock_poker_state() -> dict:
    return {
        "hole_indices": [0, 13],
        "community_indices": [],
        "player_idx": 0,
        "num_players": 2,
        "my_stack": _INITIAL_CHIPS,
        "pot": 150,
        "current_bet": 100,
        "phase": 0,
        "hand_strength": 0.5,
        "legal_actions": [0, 1, 2, 3, 4, 5],
        "done": False,
    }
