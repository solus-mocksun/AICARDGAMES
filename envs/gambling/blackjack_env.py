"""
Blackjack environment — wraps RLCard's blackjack game.

State encoding (688-dim):
  Plane 0  : player's hand (visible cards)
  Plane 1  : dealer's face-up card (one card)
  Planes 2–10: zeros (dealer hole card is hidden — strict partial info)
  Plane 11 : unknown (auto-computed)

  Scalars:
    player_idx    = 0 (always player 0 vs dealer)
    num_players   = 2 (player + dealer)
    hand_size     = current hand size / 11
    my_score      = current hand value / 21
    dealer_score  = dealer visible score / 21
    can_double    = bool
    can_split     = bool
    game_phase    = 0 (single phase: play)

Actions (RLCard native → unified slot):
  0 = hit    → 10
  1 = stand  → 11
  2 = double → 12
  3 = split  → 13

Pipeline validation: after training, agent should match the basic strategy
table in ≥90% of decisions. See agents/heuristic/blackjack_basic_strategy.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.card import cards_to_plane
from core.state_encoder import UnifiedStateEncoder
from core.types import EnvMetadata, GameID, CategoryID, OBS_DIM
from envs.base_env import BaseCardGameEnv

try:
    import rlcard
    _RLCARD_AVAILABLE = True
except ImportError:
    _RLCARD_AVAILABLE = False


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------

class BlackjackStateEncoder(UnifiedStateEncoder):
    """Converts RLCard blackjack observation to the 688-dim unified vector."""

    def __init__(self) -> None:
        super().__init__(game_id=GameID.BLACKJACK, strict_partial_info=True)

    def _fill_card_planes(self, game_state: dict, planes: np.ndarray) -> None:
        """
        RLCard blackjack state dict keys:
          "hand"         : list of card strings e.g. ["SA", "HK"]
          "dealer_hand"  : list, but only first card is visible
          "raw_obs"      : numpy array (RLCard's own encoding, not used here)
        """
        hand_indices = game_state.get("hand_indices", [])
        dealer_up_indices = game_state.get("dealer_up_indices", [])

        # Plane 0: player hand
        for idx in hand_indices:
            if 0 <= idx < 52:
                planes[0, idx] = 1.0

        # Plane 1: dealer face-up card only (hole card stays hidden)
        for idx in dealer_up_indices:
            if 0 <= idx < 52:
                planes[1, idx] = 1.0

    def _fill_scalars(self, game_state: dict, scalars: np.ndarray) -> None:
        self.set_player_idx(scalars, 0)
        self.set_num_players(scalars, 2)

        hand_size = game_state.get("hand_size", 2)
        self.set_hand_size(scalars, hand_size, max_size=11)

        my_score = game_state.get("my_score", 0)
        self.set_my_score(scalars, my_score, max_score=21)

        dealer_score = game_state.get("dealer_score", 0)
        self.set_opp_scores(scalars, [dealer_score], max_score=21)

        self.set_game_phase(scalars, 0)

        # Special flags
        scalars[58] = float(game_state.get("can_double", False))
        scalars[59] = float(game_state.get("can_split", False))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BlackjackEnv(BaseCardGameEnv):
    """
    Blackjack environment wrapping RLCard.

    Raises ImportError with a helpful message if rlcard is not installed.
    The env can still be instantiated and inspected without rlcard for
    testing purposes using mock_mode=True.
    """

    metadata = EnvMetadata(
        game_id=GameID.BLACKJACK,
        category_id=CategoryID.GAMBLING,
        min_players=1,
        max_players=1,
        has_partial_info=True,
        is_single_player=False,
        deck_size=52,
        max_hand_size=11,
        max_actions=14,
        engine="rlcard",
    )

    def __init__(
        self,
        num_players: int = 1,
        mock_mode: bool = False,
        **kwargs,
    ) -> None:
        self._mock_mode = mock_mode
        if not mock_mode and not _RLCARD_AVAILABLE:
            raise ImportError(
                "rlcard is required for BlackjackEnv. "
                "Install with: pip install rlcard"
            )

        super().__init__(**kwargs)

        if not mock_mode:
            self._rlcard_env = rlcard.make(
                "blackjack",
                config={"seed": None, "allow_step_back": False},
            )
        else:
            self._rlcard_env = None

        self._native_state: dict = {}

    def _make_encoder(self) -> BlackjackStateEncoder:
        return BlackjackStateEncoder()

    # ---- Native engine interface ----------------------------------------

    def _native_reset(self) -> None:
        if self._mock_mode:
            self._native_state = _mock_blackjack_state()
            return

        obs, _ = self._rlcard_env.reset()
        self._native_state = self._parse_rlcard_state(
            self._rlcard_env.get_state(0)
        )
        self._done = False

    def _native_step(self, native_action: int) -> None:
        if self._mock_mode:
            # Mock: stand always ends the game
            if native_action == 1:
                self._native_state["done"] = True
            return

        obs, _ = self._rlcard_env.step(native_action)
        player_id = self._rlcard_env.get_player_id()
        self._native_state = self._parse_rlcard_state(
            self._rlcard_env.get_state(player_id)
        )

    def _get_native_state(self, agent: str) -> dict:
        return self._native_state

    def _get_legal_native_actions(self, agent: str) -> list[int]:
        if self._mock_mode:
            return self._native_state.get("legal_actions", [0, 1])
        return self._native_state.get("legal_actions", [0, 1])

    def _compute_rewards(self) -> dict[str, float]:
        reward = self._native_state.get("reward", 0.0)
        return {"player_0": float(reward)}

    def _is_done(self) -> bool:
        if self._mock_mode:
            return self._native_state.get("done", False)
        return self._rlcard_env.is_over()

    # ---- RLCard state parsing -------------------------------------------

    def _parse_rlcard_state(self, rlcard_state: dict) -> dict:
        """
        Convert RLCard's raw state dict into our standardised format.
        RLCard blackjack state keys: 'hand', 'dealer_hand', 'raw_obs', 'legal_actions'
        """
        raw = rlcard_state.get("raw_obs", {})
        hand = raw.get("hand", [])
        dealer_hand = raw.get("dealer_hand", [])

        hand_indices = [_rlcard_card_to_index(c) for c in hand]
        # Only the first dealer card is visible
        dealer_up_indices = (
            [_rlcard_card_to_index(dealer_hand[0])] if dealer_hand else []
        )

        my_score = _hand_value([_rlcard_card_to_index(c) for c in hand])
        dealer_score = _hand_value(dealer_up_indices)

        legal_actions = list(rlcard_state.get("legal_actions", {}).keys())

        return {
            "hand_indices": hand_indices,
            "dealer_up_indices": dealer_up_indices,
            "hand_size": len(hand_indices),
            "my_score": my_score,
            "dealer_score": dealer_score,
            "can_double": 2 in legal_actions,
            "can_split": 3 in legal_actions,
            "legal_actions": legal_actions,
            "done": False,
            "reward": 0.0,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rlcard_card_to_index(card_str: str) -> int:
    """
    Convert RLCard card string (e.g. 'SA', 'HK', 'D10', 'C3') to 0–51 index.
    RLCard format: suit_letter + rank_str
    """
    if not card_str or len(card_str) < 2:
        return 0
    suit_char = card_str[0].upper()
    rank_str = card_str[1:].upper()

    suit_map = {"S": 0, "H": 1, "D": 2, "C": 3}
    rank_map = {
        "A": 0, "2": 1, "3": 2, "4": 3, "5": 4,
        "6": 5, "7": 6, "8": 7, "9": 8, "T": 9,
        "10": 9, "J": 10, "Q": 11, "K": 12,
    }
    suit = suit_map.get(suit_char, 0)
    rank = rank_map.get(rank_str, 0)
    return suit * 13 + rank


def _hand_value(card_indices: list[int]) -> int:
    """Compute blackjack hand value (Ace = 11 or 1) from card indices."""
    values = []
    aces = 0
    for idx in card_indices:
        rank = idx % 13
        if rank == 0:      # Ace
            aces += 1
            values.append(11)
        elif rank >= 9:    # 10, J, Q, K
            values.append(10)
        else:
            values.append(rank + 1)

    total = sum(values)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def _mock_blackjack_state() -> dict:
    """Minimal state for unit tests without rlcard installed."""
    return {
        "hand_indices": [0, 13],    # Ace of Spades + Ace of Hearts
        "dealer_up_indices": [9],   # Ten of Spades
        "hand_size": 2,
        "my_score": 12,
        "dealer_score": 10,
        "can_double": True,
        "can_split": True,
        "legal_actions": [0, 1, 2, 3],
        "done": False,
        "reward": 0.0,
    }
