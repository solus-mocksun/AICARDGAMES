"""
UnifiedStateEncoder — converts any game's native state into a fixed 688-dim float32 vector.

Layout:
  [0  : 624)  12 card planes × 52 cards each
  [624: 688)  64 scalar features (padded from 62 computed values)

Card planes (each 52-dim binary):
  0  my_hand
  1  face_up_1   (top discard / community card 1 / trick lead card)
  2  face_up_2   (community 2 / trick player-1 card / discard -2)
  3  face_up_3
  4  face_up_4
  5  face_up_5
  6  opp_0_known (cards known to be in opponent 0's hand)
  7  opp_1_known
  8  opp_2_known
  9  opp_3_known
  10 opp_4_known
  11 unknown     (full deck minus all known cards)

Scalar layout (64 floats):
  [0 : 6 )  category_id  one-hot (6)
  [6 :26 )  game_id      one-hot (20)
  [26:32 )  my_player_idx one-hot (6)
  [32]      num_players_norm  (players/6)
  [33]      hand_size_norm    (hand/max_hand)
  [34]      deck_remaining_norm
  [35]      my_score_norm
  [36:41)   opp_scores_norm   (5 floats)
  [41]      my_stack_norm
  [42]      pot_size_norm
  [43]      current_bet_norm
  [44:48)   game_phase    one-hot (4)
  [48]      tricks_taken_me_norm
  [49:54)   tricks_taken_opps_norm (5 floats)
  [54:58)   trump_suit    one-hot (4)
  [58]      cards_to_draw_norm
  [59]      special_action_pending
  [60]      round_number_norm
  [61]      step_number_norm
  [62:64)   padding (zeros)

Subclass this for each game. Override _fill_card_planes() and _fill_scalars().
Call encode(game_state) to get the 688-dim vector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.types import (
    CategoryID, GameID, GAME_TO_CATEGORY,
    CARD_PLANES, DECK_SIZE, SCALAR_DIM, OBS_DIM,
    NUM_CATEGORIES, NUM_GAMES,
)


# Scalar section index constants (for readability in subclasses)
_SC_CATEGORY   = slice(0, 6)
_SC_GAME       = slice(6, 26)
_SC_PLAYER_IDX = slice(26, 32)
_SC_NUM_PLAYERS   = 32
_SC_HAND_SIZE     = 33
_SC_DECK_REM      = 34
_SC_MY_SCORE      = 35
_SC_OPP_SCORES    = slice(36, 41)
_SC_MY_STACK      = 41
_SC_POT           = 42
_SC_BET           = 43
_SC_GAME_PHASE    = slice(44, 48)
_SC_TRICKS_ME     = 48
_SC_TRICKS_OPPS   = slice(49, 54)
_SC_TRUMP_SUIT    = slice(54, 58)
_SC_CARDS_TO_DRAW = 58
_SC_SPECIAL       = 59
_SC_ROUND_NORM    = 60
_SC_STEP_NORM     = 61
# 62, 63 = padding


class UnifiedStateEncoder(ABC):
    """
    Base encoder. Subclass per game, override _fill_card_planes and _fill_scalars.

    The encode() method always returns a (688,) float32 array.
    Unknown/unused fields default to zero — safe because the model sees
    category_id and game_id embeddings that tell it which fields are meaningful.
    """

    def __init__(self, game_id: GameID, strict_partial_info: bool = True) -> None:
        self.game_id = game_id
        self.category_id: CategoryID = GAME_TO_CATEGORY[game_id]
        self.strict_partial_info = strict_partial_info

    def encode(self, game_state: Any) -> np.ndarray:
        """Return the 688-dim observation vector for this game state."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        card_section = obs[:CARD_PLANES * DECK_SIZE].reshape(CARD_PLANES, DECK_SIZE)
        scalar_section = obs[CARD_PLANES * DECK_SIZE:]

        self._fill_card_planes(game_state, card_section)
        self._fill_scalars(game_state, scalar_section)

        # Always inject category + game embeddings into scalar section
        scalar_section[_SC_CATEGORY] = 0.0
        scalar_section[_SC_CATEGORY][int(self.category_id)] = 1.0
        scalar_section[_SC_GAME] = 0.0
        scalar_section[_SC_GAME][int(self.game_id)] = 1.0

        # Derive unknown-card plane from all other card planes
        known = card_section[:11].sum(axis=0).clip(0, 1)
        card_section[11] = 1.0 - known

        return obs

    @abstractmethod
    def _fill_card_planes(self, game_state: Any, planes: np.ndarray) -> None:
        """
        Fill planes 0–10 (plane 11 = unknown is computed automatically).
        planes shape: (12, 52) float32, pre-zeroed.
        Only set bits for cards that are observable to the current player.
        With strict_partial_info=True, hidden cards MUST remain zero.
        """

    @abstractmethod
    def _fill_scalars(self, game_state: Any, scalars: np.ndarray) -> None:
        """
        Fill the 64-dim scalar section (indices 0–63).
        scalars shape: (64,) float32, pre-zeroed.
        category_id and game_id are written AFTER this call — do not set them.
        """

    # ---- Scalar helper methods (for use in subclasses) ------------------

    @staticmethod
    def set_player_idx(scalars: np.ndarray, player_idx: int, max_players: int = 6) -> None:
        scalars[_SC_PLAYER_IDX] = 0.0
        if 0 <= player_idx < 6:
            scalars[_SC_PLAYER_IDX][player_idx] = 1.0

    @staticmethod
    def set_num_players(scalars: np.ndarray, n: int) -> None:
        scalars[_SC_NUM_PLAYERS] = n / 6.0

    @staticmethod
    def set_hand_size(scalars: np.ndarray, size: int, max_size: int) -> None:
        scalars[_SC_HAND_SIZE] = size / max(max_size, 1)

    @staticmethod
    def set_deck_remaining(scalars: np.ndarray, remaining: int, deck_size: int) -> None:
        scalars[_SC_DECK_REM] = remaining / max(deck_size, 1)

    @staticmethod
    def set_my_score(scalars: np.ndarray, score: float, max_score: float) -> None:
        scalars[_SC_MY_SCORE] = score / max(abs(max_score), 1)

    @staticmethod
    def set_opp_scores(scalars: np.ndarray, scores: list[float], max_score: float) -> None:
        for i, s in enumerate(scores[:5]):
            scalars[_SC_OPP_SCORES][i] = s / max(abs(max_score), 1)

    @staticmethod
    def set_stack(scalars: np.ndarray, stack: float, initial: float) -> None:
        scalars[_SC_MY_STACK] = stack / max(initial, 1)

    @staticmethod
    def set_pot(scalars: np.ndarray, pot: float, initial: float) -> None:
        scalars[_SC_POT] = pot / max(initial, 1)

    @staticmethod
    def set_bet(scalars: np.ndarray, bet: float, initial: float) -> None:
        scalars[_SC_BET] = bet / max(initial, 1)

    @staticmethod
    def set_game_phase(scalars: np.ndarray, phase: int) -> None:
        """phase: 0–3"""
        scalars[_SC_GAME_PHASE] = 0.0
        if 0 <= phase < 4:
            scalars[_SC_GAME_PHASE][phase] = 1.0

    @staticmethod
    def set_tricks(scalars: np.ndarray, my_tricks: int, opp_tricks: list[int],
                   total_tricks: int) -> None:
        scalars[_SC_TRICKS_ME] = my_tricks / max(total_tricks, 1)
        for i, t in enumerate(opp_tricks[:5]):
            scalars[_SC_TRICKS_OPPS][i] = t / max(total_tricks, 1)

    @staticmethod
    def set_trump_suit(scalars: np.ndarray, suit: int | None) -> None:
        """suit: 0=spades, 1=hearts, 2=diamonds, 3=clubs, None=no trump"""
        scalars[_SC_TRUMP_SUIT] = 0.0
        if suit is not None and 0 <= suit < 4:
            scalars[_SC_TRUMP_SUIT][suit] = 1.0

    @staticmethod
    def set_cards_to_draw(scalars: np.ndarray, n: int) -> None:
        scalars[_SC_CARDS_TO_DRAW] = n / 4.0   # max meaningful value ≈ 4

    @staticmethod
    def set_special_pending(scalars: np.ndarray, flag: bool) -> None:
        scalars[_SC_SPECIAL] = 1.0 if flag else 0.0

    @staticmethod
    def set_round(scalars: np.ndarray, round_num: int, max_rounds: int) -> None:
        scalars[_SC_ROUND_NORM] = round_num / max(max_rounds, 1)

    @staticmethod
    def set_step(scalars: np.ndarray, step: int, max_steps: int) -> None:
        scalars[_SC_STEP_NORM] = step / max(max_steps, 1)


# ---------------------------------------------------------------------------
# Passthrough encoder — for testing only
# ---------------------------------------------------------------------------

class PassthroughEncoder(UnifiedStateEncoder):
    """
    Accepts a dict with optional pre-filled planes/scalars.
    Used in unit tests where we want to inject exact values.

    game_state dict keys (all optional):
      "planes": np.ndarray shape (12, 52)
      "scalars": np.ndarray shape (64,)  (indices 2–63; 0-1 overwritten)
    """

    def __init__(self) -> None:
        super().__init__(game_id=GameID.BLACKJACK)

    def _fill_card_planes(self, game_state: dict, planes: np.ndarray) -> None:
        if "planes" in game_state:
            planes[:] = game_state["planes"]

    def _fill_scalars(self, game_state: dict, scalars: np.ndarray) -> None:
        if "scalars" in game_state:
            scalars[:] = game_state["scalars"]
