"""
Blackjack Basic Strategy agent.

Implements the mathematically optimal play table for a standard 6-deck
blackjack game (dealer stands on soft 17, double after split allowed).
This is the benchmark the trained AI must match in ≥90% of decisions.

Expected house edge with perfect basic strategy: ~0.5%
If the AI doesn't match this, the training pipeline has a bug.

Action codes (unified slots):
  10 = hit, 11 = stand, 12 = double down, 13 = split
"""

from __future__ import annotations

import numpy as np

from core.types import ACTION_DIM

# ---------------------------------------------------------------------------
# Strategy tables (player_total or pair → dealer_upcard → action)
# Dealer upcard: 2–10, A (indexed 0–9 where 0=2, 8=10, 9=A)
# ---------------------------------------------------------------------------

# Hard totals (no ace, or ace counted as 1)
# Rows = player hard total 8–17+ , Cols = dealer up 2,3,4,5,6,7,8,9,10,A
_HARD = {
    # total: [2,3,4,5,6,7,8,9,10,A]  H=hit S=stand D=double
    8:  list("HHHHHHHHHH"),
    9:  list("HDDDDHHHHH"),
    10: list("DDDDDDDDHH"),
    11: list("DDDDDDDDDH"),
    12: list("HHSSSHHHHH"),
    13: list("SSSSSHHHHH"),
    14: list("SSSSSHHHHH"),
    15: list("SSSSSHHHHH"),
    16: list("SSSSSHHHHH"),
    17: list("SSSSSSSSSS"),
}

# Soft totals (one ace counted as 11)
# Rows = soft total 13–21, Cols = dealer up 2,3,4,5,6,7,8,9,10,A
_SOFT = {
    13: list("HHHDDHHHHH"),
    14: list("HHHDDHHHHH"),
    15: list("HHDDDHHHHH"),
    16: list("HHDDDHHHHH"),
    17: list("HDDDDHHHHH"),
    18: list("SDDDDSSHHH"),
    19: list("SSSSSSSSS S"),
    20: list("SSSSSSSSSS"),
    21: list("SSSSSSSSSS"),
}

# Pairs (split = Y, no split = use hard total table)
# Rows = pair rank (A,2,3,...,10), Cols = dealer up 2,3,4,5,6,7,8,9,10,A
_PAIRS = {
    1:  list("YYYYYYYYYY"),  # A-A: always split
    2:  list("HYYYYYYHHH"),
    3:  list("HYYYYYYHHH"),
    4:  list("HHHHHHHHHH"),
    5:  list("DDDDDDDDHH"),  # 5-5: treat as 10, never split
    6:  list("YYYYYYHHHH"),
    7:  list("YYYYYYYHHH"),
    8:  list("YYYYYYYYYY"),  # 8-8: always split
    9:  list("YYYYYYYHYY"),
    10: list("SSSSSSSSSS"),  # 10-10: never split
}

_ACTION_MAP = {"H": 10, "S": 11, "D": 12, "Y": 13}  # unified slots


def _dealer_col(dealer_rank: int) -> int:
    """Convert dealer card rank (0=Ace, 1=2, ..., 12=King) to column 0–9."""
    if dealer_rank == 0:   # Ace
        return 9
    if dealer_rank >= 9:   # 10/J/Q/K all = 10
        return 8
    return dealer_rank - 1  # 2→0, 3→1, ..., 9→7


def basic_strategy_action(
    player_hand_indices: list[int],
    dealer_up_index: int,
    can_double: bool = True,
    can_split: bool = True,
) -> int:
    """
    Return the optimal unified action slot for the given game state.

    Args:
        player_hand_indices: card indices (0–51) in player's hand
        dealer_up_index:     card index of dealer's face-up card
        can_double:          whether doubling down is allowed this hand
        can_split:           whether splitting is allowed this hand

    Returns:
        unified action slot: 10=hit, 11=stand, 12=double, 13=split
    """
    from envs.gambling.blackjack_env import _hand_value

    ranks = [idx % 13 for idx in player_hand_indices]
    dealer_rank = dealer_up_index % 13
    col = _dealer_col(dealer_rank)

    # Check for pair
    if can_split and len(ranks) == 2 and ranks[0] == ranks[1]:
        pair_rank = ranks[0]
        key = 1 if pair_rank == 0 else (10 if pair_rank >= 9 else pair_rank + 1)
        # Normalise face cards to 10
        if key > 10:
            key = 10
        row = _PAIRS.get(key, _PAIRS[10])
        action_char = row[col]
        if action_char == "Y":
            return 13   # split
        # Fall through to hard/soft table if pair says don't split

    # Check for soft hand (has ace counted as 11)
    aces = ranks.count(0)
    total = _hand_value(player_hand_indices)
    has_soft_ace = aces > 0 and (total - 10) <= 11

    if has_soft_ace and 13 <= total <= 21:
        row = _SOFT.get(total, _SOFT[21])
        action_char = row[col]
    else:
        clamped = max(8, min(17, total))
        row = _HARD.get(clamped, _HARD[17])
        action_char = row[col]

    action = _ACTION_MAP.get(action_char, 10)

    # Downgrade double/split if not available
    if action == 12 and not can_double:
        action = 10   # hit instead
    if action == 13 and not can_split:
        action = 10   # hit instead

    return action


# ---------------------------------------------------------------------------
# Agent wrapper (matches the interface expected by ELO evaluator)
# ---------------------------------------------------------------------------

class BasicStrategyAgent:
    """Rule-based agent using perfect basic strategy. ELO anchor at 1200."""

    name = "BasicStrategyAgent"

    def act(self, obs: np.ndarray, action_mask: np.ndarray, game_state: dict) -> int:
        """
        Args:
            obs:         (688,) observation vector (not used — we read game_state)
            action_mask: (512,) legal action mask
            game_state:  dict from BlackjackEnv._native_state

        Returns:
            unified action slot (10–13)
        """
        hand = game_state.get("hand_indices", [])
        dealer_up = game_state.get("dealer_up_indices", [0])
        can_double = game_state.get("can_double", False)
        can_split = game_state.get("can_split", False)

        if not hand or not dealer_up:
            return 11   # stand as safe default

        action = basic_strategy_action(
            hand, dealer_up[0], can_double=can_double, can_split=can_split
        )

        # Respect legal action mask
        if not action_mask[action]:
            # Fall back to first legal action in priority: stand > hit
            for fallback in [11, 10]:
                if action_mask[fallback]:
                    return fallback
            return int(np.argmax(action_mask))

        return action
