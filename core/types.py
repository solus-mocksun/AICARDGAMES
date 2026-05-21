"""
Shared enums, dataclasses, and TypedDicts used across the entire project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TypedDict

import numpy as np


# ---------------------------------------------------------------------------
# Category and game identifiers
# ---------------------------------------------------------------------------

class CategoryID(IntEnum):
    GAMBLING = 0
    MATCHING_SHEDDING = 1
    TRICK_TAKING = 2
    MELDING = 3
    CLIMBING = 4
    SOLITAIRE = 5


class GameID(IntEnum):
    # Gambling (0–2)
    BLACKJACK = 0
    POKER_NLH = 1
    BACCARAT = 2
    # Matching/Shedding (3–5)
    CRAZY_EIGHTS = 3
    UNO = 4
    GO_FISH = 5
    # Trick-taking (6–8)
    HEARTS = 6
    SPADES = 7
    EUCHRE = 8
    # Melding (9–11)
    GIN_RUMMY = 9
    RUMMY_500 = 10
    CANASTA = 11
    # Climbing (12–14)
    PRESIDENT = 12
    BIG_TWO = 13
    TICHU = 14
    # Solitaire (15–17)
    FREECELL = 15
    KLONDIKE = 16
    SPIDER = 17


GAME_TO_CATEGORY: dict[GameID, CategoryID] = {
    GameID.BLACKJACK: CategoryID.GAMBLING,
    GameID.POKER_NLH: CategoryID.GAMBLING,
    GameID.BACCARAT: CategoryID.GAMBLING,
    GameID.CRAZY_EIGHTS: CategoryID.MATCHING_SHEDDING,
    GameID.UNO: CategoryID.MATCHING_SHEDDING,
    GameID.GO_FISH: CategoryID.MATCHING_SHEDDING,
    GameID.HEARTS: CategoryID.TRICK_TAKING,
    GameID.SPADES: CategoryID.TRICK_TAKING,
    GameID.EUCHRE: CategoryID.TRICK_TAKING,
    GameID.GIN_RUMMY: CategoryID.MELDING,
    GameID.RUMMY_500: CategoryID.MELDING,
    GameID.CANASTA: CategoryID.MELDING,
    GameID.PRESIDENT: CategoryID.CLIMBING,
    GameID.BIG_TWO: CategoryID.CLIMBING,
    GameID.TICHU: CategoryID.CLIMBING,
    GameID.FREECELL: CategoryID.SOLITAIRE,
    GameID.KLONDIKE: CategoryID.SOLITAIRE,
    GameID.SPIDER: CategoryID.SOLITAIRE,
}

NUM_CATEGORIES = len(CategoryID)   # 6
NUM_GAMES = len(GameID)            # 18


# ---------------------------------------------------------------------------
# Card constants
# ---------------------------------------------------------------------------

class Suit(IntEnum):
    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3


class Rank(IntEnum):
    ACE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    FIVE = 4
    SIX = 5
    SEVEN = 6
    EIGHT = 7
    NINE = 8
    TEN = 9
    JACK = 10
    QUEEN = 11
    KING = 12


RANK_NAMES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUIT_NAMES = ["♠", "♥", "♦", "♣"]
SUIT_LETTERS = ["S", "H", "D", "C"]

DECK_SIZE = 52          # Standard deck
CARD_PLANES = 12        # Observation planes
SCALAR_DIM = 64         # Padded scalar section
OBS_DIM = CARD_PLANES * DECK_SIZE + SCALAR_DIM   # 688
ACTION_DIM = 512        # Unified action space slots


# ---------------------------------------------------------------------------
# Typed observation dict
# ---------------------------------------------------------------------------

class UnifiedObs(TypedDict):
    """What the model receives each step."""
    observation: np.ndarray    # shape (OBS_DIM,) float32
    action_mask: np.ndarray    # shape (ACTION_DIM,) bool


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------

@dataclass
class EnvMetadata:
    game_id: GameID
    category_id: CategoryID
    min_players: int
    max_players: int
    has_partial_info: bool          # True if opponents have hidden cards
    is_single_player: bool          # True for solitaire
    deck_size: int = DECK_SIZE      # 52 for standard, 108 for UNO, etc.
    max_hand_size: int = 13         # Largest possible hand
    max_actions: int = ACTION_DIM   # Actual actions used (≤ 512)
    engine: str = "custom"          # "rlcard" | "open_spiel" | "pettingzoo" | "custom"


# ---------------------------------------------------------------------------
# Trajectory step (used by rollout buffer)
# ---------------------------------------------------------------------------

@dataclass
class Step:
    obs: np.ndarray             # (OBS_DIM,)
    action_mask: np.ndarray     # (ACTION_DIM,)
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool
    game_id: GameID
    player_id: int


@dataclass
class Episode:
    steps: list[Step] = field(default_factory=list)
    game_id: GameID = GameID.BLACKJACK
    winner: int = -1            # -1 = no winner yet / draw

    def __len__(self) -> int:
        return len(self.steps)
