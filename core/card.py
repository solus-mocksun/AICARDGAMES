"""
Card, Deck, and Hand primitives.

Encoding: card_index = suit * 13 + rank  (0–51)
  Spades 0–12, Hearts 13–25, Diamonds 26–38, Clubs 39–51
  Ace = 0, 2 = 1, ..., King = 12
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from core.types import Rank, Suit, RANK_NAMES, SUIT_NAMES, SUIT_LETTERS, DECK_SIZE


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank

    @property
    def index(self) -> int:
        """Integer index 0–51."""
        return self.suit * 13 + self.rank

    @classmethod
    def from_index(cls, idx: int) -> "Card":
        if not (0 <= idx < DECK_SIZE):
            raise ValueError(f"Card index must be 0–51, got {idx}")
        return cls(suit=Suit(idx // 13), rank=Rank(idx % 13))

    @classmethod
    def from_str(cls, s: str) -> "Card":
        """Parse e.g. 'AS', 'KH', '10D', '2C'."""
        s = s.strip().upper()
        suit_char = s[-1]
        rank_str = s[:-1]
        suit = Suit(SUIT_LETTERS.index(suit_char))
        rank = Rank(RANK_NAMES.index(rank_str))
        return cls(suit=suit, rank=rank)

    def __str__(self) -> str:
        return f"{RANK_NAMES[self.rank]}{SUIT_NAMES[self.suit]}"

    def __repr__(self) -> str:
        return f"Card({self!s})"

    # Blackjack point value (Ace = 11, face = 10)
    @property
    def blackjack_value(self) -> int:
        if self.rank == Rank.ACE:
            return 11
        if self.rank >= Rank.TEN:
            return 10
        return self.rank + 1   # 2→2, 3→3, ...


# Convenience: all 52 standard cards as a tuple (immutable, re-used everywhere)
ALL_CARDS: tuple[Card, ...] = tuple(
    Card.from_index(i) for i in range(DECK_SIZE)
)


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------

class Deck:
    """Mutable 52-card deck with shuffle/deal."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._cards: list[Card] = list(ALL_CARDS)
        self._rng = rng or random.Random()

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)

    def deal(self, n: int = 1) -> list[Card]:
        if n > len(self._cards):
            raise ValueError(f"Cannot deal {n} cards; only {len(self._cards)} remain")
        dealt = self._cards[-n:]
        self._cards = self._cards[:-n]
        return dealt[::-1]   # top of deck first

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    def __len__(self) -> int:
        return len(self._cards)

    def __repr__(self) -> str:
        return f"Deck({len(self)} cards remaining)"


# ---------------------------------------------------------------------------
# Hand
# ---------------------------------------------------------------------------

class Hand:
    """A player's hand of cards."""

    def __init__(self, cards: list[Card] | None = None) -> None:
        self._cards: list[Card] = list(cards or [])

    def add(self, card: Card) -> None:
        self._cards.append(card)

    def add_many(self, cards: list[Card]) -> None:
        self._cards.extend(cards)

    def remove(self, card: Card) -> None:
        self._cards.remove(card)   # raises ValueError if not present

    def contains(self, card: Card) -> bool:
        return card in self._cards

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self):
        return iter(self._cards)

    def __repr__(self) -> str:
        return f"Hand([{', '.join(str(c) for c in self._cards)}])"

    # ---- numpy helpers ------------------------------------------------

    def to_plane(self) -> np.ndarray:
        """Binary vector of shape (52,): 1 if card is in hand."""
        plane = np.zeros(DECK_SIZE, dtype=np.float32)
        for card in self._cards:
            plane[card.index] = 1.0
        return plane

    @classmethod
    def from_indices(cls, indices: list[int]) -> "Hand":
        return cls([Card.from_index(i) for i in indices])

    def sort(self) -> None:
        """Sort by suit then rank (cosmetic, for display)."""
        self._cards.sort(key=lambda c: (c.suit, c.rank))

    def copy(self) -> "Hand":
        return Hand(list(self._cards))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def cards_to_plane(cards: list[Card] | None) -> np.ndarray:
    """Convert a list of cards (or None) to a 52-dim binary plane."""
    plane = np.zeros(DECK_SIZE, dtype=np.float32)
    if cards:
        for card in cards:
            plane[card.index] = 1.0
    return plane


def indices_to_plane(indices: list[int] | None) -> np.ndarray:
    """Convert integer indices to a 52-dim binary plane."""
    plane = np.zeros(DECK_SIZE, dtype=np.float32)
    if indices:
        for idx in indices:
            plane[idx] = 1.0
    return plane
