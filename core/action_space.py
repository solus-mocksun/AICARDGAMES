"""
UnifiedActionSpace — maps game-native actions to/from a fixed 512-slot space.

Slot allocation by category:
  0   –  63   Gambling/Casino      (fold=0, call=1, raise_s=2, raise_m=3,
                                    raise_l=4, all_in=5, hit=10, stand=11,
                                    double=12, split=13, baccarat_player=20,
                                    baccarat_banker=21, baccarat_tie=22)
  64  – 127   Matching/Shedding    (play card by index, draw, skip, reverse...)
  128 – 191   Trick-Taking         (play card by index 0–51, pass bid, bid N)
  192 – 255   Melding              (draw, discard by index, knock, gin, meld...)
  256 – 319   Climbing             (pass=256, single cards 257–308, combos 309–319+)
  320 – 383   Solitaire            (move card between locations)
  384 – 511   Reserved / overflow

Each game registers an ActionMapper subclass with the registry.
The ActionMapper defines:
  - native_to_unified(native_action) -> int  (unified slot 0–511)
  - unified_to_native(slot) -> native_action
  - legal_mask(game_state) -> np.ndarray[512, bool]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.types import ACTION_DIM, GameID


# ---------------------------------------------------------------------------
# Base ActionMapper
# ---------------------------------------------------------------------------

class ActionMapper(ABC):
    """
    Subclass once per game. Registered in ActionSpaceRegistry.
    """

    game_id: GameID                    # set on subclass definition
    _num_actions_used: int = 0         # how many of the 512 slots this game uses

    @abstractmethod
    def native_to_unified(self, native_action: Any) -> int:
        """Map a game-native action to a unified slot 0–511."""

    @abstractmethod
    def unified_to_native(self, slot: int) -> Any:
        """Map a unified slot back to the game-native action."""

    @abstractmethod
    def legal_mask(self, game_state: Any) -> np.ndarray:
        """
        Return bool array of shape (512,) where True = legal action.
        Must have at least one True value whenever called.
        """

    def verify_roundtrip(self, native_action: Any) -> bool:
        """Sanity check: native -> unified -> native gives the same result."""
        slot = self.native_to_unified(native_action)
        return self.unified_to_native(slot) == native_action


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ActionSpaceRegistry:
    """Singleton registry mapping GameID -> ActionMapper instance."""

    _mappers: dict[GameID, ActionMapper] = {}

    @classmethod
    def register(cls, mapper: ActionMapper) -> None:
        cls._mappers[mapper.game_id] = mapper

    @classmethod
    def get(cls, game_id: GameID) -> ActionMapper:
        if game_id not in cls._mappers:
            raise KeyError(f"No ActionMapper registered for {game_id!r}")
        return cls._mappers[game_id]

    @classmethod
    def registered_games(cls) -> list[GameID]:
        return list(cls._mappers.keys())


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def empty_mask() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=bool)


def full_mask() -> np.ndarray:
    return np.ones(ACTION_DIM, dtype=bool)


def card_index_mask(legal_card_indices: list[int], slot_offset: int) -> np.ndarray:
    """
    Helper for games where each legal action = play card at position.
    Maps card index 0–51 to unified slot (slot_offset + card_index).
    """
    mask = empty_mask()
    for idx in legal_card_indices:
        mask[slot_offset + idx] = True
    return mask


# ---------------------------------------------------------------------------
# Placeholder mappers for Phase 0
# These are replaced by real implementations in later phases.
# ---------------------------------------------------------------------------

class _GamblingPlaceholderMapper(ActionMapper):
    """
    Gambling slot layout:
      0  fold
      1  check / call
      2  raise small  (2× BB)
      3  raise medium (pot)
      4  raise large  (2× pot)
      5  all-in
      10 hit
      11 stand
      12 double down
      13 split
      20 baccarat: bet player
      21 baccarat: bet banker
      22 baccarat: bet tie
    """
    game_id = GameID.BLACKJACK   # overridden per game

    # For Blackjack: native actions are RLCard integers 0=hit, 1=stand, 2=double, 3=split
    _BJ_MAP = {0: 10, 1: 11, 2: 12, 3: 13}
    _BJ_REVERSE = {v: k for k, v in _BJ_MAP.items()}

    def native_to_unified(self, native_action: int) -> int:
        return self._BJ_MAP.get(native_action, native_action)

    def unified_to_native(self, slot: int) -> int:
        return self._BJ_REVERSE.get(slot, slot)

    def legal_mask(self, game_state: Any) -> np.ndarray:
        mask = empty_mask()
        # Default: all blackjack actions legal (env will tighten at runtime)
        for slot in [10, 11, 12, 13]:
            mask[slot] = True
        return mask


class BlackjackActionMapper(_GamblingPlaceholderMapper):
    game_id = GameID.BLACKJACK


class PokerActionMapper(ActionMapper):
    game_id = GameID.POKER_NLH
    # 0=fold, 1=call, 2=raise_s, 3=raise_m, 4=raise_l, 5=all_in
    _NATIVE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    def native_to_unified(self, native_action: int) -> int:
        return self._NATIVE_MAP.get(native_action, native_action)

    def unified_to_native(self, slot: int) -> int:
        return slot  # identity in 0–5 range

    def legal_mask(self, game_state: Any) -> np.ndarray:
        mask = empty_mask()
        for slot in range(6):
            mask[slot] = True
        return mask


class BaccaratActionMapper(ActionMapper):
    game_id = GameID.BACCARAT
    # 3 actions only
    _MAP = {0: 20, 1: 21, 2: 22}
    _REV = {v: k for k, v in _MAP.items()}

    def native_to_unified(self, native_action: int) -> int:
        return self._MAP[native_action]

    def unified_to_native(self, slot: int) -> int:
        return self._REV[slot]

    def legal_mask(self, game_state: Any) -> np.ndarray:
        mask = empty_mask()
        for slot in [20, 21, 22]:
            mask[slot] = True
        return mask


class _CardPlayMapper(ActionMapper):
    """
    Base for games where primary action = play a card from hand.
    Maps card_index (0–51) -> unified slot (offset + card_index).
    Also supports special non-card actions mapped to high slots.
    """
    game_id: GameID
    _SLOT_OFFSET: int        # e.g. 64 for matching, 128 for trick-taking
    _PASS_SLOT: int | None = None   # unified slot for "pass" / "skip"
    _DRAW_SLOT: int | None = None

    def native_to_unified(self, native_action: int) -> int:
        # Convention: native 0–51 = play card at that index
        # native 100 = draw, native 101 = pass/skip
        if native_action == 100 and self._DRAW_SLOT is not None:
            return self._DRAW_SLOT
        if native_action == 101 and self._PASS_SLOT is not None:
            return self._PASS_SLOT
        return self._SLOT_OFFSET + native_action

    def unified_to_native(self, slot: int) -> int:
        if slot == self._DRAW_SLOT:
            return 100
        if slot == self._PASS_SLOT:
            return 101
        return slot - self._SLOT_OFFSET

    def legal_mask(self, game_state: Any) -> np.ndarray:
        mask = empty_mask()
        # Default: all 52 card slots legal (tightened by env)
        for i in range(52):
            mask[self._SLOT_OFFSET + i] = True
        return mask


class CrazyEightsActionMapper(_CardPlayMapper):
    game_id = GameID.CRAZY_EIGHTS
    _SLOT_OFFSET = 64
    _DRAW_SLOT = 116     # 64 + 52
    _PASS_SLOT = 117


class UnoActionMapper(_CardPlayMapper):
    game_id = GameID.UNO
    _SLOT_OFFSET = 64
    _DRAW_SLOT = 116
    _PASS_SLOT = 117


class GoFishActionMapper(ActionMapper):
    """Ask for a rank (0–12) = 13 possible actions, offset at slot 118."""
    game_id = GameID.GO_FISH
    _OFFSET = 118

    def native_to_unified(self, native_action: int) -> int:
        return self._OFFSET + native_action

    def unified_to_native(self, slot: int) -> int:
        return slot - self._OFFSET

    def legal_mask(self, game_state: Any) -> np.ndarray:
        mask = empty_mask()
        for i in range(13):
            mask[self._OFFSET + i] = True
        return mask


class HeartsActionMapper(_CardPlayMapper):
    game_id = GameID.HEARTS
    _SLOT_OFFSET = 128
    _PASS_SLOT = 180     # 128 + 52


class SpadesActionMapper(_CardPlayMapper):
    game_id = GameID.SPADES
    _SLOT_OFFSET = 128
    _PASS_SLOT = 180


class EuchreActionMapper(_CardPlayMapper):
    game_id = GameID.EUCHRE
    _SLOT_OFFSET = 128
    _PASS_SLOT = 180


class GinRummyActionMapper(_CardPlayMapper):
    """Draw from deck=200, draw from discard=201, discard card 0-51=202-253, knock=254, gin=255."""
    game_id = GameID.GIN_RUMMY
    _SLOT_OFFSET = 202
    _DRAW_SLOT = 200
    _PASS_SLOT = 254    # knock

    def native_to_unified(self, native_action: int) -> int:
        if native_action == 100:
            return 200   # draw deck
        if native_action == 101:
            return 201   # draw discard
        if native_action == 102:
            return 254   # knock
        if native_action == 103:
            return 255   # gin
        return self._SLOT_OFFSET + native_action

    def unified_to_native(self, slot: int) -> int:
        if slot == 200: return 100
        if slot == 201: return 101
        if slot == 254: return 102
        if slot == 255: return 103
        return slot - self._SLOT_OFFSET


class Rummy500ActionMapper(GinRummyActionMapper):
    game_id = GameID.RUMMY_500


class CanastaActionMapper(GinRummyActionMapper):
    game_id = GameID.CANASTA


class PresidentActionMapper(_CardPlayMapper):
    """Pass = slot 256, play cards 257–308 (52 cards)."""
    game_id = GameID.PRESIDENT
    _SLOT_OFFSET = 257
    _PASS_SLOT = 256


class BigTwoActionMapper(ActionMapper):
    """
    Big Two: pass = 256, single cards 257–308, combos 309–383.
    Combos are pre-enumerated per hand; for now we use a simplified
    representation: play card(s) by lead card index.
    """
    game_id = GameID.BIG_TWO
    _PASS_SLOT = 256
    _SINGLE_OFFSET = 257
    _COMBO_OFFSET = 309

    def native_to_unified(self, native_action: int) -> int:
        if native_action == 101:
            return self._PASS_SLOT
        if native_action < 52:
            return self._SINGLE_OFFSET + native_action
        return self._COMBO_OFFSET + (native_action - 52)

    def unified_to_native(self, slot: int) -> int:
        if slot == self._PASS_SLOT:
            return 101
        if slot < self._COMBO_OFFSET:
            return slot - self._SINGLE_OFFSET
        return 52 + (slot - self._COMBO_OFFSET)

    def legal_mask(self, game_state: Any) -> np.ndarray:
        mask = empty_mask()
        mask[self._PASS_SLOT] = True
        for i in range(52):
            mask[self._SINGLE_OFFSET + i] = True
        return mask


class TichuActionMapper(BigTwoActionMapper):
    game_id = GameID.TICHU


class _SolitaireMapper(ActionMapper):
    """
    Solitaire: move card from location A to location B.
    Encoded as flat index over (from_loc, to_loc) pairs.
    Locations: 0–6 tableau cols, 7–10 foundation piles, 11 stock, 12 waste.
    Max moves: 13 × 13 = 169 → fits in slots 320–488.
    """
    game_id: GameID
    _OFFSET = 320
    _N_LOCS = 13

    def native_to_unified(self, native_action: tuple[int, int]) -> int:
        from_loc, to_loc = native_action
        return self._OFFSET + from_loc * self._N_LOCS + to_loc

    def unified_to_native(self, slot: int) -> tuple[int, int]:
        idx = slot - self._OFFSET
        return (idx // self._N_LOCS, idx % self._N_LOCS)

    def legal_mask(self, game_state: Any) -> np.ndarray:
        mask = empty_mask()
        for i in range(self._N_LOCS * self._N_LOCS):
            mask[self._OFFSET + i] = True
        return mask


class FreeCellActionMapper(_SolitaireMapper):
    game_id = GameID.FREECELL


class KlondikeActionMapper(_SolitaireMapper):
    game_id = GameID.KLONDIKE


class SpiderActionMapper(_SolitaireMapper):
    game_id = GameID.SPIDER


# ---------------------------------------------------------------------------
# Auto-register all mappers
# ---------------------------------------------------------------------------

_ALL_MAPPERS: list[ActionMapper] = [
    BlackjackActionMapper(),
    PokerActionMapper(),
    BaccaratActionMapper(),
    CrazyEightsActionMapper(),
    UnoActionMapper(),
    GoFishActionMapper(),
    HeartsActionMapper(),
    SpadesActionMapper(),
    EuchreActionMapper(),
    GinRummyActionMapper(),
    Rummy500ActionMapper(),
    CanastaActionMapper(),
    PresidentActionMapper(),
    BigTwoActionMapper(),
    TichuActionMapper(),
    FreeCellActionMapper(),
    KlondikeActionMapper(),
    SpiderActionMapper(),
]

for _m in _ALL_MAPPERS:
    ActionSpaceRegistry.register(_m)
