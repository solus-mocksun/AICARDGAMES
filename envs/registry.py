"""
ENV_REGISTRY — maps game name strings to (env_factory, EnvMetadata).

Usage:
    from envs.registry import ENV_REGISTRY
    env = ENV_REGISTRY["blackjack"].make()
    meta = ENV_REGISTRY["blackjack"].metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from core.types import CategoryID, EnvMetadata, GameID

if TYPE_CHECKING:
    from envs.base_env import BaseCardGameEnv


@dataclass
class RegistryEntry:
    metadata: EnvMetadata
    _factory: Callable[..., "BaseCardGameEnv"]

    def make(self, **kwargs) -> "BaseCardGameEnv":
        return self._factory(**kwargs)


# ---------------------------------------------------------------------------
# Registry dict — populated lazily to avoid circular imports.
# Env classes are imported inside lambdas so they're only loaded on demand.
# ---------------------------------------------------------------------------

def _make_registry() -> dict[str, RegistryEntry]:
    return {
        # ---- Gambling -------------------------------------------------------
        "blackjack": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.BLACKJACK,
                category_id=CategoryID.GAMBLING,
                min_players=1,
                max_players=1,
                has_partial_info=True,
                is_single_player=False,   # dealer is opponent
                deck_size=52,
                max_hand_size=11,         # can hit to 21, theoretically up to ~11 cards
                max_actions=14,
                engine="rlcard",
            ),
            _factory=lambda **kw: _import("envs.gambling.blackjack_env", "BlackjackEnv")(**kw),
        ),
        "poker_nlh": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.POKER_NLH,
                category_id=CategoryID.GAMBLING,
                min_players=2,
                max_players=6,
                has_partial_info=True,
                is_single_player=False,
                deck_size=52,
                max_hand_size=7,   # 2 hole + 5 community
                max_actions=6,
                engine="rlcard",
            ),
            _factory=lambda **kw: _import("envs.gambling.poker_env", "PokerEnv")(**kw),
        ),
        "baccarat": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.BACCARAT,
                category_id=CategoryID.GAMBLING,
                min_players=1,
                max_players=1,
                has_partial_info=False,
                is_single_player=False,
                deck_size=52,
                max_hand_size=3,
                max_actions=3,
                engine="custom",
            ),
            _factory=lambda **kw: _import("envs.gambling.baccarat_env", "BaccaratEnv")(**kw),
        ),

        # ---- Matching/Shedding ----------------------------------------------
        "crazy_eights": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.CRAZY_EIGHTS,
                category_id=CategoryID.MATCHING_SHEDDING,
                min_players=2,
                max_players=4,
                has_partial_info=True,
                is_single_player=False,
                deck_size=52,
                max_hand_size=20,
                max_actions=54,
                engine="open_spiel",
            ),
            _factory=lambda **kw: _import("envs.matching_shedding.crazy_eights_env", "CrazyEightsEnv")(**kw),
        ),
        "uno": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.UNO,
                category_id=CategoryID.MATCHING_SHEDDING,
                min_players=2,
                max_players=4,
                has_partial_info=True,
                is_single_player=False,
                deck_size=108,
                max_hand_size=25,
                max_actions=61,
                engine="rlcard",
            ),
            _factory=lambda **kw: _import("envs.matching_shedding.uno_env", "UnoEnv")(**kw),
        ),
        "go_fish": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.GO_FISH,
                category_id=CategoryID.MATCHING_SHEDDING,
                min_players=2,
                max_players=4,
                has_partial_info=True,
                is_single_player=False,
                deck_size=52,
                max_hand_size=13,
                max_actions=13,
                engine="custom",
            ),
            _factory=lambda **kw: _import("envs.matching_shedding.go_fish_env", "GoFishEnv")(**kw),
        ),

        # ---- Trick-Taking ---------------------------------------------------
        "hearts": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.HEARTS,
                category_id=CategoryID.TRICK_TAKING,
                min_players=4,
                max_players=4,
                has_partial_info=True,
                is_single_player=False,
                deck_size=52,
                max_hand_size=13,
                max_actions=53,
                engine="open_spiel",
            ),
            _factory=lambda **kw: _import("envs.trick_taking.hearts_env", "HeartsEnv")(**kw),
        ),
        "spades": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.SPADES,
                category_id=CategoryID.TRICK_TAKING,
                min_players=4,
                max_players=4,
                has_partial_info=True,
                is_single_player=False,
                deck_size=52,
                max_hand_size=13,
                max_actions=66,   # 13 cards + 14 bid levels + pass + nil
                engine="open_spiel",
            ),
            _factory=lambda **kw: _import("envs.trick_taking.spades_env", "SpadesEnv")(**kw),
        ),
        "euchre": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.EUCHRE,
                category_id=CategoryID.TRICK_TAKING,
                min_players=4,
                max_players=4,
                has_partial_info=True,
                is_single_player=False,
                deck_size=24,
                max_hand_size=5,
                max_actions=30,
                engine="open_spiel",
            ),
            _factory=lambda **kw: _import("envs.trick_taking.euchre_env", "EuchreEnv")(**kw),
        ),

        # ---- Melding --------------------------------------------------------
        "gin_rummy": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.GIN_RUMMY,
                category_id=CategoryID.MELDING,
                min_players=2,
                max_players=2,
                has_partial_info=True,
                is_single_player=False,
                deck_size=52,
                max_hand_size=11,
                max_actions=56,
                engine="pettingzoo",
            ),
            _factory=lambda **kw: _import("envs.melding.gin_rummy_env", "GinRummyEnv")(**kw),
        ),
        "rummy_500": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.RUMMY_500,
                category_id=CategoryID.MELDING,
                min_players=2,
                max_players=4,
                has_partial_info=True,
                is_single_player=False,
                deck_size=52,
                max_hand_size=13,
                max_actions=70,
                engine="custom",
            ),
            _factory=lambda **kw: _import("envs.melding.rummy500_env", "Rummy500Env")(**kw),
        ),
        "canasta": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.CANASTA,
                category_id=CategoryID.MELDING,
                min_players=4,
                max_players=4,
                has_partial_info=True,
                is_single_player=False,
                deck_size=104,  # 2 decks
                max_hand_size=15,
                max_actions=80,
                engine="custom",
            ),
            _factory=lambda **kw: _import("envs.melding.canasta_env", "CanastaEnv")(**kw),
        ),

        # ---- Climbing -------------------------------------------------------
        "president": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.PRESIDENT,
                category_id=CategoryID.CLIMBING,
                min_players=4,
                max_players=6,
                has_partial_info=True,
                is_single_player=False,
                deck_size=52,
                max_hand_size=13,
                max_actions=54,
                engine="custom",
            ),
            _factory=lambda **kw: _import("envs.climbing.president_env", "PresidentEnv")(**kw),
        ),
        "big_two": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.BIG_TWO,
                category_id=CategoryID.CLIMBING,
                min_players=4,
                max_players=4,
                has_partial_info=True,
                is_single_player=False,
                deck_size=52,
                max_hand_size=13,
                max_actions=128,
                engine="custom",
            ),
            _factory=lambda **kw: _import("envs.climbing.big_two_env", "BigTwoEnv")(**kw),
        ),
        "tichu": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.TICHU,
                category_id=CategoryID.CLIMBING,
                min_players=4,
                max_players=4,
                has_partial_info=True,
                is_single_player=False,
                deck_size=56,   # 52 + 4 special cards
                max_hand_size=14,
                max_actions=200,
                engine="custom",
            ),
            _factory=lambda **kw: _import("envs.climbing.tichu_env", "TichuEnv")(**kw),
        ),

        # ---- Solitaire ------------------------------------------------------
        "freecell": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.FREECELL,
                category_id=CategoryID.SOLITAIRE,
                min_players=1,
                max_players=1,
                has_partial_info=False,
                is_single_player=True,
                deck_size=52,
                max_hand_size=52,
                max_actions=169,
                engine="custom",
            ),
            _factory=lambda **kw: _import("envs.solitaire.freecell_env", "FreeCellEnv")(**kw),
        ),
        "klondike": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.KLONDIKE,
                category_id=CategoryID.SOLITAIRE,
                min_players=1,
                max_players=1,
                has_partial_info=True,
                is_single_player=True,
                deck_size=52,
                max_hand_size=52,
                max_actions=169,
                engine="custom",
            ),
            _factory=lambda **kw: _import("envs.solitaire.klondike_env", "KlondikeEnv")(**kw),
        ),
        "spider": RegistryEntry(
            metadata=EnvMetadata(
                game_id=GameID.SPIDER,
                category_id=CategoryID.SOLITAIRE,
                min_players=1,
                max_players=1,
                has_partial_info=True,
                is_single_player=True,
                deck_size=104,  # 2 decks
                max_hand_size=104,
                max_actions=169,
                engine="custom",
            ),
            _factory=lambda **kw: _import("envs.solitaire.spider_env", "SpiderEnv")(**kw),
        ),
    }


def _import(module_path: str, class_name: str):
    """Lazy import to avoid loading all game dependencies at startup."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# Singleton — built once on first access
_REGISTRY: dict[str, RegistryEntry] | None = None


def get_registry() -> dict[str, RegistryEntry]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _make_registry()
    return _REGISTRY


# Convenience alias
ENV_REGISTRY = get_registry()


def list_games() -> list[str]:
    return sorted(ENV_REGISTRY.keys())


def games_by_category(category: CategoryID) -> list[str]:
    return [name for name, entry in ENV_REGISTRY.items()
            if entry.metadata.category_id == category]
