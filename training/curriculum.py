"""
Curriculum scheduler — controls which games are sampled during training
and the probability of each game at each training phase.

Gambling curriculum:
  Phase 1 (0–5M):   Blackjack only, opponent = random
  Phase 2 (5–15M):  Blackjack self-play
  Phase 3 (15–30M): Blackjack + Poker (50/50)
  Phase 4 (30–50M): All three games (BJ 40%, Poker 40%, Baccarat 20%)
  Phase 5 (50M+):   Adapter fine-tune phase (one game at a time)
"""

from __future__ import annotations

from dataclasses import dataclass

from core.types import CategoryID, GameID


@dataclass
class Phase:
    name: str
    start_step: int          # training step this phase begins
    game_weights: dict[GameID, float]  # sampling probabilities (sum to 1)
    use_self_play: bool = False


# ---------------------------------------------------------------------------
# Per-category curriculum definitions
# ---------------------------------------------------------------------------

GAMBLING_CURRICULUM: list[Phase] = [
    Phase(
        name="blackjack_random",
        start_step=0,
        game_weights={GameID.BLACKJACK: 1.0},
        use_self_play=False,
    ),
    Phase(
        name="blackjack_selfplay",
        start_step=5_000_000,
        game_weights={GameID.BLACKJACK: 1.0},
        use_self_play=True,
    ),
    Phase(
        name="blackjack_poker",
        start_step=15_000_000,
        game_weights={GameID.BLACKJACK: 0.5, GameID.POKER_NLH: 0.5},
        use_self_play=True,
    ),
    Phase(
        name="all_gambling",
        start_step=30_000_000,
        game_weights={
            GameID.BLACKJACK: 0.40,
            GameID.POKER_NLH: 0.40,
            GameID.BACCARAT:  0.20,
        },
        use_self_play=True,
    ),
]

MATCHING_CURRICULUM: list[Phase] = [
    Phase("crazy_eights_random", 0,
          {GameID.CRAZY_EIGHTS: 1.0}, use_self_play=False),
    Phase("uno_selfplay", 5_000_000,
          {GameID.UNO: 1.0}, use_self_play=True),
    Phase("matching_mixed", 25_000_000,
          {GameID.CRAZY_EIGHTS: 0.5, GameID.UNO: 0.5}, use_self_play=True),
    Phase("all_matching", 50_000_000,
          {GameID.CRAZY_EIGHTS: 0.4, GameID.UNO: 0.4, GameID.GO_FISH: 0.2},
          use_self_play=True),
]

TRICK_TAKING_CURRICULUM: list[Phase] = [
    Phase("hearts_random", 0,
          {GameID.HEARTS: 1.0}, use_self_play=False),
    Phase("hearts_selfplay", 10_000_000,
          {GameID.HEARTS: 1.0}, use_self_play=True),
    Phase("hearts_spades", 35_000_000,
          {GameID.HEARTS: 0.5, GameID.SPADES: 0.5}, use_self_play=True),
    Phase("all_trick_taking", 65_000_000,
          {GameID.HEARTS: 0.33, GameID.SPADES: 0.34, GameID.EUCHRE: 0.33},
          use_self_play=True),
]

MELDING_CURRICULUM: list[Phase] = [
    Phase("gin_rummy_random", 0,
          {GameID.GIN_RUMMY: 1.0}, use_self_play=False),
    Phase("gin_rummy_selfplay", 10_000_000,
          {GameID.GIN_RUMMY: 1.0}, use_self_play=True),
    Phase("gin_rummy500", 35_000_000,
          {GameID.GIN_RUMMY: 0.5, GameID.RUMMY_500: 0.5}, use_self_play=True),
    Phase("all_melding", 55_000_000,
          {GameID.GIN_RUMMY: 0.4, GameID.RUMMY_500: 0.4, GameID.CANASTA: 0.2},
          use_self_play=True),
]

CLIMBING_CURRICULUM: list[Phase] = [
    Phase("president_random", 0,
          {GameID.PRESIDENT: 1.0}, use_self_play=False),
    Phase("big_two_selfplay", 5_000_000,
          {GameID.BIG_TWO: 1.0}, use_self_play=True),
    Phase("climbing_mixed", 30_000_000,
          {GameID.PRESIDENT: 0.5, GameID.BIG_TWO: 0.5}, use_self_play=True),
    Phase("all_climbing", 60_000_000,
          {GameID.PRESIDENT: 0.33, GameID.BIG_TWO: 0.34, GameID.TICHU: 0.33},
          use_self_play=True),
]

SOLITAIRE_CURRICULUM: list[Phase] = [
    Phase("freecell", 0,
          {GameID.FREECELL: 1.0}, use_self_play=False),
    Phase("klondike", 5_000_000,
          {GameID.KLONDIKE: 1.0}, use_self_play=False),
    Phase("all_solitaire", 20_000_000,
          {GameID.FREECELL: 0.33, GameID.KLONDIKE: 0.34, GameID.SPIDER: 0.33},
          use_self_play=False),
]

CURRICULA: dict[CategoryID, list[Phase]] = {
    CategoryID.GAMBLING:          GAMBLING_CURRICULUM,
    CategoryID.MATCHING_SHEDDING: MATCHING_CURRICULUM,
    CategoryID.TRICK_TAKING:      TRICK_TAKING_CURRICULUM,
    CategoryID.MELDING:           MELDING_CURRICULUM,
    CategoryID.CLIMBING:          CLIMBING_CURRICULUM,
    CategoryID.SOLITAIRE:         SOLITAIRE_CURRICULUM,
}


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Returns the current Phase based on training step count.
    Call sample_game() to get which game to run next.
    """

    def __init__(self, category: CategoryID) -> None:
        self.category = category
        self._phases = CURRICULA[category]
        self._current_step = 0

    def update(self, step: int) -> None:
        self._current_step = step

    def current_phase(self) -> Phase:
        phase = self._phases[0]
        for p in self._phases:
            if self._current_step >= p.start_step:
                phase = p
        return phase

    def sample_game(self) -> GameID:
        import random
        phase = self.current_phase()
        games = list(phase.game_weights.keys())
        weights = list(phase.game_weights.values())
        return random.choices(games, weights=weights, k=1)[0]

    def use_self_play(self) -> bool:
        return self.current_phase().use_self_play

    def phase_name(self) -> str:
        return self.current_phase().name
