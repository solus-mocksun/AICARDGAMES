"""
Baccarat environment — fully custom implementation.

Rules: Player and Banker each draw 2–3 cards following fixed third-card rules.
The hand closest to 9 wins. Face cards and tens = 0. Ace = 1.
Only decision: bet on Player (0), Banker (1), or Tie (2) before the hand.

State encoding:
  Plane 0: player hand cards (face-up, fully observable)
  Plane 1: banker hand cards (face-up, fully observable)
  Scalars: my_score, banker_score, bet_placed (one-hot 3), round_norm
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from core.card import Card, Deck
from core.state_encoder import UnifiedStateEncoder
from core.types import EnvMetadata, GameID, CategoryID
from envs.base_env import BaseCardGameEnv


class BaccaratStateEncoder(UnifiedStateEncoder):
    def __init__(self) -> None:
        super().__init__(game_id=GameID.BACCARAT, strict_partial_info=False)

    def _fill_card_planes(self, state: dict, planes: np.ndarray) -> None:
        for idx in state.get("player_hand", []):
            if 0 <= idx < 52:
                planes[0, idx] = 1.0
        for idx in state.get("banker_hand", []):
            if 0 <= idx < 52:
                planes[1, idx] = 1.0

    def _fill_scalars(self, state: dict, scalars: np.ndarray) -> None:
        self.set_player_idx(scalars, 0)
        self.set_num_players(scalars, 2)
        self.set_my_score(scalars, state.get("player_score", 0), max_score=9)
        self.set_opp_scores(scalars, [state.get("banker_score", 0)], max_score=9)
        self.set_round(scalars, state.get("round", 0), max_rounds=100)

        # Encode bet as one-hot in scalars 44–46
        bet = state.get("bet", -1)
        if 0 <= bet <= 2:
            scalars[44 + bet] = 1.0


class BaccaratEnv(BaseCardGameEnv):
    metadata = EnvMetadata(
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
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng = random.Random()
        self._native_state: dict = {}
        self._round = 0

    def _make_encoder(self) -> BaccaratStateEncoder:
        return BaccaratStateEncoder()

    def _native_reset(self) -> None:
        self._round = 0
        self._native_state = {
            "phase": "betting",
            "player_hand": [],
            "banker_hand": [],
            "player_score": 0,
            "banker_score": 0,
            "bet": -1,
            "round": self._round,
            "legal_actions": [0, 1, 2],   # bet player / banker / tie
            "done": False,
            "reward": 0.0,
        }

    def _native_step(self, native_action: int) -> None:
        state = self._native_state

        if state["phase"] == "betting":
            state["bet"] = native_action
            state["phase"] = "deal"
            self._deal_and_resolve()
        # After dealing, episode is done — no further actions

    def _deal_and_resolve(self) -> None:
        deck = Deck(rng=self._rng)
        deck.shuffle()

        # Deal 2 cards each
        p_hand = [deck.deal_one(), deck.deal_one()]
        b_hand = [deck.deal_one(), deck.deal_one()]

        p_score = _baccarat_score(p_hand)
        b_score = _baccarat_score(b_hand)

        # Third card rules (standard Baccarat)
        p_drew = False
        if p_score <= 5:
            p_hand.append(deck.deal_one())
            p_score = _baccarat_score(p_hand)
            p_drew = True

        if not p_drew:
            if b_score <= 5:
                b_hand.append(deck.deal_one())
                b_score = _baccarat_score(b_hand)
        else:
            p_third = p_hand[2].rank + 1 if p_hand[2].rank < 9 else 0
            # Banker draws based on their score and player's third card
            if b_score <= 2:
                b_hand.append(deck.deal_one())
            elif b_score == 3 and p_third != 8:
                b_hand.append(deck.deal_one())
            elif b_score == 4 and p_third in [2,3,4,5,6,7]:
                b_hand.append(deck.deal_one())
            elif b_score == 5 and p_third in [4,5,6,7]:
                b_hand.append(deck.deal_one())
            elif b_score == 6 and p_third in [6,7]:
                b_hand.append(deck.deal_one())
            b_score = _baccarat_score(b_hand)

        bet = self._native_state["bet"]
        reward = _baccarat_reward(bet, p_score, b_score)

        self._native_state.update({
            "player_hand": [c.index for c in p_hand],
            "banker_hand": [c.index for c in b_hand],
            "player_score": p_score,
            "banker_score": b_score,
            "legal_actions": [],
            "done": True,
            "reward": reward,
        })
        self._round += 1
        self._native_state["round"] = self._round

    def _get_native_state(self, agent: str) -> dict:
        return self._native_state

    def _get_legal_native_actions(self, agent: str) -> list[int]:
        return self._native_state.get("legal_actions", [0, 1, 2])

    def _compute_rewards(self) -> dict[str, float]:
        return {"player_0": self._native_state.get("reward", 0.0)}

    def _is_done(self) -> bool:
        return self._native_state.get("done", False)


def _baccarat_score(cards: list[Card]) -> int:
    total = 0
    for card in cards:
        rank = card.rank
        if rank == 0:          # Ace
            total += 1
        elif rank >= 9:        # 10, J, Q, K = 0
            total += 0
        else:
            total += rank + 1
    return total % 10


def _baccarat_reward(bet: int, p_score: int, b_score: int) -> float:
    if p_score > b_score:
        winner = 0   # player wins
    elif b_score > p_score:
        winner = 1   # banker wins
    else:
        winner = 2   # tie

    if bet == winner:
        if bet == 1:   # banker bet pays 0.95:1
            return 0.95
        elif bet == 2: # tie pays 8:1
            return 8.0
        else:
            return 1.0
    elif bet == 2 and winner != 2:
        return 0.0    # tie bet pushed on non-tie (some rules return bet)
    return -1.0
