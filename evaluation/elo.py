"""
ELO rating system for tracking agent strength.

Anchors (fixed, never updated):
  RandomAgent      = 800
  BasicHeuristic   = 1000
  StrongHeuristic  = 1200

K-factor: 32 during training, 16 for human eval.
Agent ELO > 1250 triggers human play testing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

K_TRAIN = 32
K_HUMAN = 16
HUMAN_EVAL_THRESHOLD = 1250

# Fixed anchor ratings
ANCHOR_RATINGS = {
    "RandomAgent": 800.0,
    "BasicStrategyAgent": 1000.0,
    "StrongHeuristicAgent": 1200.0,
}


@dataclass
class PlayerRecord:
    name: str
    rating: float = 1000.0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.games, 1)


class ELOSystem:
    """
    Tracks ELO ratings for (game, player) pairs.
    Key format: "{game_name}/{player_name}"
    """

    def __init__(self) -> None:
        self._records: dict[str, PlayerRecord] = {}
        # Seed anchors
        for name, rating in ANCHOR_RATINGS.items():
            self._records[name] = PlayerRecord(name=name, rating=rating)

    def _key(self, game: str, player: str) -> str:
        return f"{game}/{player}"

    def get_rating(self, game: str, player: str) -> float:
        return self._records.get(self._key(game, player),
                                  PlayerRecord(player)).rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        game: str,
        agent_name: str,
        opponent_name: str,
        win_rate: float,
        n_games: int = 1,
        k: float = K_TRAIN,
    ) -> float:
        """
        Update agent's ELO based on win_rate against opponent.
        Opponent rating is treated as fixed if it's an anchor.
        Returns updated agent ELO.
        """
        agent_key = self._key(game, agent_name)
        if agent_key not in self._records:
            self._records[agent_key] = PlayerRecord(name=agent_name)

        agent_rec = self._records[agent_key]
        opp_rating = (
            ANCHOR_RATINGS.get(opponent_name)
            or self.get_rating(game, opponent_name)
        )

        expected = self.expected_score(agent_rec.rating, opp_rating)
        # Scale K by sqrt(n_games) to account for more data
        effective_k = k * math.sqrt(n_games) / math.sqrt(200)
        agent_rec.rating += effective_k * (win_rate - expected)

        # Update win/loss counts
        wins = round(win_rate * n_games)
        losses = n_games - wins
        agent_rec.wins += wins
        agent_rec.losses += losses

        return agent_rec.rating

    def should_test_human(self, game: str, agent_name: str) -> bool:
        return self.get_rating(game, agent_name) >= HUMAN_EVAL_THRESHOLD

    def summary(self, game: str) -> str:
        lines = [f"ELO ratings for {game}:"]
        relevant = {k: v for k, v in self._records.items() if game in k}
        for key, rec in sorted(relevant.items(), key=lambda x: -x[1].rating):
            lines.append(f"  {rec.name:<30} {rec.rating:>6.0f}  "
                         f"({rec.wins}W/{rec.losses}L, {rec.win_rate:.1%})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval_games(
    agent,
    game_id,
    opponents: list,
    n_games: int = 200,
) -> dict[str, float]:
    """
    Play n_games against each opponent. Returns win rates per opponent.
    Agent uses deterministic (greedy) policy during evaluation.
    """
    from envs.registry import ENV_REGISTRY
    from training.multi_task_trainer import _game_id_to_name

    game_name = _game_id_to_name(game_id)
    results = {}

    for opponent in opponents:
        wins = 0
        for _ in range(n_games):
            try:
                env = ENV_REGISTRY[game_name].make(mock_mode=False)
            except Exception:
                env = ENV_REGISTRY[game_name].make(mock_mode=True)

            env.reset()
            agent.set_game(game_id)
            steps = 0

            while steps < 500:
                ag = env.agent_selection
                obs_dict = env.observe(ag)
                obs = obs_dict["observation"]
                mask = obs_dict["action_mask"]

                if ag == "player_0":
                    action = agent.act_deterministic(obs, mask)
                else:
                    native = env._get_native_state(ag)
                    action = opponent.act(obs, mask, game_state=native)

                env.step(action)
                steps += 1

                if all(env.terminations.values()):
                    break

            reward = env._cumulative_rewards.get("player_0", 0.0)
            if reward > 0:
                wins += 1

        results[getattr(opponent, "name", str(opponent))] = wins / n_games

    return results
