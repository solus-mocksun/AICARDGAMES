"""
MultiTaskTrainer — orchestrates the full training loop for one category.

Each iteration:
  1. CurriculumScheduler picks which game to play this episode
  2. Agent adapter is swapped to that game
  3. Opponent is sampled from OpponentPool
  4. Episode runs until done, steps collected into RolloutBuffer
  5. Every ROLLOUT_STEPS steps: PPO update runs
  6. Every SNAPSHOT_INTERVAL steps: model snapshot saved to OpponentPool
  7. Every EVAL_INTERVAL steps: ELO evaluation vs baselines
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from core.types import CategoryID, GameID
from envs.registry import ENV_REGISTRY
from training.curriculum import CurriculumScheduler
from training.ppo import PPOTrainer, PPOConfig
from training.self_play import OpponentPool, RandomAgent

ROLLOUT_STEPS = 2048
SNAPSHOT_INTERVAL = 100_000
EVAL_INTERVAL = 2_000_000


@dataclass
class TrainingConfig:
    category: CategoryID
    total_steps: int = 60_000_000
    rollout_steps: int = ROLLOUT_STEPS
    snapshot_interval: int = SNAPSHOT_INTERVAL
    eval_interval: int = EVAL_INTERVAL
    ppo: PPOConfig = None
    use_wandb: bool = False
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        if self.ppo is None:
            self.ppo = PPOConfig()


class MultiTaskTrainer:
    """
    Trains one CategoryExpertModel across all games in its category.

    Usage:
        agent = build_agent()
        config = TrainingConfig(category=CategoryID.GAMBLING)
        trainer = MultiTaskTrainer(agent, config)
        trainer.train()
    """

    def __init__(self, agent, config: TrainingConfig) -> None:
        self.agent = agent
        self.config = config
        self.curriculum = CurriculumScheduler(config.category)
        self.pool = OpponentPool()
        self.ppo = PPOTrainer(agent, config.ppo)
        self._elo_system = None   # lazy import to avoid circular deps

        # Register heuristic anchors based on category
        self._register_anchors()

    def _register_anchors(self) -> None:
        self.pool.add_anchor("random", RandomAgent())
        if self.config.category == CategoryID.GAMBLING:
            from agents.heuristic.blackjack_basic_strategy import BasicStrategyAgent
            self.pool.add_anchor("basic_strategy", BasicStrategyAgent())

    def train(self) -> None:
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        if self.config.use_wandb:
            self._init_wandb()

        print(f"Starting training: {self.config.category.name} "
              f"| target={self.config.total_steps:,} steps")

        step = 0
        episode = 0
        steps_since_snapshot = 0
        steps_since_eval = 0
        t0 = time.time()

        while step < self.config.total_steps:
            self.curriculum.update(step)
            game_id = self.curriculum.sample_game()
            self.ppo.set_game(game_id)

            # Sample opponent
            opponent = self.pool.sample_opponent(game_id)

            # Run one episode
            ep_steps, ep_reward = self._run_episode(game_id, opponent)
            step += ep_steps
            episode += 1
            steps_since_snapshot += ep_steps
            steps_since_eval += ep_steps

            # PPO update when buffer is full enough
            if len(self.ppo.buffer) >= self.config.rollout_steps:
                metrics = self.ppo.update()
                self._log(step, episode, metrics, ep_reward, t0)

            # Snapshot for opponent pool
            if steps_since_snapshot >= self.config.snapshot_interval:
                self.pool.add_snapshot(step, game_id, self.agent)
                steps_since_snapshot = 0
                print(f"  [step {step:,}] Snapshot saved (pool size: {self.pool.pool_size()})")

            # ELO evaluation
            if steps_since_eval >= self.config.eval_interval:
                self._evaluate(step, game_id)
                steps_since_eval = 0

        print(f"Training complete: {step:,} steps in {(time.time()-t0)/3600:.1f}h")

    def _run_episode(
        self, game_id: GameID, opponent
    ) -> tuple[int, float]:
        """Run one full episode. Returns (steps_taken, total_reward)."""
        env = ENV_REGISTRY[_game_id_to_name(game_id)].make(mock_mode=False)
        env.reset()

        steps = 0
        total_reward = 0.0

        while True:
            agent_name = env.agent_selection
            obs_dict = env.observe(agent_name)
            obs = obs_dict["observation"]
            mask = obs_dict["action_mask"]

            is_agent_turn = (agent_name == "player_0")

            if is_agent_turn:
                action = self.ppo.collect_step(
                    obs, mask,
                    reward=env.rewards.get(agent_name, 0.0),
                    done=env.terminations.get(agent_name, False),
                    game_id=game_id,
                )
            else:
                native_state = env._get_native_state(agent_name)
                action = opponent.act(obs, mask, game_state=native_state)

            env.step(action)
            steps += 1

            if all(env.terminations.values()) or all(env.truncations.values()):
                total_reward = env._cumulative_rewards.get("player_0", 0.0)
                break

            if steps > 500:   # safety cap for infinite loops during development
                break

        return steps, total_reward

    def _evaluate(self, step: int, game_id: GameID) -> None:
        if self._elo_system is None:
            from evaluation.elo import ELOSystem
            self._elo_system = ELOSystem()

        from evaluation.elo import run_eval_games
        results = run_eval_games(
            self.agent, game_id, n_games=200,
            opponents=list(self.pool._anchors.values()),
        )
        for opp_name, win_rate in results.items():
            elo = self._elo_system.update(str(game_id), opp_name, win_rate, n_games=200)
            print(f"  [step {step:,}] ELO vs {opp_name}: {elo:.0f} (win rate {win_rate:.1%})")

        if self.config.use_wandb:
            import wandb
            wandb.log({"step": step, **{f"elo/{k}": v for k, v in results.items()}})

    def _log(
        self, step: int, episode: int, metrics: dict,
        ep_reward: float, t0: float,
    ) -> None:
        elapsed = time.time() - t0
        sps = step / max(elapsed, 1)
        phase = self.curriculum.phase_name()
        print(
            f"step={step:>10,} | ep={episode:>6,} | phase={phase:<20} | "
            f"reward={ep_reward:+.3f} | "
            f"policy_loss={metrics['policy_loss']:.4f} | "
            f"value_loss={metrics['value_loss']:.4f} | "
            f"entropy={metrics['entropy']:.4f} | "
            f"{sps:.0f} sps"
        )
        if self.config.use_wandb:
            import wandb
            wandb.log({"step": step, "episode": episode,
                       "reward": ep_reward, **metrics})

    def _init_wandb(self) -> None:
        import wandb
        wandb.init(
            project="aicardgames",
            name=f"{self.config.category.name.lower()}_training",
            config={
                "category": self.config.category.name,
                "total_steps": self.config.total_steps,
                **vars(self.config.ppo),
            },
        )


def _game_id_to_name(game_id: GameID) -> str:
    """Convert GameID enum to registry key string."""
    return game_id.name.lower().replace("_nlh", "_nlh").replace("_", "_")
