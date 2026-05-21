"""Tests for gambling environments (mock_mode — no rlcard required)."""

import numpy as np
import pytest

from core.types import OBS_DIM, ACTION_DIM, GameID


class TestBlackjackEnv:
    def setup_method(self):
        from envs.gambling.blackjack_env import BlackjackEnv
        self.env = BlackjackEnv(mock_mode=True)
        self.env.reset()

    def test_obs_shape(self):
        self.env.validate_obs_shape("player_0")

    def test_obs_dtype(self):
        obs = self.env.observe("player_0")["observation"]
        assert obs.dtype == np.float32

    def test_mask_has_legal_actions(self):
        mask = self.env.observe("player_0")["action_mask"]
        assert mask.any()

    def test_legal_actions_in_blackjack_slots(self):
        mask = self.env.observe("player_0")["action_mask"]
        # Legal BJ slots: 10=hit, 11=stand, 12=double, 13=split
        legal_slots = set(np.where(mask)[0])
        assert legal_slots.issubset({10, 11, 12, 13})

    def test_step_stand_ends_game(self):
        self.env.step(11)   # stand
        assert self.env.terminations.get("player_0", False)

    def test_dealer_card_hidden(self):
        obs = self.env.observe("player_0")["observation"]
        planes = obs[:12 * 52].reshape(12, 52)
        # Planes 2–10 should be all zero (dealer hole card hidden)
        assert planes[2:11].sum() == 0.0

    def test_reset_clears_state(self):
        self.env.step(11)
        self.env.reset()
        assert not self.env.terminations.get("player_0", True)


class TestBaccaratEnv:
    def setup_method(self):
        from envs.gambling.baccarat_env import BaccaratEnv
        self.env = BaccaratEnv()
        self.env.reset()

    def test_obs_shape(self):
        self.env.validate_obs_shape("player_0")

    def test_three_legal_actions(self):
        mask = self.env.observe("player_0")["action_mask"]
        # Baccarat: slots 20, 21, 22
        legal = set(np.where(mask)[0])
        assert legal == {20, 21, 22}

    def test_betting_ends_episode(self):
        self.env.step(20)   # bet player
        assert self.env.terminations.get("player_0", False)

    def test_reward_is_finite(self):
        self.env.step(20)
        rewards = self.env._compute_rewards()
        assert np.isfinite(rewards["player_0"])

    def test_score_conservation(self):
        """Baccarat scores must be 0–9."""
        self.env.step(21)
        s = self.env._native_state
        assert 0 <= s["player_score"] <= 9
        assert 0 <= s["banker_score"] <= 9

    def test_full_observation_both_hands_visible(self):
        self.env.step(20)
        obs = self.env.observe("player_0")["observation"]
        planes = obs[:12 * 52].reshape(12, 52)
        # Both player (plane 0) and banker (plane 1) hands should have cards
        assert planes[0].sum() >= 2.0
        assert planes[1].sum() >= 2.0


class TestBaccaratRules:
    """Verify baccarat scoring and reward logic."""

    def test_baccarat_score_ace(self):
        from envs.gambling.baccarat_env import _baccarat_score
        from core.card import Card
        from core.types import Suit, Rank
        hand = [Card(Suit.SPADES, Rank.ACE)]
        assert _baccarat_score(hand) == 1

    def test_baccarat_score_face_card(self):
        from envs.gambling.baccarat_env import _baccarat_score
        from core.card import Card
        from core.types import Suit, Rank
        hand = [Card(Suit.SPADES, Rank.KING)]
        assert _baccarat_score(hand) == 0

    def test_baccarat_score_wraps_at_10(self):
        from envs.gambling.baccarat_env import _baccarat_score
        from core.card import Card
        from core.types import Suit, Rank
        hand = [Card(Suit.SPADES, Rank.SEVEN), Card(Suit.HEARTS, Rank.EIGHT)]
        assert _baccarat_score(hand) == 5   # 15 % 10

    def test_reward_player_win(self):
        from envs.gambling.baccarat_env import _baccarat_reward
        assert _baccarat_reward(0, 7, 3) == 1.0   # bet player, player wins

    def test_reward_banker_win_pays_095(self):
        from envs.gambling.baccarat_env import _baccarat_reward
        assert _baccarat_reward(1, 3, 7) == 0.95  # bet banker, banker wins

    def test_reward_tie_pays_8(self):
        from envs.gambling.baccarat_env import _baccarat_reward
        assert _baccarat_reward(2, 5, 5) == 8.0

    def test_reward_wrong_bet(self):
        from envs.gambling.baccarat_env import _baccarat_reward
        assert _baccarat_reward(0, 3, 7) == -1.0  # bet player, banker wins


class TestBasicStrategy:
    def test_hard_16_vs_10_should_hit(self):
        from agents.heuristic.blackjack_basic_strategy import basic_strategy_action
        # 10♠ (idx 9) + 6♠ (idx 5) = hard 16
        # Dealer shows 10♠ (idx 9)
        action = basic_strategy_action([9, 5], 9)
        assert action == 10   # hit

    def test_hard_11_vs_6_should_double(self):
        from agents.heuristic.blackjack_basic_strategy import basic_strategy_action
        # 6♠ (idx 5) + 5♠ (idx 4) = hard 11
        # Dealer shows 6♠ (idx 5)
        action = basic_strategy_action([5, 4], 5, can_double=True)
        assert action == 12   # double

    def test_pair_aces_always_split(self):
        from agents.heuristic.blackjack_basic_strategy import basic_strategy_action
        action = basic_strategy_action([0, 13], 9, can_split=True)  # A♠ A♥
        assert action == 13   # split

    def test_hard_17_always_stand(self):
        from agents.heuristic.blackjack_basic_strategy import basic_strategy_action
        for dealer_up in range(52):   # any dealer upcard
            action = basic_strategy_action([9, 6], dealer_up)  # 10+7=17
            assert action == 11, f"Should stand on hard 17 vs dealer {dealer_up}"


class TestRewardNormalizer:
    def test_normalizes_to_near_zero_mean(self):
        from training.reward_normalizer import RewardNormalizer
        norm = RewardNormalizer()
        rewards = [1.0, -1.0, 1.0, -1.0, 0.0] * 20
        for r in rewards:
            norm.normalize(GameID.BLACKJACK, r)
        # After many updates, normalized values should be near N(0,1)
        last = norm.normalize(GameID.BLACKJACK, 0.0)
        assert abs(last) < 1.0

    def test_different_games_independent(self):
        from training.reward_normalizer import RewardNormalizer
        norm = RewardNormalizer()
        for _ in range(50):
            norm.normalize(GameID.BLACKJACK, 1.0)    # always +1
        for _ in range(50):
            norm.normalize(GameID.POKER_NLH, 10000.0)  # always +10000
        # Stats should be different
        bj = norm._get(GameID.BLACKJACK)
        pk = norm._get(GameID.POKER_NLH)
        assert bj.mean != pk.mean


class TestCurriculum:
    def test_gambling_starts_with_blackjack(self):
        from training.curriculum import CurriculumScheduler
        from core.types import CategoryID
        sched = CurriculumScheduler(CategoryID.GAMBLING)
        sched.update(0)
        game = sched.sample_game()
        assert game == GameID.BLACKJACK

    def test_advances_phase_at_correct_step(self):
        from training.curriculum import CurriculumScheduler
        from core.types import CategoryID
        sched = CurriculumScheduler(CategoryID.GAMBLING)
        sched.update(20_000_000)
        phase = sched.current_phase()
        assert "poker" in phase.name or "blackjack" in phase.name

    def test_all_games_in_final_phase_are_valid(self):
        from training.curriculum import CurriculumScheduler, CURRICULA
        from core.types import CategoryID
        for cat, phases in CURRICULA.items():
            last_phase = phases[-1]
            for gid in last_phase.game_weights:
                assert isinstance(gid, GameID)


class TestELOSystem:
    def test_win_increases_rating(self):
        from evaluation.elo import ELOSystem
        elo = ELOSystem()
        before = elo.get_rating("blackjack", "ai_agent")
        elo.update("blackjack", "ai_agent", "RandomAgent", win_rate=1.0, n_games=100)
        after = elo.get_rating("blackjack", "ai_agent")
        assert after > before

    def test_loss_decreases_rating(self):
        from evaluation.elo import ELOSystem
        elo = ELOSystem()
        before = elo.get_rating("blackjack", "ai_agent")
        elo.update("blackjack", "ai_agent", "RandomAgent", win_rate=0.0, n_games=100)
        after = elo.get_rating("blackjack", "ai_agent")
        assert after < before

    def test_human_eval_threshold(self):
        from evaluation.elo import ELOSystem, HUMAN_EVAL_THRESHOLD
        elo = ELOSystem()
        # Artificially set high rating
        from evaluation.elo import PlayerRecord
        elo._records["bj/strong"] = PlayerRecord("strong", rating=1300.0)
        assert elo.should_test_human("bj", "strong")

    def test_anchors_seeded_correctly(self):
        from evaluation.elo import ELOSystem, ANCHOR_RATINGS
        elo = ELOSystem()
        for name, rating in ANCHOR_RATINGS.items():
            assert elo._records[name].rating == rating
