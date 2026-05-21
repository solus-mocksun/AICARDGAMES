"""Tests for models/ — backbone, adapter, heads, full agent."""

import pytest
import numpy as np

torch = pytest.importorskip("torch", reason="torch not installed")

from core.types import (
    OBS_DIM, ACTION_DIM, NUM_CATEGORIES, NUM_GAMES,
    GameID, CategoryID, GAME_TO_CATEGORY,
)

# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class TestBackbone:
    def setup_method(self):
        from models.backbone import CategoryExpertModel
        self.model = CategoryExpertModel(n_layers=2)  # small for test speed

    def _ids(self, B=2):
        cat = torch.zeros(B, dtype=torch.long)    # GAMBLING
        game = torch.zeros(B, dtype=torch.long)   # BLACKJACK
        return cat, game

    def test_output_shape_single_step(self):
        obs = torch.randn(2, OBS_DIM)
        cat, game = self._ids()
        out = self.model(obs, cat, game)
        assert out.shape == (2, 512)

    def test_output_shape_history(self):
        obs = torch.randn(2, 8, OBS_DIM)
        cat, game = self._ids()
        out = self.model(obs, cat, game)
        assert out.shape == (2, 512)

    def test_output_dtype_float32(self):
        obs = torch.randn(1, OBS_DIM)
        cat, game = self._ids(1)
        out = self.model(obs, cat, game)
        assert out.dtype == torch.float32

    def test_different_games_different_outputs(self):
        obs = torch.randn(1, OBS_DIM)
        cat_bj = torch.tensor([int(CategoryID.GAMBLING)])
        cat_hr = torch.tensor([int(CategoryID.TRICK_TAKING)])
        game_bj = torch.tensor([int(GameID.BLACKJACK)])
        game_hr = torch.tensor([int(GameID.HEARTS)])
        out_bj = self.model(obs, cat_bj, game_bj)
        out_hr = self.model(obs, cat_hr, game_hr)
        assert not torch.allclose(out_bj, out_hr)

    def test_parameter_count_reasonable(self):
        from models.backbone import CategoryExpertModel
        full_model = CategoryExpertModel()  # 8 layers
        params = full_model.count_parameters()
        # Should be ~26M ± 20%
        assert 20_000_000 < params < 32_000_000, f"Unexpected param count: {params:,}"


# ---------------------------------------------------------------------------
# Policy + Value heads
# ---------------------------------------------------------------------------

class TestPolicyHead:
    def setup_method(self):
        from models.policy_value_head import PolicyHead
        self.head = PolicyHead()

    def test_output_shape(self):
        hidden = torch.randn(4, 512)
        mask = torch.ones(4, ACTION_DIM, dtype=torch.bool)
        out = self.head(hidden, mask)
        assert out.shape == (4, ACTION_DIM)

    def test_log_probs_sum_to_one(self):
        hidden = torch.randn(2, 512)
        mask = torch.ones(2, ACTION_DIM, dtype=torch.bool)
        log_probs = self.head(hidden, mask)
        probs = log_probs.exp().sum(dim=-1)
        assert torch.allclose(probs, torch.ones(2), atol=1e-4)

    def test_illegal_actions_get_zero_prob(self):
        hidden = torch.randn(1, 512)
        mask = torch.zeros(1, ACTION_DIM, dtype=torch.bool)
        mask[0, 10] = True   # only slot 10 is legal
        log_probs = self.head(hidden, mask)
        probs = log_probs.exp()
        assert probs[0, 10].item() > 0.99
        assert probs[0, 11].item() < 1e-5

    def test_distribution_samples_legal_action(self):
        hidden = torch.randn(1, 512)
        mask = torch.zeros(1, ACTION_DIM, dtype=torch.bool)
        legal = [10, 11, 12]
        for s in legal:
            mask[0, s] = True
        dist = self.head.get_dist(hidden, mask)
        for _ in range(20):
            sample = dist.sample().item()
            assert sample in legal


class TestValueHead:
    def setup_method(self):
        from models.policy_value_head import ValueHead
        self.head = ValueHead()

    def test_output_shape(self):
        hidden = torch.randn(4, 512)
        out = self.head(hidden)
        assert out.shape == (4, 1)

    def test_output_in_minus1_plus1(self):
        hidden = torch.randn(100, 512)
        out = self.head(hidden)
        assert (out >= -1.0).all() and (out <= 1.0).all()


# ---------------------------------------------------------------------------
# Full agent
# ---------------------------------------------------------------------------

class TestCardGameAgent:
    def setup_method(self):
        from models.full_agent import build_agent
        self.agent = build_agent(device=torch.device("cpu"))

    def test_act_returns_legal_action(self):
        self.agent.set_game(GameID.BLACKJACK)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[10] = True   # only hit is legal
        mask[11] = True   # and stand
        action, log_prob, value = self.agent.act(obs, mask)
        assert action in [10, 11]
        assert isinstance(log_prob, float)
        assert -1.0 <= value <= 1.0

    def test_act_deterministic_picks_best(self):
        self.agent.set_game(GameID.BLACKJACK)
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[11] = True   # only stand
        action = self.agent.act_deterministic(obs, mask)
        assert action == 11

    def test_evaluate_batch(self):
        self.agent.set_game(GameID.BLACKJACK)
        B = 8
        obs = torch.randn(B, OBS_DIM)
        mask = torch.ones(B, ACTION_DIM, dtype=torch.bool)
        actions = torch.randint(0, ACTION_DIM, (B,))
        log_probs, entropy, values = self.agent.evaluate(obs, mask, actions)
        assert log_probs.shape == (B,)
        assert entropy.shape == (B,)
        assert values.shape == (B,)
        assert (entropy > 0).all()

    def test_set_game_switches_adapter(self):
        self.agent.set_game(GameID.BLACKJACK)
        assert self.agent._current_game == GameID.BLACKJACK
        self.agent.set_game(GameID.BACCARAT)
        assert self.agent._current_game == GameID.BACCARAT


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

class TestAdapterRegistry:
    def setup_method(self):
        from models.backbone import CategoryExpertModel
        from models.adapter import AdapterRegistry
        self.backbone = CategoryExpertModel(n_layers=2)
        self.registry = AdapterRegistry(self.backbone)

    def test_creates_adapter_on_demand(self):
        adapter = self.registry.get_or_create(GameID.BLACKJACK)
        assert adapter is not None
        assert adapter.game_id == GameID.BLACKJACK

    def test_adapter_has_small_params(self):
        adapter = self.registry.get_or_create(GameID.BLACKJACK)
        params = adapter.num_parameters()
        # Should be ~0.52M — much smaller than backbone's 26M
        assert params < 2_000_000, f"Adapter too large: {params:,}"

    def test_different_games_different_adapters(self):
        a1 = self.registry.get_or_create(GameID.BLACKJACK)
        a2 = self.registry.get_or_create(GameID.POKER_NLH)
        assert a1 is not a2
