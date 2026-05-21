"""Tests for core/state_encoder.py"""

import numpy as np
import pytest

from core.state_encoder import PassthroughEncoder, UnifiedStateEncoder
from core.types import (
    CARD_PLANES, DECK_SIZE, SCALAR_DIM, OBS_DIM,
    ACTION_DIM, CategoryID, GameID,
)


class TestEncoderShape:
    def setup_method(self):
        self.enc = PassthroughEncoder()

    def test_output_shape(self):
        obs = self.enc.encode({})
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_ods_dim_constant(self):
        assert OBS_DIM == CARD_PLANES * DECK_SIZE + SCALAR_DIM
        assert OBS_DIM == 688

    def test_empty_state_all_zeros_except_embeddings(self):
        obs = self.enc.encode({})
        # Card planes 0–10 should be all zero
        card_section = obs[:CARD_PLANES * DECK_SIZE].reshape(CARD_PLANES, DECK_SIZE)
        assert card_section[:11].sum() == 0.0
        # Plane 11 (unknown) should be all ones (all cards unknown)
        assert card_section[11].sum() == float(DECK_SIZE)

    def test_category_embedding_set(self):
        obs = self.enc.encode({})
        scalars = obs[CARD_PLANES * DECK_SIZE:]
        # Category one-hot: PassthroughEncoder uses BLACKJACK = GAMBLING = 0
        category_vec = scalars[0:6]
        assert category_vec[int(CategoryID.GAMBLING)] == 1.0
        assert category_vec.sum() == 1.0

    def test_game_embedding_set(self):
        obs = self.enc.encode({})
        scalars = obs[CARD_PLANES * DECK_SIZE:]
        game_vec = scalars[6:26]
        assert game_vec[int(GameID.BLACKJACK)] == 1.0
        assert game_vec.sum() == 1.0


class TestUnknownPlane:
    def setup_method(self):
        self.enc = PassthroughEncoder()

    def test_unknown_derived_from_known(self):
        # Put 2 cards in my_hand (plane 0)
        planes = np.zeros((CARD_PLANES, DECK_SIZE), dtype=np.float32)
        planes[0, 0] = 1.0   # Ace of Spades
        planes[0, 1] = 1.0   # Two of Spades
        obs = self.enc.encode({"planes": planes})
        card_section = obs[:CARD_PLANES * DECK_SIZE].reshape(CARD_PLANES, DECK_SIZE)
        # plane 11 should have 0 in positions 0 and 1, 1 everywhere else
        assert card_section[11, 0] == 0.0
        assert card_section[11, 1] == 0.0
        assert card_section[11, 2:].sum() == float(DECK_SIZE - 2)

    def test_full_hand_leaves_zero_unknown(self):
        planes = np.zeros((CARD_PLANES, DECK_SIZE), dtype=np.float32)
        planes[0, :] = 1.0   # All 52 cards in hand
        obs = self.enc.encode({"planes": planes})
        card_section = obs[:CARD_PLANES * DECK_SIZE].reshape(CARD_PLANES, DECK_SIZE)
        assert card_section[11].sum() == 0.0


class TestScalarHelpers:
    def setup_method(self):
        self.enc = PassthroughEncoder()

    def _scalars(self, **kwargs) -> np.ndarray:
        import numpy as np
        scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
        if "player_idx" in kwargs:
            UnifiedStateEncoder.set_player_idx(scalars, kwargs["player_idx"])
        if "num_players" in kwargs:
            UnifiedStateEncoder.set_num_players(scalars, kwargs["num_players"])
        if "hand_size" in kwargs:
            UnifiedStateEncoder.set_hand_size(scalars, kwargs["hand_size"]["size"],
                                               kwargs["hand_size"]["max"])
        return scalars

    def test_player_idx_one_hot(self):
        s = self._scalars(player_idx=2)
        # indices 26–31 are player one-hot
        assert s[28] == 1.0   # player_idx 2 → offset 26+2=28
        assert s[26:32].sum() == 1.0

    def test_num_players_normalized(self):
        s = self._scalars(num_players=4)
        assert abs(s[32] - 4/6) < 1e-5

    def test_hand_size_normalized(self):
        s = self._scalars(hand_size={"size": 7, "max": 13})
        assert abs(s[33] - 7/13) < 1e-5


class TestCardConservation:
    """Each card should appear in at most one of planes 0–10."""

    def setup_method(self):
        self.enc = PassthroughEncoder()

    def test_no_overlap_multiple_planes(self):
        planes = np.zeros((CARD_PLANES, DECK_SIZE), dtype=np.float32)
        planes[0, :13] = 1.0   # first 13 cards in my_hand
        planes[6, 13:26] = 1.0  # next 13 in opp_0_known
        obs = self.enc.encode({"planes": planes})
        card_section = obs[:CARD_PLANES * DECK_SIZE].reshape(CARD_PLANES, DECK_SIZE)
        totals = card_section[:11].sum(axis=0)
        assert (totals <= 1).all(), "Card appears in multiple planes"
