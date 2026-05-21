"""Tests for core/action_space.py"""

import numpy as np
import pytest

from core.action_space import ActionSpaceRegistry, empty_mask
from core.types import ACTION_DIM, GameID


ALL_GAME_IDS = list(GameID)


class TestRegistration:
    def test_all_games_registered(self):
        registered = ActionSpaceRegistry.registered_games()
        for gid in ALL_GAME_IDS:
            assert gid in registered, f"{gid!r} not registered in ActionSpaceRegistry"

    def test_get_returns_mapper(self):
        mapper = ActionSpaceRegistry.get(GameID.BLACKJACK)
        assert mapper is not None
        assert mapper.game_id == GameID.BLACKJACK


class TestBlackjackMapper:
    def setup_method(self):
        self.mapper = ActionSpaceRegistry.get(GameID.BLACKJACK)

    def test_hit_maps_to_slot_10(self):
        assert self.mapper.native_to_unified(0) == 10   # hit

    def test_stand_maps_to_slot_11(self):
        assert self.mapper.native_to_unified(1) == 11   # stand

    def test_double_maps_to_slot_12(self):
        assert self.mapper.native_to_unified(2) == 12

    def test_split_maps_to_slot_13(self):
        assert self.mapper.native_to_unified(3) == 13

    def test_roundtrip_all_actions(self):
        for native in [0, 1, 2, 3]:
            slot = self.mapper.native_to_unified(native)
            assert self.mapper.unified_to_native(slot) == native

    def test_legal_mask_shape(self):
        mask = self.mapper.legal_mask(None)
        assert mask.shape == (ACTION_DIM,)
        assert mask.dtype == bool

    def test_legal_mask_has_legal_actions(self):
        mask = self.mapper.legal_mask(None)
        assert mask.any()


class TestBaccaratMapper:
    def setup_method(self):
        self.mapper = ActionSpaceRegistry.get(GameID.BACCARAT)

    def test_three_actions_in_slots_20_22(self):
        assert self.mapper.native_to_unified(0) == 20
        assert self.mapper.native_to_unified(1) == 21
        assert self.mapper.native_to_unified(2) == 22

    def test_roundtrip(self):
        for native in [0, 1, 2]:
            assert self.mapper.unified_to_native(self.mapper.native_to_unified(native)) == native


class TestPokerMapper:
    def setup_method(self):
        self.mapper = ActionSpaceRegistry.get(GameID.POKER_NLH)

    def test_all_six_actions_in_0_5(self):
        for i in range(6):
            assert self.mapper.native_to_unified(i) == i

    def test_roundtrip(self):
        for i in range(6):
            assert self.mapper.unified_to_native(i) == i


class TestCardPlayMappers:
    @pytest.mark.parametrize("game_id,offset", [
        (GameID.CRAZY_EIGHTS, 64),
        (GameID.UNO, 64),
        (GameID.HEARTS, 128),
        (GameID.SPADES, 128),
        (GameID.EUCHRE, 128),
    ])
    def test_card_index_maps_to_offset_plus_index(self, game_id, offset):
        mapper = ActionSpaceRegistry.get(game_id)
        for card_idx in [0, 13, 25, 51]:
            slot = mapper.native_to_unified(card_idx)
            assert slot == offset + card_idx

    def test_draw_action_maps_outside_card_range(self):
        mapper = ActionSpaceRegistry.get(GameID.CRAZY_EIGHTS)
        draw_slot = mapper.native_to_unified(100)
        assert draw_slot not in range(64, 116)   # not in the card range


class TestSolitaireMapper:
    def setup_method(self):
        self.mapper = ActionSpaceRegistry.get(GameID.FREECELL)

    def test_move_maps_and_roundtrips(self):
        action = (2, 7)   # from tableau 2 to foundation 3
        slot = self.mapper.native_to_unified(action)
        assert 320 <= slot < 320 + 169
        recovered = self.mapper.unified_to_native(slot)
        assert recovered == action

    def test_all_location_pairs_have_unique_slots(self):
        slots = set()
        for f in range(13):
            for t in range(13):
                slot = self.mapper.native_to_unified((f, t))
                assert slot not in slots
                slots.add(slot)
        assert len(slots) == 169


class TestMaskShape:
    @pytest.mark.parametrize("game_id", ALL_GAME_IDS)
    def test_mask_correct_shape_and_type(self, game_id):
        mapper = ActionSpaceRegistry.get(game_id)
        mask = mapper.legal_mask(None)
        assert mask.shape == (ACTION_DIM,), f"{game_id}: mask shape {mask.shape}"
        assert mask.dtype == bool, f"{game_id}: mask dtype {mask.dtype}"
        assert mask.any(), f"{game_id}: mask has no legal actions"


class TestEmptyMask:
    def test_empty_mask_all_false(self):
        m = empty_mask()
        assert not m.any()
        assert m.shape == (ACTION_DIM,)
