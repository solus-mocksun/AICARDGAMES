"""Tests for core/card.py"""

import pytest
import numpy as np

from core.card import Card, Deck, Hand, ALL_CARDS, cards_to_plane, indices_to_plane
from core.types import Suit, Rank, DECK_SIZE


class TestCard:
    def test_index_roundtrip(self):
        for i in range(DECK_SIZE):
            assert Card.from_index(i).index == i

    def test_all_cards_unique(self):
        indices = [c.index for c in ALL_CARDS]
        assert len(set(indices)) == DECK_SIZE

    def test_encoding_layout(self):
        # Spades: 0–12, Hearts: 13–25
        assert Card(Suit.SPADES, Rank.ACE).index == 0
        assert Card(Suit.SPADES, Rank.KING).index == 12
        assert Card(Suit.HEARTS, Rank.ACE).index == 13
        assert Card(Suit.CLUBS, Rank.KING).index == 51

    def test_from_str(self):
        assert Card.from_str("AS") == Card(Suit.SPADES, Rank.ACE)
        assert Card.from_str("KH") == Card(Suit.HEARTS, Rank.KING)
        assert Card.from_str("10D") == Card(Suit.DIAMONDS, Rank.TEN)
        assert Card.from_str("2C") == Card(Suit.CLUBS, Rank.TWO)

    def test_blackjack_values(self):
        assert Card(Suit.SPADES, Rank.ACE).blackjack_value == 11
        assert Card(Suit.SPADES, Rank.KING).blackjack_value == 10
        assert Card(Suit.SPADES, Rank.QUEEN).blackjack_value == 10
        assert Card(Suit.SPADES, Rank.TEN).blackjack_value == 10
        assert Card(Suit.SPADES, Rank.TWO).blackjack_value == 2
        assert Card(Suit.SPADES, Rank.NINE).blackjack_value == 9

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError):
            Card.from_index(52)
        with pytest.raises(ValueError):
            Card.from_index(-1)


class TestDeck:
    def test_full_deck_size(self):
        d = Deck()
        assert len(d) == DECK_SIZE

    def test_deal_reduces_size(self):
        d = Deck()
        d.deal(5)
        assert len(d) == DECK_SIZE - 5

    def test_deal_returns_correct_count(self):
        d = Deck()
        cards = d.deal(7)
        assert len(cards) == 7

    def test_deal_one(self):
        d = Deck()
        c = d.deal_one()
        assert isinstance(c, Card)
        assert len(d) == DECK_SIZE - 1

    def test_deal_too_many_raises(self):
        d = Deck()
        with pytest.raises(ValueError):
            d.deal(DECK_SIZE + 1)

    def test_shuffle_changes_order(self):
        import random
        d1 = Deck(rng=random.Random(42))
        d2 = Deck(rng=random.Random(99))
        d1.shuffle()
        d2.shuffle()
        c1 = d1.deal(DECK_SIZE)
        c2 = d2.deal(DECK_SIZE)
        assert c1 != c2   # astronomically unlikely to match

    def test_no_duplicate_cards(self):
        d = Deck()
        d.shuffle()
        cards = d.deal(DECK_SIZE)
        assert len(set(c.index for c in cards)) == DECK_SIZE


class TestHand:
    def test_to_plane_correct_bits(self):
        c1 = Card(Suit.SPADES, Rank.ACE)   # index 0
        c2 = Card(Suit.HEARTS, Rank.KING)  # index 25
        hand = Hand([c1, c2])
        plane = hand.to_plane()
        assert plane.shape == (DECK_SIZE,)
        assert plane[0] == 1.0
        assert plane[25] == 1.0
        assert plane.sum() == 2.0

    def test_empty_hand_plane_all_zeros(self):
        hand = Hand()
        assert hand.to_plane().sum() == 0.0

    def test_add_remove(self):
        hand = Hand()
        c = Card(Suit.SPADES, Rank.ACE)
        hand.add(c)
        assert hand.contains(c)
        hand.remove(c)
        assert not hand.contains(c)

    def test_from_indices(self):
        hand = Hand.from_indices([0, 13, 26, 39])
        assert len(hand) == 4


class TestPlaneHelpers:
    def test_cards_to_plane_none(self):
        plane = cards_to_plane(None)
        assert plane.sum() == 0.0

    def test_indices_to_plane(self):
        plane = indices_to_plane([0, 51])
        assert plane[0] == 1.0
        assert plane[51] == 1.0
        assert plane.sum() == 2.0
