"""
Omi rules engine and helper utilities.

Implements deck creation, shuffling, dealing, legal move masks (must follow
suit), trick resolution, scoring, and terminal checks. The environment builds
on these helpers to provide a PettingZoo-compatible API.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

SUITS: Sequence[str] = ("C", "D", "H", "S")  # Clubs, Diamonds, Hearts, Spades

# Omi (32-card) ranks: 7..A (8 ranks × 4 suits = 32 cards)
RANKS: Sequence[str] = ("7", "8", "9", "10", "J", "Q", "K", "A")

NUM_CARDS = 32
HAND_SIZE = 8
TRICKS_PER_HAND = 8
DEAL_FIRST = 4
DEAL_SECOND = 4

# Action space: 32 card plays + 4 trump declarations (suits)
ACTION_TRUMP_OFFSET = NUM_CARDS
ACTION_DIM = NUM_CARDS + 4


@dataclass(frozen=True)
class Card:
    suit: str
    rank: str

    @property
    def value(self) -> int:
        """Returns numeric rank ordering for comparison (2 low, Ace high)."""
        return RANKS.index(self.rank) + 2


def index_to_card(idx: int) -> Card:
    if idx < 0 or idx >= NUM_CARDS:
        raise ValueError(f"Card index out of range: {idx}")
    suit = SUITS[idx // len(RANKS)]
    rank = RANKS[idx % len(RANKS)]
    return Card(suit, rank)


def card_to_index(card: Card) -> int:
    suit_idx = SUITS.index(card.suit)
    rank_idx = RANKS.index(card.rank)
    return suit_idx * len(RANKS) + rank_idx


def shuffle_deck(rng: random.Random) -> List[int]:
    deck = list(range(NUM_CARDS))
    rng.shuffle(deck)
    return deck


def deal_first_four(deck: Sequence[int]) -> Tuple[List[List[int]], List[int]]:
    """Deal first 4 cards to each player, return (hands, remaining_deck)."""
    if len(deck) != NUM_CARDS:
        raise ValueError(f"Deck must contain {NUM_CARDS} cards")
    hands = [list(deck[i * DEAL_FIRST : (i + 1) * DEAL_FIRST]) for i in range(4)]
    remaining = list(deck[4 * DEAL_FIRST :])
    return hands, remaining


def deal_remaining_four(hands: List[List[int]], remaining_deck: Sequence[int]) -> List[List[int]]:
    """Deal the remaining 4 cards to each player (expects 16 cards remaining)."""
    if len(remaining_deck) != 4 * DEAL_SECOND:
        raise ValueError(f"Expected {4 * DEAL_SECOND} remaining cards, got {len(remaining_deck)}")
    out = [list(h) for h in hands]
    for i in range(4):
        out[i].extend(list(remaining_deck[i * DEAL_SECOND : (i + 1) * DEAL_SECOND]))
    return out


def legal_card_mask(hand: Sequence[int], lead_suit: Optional[str]) -> List[int]:
    """
    Compute a legal action mask for playing a card under must-follow-suit rules.

    Returns a length-NUM_CARDS list of 0/1 flags for each card index.
    """
    mask = [0] * NUM_CARDS
    if not hand:
        return mask

    suit_matches = []
    if lead_suit is not None:
        suit_matches = [c for c in hand if index_to_card(c).suit == lead_suit]

    playable = suit_matches if suit_matches else list(hand)
    for card_idx in playable:
        mask[card_idx] = 1
    return mask


def legal_trump_mask() -> List[int]:
    """Legal mask for trump declaration actions (only 4 suit choices)."""
    mask = [0] * ACTION_DIM
    for i in range(4):
        mask[ACTION_TRUMP_OFFSET + i] = 1
    return mask


def is_trump_action(action: int) -> bool:
    return action >= ACTION_TRUMP_OFFSET and action < ACTION_TRUMP_OFFSET + 4


def resolve_trick(
    trick: Sequence[Tuple[int, int]], lead_suit: str, trump_suit: Optional[str]
) -> int:
    """
    Determine trick winner.

    Args:
        trick: sequence of (player_id, card_index) in play order.
        lead_suit: suit that initiated the trick.
        trump_suit: optional trump suit. If provided, highest trump wins; otherwise
            highest card of the lead suit wins.

    Returns:
        winning player id.
    """
    if not trick:
        raise ValueError("Trick cannot be empty")

    winning_player, winning_card = trick[0]
    winning_card_obj = index_to_card(winning_card)

    for player, card_idx in trick[1:]:
        card = index_to_card(card_idx)
        if trump_suit is not None:
            if card.suit == trump_suit:
                if winning_card_obj.suit != trump_suit or card.value > winning_card_obj.value:
                    winning_player, winning_card_obj = player, card
                continue
            if winning_card_obj.suit == trump_suit:
                continue

        # No trump or neither card is trump: compare within lead suit
        if card.suit == winning_card_obj.suit == lead_suit and card.value > winning_card_obj.value:
            winning_player, winning_card_obj = player, card
        elif winning_card_obj.suit != lead_suit and card.suit == lead_suit:
            winning_player, winning_card_obj = player, card

    return winning_player


def team_for_player(player_id: int) -> int:
    """Team assignment: players 0/2 vs 1/3."""
    return 0 if player_id % 2 == 0 else 1


def is_terminal(tricks_won: Tuple[int, int], cards_remaining: int) -> bool:
    """Episode ends when all cards have been played (32 plays)."""
    return cards_remaining == 0


def compute_winner(tricks_won: Tuple[int, int]) -> int:
    """Return 0 if team A wins, 1 if team B wins, -1 for tie."""
    if tricks_won[0] > tricks_won[1]:
        return 0
    if tricks_won[1] > tricks_won[0]:
        return 1
    return -1
