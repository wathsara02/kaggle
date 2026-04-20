"""
Observation and action encoding utilities for the Omi environment.

The observation structure mirrors PettingZoo's AEC/AIO pattern:
{
    "observation": np.ndarray,  # flat vector for policy input
    "action_mask": np.ndarray,  # legal moves (includes trump declaration)
    "history": np.ndarray       # sequence features for recurrent policies
}
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import rules

# History configuration: number of past plays to encode
HISTORY_LEN = 32  # total plays in a 32-card hand (8 tricks × 4 plays)
HISTORY_FEAT_DIM = rules.NUM_CARDS + 12  # card one-hot + 4 player + 4 lead + 4 trump


def card_one_hot(card_idx: int) -> np.ndarray:
    vec = np.zeros(rules.NUM_CARDS, dtype=np.float32)
    vec[card_idx] = 1.0
    return vec


def one_hot(idx: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def encode_history(
    history: Sequence[Tuple[int, int, Optional[str], Optional[str]]]
) -> np.ndarray:
    """
    Encode a list of past plays into a fixed-length sequence.

    Args:
        history: iterable of tuples (player_id, card_idx, lead_suit, trump_suit)
            ordered from oldest to newest.
    """
    encoded = np.zeros((HISTORY_LEN, HISTORY_FEAT_DIM), dtype=np.float32)
    start = max(0, len(history) - HISTORY_LEN)
    slice_hist = history[start:]
    offset = HISTORY_LEN - len(slice_hist)
    for i, (player, card_idx, lead_suit, trump_suit) in enumerate(slice_hist):
        row = np.concatenate(
            [
                card_one_hot(card_idx),
                one_hot(player, 4),
                one_hot(rules.SUITS.index(lead_suit), 4)
                if lead_suit is not None
                else np.zeros(4, dtype=np.float32),
                one_hot(rules.SUITS.index(trump_suit), 4)
                if trump_suit is not None
                else np.zeros(4, dtype=np.float32),
            ]
        )
        encoded[offset + i] = row
    return encoded


def compute_void_matrix(
    history: Sequence[Tuple[int, int, Optional[str], Optional[str]]]
) -> np.ndarray:
    """
    Compute a (4 players × 4 suits) void probability matrix from play history.

    Combines two signals:
    1. Confirmed voids (value=1.0): player failed to follow the led suit —
       they are definitively void in that suit from this point forward.
    2. Soft probability (value in (0,1)): if a player has not yet played a
       suit but a fraction of that suit's cards are already accounted for by
       other players, those cards cannot be in this player's hand.  The
       probability equals (cards_of_suit_seen_by_others / cards_per_suit).

    This gives the network a richer signal than a binary void flag, enabling
    earlier, more confident void-based inferences before a suit failure occurs.
    """
    void_matrix = np.zeros((4, 4), dtype=np.float32)
    cards_per_suit = rules.NUM_CARDS // len(rules.SUITS)  # 8

    # Track which cards of each suit have been played, and whether each player
    # has played at least one card of each suit.
    suit_cards_played: List[set] = [set() for _ in range(len(rules.SUITS))]
    player_suit_played = [[False] * len(rules.SUITS) for _ in range(4)]

    for player, card_idx, lead_suit, _ in history:
        card_suit = rules.index_to_card(card_idx).suit
        s = rules.SUITS.index(card_suit)
        suit_cards_played[s].add(card_idx)
        player_suit_played[player][s] = True

        # Confirmed void: failed to follow the led suit
        if lead_suit is not None and card_suit != lead_suit:
            lead_s = rules.SUITS.index(lead_suit)
            void_matrix[player][lead_s] = 1.0

    # Soft probability: if player hasn't played suit S, estimate void likelihood
    # from how many suit-S cards are already accounted for by other players.
    # Since player_suit_played[p][s] is False here, all cards in
    # suit_cards_played[s] were played by others and cannot be in player p's hand.
    for p in range(4):
        for s in range(len(rules.SUITS)):
            if void_matrix[p][s] == 1.0:
                continue  # already confirmed void
            if not player_suit_played[p][s] and suit_cards_played[s]:
                void_matrix[p][s] = len(suit_cards_played[s]) / cards_per_suit

    return void_matrix  # shape (4, 4) — flatten when adding to obs


def encode_observation(
    agent_id: int,
    hand: Sequence[int],
    trump_suit: Optional[str],
    lead_suit: Optional[str],
    current_trick: Sequence[Tuple[int, int]],
    scores: Tuple[int, int],
    action_mask: Sequence[int],
    history: Sequence[Tuple[int, int, Optional[str], Optional[str]]],
) -> dict:
    """
    Build the observation dictionary for the current agent.

    Args:
        agent_id: active agent id (0-3).
        hand: list of card indices for the agent.
        trump_suit: current trump suit or None if not declared yet.
        lead_suit: suit of the current trick leader.
        current_trick: list of (player_id, card_idx) tuples already played in this trick.
        scores: tuple of team scores (team 0, team 1).
        action_mask: legal action mask aligned with rules.ACTION_DIM.
        history: past play tuples (player_id, card_idx, lead_suit, trump_suit).
    """
    hand_vec = np.zeros(rules.NUM_CARDS, dtype=np.float32)
    for c in hand:
        hand_vec[c] = 1.0

    # Normalized suit proportions: 4-dim vector, one entry per suit.
    # Lets the network read suit distribution directly without summing the
    # 32-dim one-hot — especially valuable at trump declaration time.
    suit_counts = np.zeros(4, dtype=np.float32)
    for c in hand:
        suit_counts[rules.SUITS.index(rules.index_to_card(c).suit)] += 1.0
    if hand:
        suit_counts /= float(len(hand))

    # Hand strength: normalized sum of card rank values.
    # Card.value ranges from 2 (rank '7') to 9 (rank 'A').
    # Normalised by theoretical max (9 * HAND_SIZE) so the value is in [0, 1].
    # Gives the network an immediate scalar signal for hand quality rather than
    # requiring it to infer strength from the 32-dim card one-hot vector.
    _max_card_value = len(rules.RANKS) + 1  # 9 (RANKS.index('A') + 2)
    hand_strength = np.array(
        [sum(rules.index_to_card(c).value for c in hand) / (_max_card_value * rules.HAND_SIZE)],
        dtype=np.float32,
    )

    # Void matrix: 4 players × 4 suits = 16 values.
    # void[p][s] = 1 if player p has been observed failing to follow suit s,
    # meaning they are definitely void in suit s from that point forward.
    # Pre-computed from history so the network reads it directly rather than
    # having to discover the pattern across 1408 raw history features.
    void_flat = compute_void_matrix(history).reshape(-1)  # 16 values

    trump_vec = (
        one_hot(rules.SUITS.index(trump_suit), 4) if trump_suit is not None else np.zeros(4, dtype=np.float32)
    )
    lead_vec = (
        one_hot(rules.SUITS.index(lead_suit), 4) if lead_suit is not None else np.zeros(4, dtype=np.float32)
    )

    trick_vecs: List[np.ndarray] = []
    # Up to 4 cards per trick; pad to 4
    for _, card_idx in current_trick:
        trick_vecs.append(card_one_hot(card_idx))
    while len(trick_vecs) < 4:
        trick_vecs.append(np.zeros(rules.NUM_CARDS, dtype=np.float32))
    trick_flat = np.concatenate(trick_vecs, axis=0)

    # normalize to [0,1] for an 8-trick hand
    score_vec = np.array(scores, dtype=np.float32) / float(rules.TRICKS_PER_HAND)
    player_vec = one_hot(agent_id, 4)

    observation_vec = np.concatenate(
        [
            hand_vec,
            trump_vec,
            lead_vec,
            trick_flat,
            score_vec,
            player_vec,
            suit_counts,
            void_flat,
            hand_strength,
        ],
        axis=0,
    ).astype(np.float32)

    return {
        "observation": observation_vec,
        "action_mask": np.array(action_mask, dtype=np.float32),
        "history": encode_history(history),
    }


def decode_action(action: int) -> Tuple[bool, int]:
    """
    Decode an action index.

    Returns:
        (is_trump_action, payload) where payload is suit index (0-3) if trump,
        otherwise card index (0-{rules.NUM_CARDS - 1}).
    """
    if rules.is_trump_action(action):
        return True, action - rules.ACTION_TRUMP_OFFSET
    if action < 0 or action >= rules.NUM_CARDS:
        raise ValueError(f"Invalid action {action}")
    return False, action


def observation_length() -> int:
    """Length of the flat observation vector."""
    # hand one-hot (32) + trump (4) + lead (4) + current trick 4×32 (128)
    # + score (2) + player id (4) + suit proportions (4) + void matrix 4×4 (16)
    # + hand strength (1)
    return rules.NUM_CARDS + 4 + 4 + (4 * rules.NUM_CARDS) + 2 + 4 + 4 + 16 + 1
