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

# One full hand has 32 played cards.
HISTORY_LEN = 32
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
    Compute a 4-player by 4-suit void matrix from play history.
    """
    void_matrix = np.zeros((4, 4), dtype=np.float32)
    cards_per_suit = rules.NUM_CARDS // len(rules.SUITS)  # 8

    # Track played suits per player.
    suit_cards_played: List[set] = [set() for _ in range(len(rules.SUITS))]
    player_suit_played = [[False] * len(rules.SUITS) for _ in range(4)]

    for player, card_idx, lead_suit, _ in history:
        card_suit = rules.index_to_card(card_idx).suit
        s = rules.SUITS.index(card_suit)
        suit_cards_played[s].add(card_idx)
        player_suit_played[player][s] = True

        # Failing to follow the led suit confirms a void.
        if lead_suit is not None and card_suit != lead_suit:
            lead_s = rules.SUITS.index(lead_suit)
            void_matrix[player][lead_s] = 1.0

    # Estimate likely voids from cards already played by others.
    for p in range(4):
        for s in range(len(rules.SUITS)):
            if void_matrix[p][s] == 1.0:
                continue
            if not player_suit_played[p][s] and suit_cards_played[s]:
                void_matrix[p][s] = len(suit_cards_played[s]) / cards_per_suit

    return void_matrix


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

    # Suit mix helps with trump selection.
    suit_counts = np.zeros(4, dtype=np.float32)
    for c in hand:
        suit_counts[rules.SUITS.index(rules.index_to_card(c).suit)] += 1.0
    if hand:
        suit_counts /= float(len(hand))

    # Scalar hand strength in [0, 1].
    _max_card_value = len(rules.RANKS) + 1
    hand_strength = np.array(
        [sum(rules.index_to_card(c).value for c in hand) / (_max_card_value * rules.HAND_SIZE)],
        dtype=np.float32,
    )

    # Known and likely void suits.
    void_flat = compute_void_matrix(history).reshape(-1)

    trump_vec = (
        one_hot(rules.SUITS.index(trump_suit), 4) if trump_suit is not None else np.zeros(4, dtype=np.float32)
    )
    lead_vec = (
        one_hot(rules.SUITS.index(lead_suit), 4) if lead_suit is not None else np.zeros(4, dtype=np.float32)
    )

    trick_vecs: List[np.ndarray] = []
    # Pad current trick to four cards.
    for _, card_idx in current_trick:
        trick_vecs.append(card_one_hot(card_idx))
    while len(trick_vecs) < 4:
        trick_vecs.append(np.zeros(rules.NUM_CARDS, dtype=np.float32))
    trick_flat = np.concatenate(trick_vecs, axis=0)

    # Normalize trick score.
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
    # Keep this in sync with encode_observation().
    return rules.NUM_CARDS + 4 + 4 + (4 * rules.NUM_CARDS) + 2 + 4 + 4 + 16 + 1
