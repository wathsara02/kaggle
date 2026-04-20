from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from omi_env import encoding, rules


def encode_central_state(state: dict) -> torch.Tensor:
    """
    Encode centralized state for the critic.

    Args:
        state: dict from env.state()
    """
    hands = state["hands"]
    trump = state.get("trump_suit")
    lead = state.get("lead_suit")
    current_trick = state.get("current_trick", [])
    history = state.get("history", [])
    tricks_won = state.get("tricks_won", (0, 0))

    # Reuse encoding helpers instead of duplicating one-hot logic here
    hand_vecs = []
    for h in hands:
        vec = [0.0] * rules.NUM_CARDS
        for c in h:
            vec[c] = 1.0
        hand_vecs.append(vec)

    trump_vec = (
        encoding.one_hot(rules.SUITS.index(trump), 4).tolist()
        if trump is not None
        else [0.0] * 4
    )
    lead_vec = (
        encoding.one_hot(rules.SUITS.index(lead), 4).tolist()
        if lead is not None
        else [0.0] * 4
    )

    trick_vecs = []
    for _, card_idx in current_trick:
        trick_vecs.append(encoding.card_one_hot(card_idx).tolist())
    while len(trick_vecs) < 4:
        trick_vecs.append([0.0] * rules.NUM_CARDS)
    trick_flat = [x for vec in trick_vecs for x in vec]

    score_vec = [
        tricks_won[0] / float(rules.TRICKS_PER_HAND),
        tricks_won[1] / float(rules.TRICKS_PER_HAND),
    ]

    hist_arr = encoding.encode_history(history).reshape(-1)
    features = (
        hand_vecs[0] + hand_vecs[1] + hand_vecs[2] + hand_vecs[3]
        + trump_vec + lead_vec + trick_flat + score_vec
    )
    return torch.tensor(features + hist_arr.tolist(), dtype=torch.float32)


class CentralCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size

        # Split point: state features vs flattened history in the input vector.
        # State features = 4 hands + trump + lead + trick + score
        #   = 8 * NUM_CARDS + 4 + 4 + 2 = 8*32 + 10 = 266 dims
        # History = HISTORY_LEN * HISTORY_FEAT_DIM = 32 * 44 = 1408 dims
        self._state_dim = 8 * rules.NUM_CARDS + 10
        self._hist_len = encoding.HISTORY_LEN
        self._hist_feat = encoding.HISTORY_FEAT_DIM

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(self._state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )

        # History encoder: project each timestep's features into hidden space
        self.hist_proj = nn.Linear(self._hist_feat, hidden_size)

        # Single-head attention: state embedding queries into history keys/values
        self.attn_query = nn.Linear(hidden_size, hidden_size)
        self.attn_key = nn.Linear(hidden_size, hidden_size)

        # Value head: fuses attended history with state embedding
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Split flat input back into state features and history sequence
        state = x[:, :self._state_dim]
        hist = x[:, self._state_dim:].reshape(B, self._hist_len, self._hist_feat)

        state_emb = self.state_encoder(state)           # (B, H)
        hist_proj = torch.tanh(self.hist_proj(hist))    # (B, L, H)

        # Scaled dot-product attention: state queries, history keys/values
        q = self.attn_query(state_emb).unsqueeze(1)     # (B, 1, H)
        k = self.attn_key(hist_proj)                    # (B, L, H)
        scale = self.hidden_size ** -0.5
        weights = torch.softmax(
            (q @ k.transpose(1, 2)) * scale, dim=-1
        )                                               # (B, 1, L)
        attended = (weights @ hist_proj).squeeze(1)     # (B, H)

        combined = torch.cat([state_emb, attended], dim=-1)  # (B, 2H)
        return self.value_head(combined).squeeze(-1)
