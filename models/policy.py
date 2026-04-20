"""
Policy network.

Supports two modes controlled by ``recurrent_type`` in the config:

* ``"none"`` (default) -- feed-forward policy.  The explicitly encoded
  history vector is injected as a static input feature.  Simple and fast.

* ``"lstm"`` -- the history sequence is processed step-by-step through an
  LSTM.  The hidden state ``(h, c)`` is carried between turns within the
  same hand/episode and reset at the start of every new hand.  This allows
  the agent to reason *sequentially* about which cards have been played and
  infer what cards opponents may still hold.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        history_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        recurrent_type: str = "none",
        hist_feat_dim: int = 0,   # per-step feature dim for LSTM (ignored in FF mode)
    ):
        super().__init__()
        self.recurrent_type = recurrent_type.lower()
        self.hidden_size = hidden_size

        # Observation encoder (shared by both modes)
        self.obs_encoder = nn.Linear(obs_dim, hidden_size)

        if self.recurrent_type == "lstm":
            # history_dim is the flat size (HISTORY_LEN * HISTORY_FEAT_DIM).
            # The LSTM processes one step at a time, so input_size = HISTORY_FEAT_DIM.
            # We derive it from the encoded history shape in forward(); store for reference.
            self._history_feat_dim = hist_feat_dim
            self.lstm = nn.LSTM(
                input_size=hist_feat_dim,      # per-step feature dim
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            core_in = hidden_size * 2   # obs_emb + lstm_out
        else:
            # Feed-forward: 2-layer MLP to capture structure in the flattened history.
            # A single linear layer crushes 1408 dims into 256 with no intermediate
            # reasoning; the extra layer + norms give the encoder representational depth.
            self.hist_encoder = nn.Sequential(
                nn.Linear(history_dim, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            core_in = hidden_size * 2

        self.core = nn.Sequential(
            nn.Linear(core_in, hidden_size),
            nn.LayerNorm(hidden_size),   # OPTIMISATION: LayerNorm stabilises training with larger hidden size
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)

    # ------------------------------------------------------------------
    # Hidden-state management (only relevant for LSTM mode)
    # ------------------------------------------------------------------

    def init_hidden(self, batch_size: int, device: torch.device):
        """Return a fresh (h_0, c_0) tuple for LSTM mode, or None otherwise."""
        if self.recurrent_type == "lstm":
            h = torch.zeros(1, batch_size, self.hidden_size, device=device)
            c = torch.zeros(1, batch_size, self.hidden_size, device=device)
            return (h, c)
        return None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,
        history: torch.Tensor,
        hidden_state=None,
        action_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            obs:          (B, obs_dim)
            history:      (B, H, F) sequence  OR  (B, H*F) flat tensor.
            hidden_state: (h, c) tuple for LSTM, or None for feed-forward.
            action_mask:  (B, action_dim) — 1 = legal, 0 = illegal.

        Returns:
            logits:       (B, action_dim)
            hidden_out:   updated (h, c) for LSTM; None for feed-forward.
        """
        B = obs.shape[0]
        obs_emb = torch.tanh(self.obs_encoder(obs))  # (B, H)

        if self.recurrent_type == "lstm":
            # Ensure (B, seq_len, feat_dim) shape
            if history.dim() == 2:
                # Infer sequence shape: flat → (B, HISTORY_LEN, FEAT_DIM)
                feat_dim = self.lstm.input_size
                history = history.view(B, -1, feat_dim)

            if hidden_state is None:
                hidden_state = self.init_hidden(B, obs.device)

            # Detach to prevent BPTT beyond a single episode step during update
            h, c = hidden_state
            h = h.detach()
            c = c.detach()

            _, (h_new, c_new) = self.lstm(history, (h, c))
            lstm_out = h_new.squeeze(0)          # (B, hidden_size)
            hidden_out = (h_new, c_new)
            x = torch.cat([obs_emb, lstm_out], dim=-1)
        else:
            # Feed-forward mode — flatten history
            if history.dim() == 3:
                history = history.reshape(B, -1)
            hist_emb = self.hist_encoder(history)  # MLP has its own norms/activations
            x = torch.cat([obs_emb, hist_emb], dim=-1)
            hidden_out = None

        x = self.core(x)
        logits = self.actor(x)
        if action_mask is not None:
            logits = mask_logits(logits, action_mask)
        return logits, hidden_out


def mask_logits(logits: torch.Tensor, mask: torch.Tensor, mask_value: float = -1e9) -> torch.Tensor:
    """Apply an action mask to logits.

    mask is expected to be 1 for legal actions, 0 for illegal actions.
    """
    expanded_mask = (mask > 0).float()
    return logits + (1.0 - expanded_mask) * mask_value
