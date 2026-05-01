"""
Inference helper for deploying the trained Intelligent Omi agent.

Usage:
    agent = load_agent("artifacts/policy_agent.pt", "artifacts/config.json")
    action, hidden = agent.act(obs_vector, legal_mask, history_array, hidden_state=None, deterministic=True)
"""
from __future__ import annotations

import json
from typing import Optional, Tuple

import torch

from models.policy import PolicyNet
from omi_env import encoding, rules
from utils import clean_state_dict, get_device


class InferenceAgent:
    def __init__(self, policy: PolicyNet, device: torch.device):
        self.policy = policy.to(device)
        self.device = device

    def act(
        self,
        obs: torch.Tensor,
        legal_mask: torch.Tensor,
        history: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ):
        """
        Choose an action.

        temperature controls difficulty when deterministic=False:
            < 1.0  →  harder  (agent picks best move more consistently)
            1.0    →  default
            > 1.0  →  easier  (more random)
        deterministic=True overrides temperature and always picks argmax.
        """
        self.policy.eval()
        obs = obs.to(self.device).unsqueeze(0)
        legal_mask = legal_mask.to(self.device).unsqueeze(0)
        history = history.to(self.device).unsqueeze(0)
        # Feed-forward policies do not use hidden state.
        if hidden_state is None and hasattr(self.policy, "init_hidden"):
            hidden_state = self.policy.init_hidden(1, self.device)
        with torch.no_grad():
            logits, hidden_out = self.policy(obs, history, hidden_state, action_mask=legal_mask)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                scaled_logits = logits / max(temperature, 1e-6)
                probs = torch.softmax(scaled_logits, dim=-1)
                action = torch.distributions.Categorical(probs).sample()
        return int(action.item()), hidden_out


def load_agent(weights_path: str, config_path: str, device: Optional[torch.device] = None) -> InferenceAgent:
    device = device or get_device()
    with open(config_path, "r") as f:
        cfg = json.load(f)
    policy = PolicyNet(
        obs_dim=cfg["obs_dim"],
        history_dim=cfg["history_dim"],
        action_dim=cfg["action_dim"],
        hidden_size=cfg["model"].get("recurrent_hidden_size", 128),
        recurrent_type=cfg["model"]["recurrent_type"],
        hist_feat_dim=encoding.HISTORY_FEAT_DIM,
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    policy.load_state_dict(clean_state_dict(state_dict))
    return InferenceAgent(policy, device)
