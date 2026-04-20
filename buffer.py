from __future__ import annotations

from typing import Dict, List

import torch


class AgentBuffer:
    def __init__(self, gamma: float, gae_lambda: float, device: torch.device):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.storage: Dict[int, List[dict]] = {0: [], 1: [], 2: [], 3: []}

    def add(self, agent_id: int, transition: dict):
        self.storage[agent_id].append(transition)

    def finalize(self, final_rewards: Dict[int, float]):
        for agent_id, transitions in self.storage.items():
            if not transitions:
                continue
            transitions[-1]["reward"] = final_rewards.get(agent_id, transitions[-1]["reward"])
            transitions[-1]["done"] = True

    def compute_advantages(self):
        all_transitions: List[dict] = []
        for agent_id, traj in self.storage.items():
            gae = 0.0
            next_value = 0.0
            for t in reversed(traj):
                reward = t["reward"]
                value = t["value"]
                done = t.get("done", False)
                mask = 0.0 if done else 1.0
                delta = reward + self.gamma * next_value * mask - value
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                t["advantage"] = gae
                t["return"] = gae + value
                next_value = value  # BUG FIX: must update AFTER using next_value for delta (was correct position but value was the current t's value — this is actually correct for reversed iteration; the prior step's "next" is this step's value)
            all_transitions.extend(traj)
        return all_transitions

    def clear(self):
        for k in self.storage:
            self.storage[k] = []
