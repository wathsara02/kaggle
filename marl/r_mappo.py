from __future__ import annotations

import copy
import random as _random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baselines.rule_based_agent import RuleBasedAgent
from baselines.random_agent import RandomLegalAgent
from buffer import AgentBuffer
from models.critic import CentralCritic, encode_central_state
from models.policy import PolicyNet
from utils import masked_sample


class MAPPOTrainer:
    def __init__(
        self,
        policy: PolicyNet,
        critic: CentralCritic,
        config: dict,
        device: torch.device,
    ):
        self.policy = policy.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.cfg = config
        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=config["lr"])
        self.optimizer_v = optim.Adam(self.critic.parameters(), lr=config["lr"])
        self.clip_range = config["clip_range"]
        self.entropy_coef = config["entropy_coef"]
        self.entropy_coef_end = config.get("entropy_coef_end", config["entropy_coef"])
        self.initial_entropy_coef = config["entropy_coef"]
        self.value_coef = config["value_coef"]
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.initial_lr = config["lr"]
        self.lr_min = config.get("lr_min", 1e-5)
        self.lr_annealing = config.get("lr_annealing", True)
        # OPTIMISATION: AMP grad scaler for mixed-precision training on T4 GPU
        self._use_amp = (device.type == "cuda")
        self.scaler_pi = torch.amp.GradScaler("cuda", enabled=self._use_amp)
        self.scaler_v = torch.amp.GradScaler("cuda", enabled=self._use_amp)
        # Curriculum training state
        # opponent_mode: "self_play" | "random" | "frozen"
        self.opponent_mode: str = "self_play"
        self.frozen_policy: Optional[PolicyNet] = None
        self._random_agent = RandomLegalAgent()

    def set_frozen_policy(self) -> None:
        """Snapshot current policy weights as the frozen opponent for Phase 2."""
        self.frozen_policy = copy.deepcopy(self.policy).to(self.device)
        self.frozen_policy.eval()
        for p in self.frozen_policy.parameters():
            p.requires_grad_(False)
        print("[CURRICULUM] Frozen policy updated from current policy weights.")

    def anneal_lr(self, fraction: float) -> None:
        """Linearly decay LR and entropy coefficient from initial values → minimums as fraction goes 0 → 1."""
        if not self.lr_annealing:
            return
        new_lr = self.initial_lr + fraction * (self.lr_min - self.initial_lr)
        for pg in self.optimizer_pi.param_groups:
            pg["lr"] = new_lr
        for pg in self.optimizer_v.param_groups:
            pg["lr"] = new_lr
        # Decay entropy coefficient: high early (exploration) → low late (exploitation)
        self.entropy_coef = self.initial_entropy_coef + fraction * (
            self.entropy_coef_end - self.initial_entropy_coef
        )

    def collect_episode(self, env) -> Tuple[List[dict], dict]:
        is_vector = hasattr(env, "num_envs")
        num_envs = getattr(env, "num_envs", 1)
        is_recurrent = self.policy.recurrent_type == "lstm"
        rule_agent = RuleBasedAgent()

        # ── Opponent assignment per env ───────────────────────────────────────
        # "random"  → players 1 & 3 always use RandomLegalAgent
        # "frozen"  → players 1 & 3 always use frozen policy snapshot
        # "self_play" → existing rule_mix_prob behaviour (diverse mixing)
        env_opp_agents: List[set] = []
        if self.opponent_mode in ("random", "frozen"):
            env_opp_agents = [{1, 3} for _ in range(num_envs)]
        else:
            rule_mix_prob = self.cfg.get("rule_mix_prob", 0.0)
            for _ in range(num_envs):
                rule_set: set = set()
                if rule_mix_prob > 0 and _random.random() < rule_mix_prob:
                    mode = _random.choice(["opponents", "teammate", "one"])
                    if mode == "opponents":
                        rule_set = {1, 3}
                    elif mode == "teammate":
                        rule_set = {2}
                    else:
                        rule_set = {_random.randint(0, 3)}
                env_opp_agents.append(rule_set)

        buffers = [AgentBuffer(self.gamma, self.gae_lambda, self.device) for _ in range(num_envs)]

        # One hidden state per agent per env (meaningful only in LSTM mode)
        hidden_states = [
            {i: self.policy.init_hidden(1, self.device) for i in range(4)}
            for _ in range(num_envs)
        ]

        if is_vector:
            env.reset()
        else:
            env.reset()
        last_cumulative_rewards = [
            {f"player_{j}": 0.0 for j in range(4)}
            for _ in range(num_envs)
        ]
        active_envs = list(range(num_envs))
        episode_infos = []

        while active_envs:
            # ── Observe ───────────────────────────────────────────────────────
            if is_vector:
                agent_names = env.agent_selection(active_envs)
                obs_list = env.observe(agent_names, active_envs)
            else:
                agent_names = [env.agent_selection]
                obs_list = [env.observe(agent_names[0])]

            agent_ids = [int(name.split("_")[1]) for name in agent_names]

            # Split this batch into policy slots and opponent slots
            pol_slots  = [i for i, (ei, ai) in enumerate(zip(active_envs, agent_ids))
                          if ai not in env_opp_agents[ei]]
            opp_slots  = [i for i, (ei, ai) in enumerate(zip(active_envs, agent_ids))
                          if ai in env_opp_agents[ei]]
            pol_slots_set = set(pol_slots)

            # Final action per slot (aligned with active_envs)
            final_actions: List = [None] * len(active_envs)

            # ── Opponent actions ──────────────────────────────────────────────
            if self.opponent_mode == "random":
                for i in opp_slots:
                    final_actions[i] = self._random_agent.act(obs_list[i])
            elif self.opponent_mode == "frozen" and self.frozen_policy is not None and opp_slots:
                obs_f    = torch.from_numpy(np.array([obs_list[i]["observation"] for i in opp_slots])).float().to(self.device, non_blocking=True)
                hist_f   = torch.from_numpy(np.array([obs_list[i]["history"]     for i in opp_slots])).float().to(self.device, non_blocking=True)
                mask_f   = torch.from_numpy(np.array([obs_list[i]["action_mask"] for i in opp_slots])).float().to(self.device, non_blocking=True)
                with torch.no_grad():
                    logits_f, _ = self.frozen_policy(obs_f, hist_f, None, action_mask=mask_f)
                    actions_f, _ = masked_sample(logits_f, mask_f, deterministic=False)
                actions_f_np = actions_f.cpu().numpy()
                for j, i in enumerate(opp_slots):
                    final_actions[i] = int(actions_f_np[j].item())
            else:
                # self_play or frozen not yet set — fall back to rule-based
                for i in opp_slots:
                    final_actions[i] = rule_agent.act(obs_list[i])

            # ── Policy actions (batched forward pass) ─────────────────────────
            logprobs_map: Dict[int, float] = {}
            values_map:   Dict[int, float] = {}
            cs_map:       Dict[int, np.ndarray] = {}

            if pol_slots:
                obs_tensor  = torch.from_numpy(np.array([obs_list[i]["observation"] for i in pol_slots])).float().to(self.device, non_blocking=True)
                hist_tensor = torch.from_numpy(np.array([obs_list[i]["history"]     for i in pol_slots])).float().to(self.device, non_blocking=True)
                mask_tensor = torch.from_numpy(np.array([obs_list[i]["action_mask"] for i in pol_slots])).float().to(self.device, non_blocking=True)

                if is_recurrent:
                    h_list, c_list = [], []
                    for i in pol_slots:
                        h, c = hidden_states[active_envs[i]][agent_ids[i]]
                        h_list.append(h); c_list.append(c)
                    batch_hidden = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1))
                else:
                    batch_hidden = None

                with torch.no_grad():
                    logits, new_hidden = self.policy(
                        obs_tensor, hist_tensor, batch_hidden, action_mask=mask_tensor
                    )

                    if is_recurrent and new_hidden is not None:
                        new_h, new_c = new_hidden
                        for j, i in enumerate(pol_slots):
                            hidden_states[active_envs[i]][agent_ids[i]] = (
                                new_h[:, j:j+1, :].clone(),
                                new_c[:, j:j+1, :].clone(),
                            )

                    # BUG FIX (logprob consistency): use Categorical.log_prob
                    actions_pol, probs_pol = masked_sample(logits, mask_tensor, deterministic=False)
                    dist = torch.distributions.Categorical(probs_pol)
                    logprobs_pol = dist.log_prob(actions_pol)

                    env_state_indices = [active_envs[i] for i in pol_slots]
                    if is_vector:
                        env_states = env.get_state(env_state_indices)
                    else:
                        env_states = [env.state()]

                    central_states = torch.stack(
                        [encode_central_state(s) for s in env_states]
                    ).to(self.device, non_blocking=True)
                    values = self.critic(central_states)
                    if values.dim() > 1:
                        values = values.squeeze(-1)

                actions_pol_np = actions_pol.cpu().numpy()
                for j, i in enumerate(pol_slots):
                    final_actions[i] = int(actions_pol_np[j].item())
                    logprobs_map[i]   = logprobs_pol[j].item()
                    values_map[i]     = values[j].item()
                    cs_map[i]         = central_states[j].cpu().numpy()

            # ── Add policy transitions to buffer (rule-based → no buffer) ─────
            for i in pol_slots:
                env_idx = active_envs[i]
                a_id = agent_ids[i]
                buffers[env_idx].add(a_id, {
                    "obs":         obs_list[i]["observation"],
                    "history":     obs_list[i]["history"],
                    "action_mask": obs_list[i]["action_mask"],
                    "action":      final_actions[i],
                    "logprob":     logprobs_map[i],
                    "value":       values_map[i],
                    "reward":      0.0,
                    "done":        False,
                    "agent_id":    a_id,
                    "central_state": cs_map[i],
                })

            # ── Step all envs ─────────────────────────────────────────────────
            if is_vector:
                env.step(final_actions, active_envs)
                cumulative_rewards_list = env.get_cumulative_rewards(active_envs)
                terminations_list = env.get_terminations(active_envs)
            else:
                env.step(int(final_actions[0]))
                cumulative_rewards_list = [env._cumulative_rewards]
                terminations_list = [env.terminations]

            # ── Assign rewards and handle episode completion ───────────────────
            next_active_envs = []
            for i, env_idx in enumerate(active_envs):
                current_rewards = cumulative_rewards_list[i]
                reward_deltas = {
                    j: current_rewards.get(f"player_{j}", 0.0)
                    - last_cumulative_rewards[env_idx].get(f"player_{j}", 0.0)
                    for j in range(4)
                }
                last_cumulative_rewards[env_idx] = dict(current_rewards)

                # Team rewards can be granted to non-acting agents. Credit each
                # trained agent's latest transition with the reward delta since
                # the previous environment step.
                for rewarded_agent_id, reward_delta in reward_deltas.items():
                    if buffers[env_idx].storage[rewarded_agent_id]:
                        buffers[env_idx].storage[rewarded_agent_id][-1]["reward"] += reward_delta

                done = all(terminations_list[i].values())
                if done:
                    buffers[env_idx].finalize({})
                    if is_vector:
                        episode_infos.append(
                            next(iter(env.get_infos([env_idx])[0].values()), {})
                        )
                    else:
                        episode_infos.append(next(iter(env.infos.values()), {}))
                else:
                    next_active_envs.append(env_idx)

            active_envs = next_active_envs

        # ── Aggregate transitions from all buffers ────────────────────────────
        all_transitions = []
        for b in buffers:
            all_transitions.extend(b.compute_advantages())
            b.clear()

        return all_transitions, episode_infos

    def update(self, transitions: List[dict]):
        if not transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        def batch_tensor(key: str) -> torch.Tensor:
            tensor = torch.from_numpy(np.array([t[key] for t in transitions])).float()
            if self.device.type == "cuda":
                tensor = tensor.pin_memory()
            return tensor.to(self.device, non_blocking=(self.device.type == "cuda"))

        obs        = batch_tensor("obs")
        hist       = batch_tensor("history")
        masks      = batch_tensor("action_mask")
        states     = batch_tensor("central_state")
        actions    = torch.tensor([t["action"]   for t in transitions], dtype=torch.long,  device=self.device)
        old_logprobs = torch.tensor([t["logprob"] for t in transitions], dtype=torch.float32, device=self.device)
        returns    = torch.tensor([t["return"]   for t in transitions], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([t["advantage"] for t in transitions], dtype=torch.float32, device=self.device)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(transitions)
        batch_size = self.cfg["batch_size"]
        ppo_epochs = self.cfg["ppo_epochs"]
        indices = np.arange(dataset_size)
        for _ in range(ppo_epochs):
            # LSTM relies on temporal ordering within sequences; shuffling breaks
            # the hidden-state carry-over that collection builds up, creating a
            # train/inference mismatch. FF mode is unaffected by order so always shuffle.
            if self.policy.recurrent_type != "lstm":
                np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                obs_mb = obs[mb_idx]
                hist_mb = hist[mb_idx]
                mask_mb = masks[mb_idx]
                actions_mb = actions[mb_idx]
                old_logprobs_mb = old_logprobs[mb_idx]
                returns_mb = returns[mb_idx]
                adv_mb = advantages[mb_idx]
                states_mb = states[mb_idx]

                # Use torch.amp.autocast (torch.cuda.amp.autocast is deprecated in PyTorch 2+)
                with torch.amp.autocast("cuda", enabled=self._use_amp):
                    logits, _ = self.policy(obs_mb, hist_mb, None, action_mask=mask_mb)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    logprobs = dist.log_prob(actions_mb)
                    entropy = dist.entropy().mean()

                    ratios = torch.exp(logprobs - old_logprobs_mb)
                    surr1 = ratios * adv_mb
                    surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()
                    policy_loss_total = policy_loss - self.entropy_coef * entropy

                # BUG FIX: update policy and critic separately to avoid cross-gradient contamination
                self.optimizer_pi.zero_grad()
                self.scaler_pi.scale(policy_loss_total).backward()
                self.scaler_pi.unscale_(self.optimizer_pi)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.scaler_pi.step(self.optimizer_pi)
                self.scaler_pi.update()

                with torch.amp.autocast("cuda", enabled=self._use_amp):
                    values = self.critic(states_mb)
                    # BUG FIX: ensure values shape matches returns_mb (both 1D)
                    if values.dim() > 1:
                        values = values.squeeze(-1)
                    value_loss = self.value_coef * (returns_mb - values).pow(2).mean()

                self.optimizer_v.zero_grad()
                self.scaler_v.scale(value_loss).backward()
                self.scaler_v.unscale_(self.optimizer_v)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.scaler_v.step(self.optimizer_v)
                self.scaler_v.update()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }
