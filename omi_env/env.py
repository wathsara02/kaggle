"""
PettingZoo AEC environment for the Omi trick-taking game.

Key features:
- Must-follow-suit enforcement with action masks.
- Trump declaration phase (player 0 chooses a suit by default).
- Observation includes private hand, public info, and history for recurrent agents.
- CTDE-friendly info dict exposes winner and final score at terminal.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from . import encoding, rules

# Default reward values (used if not provided in config)
DEFAULT_REWARDS = {
    "illegal_action_penalty": -0.1,   # BUG FIX: was +0.1 (positive penalty is a reward!)
    "overplay_penalty": -0.05,         # BUG FIX: was +0.05
    "trick_reward": 0.1,
    "cap_bonus": 0.5,
    "declarer_bonus": 0.1,
    "cap_penalty": -0.5,              # BUG FIX: was +0.5 (positive penalty is a reward!)
    "trump_quality_bonus": 0.2,       # Immediate reward for picking the suit you hold most of
    "declarer_team_win_bonus": 0.0,
    "declarer_team_loss_penalty": 0.0,
    "late_trick_reward": 0.0,
    "trump_cut_reward": 0.0,
    "wasted_trump_penalty": 0.0,
    "partner_save_reward": 0.0,
}


class OmiEnv(AECEnv):
    metadata = {"render.modes": ["human"], "name": "omi_v0"}

    def __init__(self, seed: int = 0, reward_shaping: bool = False, rewards_dict: Optional[dict] = None):
        super().__init__()
        self._seed = seed
        self.reward_shaping = reward_shaping
        self.rewards_cfg = DEFAULT_REWARDS.copy()
        if rewards_dict:
            self.rewards_cfg.update(rewards_dict)

        self.rng = random.Random(seed)
        self.agents = [f"player_{i}" for i in range(4)]
        self.possible_agents = self.agents[:]
        # Trump declarer rotates each hand.
        # Per your rule: the player to the right-hand side of the previous declarer
        # declares trumps next hand (we interpret this as clockwise rotation: +1).
        self.start_player = 0
        self.reinit()

        obs_len = encoding.observation_length()
        self.observation_spaces: Dict[str, spaces.Space] = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32),
                    "action_mask": spaces.Box(low=0.0, high=1.0, shape=(rules.ACTION_DIM,), dtype=np.float32),
                    "history": spaces.Box(low=0.0, high=1.0, shape=(encoding.HISTORY_LEN, encoding.HISTORY_FEAT_DIM), dtype=np.float32),
                }
            )
            for agent in self.possible_agents
        }
        self.action_spaces: Dict[str, spaces.Space] = {
            agent: spaces.Discrete(rules.ACTION_DIM) for agent in self.possible_agents
        }

    def seed(self, seed: Optional[int] = None):
        # BUG FIX (Bug 4): Previously this always called random.Random(self._seed),
        # resetting rng to the same state on every reset() call.
        # That made every episode replay the exact same hand — zero diversity.
        # Now rng is only re-created when a new seed is explicitly provided.
        # The rng advances naturally across episodes, producing diverse hands.
        if seed is not None:
            self._seed = seed
            self.rng = random.Random(self._seed)

    def reinit(self):
        self.trump_suit: Optional[str] = None
        self.lead_suit: Optional[str] = None
        self.hands: List[List[int]] = [[], [], [], []]
        self._remaining_deck: List[int] = []
        self.current_trick: List[Tuple[int, int]] = []
        self.tricks_won: Tuple[int, int] = (0, 0)
        self.history: List[Tuple[int, int, Optional[str], Optional[str]]] = []
        # Stages: "trump" (after initial 4-card deal) -> "play" (after remaining deal)
        self.stage: str = "trump"
        self._terminated = False
        self._illegal_actions = 0
        self.episode_length = 0
        self._trump_quality: float = 0.0
        self._shaping_events = {
            "partner_save": 0,
            "trump_cut": 0,
            "wasted_trump": 0,
            "late_trick": 0,
            "declarer_team_win": 0,
            "declarer_team_loss": 0,
        }
        self._init_selector(start=self.start_player)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.has_reset = False

    def _init_selector(self, start: int, *, only_one: bool = False):
        if only_one:
            order = [f"player_{start % 4}"]
        else:
            order = [f"player_{(start + i) % 4}" for i in range(4)]
        self.agent_selector = agent_selector(order)
        self.agent_selection = self.agent_selector.reset()

    def reset(self, seed: Optional[int] = None, options=None):
        # BUG FIX (Bug 4 cont.): Only pass a seed when explicitly given.
        # The old code passed self._seed unconditionally, which re-created rng
        # from scratch every episode, replaying the same hand forever.
        # Now when seed=None, we call self.seed(None) which does NOT reset rng,
        # so it advances naturally and produces a fresh hand each episode.
        self.seed(seed)
        self.reinit()
        deck = rules.shuffle_deck(self.rng)
        self.hands, self._remaining_deck = rules.deal_first_four(deck)
        # Trump declaration: only the current declarer takes an action
        self._init_selector(start=self.start_player, only_one=True)
        self.has_reset = True
        # Rotate clockwise for the next hand (right-hand side of previous declarer)
        self.start_player = (self.start_player + 1) % 4
        return self.observe(self.agent_selection)

    # PettingZoo requirement
    def observe(self, agent: str):
        agent_id = self.agents.index(agent)
        mask = self._action_mask(agent_id)
        return encoding.encode_observation(
            agent_id,
            self.hands[agent_id],
            self.trump_suit,
            self.lead_suit,
            self.current_trick,
            self.tricks_won,
            mask,
            self.history,
        )

    def _action_mask(self, agent_id: int) -> List[int]:
        if self.stage == "trump":
            return rules.legal_trump_mask()

        card_mask = rules.legal_card_mask(self.hands[agent_id], self.lead_suit)
        return card_mask + [0] * 4

    def step(self, action: int):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        agent = self.agent_selection
        agent_id = self.agents.index(agent)
        self._cumulative_rewards[agent] = 0.0

        mask = self._action_mask(agent_id)
        
        # Reward shaping: Illegal action penalty
        if mask[action] == 0:
            self._illegal_actions += 1
            if self.reward_shaping:
                self.rewards[agent] += self.rewards_cfg["illegal_action_penalty"]
            # For strictness, ignore the action by selecting a legal fallback
            legal_indices = [i for i, v in enumerate(mask) if v == 1]
            action = legal_indices[0]

        # Reward shaping: Over-play penalty
        if self.reward_shaping and self.stage == "play" and len(self.current_trick) > 0:
            current_winner = rules.resolve_trick(self.current_trick, self.lead_suit, self.trump_suit)
            if rules.team_for_player(current_winner) == rules.team_for_player(agent_id):
                # Teammate is winning. Check if we had a non-winning alternative.
                mask = self._action_mask(agent_id)
                legal_actions = [i for i, v in enumerate(mask) if v == 1]
                has_safe_move = False
                for la in legal_actions:
                    temp_trick = self.current_trick + [(agent_id, la)]
                    temp_winner = rules.resolve_trick(temp_trick, self.lead_suit, self.trump_suit)
                    if rules.team_for_player(temp_winner) == rules.team_for_player(current_winner):
                        has_safe_move = True
                        break
                
                if has_safe_move:
                    actual_temp_trick = self.current_trick + [(agent_id, action)]
                    actual_temp_winner = rules.resolve_trick(actual_temp_trick, self.lead_suit, self.trump_suit)
                    # Penalise two distinct mistakes when a safe move existed:
                    #   1. Agent steals the trick from partner (actual_temp_winner == agent_id)
                    #   2. Agent gives the trick to the opponent team (worse — a trick that
                    #      was winnable is now lost)
                    if (actual_temp_winner == agent_id or
                            rules.team_for_player(actual_temp_winner) != rules.team_for_player(agent_id)):
                        self.rewards[agent] += self.rewards_cfg["overplay_penalty"]

        advance_turn = True

        if self.stage == "trump":
            # Only the current declarer acts in this stage.
            if not rules.is_trump_action(action):
                raise ValueError("During trump declaration only trump actions are legal")
            suit_idx = action - rules.ACTION_TRUMP_OFFSET
            self.trump_suit = rules.SUITS[suit_idx]
            # Deal remaining 16 cards (4 to each) after trump is declared
            self.hands = rules.deal_remaining_four(self.hands, self._remaining_deck)
            self._remaining_deck = []

            # Store trump quality for deferred reward at episode end.
            # Deferring avoids the credit-assignment mismatch of rewarding at
            # step 0 for an outcome that resolves 32 steps later. The bonus is
            # applied at terminal, scaled by the team's actual win fraction.
            if self.reward_shaping:
                full_hand = self.hands[agent_id]
                trump_count = sum(
                    1 for c in full_hand
                    if rules.index_to_card(c).suit == self.trump_suit
                )
                self._trump_quality = trump_count / float(rules.HAND_SIZE)
            self.stage = "play"
            # Switch turn order back to all 4 players; declarer leads first trick
            self._init_selector(start=agent_id)
            advance_turn = False
        else:
            is_trump, card_idx = encoding.decode_action(action)
            if is_trump:
                raise ValueError("Trump action after trump selection is illegal")

            pre_trick = list(self.current_trick)
            pre_lead_suit = self.lead_suit
            card_obj = rules.index_to_card(card_idx)

            if self.reward_shaping and pre_trick:
                current_winner = rules.resolve_trick(pre_trick, pre_lead_suit, self.trump_suit)
                actual_temp_trick = pre_trick + [(agent_id, card_idx)]
                actual_temp_winner = rules.resolve_trick(actual_temp_trick, pre_lead_suit, self.trump_suit)
                partner_winning = (
                    current_winner != agent_id
                    and rules.team_for_player(current_winner) == rules.team_for_player(agent_id)
                )

                if partner_winning:
                    legal_actions = [i for i, v in enumerate(self._action_mask(agent_id)) if v == 1 and i < rules.NUM_CARDS]
                    had_overtake = any(
                        rules.resolve_trick(pre_trick + [(agent_id, la)], pre_lead_suit, self.trump_suit) == agent_id
                        for la in legal_actions
                    )
                    if had_overtake and actual_temp_winner == current_winner:
                        self.rewards[agent] += self.rewards_cfg.get("partner_save_reward", 0.0)
                        self._shaping_events["partner_save"] += 1

                if (
                    self.trump_suit is not None
                    and pre_lead_suit is not None
                    and pre_lead_suit != self.trump_suit
                    and card_obj.suit == self.trump_suit
                    and (actual_temp_winner != agent_id or partner_winning)
                ):
                    self.rewards[agent] += self.rewards_cfg.get("wasted_trump_penalty", 0.0)
                    self._shaping_events["wasted_trump"] += 1

            # Remove card from hand
            try:
                self.hands[agent_id].remove(card_idx)
            except ValueError as exc:
                raise ValueError(f"Card {card_idx} not in hand for player {agent_id}") from exc

            if self.lead_suit is None:
                self.lead_suit = rules.index_to_card(card_idx).suit
            self.current_trick.append((agent_id, card_idx))
            self.history.append((agent_id, card_idx, self.lead_suit, self.trump_suit))
            self.episode_length += 1

            # Resolve trick if complete
            if len(self.current_trick) == 4:
                winner = rules.resolve_trick(self.current_trick, self.lead_suit, self.trump_suit)
                winning_card = next(card for player, card in self.current_trick if player == winner)
                winning_card_obj = rules.index_to_card(winning_card)
                team = rules.team_for_player(winner)
                if team == 0:
                    self.tricks_won = (self.tricks_won[0] + 1, self.tricks_won[1])
                else:
                    self.tricks_won = (self.tricks_won[0], self.tricks_won[1] + 1)

                # Reward shaping: Trick winner reward
                if self.reward_shaping:
                    team_winning = rules.team_for_player(winner)
                    for ag_id, ag_name in enumerate(self.agents):
                        if rules.team_for_player(ag_id) == team_winning:
                            self.rewards[ag_name] += self.rewards_cfg["trick_reward"]

                    trick_number = sum(self.tricks_won)
                    if trick_number >= 6:
                        for ag_id, ag_name in enumerate(self.agents):
                            if rules.team_for_player(ag_id) == team_winning:
                                self.rewards[ag_name] += self.rewards_cfg.get("late_trick_reward", 0.0)
                        self._shaping_events["late_trick"] += 1

                    if (
                        self.trump_suit is not None
                        and self.lead_suit != self.trump_suit
                        and winning_card_obj.suit == self.trump_suit
                    ):
                        self.rewards[self.agents[winner]] += self.rewards_cfg.get("trump_cut_reward", 0.0)
                        self._shaping_events["trump_cut"] += 1

                self.current_trick = []
                self.lead_suit = None
                # Next trick led by winner
                self._init_selector(start=winner)
                advance_turn = False

        # Advance to next agent unless game just ended
        if advance_turn and not self._terminated:
            self.agent_selection = self.agent_selector.next()

        # Terminal check
        cards_remaining = sum(len(h) for h in self.hands)
        self._terminated = rules.is_terminal(self.tricks_won, cards_remaining)

        if self._terminated:
            winner_team = rules.compute_winner(self.tricks_won)
            declarer_id = (self.start_player - 1) % 4
            declarer_team = rules.team_for_player(declarer_id)
            trump_str = self.trump_suit
            tricks_str = []
            for i in range(0, len(self.history), 4):
                trick = self.history[i:i+4]
                plays = [f"p{t[0]}(team{t[0]%2}):{rules.index_to_card(t[1]).rank}{rules.index_to_card(t[1]).suit}" for t in trick]
                tricks_str.append(f"T{i//4 + 1}: [{', '.join(plays)}]")
            
            trace_str = f"Trump: {trump_str} by player_{declarer_id}. Plays: {' | '.join(tricks_str)}. Score: {self.tricks_won}. Winner Team: {winner_team}"
            
            for ag in self.agents:
                ag_id = self.agents.index(ag)
                ag_team = rules.team_for_player(ag_id)
                if winner_team == -1:
                    reward = self.rewards_cfg.get("draw_penalty", 0.0)
                elif ag_team == winner_team:
                    if self.reward_shaping:
                        # Margin-based reward scaled to 0–2.0 so terminal signal
                        # dominates accumulated trick rewards (max ~0.4 at 0.05/trick)
                        my_team_tricks = self.tricks_won[ag_team]
                        reward = (my_team_tricks - 4) / 4.0 * 2.0
                        if my_team_tricks == 8:
                            reward += self.rewards_cfg["cap_bonus"]

                        # Trump declarer bonus
                        if ag_id == declarer_id:
                            reward += self.rewards_cfg["declarer_bonus"]
                            # Deferred trump quality bonus: reward good suit selection,
                            # scaled by actual win fraction so credit aligns with outcome
                            win_fraction = my_team_tricks / float(rules.TRICKS_PER_HAND)
                            reward += (
                                self.rewards_cfg.get("trump_quality_bonus", 0.0)
                                * self._trump_quality
                                * win_fraction
                            )
                    else:
                        reward = 1.0
                else:
                    if self.reward_shaping:
                        opp_team_tricks = self.tricks_won[1 - ag_team]
                        # Symmetric scale with winner: 0 to -2.0
                        reward = -(opp_team_tricks - 4) / 4.0 * 2.0
                        if opp_team_tricks == 8:
                            reward += self.rewards_cfg["cap_penalty"]
                    else:
                        reward = -1.0
                if self.reward_shaping and winner_team != -1 and ag_team == declarer_team:
                    if winner_team == declarer_team:
                        reward += self.rewards_cfg.get("declarer_team_win_bonus", 0.0)
                        self._shaping_events["declarer_team_win"] += 1
                    else:
                        reward += self.rewards_cfg.get("declarer_team_loss_penalty", 0.0)
                        self._shaping_events["declarer_team_loss"] += 1
                self.rewards[ag] += reward
                self.terminations[ag] = True
                self.infos[ag] = {
                    "winner_team": winner_team,
                    "final_score": self.tricks_won,
                    "episode_length": self.episode_length,
                    "illegal_actions": self._illegal_actions,
                    "match_trace": trace_str,
                    "shaping_events": dict(self._shaping_events),
                }
        
        self._accumulate_rewards()
        return None  # AEC usually uses env.last() for observability

    def _accumulate_rewards(self):
        """Adds .rewards to ._cumulative_rewards and resets .rewards."""
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
        self.rewards = {agent: 0.0 for agent in self.agents}

    def _was_dead_step(self, action):
        """Called by PettingZoo when a terminated agent is stepped.

        Per PettingZoo convention the action is silently ignored; the agent
        simply advances the turn without side-effects.
        """
        agent = self.agent_selection
        self.rewards[agent] = 0.0
        self._cumulative_rewards[agent] = 0.0
        self.agent_selection = self.agent_selector.next()
        self._accumulate_rewards()
        return None

    def render(self):
        print(f"Stage: {self.stage}, Trump: {self.trump_suit}, Lead: {self.lead_suit}")
        print(f"Current trick: {[(p, rules.index_to_card(c)) for p, c in self.current_trick]}")
        print(f"Scores: {self.tricks_won}")

    def close(self):
        pass

    def state(self) -> Dict[str, object]:
        """
        Return a centralized state representation for training the critic.
        Includes public info plus all hands (for CTDE).
        """
        return {
            "hands": [list(h) for h in self.hands],
            "trump_suit": self.trump_suit,
            "lead_suit": self.lead_suit,
            "current_trick": list(self.current_trick),
            "tricks_won": self.tricks_won,
            "history": list(self.history),
        }
