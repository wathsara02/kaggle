# Omi MARL — Repository Guide

A complete explanation of how this project works: game rules, architecture, training
algorithm, observation design, and reward structure.

---

## 1. The Game — Omi

Omi is a Sri Lankan trick-taking card game for 4 players in two fixed teams:
- **Team 0**: players 0 and 2
- **Team 1**: players 1 and 3

**Deck**: 32 cards — ranks 7, 8, 9, 10, J, Q, K, A across 4 suits (Clubs, Diamonds, Hearts, Spades).

**Each hand (episode) proceeds in two stages:**

1. **Trump declaration** — each player is dealt 4 cards. The designated declarer (rotates clockwise each hand) picks a trump suit by looking at their 4 cards. The remaining 4 cards per player are then dealt.

2. **Play** — 8 tricks are played. Players must follow the led suit if possible (must-follow-suit rule). If void in the led suit, any card may be played. The highest trump card wins, otherwise the highest card of the led suit wins.

**Scoring**: the team winning more than 4 tricks wins the hand. Winning all 8 tricks ("cap") scores a bonus.

---

## 2. Project Structure

```
Omiya-main/
│
├── omi_env/              # The game environment (PettingZoo AEC)
│   ├── env.py            # OmiEnv — step(), reset(), observe(), reward logic
│   ├── rules.py          # Pure game logic: deck, dealing, masks, trick resolution
│   └── encoding.py       # Observation & history encoding (produces the 194-dim vector)
│
├── models/               # Neural network definitions
│   ├── policy.py         # PolicyNet — FF or LSTM actor
│   └── critic.py         # CentralCritic — centralized value function (CTDE)
│
├── marl/                 # Multi-agent training
│   ├── r_mappo.py        # MAPPOTrainer — collect_episode() + update()
│   └── vector_env.py     # CloudVectorEnv — subprocess-based parallel envs
│
├── buffer.py             # AgentBuffer — stores transitions, computes GAE advantages
├── utils.py              # Shared helpers: config loading, seeding, CSV writing, etc.
│
├── baselines/            # Non-learning opponents for eval and mixed training
│   ├── rule_based_agent.py  # Heuristic: longest suit as trump, highest card to follow
│   └── random_agent.py      # Uniformly random legal action
│
├── scripts/              # Runnable entry points
│   ├── train.py          # Main training loop
│   ├── eval.py           # Evaluate trained policy vs baseline
│   ├── export.py         # Package weights + config for deployment
│   └── plot_training.py  # Generate training graphs from CSV
│
├── inference/
│   └── inference.py      # InferenceAgent — loads exported model, serves web app
│
├── configs/
│   ├── default.yaml      # Master config (T4 GPU, FF policy)
│   ├── lstm.yaml         # Override for LSTM policy
│   └── small.yaml        # Override for quick smoke-tests
│
└── tests/                # Unit tests (pytest)
    ├── test_rules.py
    ├── test_env.py
    └── conftest.py
```

---

## 3. Observation Space — 194 dimensions

Every agent receives a flat observation vector built in `omi_env/encoding.py`:

| Segment | Size | Description |
|---------|------|-------------|
| `hand_vec` | 32 | One-hot of cards in the agent's own hand |
| `trump_vec` | 4 | One-hot of the declared trump suit (zeros before declaration) |
| `lead_vec` | 4 | One-hot of the current trick's lead suit |
| `trick_flat` | 128 | Up to 4 cards already played this trick (4 × 32 one-hot, padded) |
| `score_vec` | 2 | Normalised trick count per team (0–1) |
| `player_vec` | 4 | One-hot of this agent's player ID |
| `suit_counts` | 4 | Normalised proportion of each suit in the agent's current hand |
| `void_flat` | 16 | 4×4 void matrix: `void[player][suit] = 1` if that player has been observed to be void in that suit |
| **Total** | **194** | |

The **void matrix** is the most strategically important addition. When a player fails to follow a led suit it proves they are void in it. Rather than requiring the network to discover this pattern across 1408 raw history features, the encoding pre-computes it and hands it over directly. This gives the feed-forward policy the same void-tracking capability that an LSTM would build incrementally.

Each agent also receives a `history` array of shape `(32, 44)` for the optional LSTM path — every past play encoded as `[card one-hot (32) | player one-hot (4) | lead suit (4) | trump suit (4)]`.

---

## 4. Action Space — 36 actions

| Indices | Meaning |
|---------|---------|
| 0 – 31 | Play card with that index |
| 32 – 35 | Declare trump suit (C=32, D=33, H=34, S=35) |

During the trump phase the action mask allows only 32–35. During play only 0–31 (further restricted by must-follow-suit). Illegal actions are replaced by a legal fallback and penalised.

---

## 5. Reward Structure

Rewards are assigned at the end of each episode plus optional per-step shaping:

### Terminal rewards (always on)
| Outcome | Reward |
|---------|--------|
| Winning team | `(tricks_won - 4) / 4.0` (scales 0 → 1.0) |
| Losing team | `-(opponent_tricks - 4) / 4.0` |
| Cap (all 8 tricks) — winners | `+cap_bonus` |
| Cap (all 8 tricks) — losers | `+cap_penalty` (negative) |
| Trump declarer bonus | `+declarer_bonus` if on winning team |

### Shaping rewards (`reward_shaping.enabled: true`)
| Event | Reward |
|-------|--------|
| Illegal action | `illegal_action_penalty` (−0.1) |
| Team wins a trick | `trick_reward` (+0.2) to both teammates |
| Trump declarer: quality of suit chosen | `trump_quality_bonus × (trump_cards_in_full_hand / 8)` — fires immediately after remaining cards are dealt, cutting credit-assignment delay from 32 steps to 0 |
| Overplay (teammate winning, agent had safe alternative but caused team loss or stole trick from partner) | `overplay_penalty` (−0.2) |

---

## 6. Policy Network (`models/policy.py`)

```
obs (194) ──► Linear(194→256) ──► Tanh          ─┐
                                                   ├─► concat(512) ──► Linear(512→256) ──► LayerNorm ──► Tanh
history:                                           │                ──► Linear(256→256) ──► LayerNorm ──► Tanh
  FF:   flatten(32×44=1408) ──► Linear(1408→256) ─┘                ──► Linear(256→36) = logits
  LSTM: LSTM(44→256, carried h,c) ──► h_last ─────┘
```

**Feed-forward mode** (`recurrent_type: none`): history is flattened and encoded with a single linear layer. Simple, fast, and sufficient when combined with the explicit void matrix and suit-count features in the observation.

**LSTM mode** (`recurrent_type: lstm`): the 32-step history sequence is processed step-by-step. Hidden state `(h, c)` is carried between turns within a hand and reset at the start of each new hand. Better at sequential inference but slower and more complex to deploy.

The action mask is applied by setting illegal action logits to −10⁹ before the softmax, ensuring they are never sampled.

---

## 7. Critic Network (`models/critic.py`)

The critic is **centralised** (Centralised Training, Decentralised Execution — CTDE): it receives the full game state including all four players' hands. This removes the partial-observability problem from value estimation during training.

The central state encodes all hands, trump, lead, current trick, scores, and full play history into one flat vector. The critic maps this to a single scalar value using a 3-layer MLP with LayerNorm.

At deployment only the policy (actor) is used — the critic is discarded.

---

## 8. Training Algorithm — MAPPO

Multi-Agent PPO with a shared policy and centralised critic.

**All 4 agents share the same policy weights.** This means:
- The policy learns to play from any of the 4 seat positions
- The opponent's strategy is the policy's own past self (self-play)
- A single checkpoint serves as all 3 AI players in the web app

**Training loop (one iteration):**

1. `collect_episode()` — run N parallel episodes (one per env worker). For each agent turn: run policy forward pass, sample action, step env, store transition.
2. `update()` — compute GAE advantages, run PPO update for `ppo_epochs` epochs over all collected transitions.
3. Anneal learning rate linearly from `lr` → `lr_min`.

**Mixed-opponent training** (`rule_mix_prob: 0.3`): 30% of episodes randomly assign some agent seats to the rule-based heuristic agent. Three modes rotate:
- *opponents* — players 1 & 3 are rule-based (teaches the policy to beat consistent heuristics)
- *teammate* — player 2 is rule-based (teaches the policy to cooperate with non-AI partners, crucial for web app where player 2 is the human's AI teammate)
- *one* — one random seat is rule-based (general diversity)

Rule-based agents' transitions are **not** added to the training buffer — only the learned policy's turns contribute gradients.

---

## 9. Buffer and GAE (`buffer.py`)

Each env maintains an `AgentBuffer` with separate trajectory storage per agent (0–3). After an episode ends, `compute_advantages()` runs backwards through each agent's trajectory computing Generalised Advantage Estimation:

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done) - V(s_t)
A_t = δ_t + γλ · A_{t+1}
return_t = A_t + V(s_t)
```

Advantages are then normalised to zero mean unit variance before the PPO update.

---

## 10. Vector Environment (`marl/vector_env.py`)

`CloudVectorEnv` spawns N worker processes (one per env), each running an independent `OmiEnv`. Communication is via Python `multiprocessing.Pipe`. All envs step in parallel, and the trainer batches their observations into a single GPU forward pass — this is where the T4 GPU acceleration comes from.

On Linux (Kaggle), `forkserver` context is used to avoid CUDA deadlocks. On Windows, `spawn` is used instead.

---

## 11. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Shared policy (all 4 agents) | Single checkpoint to deploy; policy must be good from any seat |
| FF over LSTM | Simpler deployment (stateless), void matrix closes most of the information gap |
| Void matrix in observation | Converts a complex sequential inference into a direct readable feature |
| Immediate trump quality reward | Cuts credit-assignment chain from 32 steps to 0 for the most impactful early decision |
| Centralised critic | Removes partial observability from value estimation without affecting actor at inference |
| Mixed rule-based training | Prevents the agent from learning conventions that only work against itself; improves teammate behaviour with human players |
