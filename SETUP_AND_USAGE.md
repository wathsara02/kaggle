# Omi MARL — Setup & Usage Guide

Step-by-step instructions for installation, training, evaluation, and web app export.

---

## Prerequisites

- Python 3.10 or later
- pip
- An NVIDIA GPU is strongly recommended for training. CPU training works but is ~20× slower.
- For Kaggle: use a T4 GPU accelerator notebook.

---

## 1. Installation

```bash
# Clone or unzip the project, then cd into it
cd Omiya-main

# Install all dependencies
pip install -r requirements.txt
```

`requirements.txt` pulls the CUDA 12.1 build of PyTorch by default.
If you are on CPU only, change the first line of `requirements.txt` to:
```
--extra-index-url https://download.pytorch.org/whl/cpu
```

### Verify the install

Run a 20-episode smoke-test (takes ~10 seconds on CPU):
```bash
python scripts/train.py --config configs/small.yaml
```

You should see two `[TRAINING PROGRESS]` blocks followed by `[TRAINING COMPLETE]` with no errors.

---

## 2. Training

### On a laptop GPU (GTX 1650 / 4 GB VRAM)

```bash
python scripts/train.py --config configs/laptop.yaml
```

- Feed-forward policy, 6 parallel environments, hidden size 128 — fits in 4 GB VRAM.
- Saves output to `runs/laptop_ff/`.
- Runs for 2 million episodes (≈ 1–2 days on a 1650). Resume to add more.

### On Kaggle / T4 GPU (16 GB VRAM)

```bash
python scripts/train.py --config configs/default.yaml
```

- Feed-forward policy, 16 parallel environments, hidden size 256.
- Saves output to `runs/webapp_ff/`.
- Runs for 5 million episodes (~1–2 Kaggle sessions).

### Resume after a session ends or crash

```bash
# Laptop
python scripts/train.py --config configs/laptop.yaml --resume

# Kaggle T4
python scripts/train.py --config configs/default.yaml --resume
```

Resumes from the latest checkpoint in the run folder. Checkpoints are saved automatically (every 2000 episodes on laptop, 5000 on T4).

### LSTM variant (slower, slightly better sequential reasoning)

```bash
python scripts/train.py --config configs/lstm.yaml
```

> Note: The LSTM model requires tracking hidden states per player per game session in your web app. The FF model is recommended for deployment.

### What gets saved during training

All output lands in `runs/<exp_name>/` (default: `runs/webapp_ff/`):

| File | When saved | Contents |
|------|-----------|----------|
| `checkpoint_latest.pt` | Every 5000 episodes | Full training state (policy + critic + optimisers + episode count) |
| `policy_1_3.pt` | At 33% of training | Policy weights — use as **Easy** difficulty |
| `policy_2_3.pt` | At 66% of training | Policy weights — use as **Medium** difficulty |
| `policy_last.pt` | End of training | Final policy weights — use as **Hard** difficulty |
| `critic_last.pt` | End of training | Final critic weights |
| `training_summary.csv` | Every 10 episodes | Win rates, losses, entropy, illegal actions |
| `training_progress.png` | Every 10 episodes | Auto-generated training graphs |

### On Kaggle specifically

1. Upload this project folder as a Kaggle dataset or use the notebook editor.
2. Enable the T4 GPU accelerator (Settings → Accelerator → GPU T4 x2).
3. Run the training cell. Each Kaggle session is up to 12 hours.
4. After session 1, download `checkpoint_latest.pt` and upload it for session 2.
5. Session 2: run with `--resume` to continue.

---

## 3. Monitoring Training

Training graphs are saved automatically as `training_progress.png` in the run folder after every log block. They show:

1. **Win Rate** — Team A vs Team B over training. You want Team A rising above 50%.
2. **Policy & Value Loss** — both should fall and stabilise. Spikes indicate instability.
3. **Policy Entropy** — starts high (~3.5), should decay as the agent becomes more decisive.
4. **Illegal Actions per Episode** — should drop toward 0 quickly.

To regenerate graphs from a saved CSV at any time:
```bash
python scripts/plot_training.py --csv runs/webapp_ff/training_summary.csv
```

---

## 4. Evaluation

Test your trained agent against a baseline opponent:

### Against the rule-based heuristic agent (recommended)

```bash
python scripts/eval.py \
    --config configs/default.yaml \
    --weights runs/webapp_ff/policy_last.pt \
    --episodes 200 \
    --deterministic
```

### Against a random agent

```bash
python scripts/eval.py \
    --config configs/default.yaml \
    --weights runs/webapp_ff/policy_last.pt \
    --baseline random \
    --episodes 200
```

### Interpreting results

| Win rate vs rule-based | Interpretation |
|------------------------|---------------|
| < 45% | Agent has not learned well — extend training or check config |
| 45–55% | Competitive with the heuristic — acceptable |
| > 55% | Agent has learned genuine strategy beyond the heuristic |

Evaluation results are printed to the terminal and saved to `runs/webapp_ff/evaluation_summary.csv`.

---

## 5. Export for Web App

Package the trained model into a deployable artifact:

```bash
python scripts/export.py \
    --config configs/default.yaml \
    --weights runs/webapp_ff/policy_last.pt \
    --output_dir artifacts/
```

This creates:
```
artifacts/
  policy_agent.pt   — PyTorch weights
  config.json       — Model dimensions and difficulty settings
  VERSION           — Version tag
```

### Optional: ONNX export (for JavaScript / browser frontends)

```bash
python scripts/export.py \
    --config configs/default.yaml \
    --weights runs/webapp_ff/policy_last.pt \
    --onnx
```

Adds `policy_agent.onnx` to the artifacts folder. Load this with ONNX Runtime in a Node.js or browser backend.

---

## 6. Using the Exported Model in Your Web App

```python
from inference.inference import load_agent
import torch
import numpy as np

# Load once at server startup
agent = load_agent("artifacts/policy_agent.pt", "artifacts/config.json")

# Each time an AI player needs to act (per HTTP request / game turn):
obs     = torch.from_numpy(obs_array).float()     # shape (194,)
mask    = torch.from_numpy(mask_array).float()    # shape (36,)
history = torch.from_numpy(history_array).float() # shape (32, 44)

action, _ = agent.act(obs, mask, history, deterministic=True)
# action is an int: 0-31 = play card, 32-35 = declare trump suit
```

### Difficulty levels

Control difficulty via the `temperature` parameter (no retraining required):

```python
# Easy — more random
action, _ = agent.act(obs, mask, history, temperature=2.0)

# Medium — default
action, _ = agent.act(obs, mask, history, temperature=1.0)

# Hard — very decisive
action, _ = agent.act(obs, mask, history, temperature=0.3)

# Expert — always picks highest-probability action
action, _ = agent.act(obs, mask, history, deterministic=True)
```

Alternatively, use the three milestone checkpoints as built-in difficulty tiers:

| Checkpoint | Difficulty |
|------------|-----------|
| `policy_1_3.pt` | Easy |
| `policy_2_3.pt` | Medium |
| `policy_last.pt` | Hard / Expert |

### Player setup in the web app

The AI controls 3 of the 4 seats. The human takes one seat (typically player 0):

| Seat | Who plays |
|------|----------|
| Player 0 | Human |
| Player 1 | AI (opponent) |
| Player 2 | AI (human's teammate) |
| Player 3 | AI (opponent) |

Teams: Player 0 & 2 vs Player 1 & 3.

To get the observation for an AI player, use the environment's `encode_observation` function with the current game state.

---

## 7. Running Unit Tests

```bash
pytest
```

Verifies game rules, must-follow-suit enforcement, trick resolution, and reward logic. All tests should pass before training.

---

## 8. Configuration Reference

All configs live in `configs/`. The `default.yaml` is the master; other files override only what differs.

| Config | Target hardware | Key differences |
|--------|----------------|----------------|
| `default.yaml` | Kaggle T4 (16 GB VRAM) | batch=2048, envs=16, hidden=256, 5M episodes |
| `laptop.yaml` | GTX 1650 / 4 GB VRAM | batch=512, envs=6, hidden=128, 2M episodes |
| `lstm.yaml` | Any (LSTM variant) | recurrent_type=lstm |
| `small.yaml` | CPU smoke-test | 20 episodes, hidden=64 |

Key parameters in `configs/default.yaml`:

```yaml
device: cuda                  # "cuda" for GPU, "cpu" for CPU-only

model:
  recurrent_type: none        # "none" = FF (recommended), "lstm" = LSTM
  recurrent_hidden_size: 256  # hidden units for policy network
  critic_hidden_size: 256     # hidden units for critic network

algo:
  lr: 0.0003                  # Adam learning rate (anneals to lr_min)
  lr_min: 0.00001
  clip_range: 0.2             # PPO clipping epsilon
  entropy_coef: 0.01          # entropy bonus (0.01 = decisive deployment agent)
  value_coef: 0.5             # critic loss weight
  gamma: 0.99                 # discount factor
  gae_lambda: 0.95            # GAE lambda
  batch_size: 2048            # minibatch size per PPO epoch
  ppo_epochs: 4               # PPO update passes per collected batch
  rule_mix_prob: 0.3          # fraction of episodes with rule-based opponents mixed in

training:
  episodes: 5000000           # total episodes (~2 Kaggle sessions)
  num_envs: 16                # parallel environments (suits T4 GPU)
  exp_name: webapp_ff         # output folder under runs/
  checkpoint_interval: 5000  # episodes between resumable checkpoints

reward_shaping:
  enabled: true
  trick_reward: 0.2           # reward per trick won (both teammates)
  trump_quality_bonus: 0.2    # immediate reward for good trump declaration
  overplay_penalty: -0.2      # penalty for losing a winnable trick or stealing from partner
  cap_bonus: 0.5              # bonus for winning all 8 tricks
  cap_penalty: -0.5           # penalty for losing all 8 tricks
  declarer_bonus: 0.1         # bonus to trump declarer on win
  illegal_action_penalty: -0.1
```

---

## 9. Troubleshooting

**`CUDA out of memory`**
Reduce `batch_size` to 1024 or `num_envs` to 8 in `default.yaml`.

**`FileNotFoundError` on checkpoint resume**
Make sure `runs/webapp_ff/checkpoint_latest.pt` exists. If starting fresh, remove `--resume`.

**Win rate stuck near 50%**
This is normal early in training — both teams are learning simultaneously. Check that entropy is decreasing (agent becoming less random) and illegal actions are dropping.

**Win rate never rises above 50%**
Try reducing `rule_mix_prob` to 0.1 or disabling reward shaping and running pure self-play for 1M episodes first to bootstrap basic card play.

**Tests failing**
Run `pip install -r requirements.txt` again to ensure all dependencies are up to date.
