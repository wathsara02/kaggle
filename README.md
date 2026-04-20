# Intelligent Omi: Cooperative Multi-Agent Reinforcement Learning

Welcome to the **Intelligent Omi** project! This is an advanced artificial intelligence environment designed to teach agents how to master the Sri Lankan trick-taking card game, **Omi**, through deep reinforcement learning.

## 🌟 Project Vision
The goal of this project is to build a cooperative AI that learns complex strategy, teamwork, and card-counting without any hard-coded rules for strategy. Using **Deep Reinforcement Learning (MAPPO)** and **Long Short-Term Memory (LSTM)**, the agents learn from purely randomized play to become expert Omi players over millions of matches.

---

## 🛠 Features

*   **PettingZoo AEC Environment**: A strict, turn-based referee system that enforces Omi rules (must-follow-suit, trump hierarchy).
*   **CTDE Architecture**: "Centralized Training, Decentralized Execution." The AI is trained by an omniscient critic, but plays fair matches during execution using only its private hand and memory.
*   **Memory-Augmented Observation**: The policy receives a full history of the last 32 moves as a structured sequence. In **LSTM mode**, this is processed step-by-step so agents reason sequentially across the hand — just like a human tracking played cards.
*   **Action Masking**: The AI is physically blocked from making illegal moves, focusing 100% of its power on strategy.
*   **Dense Reward System**: A toggleable, high-frequency feedback system for lightning-fast training.

---

## 📁 Repository Structure

### ⚖️ The Game Engine (`omi_env/`)
*   `rules.py`: The core Omi logic (shuffling, dealing, trick resolution, legal move masking).
*   `env.py`: The "Referee" wrapper. Manages turns, rewards, and the current game stage (Trump vs. Play).
*   `encoding.py`: Translates cards and game states into numbers for the AI's neural network.

### 🧠 The AI Brains (`models/`)
*   `policy.py` (The Actor): Each player's decentralized brain. Supports **feed-forward** (default, fast) and **LSTM** (sequential memory, stronger) modes, switchable via config.
*   `critic.py` (The Critic): The omniscient coach used only during training to grade plays.

### 🎓 Training Loop (`marl/` & `scripts/`)
*   `r_mappo.py`: The math behind the policy updates (Proximal Policy Optimization).
*   `train.py`: The main script to start a training session.
*   `eval.py`: Compare your trained AI against random or rule-based bots.
*   `inference/`: Minimal scripts to run a single trained model for testing or demos.

---

All rewards are fully tunable in `configs/default.yaml`. To use these shaped rewards, ensure `reward_shaping -> enabled: true` is set.

*   **Trick Rewards (+0.1)**: Immediate feedback for winning a trick as a team.
*   **Illegal Move Penalty (-0.1)**: Teaches the AI the game rules faster.
*   **Over-playing Penalty (-0.05)**: Discourages wasting high cards when a teammate is already winning the trick.
*   **Margin-based Final Wins**: Rewards are scaled by the margin of victory (e.g., winning 7-1 is better than 5-3).
*   **Trump Declarer Bonus (+0.1)**: Incentivizes the declarer to call the most effective trump suit.
*   **Cap Penalty (-0.5)**: A major penalty if a team loses all 8 tricks.

---

## 🚀 Quickstart Guide

### 1️⃣ Installation
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Unit Tests
Verify the environment and rules are working perfectly.
```bash
pytest
```

### 3️⃣ Train Your AI
Run a fast smoke test to verify the setup:
```powershell
python scripts/train.py --config configs/small.yaml
```
Run the full training session (feed-forward, fast):
```powershell
python scripts/train.py --config configs/default.yaml
```
Run with **LSTM for smarter agents** (recommended for best results):
```powershell
python scripts/train.py --config configs/lstm.yaml
```
Agents trained with LSTM build memory within each 8-trick hand, tracking which cards opponents have played — exactly as a skilled human would.

### 4️⃣ Evaluate Performance
Compare your trained AI's win rate against a rule-based baseline:
```powershell
python scripts/eval.py --weights runs/lstm_cpu/policy_last.pt --episodes 100
```

---

## ⚙️ Configuration & Config Inheritance

All parameters live in `configs/default.yaml`. **Partial configs** like `lstm.yaml` only specify what's *different*, and the system deep-merges them automatically:

| Config | Purpose |
|---|---|
| `configs/default.yaml` | Single source of truth for ALL parameters |
| `configs/lstm.yaml` | Override: enables LSTM memory, saves to `runs/lstm_cpu/` |
| `configs/small.yaml` | Override: 20 episodes, for quick smoke-tests |

**How it works:** running `--config configs/lstm.yaml` first loads *all* of `default.yaml`, then only overwrites the keys defined in `lstm.yaml`. So changing the learning rate in `default.yaml` applies to every config automatically.

Key parameters to tune in `default.yaml`:
*   Change `episodes` to train for longer.
*   Adjust `lr` (learning rate) for faster or more stable convergence.
*   Configure all reward values in the `reward_shaping` section.
*   Set `enabled: true` under `reward_shaping` to use the dense feedback system.

---

## 🛡️ License
Designed for researchers and Omi enthusiasts. Built with PettingZoo and PyTorch.
