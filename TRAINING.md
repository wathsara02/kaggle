# Training & Tweaking Guide

This file is meant to help you **present** and **reproduce** results.

## 1) Training (self-play)

### Minimal training
```bash
python scripts/train.py --config configs/small.yaml
```

### Standard training
```bash
python scripts/train.py --config configs/default.yaml
```

Training writes checkpoints and logs to the run directory (see your config).

## 2) Evaluation

### Evaluate a saved policy vs baselines
```bash
python scripts/eval.py --config configs/default.yaml --weights runs/default_cpu/policy_last.pt
```

You can swap the baseline inside `scripts/eval.py` (random or rule-based).

## 3) Where to change what (common edits)

### A) Game rules / realism
**File:** `omi_env/rules.py`
- Rank order, trump logic, must-follow-suit constraints
- Trick winner logic
- Terminal condition (e.g., end after 8 tricks / all cards played)
- Scoring method (tricks-only vs card-points)

**File:** `omi_env/env.py`
- Who declares trump (`start_player`) and how it rotates
- Whether trump selection is manual (agent action) vs automatic (rule-based)
- Reward shaping (currently terminal only)

### B) Observation design (what the agent can “see”)
**File:** `omi_env/encoding.py`
- Observation vector composition
- History length (`HISTORY_LEN`) and feature set
- Any extra public features you want to add

### C) Model architecture
**File:** `models/policy.py`
- LSTM/GRU choice
- Hidden size / number of layers
- Action masking application (keep this correct!)

**File:** `models/critic.py`
- Centralized critic inputs (global state features)
- Hidden size / layers

### D) Learning algorithm / hyperparameters
**File:** `marl/r_mappo.py`
- PPO update logic
- Advantage normalization, clipping, entropy bonus, gradient clipping

**Configs:** `configs/*.yaml`
- `lr`, `gamma`, `gae_lambda`
- `clip_range`, `entropy_coef`, `value_coef`
- `rollout_len`, `batch_size`, `ppo_epochs`
- `recurrent_hidden_size`, `recurrent_type`

See also: `docs/TUNING_GUIDE.md`.

## 4) Recommended tuning order (fastest wins)

1. **Verify legality & masks** (if masks are wrong, learning fails silently)
2. Increase training signal:
   - run longer (more episodes per update)
   - optionally add small reward shaping (per-trick +0.1 / -0.1 max)
3. Tune:
   - `entropy_coef` (exploration)
   - `lr` + `clip_range` (stability)
   - `recurrent_hidden_size` (memory capacity)

## 5) Common mistakes to avoid

- Changing rules but forgetting to update tests (`tests/`).
- Adding reward shaping with large magnitude (destabilizes PPO).
- Removing action masking (policy will learn illegal moves and crash exploration).
- Comparing runs with different seeds without recording configs.

## 6) Reproducibility checklist (for your report)

- Mention:
  - config file used (YAML)
  - seed(s)
  - number of training episodes / updates
  - evaluation opponents (random / rule-based)
- Save:
  - `policy_last.pt`, `critic_last.pt`
  - the exact config file used for that run


## Changing rules, scoring, or match length
See `CONFIGURATION.md` for a step-by-step map of which files control scoring/rewards, tricks per hand, and multi-hand matches.
