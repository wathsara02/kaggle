# Configuration & Rule Tweaks (Where to Change What)

This document shows *exactly where* to modify core game rules and training behavior.

## Quick map

- **Game rules / constants (deck size, deal pattern, hand size, tricks per hand)**  
  `omi_env/rules.py`

- **Rewards / scoring signal for RL (win/loss reward, per-trick shaping, penalties)**  
  `omi_env/env.py` (terminal reward assignment + where tricks are incremented)

- **Observation encoding dependencies (history length, score normalization, vector sizes)**  
  `omi_env/encoding.py` and `models/critic.py`

- **Training hyperparameters (PPO/MAPPO settings, rollout length, lr, etc.)**  
  `configs/*.yaml` and `marl/*`

---

## 1) Changing the *game structure* (hand size, tricks, dealing)

File: `omi_env/rules.py`

Key constants:

- `NUM_CARDS = 32`  
- `HAND_SIZE = 8`  
- `TRICKS_PER_HAND = 8`  
- `DEAL_FIRST = 4`  
- `DEAL_SECOND = 4`

### Example: change to a different deal pattern
If you ever change `DEAL_FIRST` / `DEAL_SECOND`, ensure:

- `DEAL_FIRST + DEAL_SECOND == HAND_SIZE`
- Remaining deck size after first deal equals `4 * DEAL_SECOND`

> **Important:** If you change `HAND_SIZE` or `TRICKS_PER_HAND`, you must also update history and score normalization (see sections 3 & 4).

---

## 2) Changing *scoring / rewards* (points per trick, win bonus, penalties)

There are two different ideas people call “scoring”:

### A) **Game scoring** (how you decide who wins)
Currently, the environment decides the winner by **tricks won**:
- Team 0 = players (0,2)
- Team 1 = players (1,3)
- Winner = team with more tricks after all cards are played

Files:
- Winner logic: `omi_env/rules.py`
  - `compute_winner(tricks_won)`
- Trick increment happens in: `omi_env/env.py` inside `step()` after `resolve_trick()`

### B) **RL reward signal** (what the agent learns from)
Right now, reward is **terminal only** in `omi_env/env.py`:

- winning team agents get `+1`
- losing team agents get `-1`
- tie gets `0`

This happens here (terminal block):
```python
if self._terminated:
    winner_team = rules.compute_winner(self.tricks_won)
    ...
    self.rewards[ag] = reward
```

#### If you want “X points per trick won”
Add reward shaping in `omi_env/env.py` **right when a trick is resolved**, right after:
```python
team = rules.team_for_player(winner)
self.tricks_won = (...)
```

Typical options:

- **Per-trick dense reward:**  
  winning team gets `+TRICK_REWARD`, losing team gets `-TRICK_REWARD`

- **Winner-only per-trick reward:**  
  only the trick-winning team gets `+TRICK_REWARD`, other team gets `0`

- **Keep terminal ±1 plus small per-trick shaping:**  
  helps learning without changing the “final objective”

> Recommendation for stability: keep terminal ±1 and add a small per-trick value like `0.05`–`0.15`.

#### If you want *card-point* Omi scoring (A=4, K=3, Q=2, J=1)
You would:
1) add a function in `omi_env/rules.py` to compute **points in a trick**
2) maintain `team_points` in `omi_env/env.py`
3) update winner logic to use points instead of trick count (or use both)

---

## 3) Changing “rounds per game” / multi-hand matches

**Current behavior:** 1 hand (8 tricks) = 1 episode.

If you want a match made of multiple hands, there are two clean approaches:

### Option 1 (recommended): **Create a wrapper MatchEnv**
- Leave `OmiEnv` as “one hand”
- Create a wrapper that:
  - runs multiple hands
  - accumulates match points
  - terminates when `target_points` or `num_hands` reached

This keeps training stable and keeps the core env simple.

### Option 2: modify `OmiEnv` directly
You would add new state:
- `self.match_points = (0,0)`
- `self.hands_played`
- `self.max_hands` or `self.target_points`

Then at terminal:
- update match_points
- if match not finished: reset hands and continue (without ending the PettingZoo episode)
- else: terminate

> This is more invasive because PettingZoo expects clear episode boundaries.

---

## 4) Critical dependency updates when you change rules

If you change **hand size** or **cards per hand**, update these:

### `omi_env/encoding.py`
- `HISTORY_LEN` currently assumes **32 plays per hand**
  - `HISTORY_LEN = 32` (8 tricks × 4 plays)
- `score_vec` normalization uses `rules.TRICKS_PER_HAND`

If `HAND_SIZE` changes, update:
- `HISTORY_LEN = 4 * TRICKS_PER_HAND` (or equivalently `4 * HAND_SIZE/??` depending on your format)

### `models/critic.py`
- central critic score normalization also divides by `rules.TRICKS_PER_HAND`
- critic input size depends on history length and feature dimensions

### Tests
Update unit tests under `tests/` if they assume 8 tricks / 32 plays.

---

## 5) “Which file do I change?” cheatsheet

### Change trick winner logic / trump rules / follow-suit rules
- `omi_env/rules.py` (functions: `resolve_trick`, `legal_card_mask`, trump helpers)

### Change how rewards are given to the agent
- `omi_env/env.py` (inside `step()` trick resolution block and terminal block)

### Change observation features / history encoding
- `omi_env/encoding.py`

### Change training parameters
- `configs/*.yaml`
- trainer code under `marl/`

---

## Suggested safe workflow for rule changes

1. Change constants / rules in `omi_env/rules.py`
2. Update env mechanics in `omi_env/env.py`
3. Update encoding (`omi_env/encoding.py`) and critic (`models/critic.py`) if needed
4. Run tests: `pytest -q`
5. Do a short training sanity run (few thousand steps) to confirm learning still works
