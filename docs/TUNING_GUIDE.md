Tuning Guide
============

What to tune first
------------------
- `lr`: decrease if updates are unstable, increase slowly if learning is too slow.
- `clip_range`: tighten (e.g., 0.1) for stability or loosen (0.25) for faster adaptation.
- `entropy_coef`: raise to encourage exploration; lower when policy is too random late in training.
- `value_coef`: raise if values underfit; lower if value loss dominates.
- `gae_lambda`: increase toward 1.0 for smoother advantages; reduce if variance is too high.
- `gamma`: default 0.99; try 0.995 for longer horizons or 0.97 for shorter ones.
- `rollout_len` / effective episode samples: increase by running more episodes per update for better gradients.
- `batch_size` and `ppo_epochs`: larger batch or more epochs improves sample efficiency but raises compute cost.
- `recurrent_hidden_size`: scale up if the agent underfits long histories; scale down for speed on CPU.
- `recurrent_type`: try `gru` if LSTM states feel heavy; change in `configs/*.yaml`.

Reward shaping tuning
----------------------
The new dense reward system (`reward_shaping: true`) adds several new variables to tune in `omi_env/env.py`:
- **`TRICK_REWARD` (0.1)**: If the agent focuses *only* on the current trick and ignores the total game outcome, lower this value.
- **`OVERPLAY_PENALTY` (0.05)**: If the agent becomes too hesitant to take a trick even when it's strategically necessary, lower this. If it wastes too many high cards, raise it.
- **Margin scaling**: The final reward `(tricks - 4) / 4` provides a smoother gradient. If you want the agent to prioritize *any* win over a "big" win, switch back to a flat `+1/-1`.
- **`DECLARER_BONUS` (0.1)**: Raise this if you want the trump-calling player to be more aggressive in their suit selection.

Symptom → Fix
-------------
- Policy collapses to repetitive/obvious play: increase `entropy_coef`, verify action masks are correct, consider reducing `lr`.
- Value loss explodes or dominates: lower `lr`, ensure `max_grad_norm` is active, and optionally normalize rewards.
- Learning stagnant/flat returns: increase episodes per update, raise `batch_size`, or enlarge `recurrent_hidden_size`; verify advantage normalization.
- Illegal actions appear: ensure `action_mask` is applied before sampling (both env and policy); check action mapping alignment with `docs/OBS_ACTION_SPEC.md`.
- Overfitting to one trump choice: mix seeds, raise entropy early, or bias trump declaration sampling toward uniform when counts tie.

CPU tips
--------
- Use `configs/small.yaml` to validate plumbing; scale to `default.yaml` once stable.
- Keep `batch_size` moderate (64–128) to avoid CPU thrash; adjust `ppo_epochs` to maintain total gradient steps.
- Disable CUDA in configs unless available; models are small enough for CPU-first iteration.
