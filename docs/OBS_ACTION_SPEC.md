Observation and Action Specification
====================================

Observation dictionary
----------------------
Each agent receives a dict:
- `observation` (float32, shape `(174,)`): concatenated features
  - `[0:32)` hand mask (1 if the agent holds the card).
  - `[32:36)` trump suit one-hot (`C,D,H,S`), zeros before declaration.
  - `[36:40)` lead suit one-hot for the current trick, zeros if none.
  - `[40:168)` current trick cards (4 slots × 32 one-hot each, padded with zeros).
  - `[168:170)` normalized trick scores `(team_a, team_b)` divided by 8.
  - `[170:174)` current agent id one-hot.
- `action_mask` (float32, shape `(36,)`): legal action flags. Must-follow-suit and trump-declaration legality are enforced here.
- `history` (float32, shape `(32, 44)`): sequence encoding of the last 32 plays. Each row is `[card_one_hot(32), player_one_hot(4), lead_suit_one_hot(4), trump_suit_one_hot(4)]`, zero-padded on the left when shorter.

Action space
------------
Discrete of size 36:
- `0..31`: play the corresponding card (index = suit * 8 + rank). Suits ordered `C,D,H,S`; ranks `7,8,9,10,J,Q,K,A`.
- `32..35`: declare trump (`32=C`, `33=D`, `34=H`, `35=S`). Only legal during the trump phase.

Legal masking
-------------
- Trump phase: only indices `32..35` are marked legal.
- Play phase: must-follow-suit mask. If the agent holds any card of the lead suit, only those cards are legal; otherwise, any card in hand is legal. Trump actions are illegal and masked out.

Terminal info contract
----------------------
At termination every agent's `info` contains:
```
{
  "winner_team": int,          # 0 for players (0,2), 1 for players (1,3), -1 for tie
  "final_score": (team_a, team_b),
  "episode_length": int,
  "illegal_actions": int
}
```
