import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from baselines.rule_based_agent import RuleBasedAgent
from baselines.random_agent import RandomLegalAgent
from omi_env.env import OmiEnv
from omi_env import rules, encoding
from utils import (
    build_policy,
    bootstrap_confidence_interval,
    clean_state_dict,
    ensure_dir,
    get_device,
    load_config,
    set_seed,
    write_csv_row,
)


def load_policy(cfg: dict, device: torch.device, weights: str):
    policy, _, _ = build_policy(cfg, device)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        state_dict = ckpt["policy_state_dict"]
        print(f"[EVAL] Loaded from full checkpoint (episode {ckpt.get('episode', '?')})")
    else:
        state_dict = ckpt
    policy.load_state_dict(clean_state_dict(state_dict))
    policy.eval()
    return policy


def log_block(progress_pct, episodes_done, block_count, agent_wins, baseline_wins, draws, lengths, illegal_actions, csv_path):
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    agent_rate = (agent_wins / block_count) * 100 if block_count > 0 else 0.0
    baseline_rate = (baseline_wins / block_count) * 100 if block_count > 0 else 0.0
    draw_rate = (draws / block_count) * 100 if block_count > 0 else 0.0
    print(
        f"[EVALUATION — {progress_pct}% COMPLETE]\n"
        f"Episodes evaluated: {episodes_done}\n"
        f"Block episodes: {block_count}\n"
        f"Learned agent wins: {agent_wins}\n"
        f"Baseline wins: {baseline_wins}\n"
        f"Draws: {draws}\n"
        f"Learned agent win rate: {agent_rate:.1f}%\n"
        f"Baseline win rate: {baseline_rate:.1f}%\n"
        f"Draw rate: {draw_rate:.1f}%\n"
        f"Avg episode length: {avg_len:.1f}\n"
        f"Illegal actions: {illegal_actions}"
    )
    headers = (
        "progress_pct",
        "episodes_completed",
        "block_episodes",
        "agent_wins",
        "baseline_wins",
        "draws",
        "agent_win_rate",
        "baseline_win_rate",
        "draw_rate",
        "avg_episode_length",
        "illegal_actions",
    )
    row = {
        "progress_pct": progress_pct,
        "episodes_completed": episodes_done,
        "block_episodes": block_count,
        "agent_wins": agent_wins,
        "baseline_wins": baseline_wins,
        "draws": draws,
        "agent_win_rate": round(agent_rate, 2),
        "baseline_win_rate": round(baseline_rate, 2),
        "draw_rate": round(draw_rate, 2),
        "avg_episode_length": round(avg_len, 2),
        "illegal_actions": illegal_actions,
    }
    write_csv_row(csv_path, headers, row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to policy weights")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--baseline", type=str, choices=["rule", "random"], default="rule")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cpu") == "cuda")
    policy = load_policy(cfg, device, args.weights)
    env = OmiEnv(seed=cfg["seed"])
    baseline_agent = RuleBasedAgent() if args.baseline == "rule" else RandomLegalAgent()

    total_eps = args.episodes
    block_size = max(1, math.ceil(total_eps / 10))
    exp_name = cfg["training"].get("exp_name", "default_run")
    run_dir = Path("runs") / exp_name
    ensure_dir(run_dir)
    csv_path = run_dir / "evaluation_summary.csv"
    match_csv_path = run_dir / "evaluation_match_traces.csv"

    wins_agent = 0
    wins_baseline = 0
    draws_total = 0
    lengths = []
    block_stats = {"agent": 0, "baseline": 0, "draws": 0, "lengths": [], "count": 0, "illegal": 0}
    illegal_total = 0
    win_flags = []

    for ep in range(total_eps):
        env.reset(seed=cfg["seed"] + ep)
        done = False
        hidden_states = {i: policy.init_hidden(1, device) for i in range(4)}

        while not done:
            agent_name = env.agent_selection
            agent_id = int(agent_name.split("_")[1])
            obs = env.observe(agent_name)
            mask = torch.from_numpy(obs["action_mask"]).float().unsqueeze(0).to(device)
            obs_tensor = torch.from_numpy(obs["observation"]).float().unsqueeze(0).to(device)
            hist_tensor = torch.from_numpy(obs["history"]).float().unsqueeze(0).to(device)

            if agent_id in (1, 3):
                action = baseline_agent.act(obs)
            else:
                with torch.no_grad():
                    logits, new_hidden = policy(
                        obs_tensor, hist_tensor,
                        hidden_states[agent_id],
                        action_mask=mask,
                    )
                    hidden_states[agent_id] = new_hidden
                    probs = torch.softmax(logits, dim=-1)
                    if args.deterministic:
                        action = torch.argmax(probs, dim=-1).item()
                    else:
                        action = torch.distributions.Categorical(probs).sample().item()

            env.step(int(action))
            done = all(env.terminations.values())

        info = next(iter(env.infos.values()))
        winner = info.get("winner_team", -1)
        if winner == 0:
            wins_agent += 1
            block_stats["agent"] += 1
            win_flags.append(1)
        elif winner == 1:
            wins_baseline += 1
            block_stats["baseline"] += 1
            win_flags.append(0)
        else:
            draws_total += 1
            block_stats["draws"] += 1

        lengths.append(info.get("episode_length", 0))
        block_stats["lengths"].append(info.get("episode_length", 0))
        illegal_total += info.get("illegal_actions", 0)
        block_stats["illegal"] += info.get("illegal_actions", 0)
        block_stats["count"] += 1

        shaping_events = info.get("shaping_events", {})
        match_headers = (
            "episode",
            "winner_team",
            "final_score",
            "episode_length",
            "illegal_actions",
            "partner_save_events",
            "trump_cut_events",
            "wasted_trump_events",
            "late_trick_events",
            "declarer_team_win_events",
            "declarer_team_loss_events",
            "match_trace",
        )
        write_csv_row(match_csv_path, match_headers, {
            "episode": ep + 1,
            "winner_team": winner,
            "final_score": info.get("final_score", ""),
            "episode_length": info.get("episode_length", 0),
            "illegal_actions": info.get("illegal_actions", 0),
            "partner_save_events": shaping_events.get("partner_save", 0),
            "trump_cut_events": shaping_events.get("trump_cut", 0),
            "wasted_trump_events": shaping_events.get("wasted_trump", 0),
            "late_trick_events": shaping_events.get("late_trick", 0),
            "declarer_team_win_events": shaping_events.get("declarer_team_win", 0),
            "declarer_team_loss_events": shaping_events.get("declarer_team_loss", 0),
            "match_trace": info.get("match_trace", ""),
        })

        if block_stats["count"] >= block_size or ep == total_eps - 1:
            progress = int(((ep + 1) / total_eps) * 100)
            log_block(
                progress,
                ep + 1,
                block_stats["count"],
                block_stats["agent"],
                block_stats["baseline"],
                block_stats["draws"],
                block_stats["lengths"],
                block_stats["illegal"],
                csv_path,
            )
            block_stats = {"agent": 0, "baseline": 0, "draws": 0, "lengths": [], "count": 0, "illegal": 0}

    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    decisive = wins_agent + wins_baseline
    ci_low, ci_high = bootstrap_confidence_interval(win_flags) if win_flags else (0.0, 0.0)
    decisive_rate = (
        f"Win rate among decisive games: {(wins_agent / decisive * 100):.1f}%\n"
        if decisive > 0
        else ""
    )
    print(
        "[EVALUATION — 100% COMPLETE]\n"
        f"Episodes evaluated: {total_eps}\n"
        f"Learned agent wins: {wins_agent}\n"
        f"Baseline wins: {wins_baseline}\n"
        f"Draws (4-4 tie): {draws_total}\n"
        f"Learned agent win rate: {(wins_agent / total_eps) * 100:.1f}%\n"
        f"Baseline win rate: {(wins_baseline / total_eps) * 100:.1f}%\n"
        f"Draw rate: {(draws_total / total_eps) * 100:.1f}%\n"
        f"{decisive_rate}"
        f"Avg episode length: {avg_len:.1f}\n"
        f"Illegal actions: {illegal_total}\n"
        f"Win rate 95% CI (decisive only): ({ci_low:.3f}, {ci_high:.3f})"
    )
    if illegal_total != 0:
        print("WARNING: Non-zero illegal actions detected. Check action masking.")


if __name__ == "__main__":
    main()
