import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from omi_env.env import OmiEnv
from utils import (
    bootstrap_confidence_interval,
    build_policy,
    clean_state_dict,
    ensure_dir,
    get_device,
    load_config,
    set_seed,
    write_csv_row,
)


def load_policy(cfg: dict, device: torch.device, weights: str, label: str):
    policy, _, _ = build_policy(cfg, device)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        state_dict = ckpt["policy_state_dict"]
        print(f"[{label}] Loaded full checkpoint from episode {ckpt.get('episode', '?')}")
    else:
        state_dict = ckpt
        print(f"[{label}] Loaded policy weights")
    policy.load_state_dict(clean_state_dict(state_dict))
    policy.eval()
    return policy


def choose_action(policy, obs, hidden_state, device, deterministic: bool):
    mask = torch.from_numpy(obs["action_mask"]).float().unsqueeze(0).to(device)
    obs_tensor = torch.from_numpy(obs["observation"]).float().unsqueeze(0).to(device)
    hist_tensor = torch.from_numpy(obs["history"]).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits, new_hidden = policy(
            obs_tensor,
            hist_tensor,
            hidden_state,
            action_mask=mask,
        )
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
        else:
            action = torch.distributions.Categorical(probs).sample().item()
    return int(action), new_hidden


def main():
    parser = argparse.ArgumentParser(description="Evaluate two trained Omi policies against each other.")
    parser.add_argument("--config-a", type=str, default="configs/new.yaml", help="Config for Team A policy")
    parser.add_argument("--config-b", type=str, default=None, help="Config for Team B policy; defaults to config-a")
    parser.add_argument("--weights-a", type=str, required=True, help="Weights/checkpoint for Team A policy, players 0 and 2")
    parser.add_argument("--weights-b", type=str, required=True, help="Weights/checkpoint for Team B policy, players 1 and 3")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    cfg_a = load_config(args.config_a)
    cfg_b = load_config(args.config_b or args.config_a)
    if args.seed is not None:
        cfg_a["seed"] = args.seed
        cfg_b["seed"] = args.seed

    set_seed(cfg_a["seed"])
    device = get_device(cfg_a.get("device", "cpu") == "cuda")
    policy_a = load_policy(cfg_a, device, args.weights_a, "TEAM_A")
    policy_b = load_policy(cfg_b, device, args.weights_b, "TEAM_B")

    reward_cfg = cfg_a.get("reward_shaping", {})
    env = OmiEnv(
        seed=cfg_a["seed"],
        reward_shaping=reward_cfg.get("enabled", False),
        rewards_dict=reward_cfg,
    )

    total_eps = args.episodes
    block_size = max(1, math.ceil(total_eps / 10))
    out_dir = Path(args.out_dir or Path("runs") / cfg_a["training"].get("exp_name", "policy_vs_policy"))
    ensure_dir(out_dir)
    csv_path = out_dir / "policy_vs_policy_summary.csv"
    match_csv_path = out_dir / "policy_vs_policy_match_traces.csv"

    wins_a = 0
    wins_b = 0
    draws = 0
    illegal_total = 0
    lengths = []
    win_flags = []
    block = {"a": 0, "b": 0, "draws": 0, "count": 0, "illegal": 0, "lengths": []}

    for ep in range(total_eps):
        env.reset(seed=cfg_a["seed"] + ep)
        done = False
        hidden_a = {i: policy_a.init_hidden(1, device) for i in (0, 2)}
        hidden_b = {i: policy_b.init_hidden(1, device) for i in (1, 3)}

        while not done:
            agent_name = env.agent_selection
            agent_id = int(agent_name.split("_")[1])
            obs = env.observe(agent_name)
            if agent_id in (0, 2):
                action, hidden_a[agent_id] = choose_action(
                    policy_a, obs, hidden_a[agent_id], device, args.deterministic
                )
            else:
                action, hidden_b[agent_id] = choose_action(
                    policy_b, obs, hidden_b[agent_id], device, args.deterministic
                )
            env.step(action)
            done = all(env.terminations.values())

        info = next(iter(env.infos.values()))
        winner = info.get("winner_team", -1)
        if winner == 0:
            wins_a += 1
            block["a"] += 1
            win_flags.append(1)
        elif winner == 1:
            wins_b += 1
            block["b"] += 1
            win_flags.append(0)
        else:
            draws += 1
            block["draws"] += 1

        length = info.get("episode_length", 0)
        illegal = info.get("illegal_actions", 0)
        lengths.append(length)
        illegal_total += illegal
        block["lengths"].append(length)
        block["illegal"] += illegal
        block["count"] += 1

        headers = (
            "episode",
            "winner_team",
            "final_score",
            "episode_length",
            "illegal_actions",
            "match_trace",
        )
        write_csv_row(match_csv_path, headers, {
            "episode": ep + 1,
            "winner_team": winner,
            "final_score": info.get("final_score", ""),
            "episode_length": length,
            "illegal_actions": illegal,
            "match_trace": info.get("match_trace", ""),
        })

        if block["count"] >= block_size or ep == total_eps - 1:
            episodes_done = ep + 1
            progress = int((episodes_done / total_eps) * 100)
            avg_len = sum(block["lengths"]) / len(block["lengths"]) if block["lengths"] else 0.0
            a_rate = block["a"] / block["count"] * 100.0
            b_rate = block["b"] / block["count"] * 100.0
            draw_rate = block["draws"] / block["count"] * 100.0
            print(
                f"[POLICY VS POLICY - {progress}% COMPLETE]\n"
                f"Episodes evaluated: {episodes_done}\n"
                f"Block episodes: {block['count']}\n"
                f"Team A wins: {block['a']} ({a_rate:.1f}%)\n"
                f"Team B wins: {block['b']} ({b_rate:.1f}%)\n"
                f"Draws: {block['draws']} ({draw_rate:.1f}%)\n"
                f"Avg episode length: {avg_len:.1f}\n"
                f"Illegal actions: {block['illegal']}"
            )
            summary_headers = (
                "progress_pct",
                "episodes_completed",
                "block_episodes",
                "team_a_wins",
                "team_b_wins",
                "draws",
                "team_a_win_rate",
                "team_b_win_rate",
                "draw_rate",
                "avg_episode_length",
                "illegal_actions",
            )
            write_csv_row(csv_path, summary_headers, {
                "progress_pct": progress,
                "episodes_completed": episodes_done,
                "block_episodes": block["count"],
                "team_a_wins": block["a"],
                "team_b_wins": block["b"],
                "draws": block["draws"],
                "team_a_win_rate": round(a_rate, 2),
                "team_b_win_rate": round(b_rate, 2),
                "draw_rate": round(draw_rate, 2),
                "avg_episode_length": round(avg_len, 2),
                "illegal_actions": block["illegal"],
            })
            block = {"a": 0, "b": 0, "draws": 0, "count": 0, "illegal": 0, "lengths": []}

    decisive = wins_a + wins_b
    ci_low, ci_high = bootstrap_confidence_interval(win_flags) if win_flags else (0.0, 0.0)
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    print(
        "[POLICY VS POLICY - 100% COMPLETE]\n"
        f"Episodes evaluated: {total_eps}\n"
        f"Team A wins: {wins_a}\n"
        f"Team B wins: {wins_b}\n"
        f"Draws: {draws}\n"
        f"Team A win rate: {(wins_a / total_eps) * 100:.1f}%\n"
        f"Team B win rate: {(wins_b / total_eps) * 100:.1f}%\n"
        f"Draw rate: {(draws / total_eps) * 100:.1f}%\n"
        f"Team A decisive win rate: {(wins_a / decisive * 100):.1f}%\n" if decisive > 0 else ""
        f"Avg episode length: {avg_len:.1f}\n"
        f"Illegal actions: {illegal_total}\n"
        f"Team A decisive 95% CI: ({ci_low:.3f}, {ci_high:.3f})"
    )


if __name__ == "__main__":
    main()
