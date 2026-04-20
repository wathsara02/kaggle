import argparse
import math
import os
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from marl.r_mappo import MAPPOTrainer
from models.critic import CentralCritic, encode_central_state
from omi_env.env import OmiEnv
from omi_env import rules, encoding
from utils import build_policy, ensure_dir, get_device, load_config, set_seed, write_csv_row
from marl.vector_env import CloudVectorEnv
from scripts.plot_training import plot_training
from functools import partial

# ── T4 GPU performance flags ──────────────────────────────────────────────────
# Allow TF32 on Ampere/Turing GPUs for faster matrix ops (harmless on T4)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Let cuDNN auto-select the fastest algorithm for fixed input sizes
torch.backends.cudnn.benchmark = True




def build_trainer(cfg: dict, device: torch.device):
    reward_cfg = cfg.get("reward_shaping", {})
    num_envs = cfg["training"].get("num_envs", 1)
    
    if num_envs > 1:
        env_fns = [
            partial(
                OmiEnv,
                seed=cfg["seed"]+i,
                reward_shaping=reward_cfg.get("enabled", False),
                rewards_dict=reward_cfg
            ) for i in range(num_envs)
        ]
        env = CloudVectorEnv(env_fns)
        env.reset([cfg["seed"] + i for i in range(num_envs)])
        dummy_state = env.get_state([0])[0]
    else:
        env = OmiEnv(
            seed=cfg["seed"],
            reward_shaping=reward_cfg.get("enabled", False),
            rewards_dict=reward_cfg
        )
        env.reset()
        dummy_state = env.state()
        
    policy, _, _ = build_policy(cfg, device)
    encoded_state = encode_central_state(dummy_state)
    critic = CentralCritic(input_dim=encoded_state.shape[0], hidden_size=cfg["model"]["critic_hidden_size"])

    # torch.compile gives 20-40% speedup on T4 via kernel fusion and graph optimisation.
    # Only applied when CUDA is available; silently skipped on CPU.
    if device.type == "cuda":
        try:
            policy = torch.compile(policy, mode="reduce-overhead")
            critic = torch.compile(critic, mode="reduce-overhead")
            print("[COMPILE] torch.compile applied to policy and critic.")
        except Exception as e:
            print(f"[COMPILE] torch.compile skipped: {e}")

    trainer = MAPPOTrainer(policy, critic, cfg["algo"], device)
    return trainer, env


def log_block(progress_pct, episodes_done, total_episodes, block_count, team_a, team_b,
              lengths, illegal_actions, csv_path, sample_traces, losses=None):
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    team_a_rate = (team_a / block_count) * 100 if block_count > 0 else 0.0
    team_b_rate = (team_b / block_count) * 100 if block_count > 0 else 0.0
    policy_loss = losses.get("policy_loss", 0.0) if losses else 0.0
    value_loss  = losses.get("value_loss",  0.0) if losses else 0.0
    entropy     = losses.get("entropy",     0.0) if losses else 0.0
    print(
        f"[TRAINING PROGRESS — {progress_pct}% COMPLETE]\n"
        f"Episodes run: {episodes_done} / {total_episodes}\n"
        f"Block episodes: {block_count}\n"
        f"Team A wins: {team_a}\n"
        f"Team B wins: {team_b}\n"
        f"Team A win rate: {team_a_rate:.1f}%\n"
        f"Team B win rate: {team_b_rate:.1f}%\n"
        f"Avg episode length: {avg_len:.1f}\n"
        f"Illegal actions: {illegal_actions}\n"
        f"Policy loss: {policy_loss:.4f} | Value loss: {value_loss:.4f} | Entropy: {entropy:.4f}"
    )
    headers = (
        "progress_pct",
        "episodes_completed",
        "block_episodes",
        "team_a_wins",
        "team_b_wins",
        "team_a_win_rate",
        "team_b_win_rate",
        "avg_episode_length",
        "illegal_actions",
        "policy_loss",
        "value_loss",
        "entropy",
        "sample_1",
        "sample_2",
        "sample_3",
    )
    row = {
        "progress_pct": progress_pct,
        "episodes_completed": episodes_done,
        "block_episodes": block_count,
        "team_a_wins": team_a,
        "team_b_wins": team_b,
        "team_a_win_rate": round(team_a_rate, 2),
        "team_b_win_rate": round(team_b_rate, 2),
        "avg_episode_length": round(avg_len, 2),
        "illegal_actions": illegal_actions,
        "policy_loss": round(policy_loss, 6),
        "value_loss":  round(value_loss,  6),
        "entropy":     round(entropy,     6),
        "sample_1": sample_traces[0] if len(sample_traces) > 0 else "",
        "sample_2": sample_traces[1] if len(sample_traces) > 1 else "",
        "sample_3": sample_traces[2] if len(sample_traces) > 2 else "",
    }
    write_csv_row(csv_path, headers, row)


def save_checkpoint(path: Path, trainer, ep: int, totals: dict):
    """Save full training state so we can resume later."""
    torch.save({
        "episode": ep,
        "policy_state_dict": trainer.policy.state_dict(),
        "critic_state_dict": trainer.critic.state_dict(),
        "optimizer_pi_state_dict": trainer.optimizer_pi.state_dict(),
        "optimizer_v_state_dict": trainer.optimizer_v.state_dict(),
        "totals": {
            "team_a": totals["team_a"],
            "team_b": totals["team_b"],
            "illegal": totals["illegal"],
            # lengths list can be huge; save just the count to keep file small
            "lengths_count": len(totals["lengths"]),
            "lengths_sum": sum(totals["lengths"]),
        },
    }, path)
    print(f"[CHECKPOINT] Saved to {path} (episode {ep})")


def load_checkpoint(path: Path, trainer):
    """Load training state in-place. Returns (episode, totals) tuple."""
    ckpt = torch.load(path, map_location=trainer.device, weights_only=False)
    trainer.policy.load_state_dict(ckpt["policy_state_dict"])
    trainer.critic.load_state_dict(ckpt["critic_state_dict"])
    trainer.optimizer_pi.load_state_dict(ckpt["optimizer_pi_state_dict"])
    trainer.optimizer_v.load_state_dict(ckpt["optimizer_v_state_dict"])
    ep = ckpt["episode"]
    raw = ckpt["totals"]
    # Reconstruct totals; lengths list is approximated from saved sum/count
    avg_len = raw["lengths_sum"] / raw["lengths_count"] if raw["lengths_count"] > 0 else 0.0
    totals = {
        "team_a": raw["team_a"],
        "team_b": raw["team_b"],
        "illegal": raw["illegal"],
        "lengths": [avg_len] * raw["lengths_count"],
    }
    print(f"[CHECKPOINT] Resumed from {path} at episode {ep}")
    return ep, totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint_latest.pt in the run directory if it exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    requested_device = cfg.get("device", "cpu").lower()
    prefer_cuda = requested_device in ["cuda", "gpu"]
    device = get_device(prefer_cuda)

    # OPTIMISATION: faster float32 matmul on GPUs that support it
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"[DEVICE] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    trainer, env = build_trainer(cfg, device)

    total_episodes = cfg["training"]["episodes"]
    ckpt_1_3 = total_episodes // 3
    ckpt_2_3 = (total_episodes * 2) // 3
    block_size = 10
    exp_name = cfg["training"].get("exp_name", "default_run")
    run_dir = Path("runs") / exp_name
    ensure_dir(run_dir)
    csv_path = run_dir / "training_summary.csv"

    # How often (in episodes) to save a resumable checkpoint
    checkpoint_interval = cfg["training"].get("checkpoint_interval", 1000)
    latest_ckpt_path = run_dir / "checkpoint_latest.pt"

    totals = {"team_a": 0, "team_b": 0, "lengths": [], "illegal": 0}
    block_stats = {"team_a": 0, "team_b": 0, "lengths": [], "illegal": 0, "count": 0, "traces": [],
                   "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}

    ep = 0
    num_envs = getattr(env, "num_envs", 1)

    # ── Curriculum training setup ──────────────────────────────────────────────
    curriculum_cfg = cfg.get("curriculum", {})
    curriculum_enabled = curriculum_cfg.get("enabled", False)
    phase1_threshold     = curriculum_cfg.get("phase1_win_rate_threshold", 0.65)
    phase1_window        = curriculum_cfg.get("phase1_window", 500)
    frozen_update_interval = curriculum_cfg.get("frozen_update_interval", 2000)
    curriculum_phase = 1
    win_rate_window: deque = deque(maxlen=phase1_window)
    last_frozen_update_ep = 0

    if curriculum_enabled:
        trainer.opponent_mode = "random"
        print(
            f"[CURRICULUM] Phase 1 — Training against random opponents.\n"
            f"             Will switch to frozen-policy opponents when Team A "
            f"win rate >= {phase1_threshold:.0%} over {phase1_window} episodes."
        )

    # ── Resume from checkpoint if requested (or auto-detect) ──────────────────
    if latest_ckpt_path.exists() and args.resume:
        ep, totals = load_checkpoint(latest_ckpt_path, trainer)
    elif latest_ckpt_path.exists():
        print(f"[CHECKPOINT] Found {latest_ckpt_path}. Run with --resume to continue from episode {torch.load(latest_ckpt_path, map_location='cpu')['episode']}.")

    while ep < total_episodes:
        transitions, infos = trainer.collect_episode(env)
        losses = trainer.update(transitions)
        trainer.anneal_lr(ep / total_episodes)

        # Accumulate losses for block logging
        block_stats["policy_loss"] += losses.get("policy_loss", 0.0)
        block_stats["value_loss"]  += losses.get("value_loss",  0.0)
        block_stats["entropy"]     += losses.get("entropy",     0.0)
        block_stats["updates"]     += 1

        if not isinstance(infos, list):
            infos = [infos]

        for info in infos:
            winner = info.get("winner_team", -1)
            if winner == 0:
                totals["team_a"] += 1
                block_stats["team_a"] += 1
            elif winner == 1:
                totals["team_b"] += 1
                block_stats["team_b"] += 1
            
            length = info.get("episode_length", 0)
            totals["lengths"].append(length)
            block_stats["lengths"].append(length)
            
            illegal = info.get("illegal_actions", 0)
            totals["illegal"] += illegal
            block_stats["illegal"] += illegal
            
            block_stats["count"] += 1
            trace = info.get("match_trace")
            if trace and len(block_stats["traces"]) < 3:
                block_stats["traces"].append(trace)

            if block_stats["count"] >= block_size or ep + block_stats["count"] >= total_episodes:
                progress = int(((ep + block_stats["count"]) / total_episodes) * 100)
                n_updates = max(block_stats["updates"], 1)
                avg_losses = {
                    "policy_loss": block_stats["policy_loss"] / n_updates,
                    "value_loss":  block_stats["value_loss"]  / n_updates,
                    "entropy":     block_stats["entropy"]     / n_updates,
                }
                log_block(
                    progress,
                    min(ep + block_stats["count"], total_episodes),
                    total_episodes,
                    block_stats["count"],
                    block_stats["team_a"],
                    block_stats["team_b"],
                    block_stats["lengths"],
                    block_stats["illegal"],
                    csv_path,
                    block_stats["traces"],
                    losses=avg_losses,
                )
                block_stats = {"team_a": 0, "team_b": 0, "lengths": [], "illegal": 0, "count": 0, "traces": [],
                               "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}

        # BUG FIX (Bug 3): increment by actual completed episodes, not assumed parallel count.
        # With num_envs=16, ep jumped by 16 each iteration regardless of how many episodes
        # finished — causing milestone checkpoints to be skipped entirely.
        ep_step = len(infos)
        ep += ep_step

        # ── Curriculum phase management ────────────────────────────────────────
        if curriculum_enabled:
            for info in infos:
                winner = info.get("winner_team", -1)
                if winner != -1:
                    win_rate_window.append(1 if winner == 0 else 0)

            if curriculum_phase == 1 and len(win_rate_window) >= phase1_window:
                rolling_wr = sum(win_rate_window) / len(win_rate_window)
                if rolling_wr >= phase1_threshold:
                    curriculum_phase = 2
                    trainer.set_frozen_policy()
                    trainer.opponent_mode = "frozen"
                    last_frozen_update_ep = ep
                    print(
                        f"[CURRICULUM] Phase 2 — Switched to frozen-policy opponents "
                        f"at episode {ep} (rolling win rate: {rolling_wr:.1%}).\n"
                        f"             Frozen policy will refresh every {frozen_update_interval} episodes."
                    )

            elif curriculum_phase == 2 and (ep - last_frozen_update_ep) >= frozen_update_interval:
                trainer.set_frozen_policy()
                last_frozen_update_ep = ep

        if ep >= ckpt_1_3 and (ep - ep_step) < ckpt_1_3:
            torch.save(trainer.policy.state_dict(), run_dir / "policy_1_3.pt")
            torch.save(trainer.critic.state_dict(), run_dir / "critic_1_3.pt")
            print(f"Saved 1/3 checkpoint at episode {ep}")
        elif ep >= ckpt_2_3 and (ep - ep_step) < ckpt_2_3:
            torch.save(trainer.policy.state_dict(), run_dir / "policy_2_3.pt")
            torch.save(trainer.critic.state_dict(), run_dir / "critic_2_3.pt")
            print(f"Saved 2/3 checkpoint at episode {ep}")

        # ── Periodic plot save (every 5000 episodes) ──────────────────────────
        prev_plot = (ep - ep_step) // 5000
        curr_plot = ep // 5000
        if curr_plot > prev_plot:
            plot_training(csv_path, run_dir)

        # ── Periodic resumable checkpoint ──────────────────────────────────────
        prev_interval = (ep - ep_step) // checkpoint_interval
        curr_interval = ep // checkpoint_interval
        if curr_interval > prev_interval:
            save_checkpoint(latest_ckpt_path, trainer, ep, totals)

    # Final summary
    total_len = sum(totals["lengths"]) / len(totals["lengths"]) if totals["lengths"] else 0.0
    print(
        "[TRAINING COMPLETE]\n"
        f"Total episodes: {total_episodes}\n"
        f"Team A total wins: {totals['team_a']}\n"
        f"Team B total wins: {totals['team_b']}\n"
        f"Team A win rate: {(totals['team_a'] / total_episodes) * 100:.1f}%\n"
        f"Team B win rate: {(totals['team_b'] / total_episodes) * 100:.1f}%\n"
        f"Avg episode length: {total_len:.1f}\n"
        f"Illegal actions: {totals['illegal']}"
    )

    # Save latest weights
    torch.save(trainer.policy.state_dict(), run_dir / "policy_last.pt")
    torch.save(trainer.critic.state_dict(), run_dir / "critic_last.pt")
    
    # Save the 3/3 checkpoint
    torch.save(trainer.policy.state_dict(), run_dir / "policy_3_3.pt")
    torch.save(trainer.critic.state_dict(), run_dir / "critic_3_3.pt")


if __name__ == "__main__":
    main()
