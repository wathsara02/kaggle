import argparse
import math
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

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
from functools import partial

# CUDA speed settings.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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

    # Compile only on CUDA.
    if device.type == "cuda":
        try:
            policy = torch.compile(policy, mode="reduce-overhead")
            critic = torch.compile(critic, mode="reduce-overhead")
            print("[COMPILE] torch.compile applied to policy and critic.")
        except Exception as e:
            print(f"[COMPILE] torch.compile skipped: {e}")

    trainer = MAPPOTrainer(policy, critic, cfg["algo"], device)
    return trainer, env


SHAPING_EVENT_KEYS = (
    "partner_save",
    "trump_cut",
    "wasted_trump",
    "late_trick",
    "declarer_team_win",
    "declarer_team_loss",
)


def log_block(progress_pct, episodes_done, total_episodes, block_count, team_a, team_b,
              lengths, illegal_actions, csv_path, sample_traces, losses=None, shaping_events=None):
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
        "partner_save_events",
        "trump_cut_events",
        "wasted_trump_events",
        "late_trick_events",
        "declarer_team_win_events",
        "declarer_team_loss_events",
        "sample_1",
        "sample_2",
        "sample_3",
    )
    shaping_events = shaping_events or {}
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
        "partner_save_events": shaping_events.get("partner_save", 0),
        "trump_cut_events": shaping_events.get("trump_cut", 0),
        "wasted_trump_events": shaping_events.get("wasted_trump", 0),
        "late_trick_events": shaping_events.get("late_trick", 0),
        "declarer_team_win_events": shaping_events.get("declarer_team_win", 0),
        "declarer_team_loss_events": shaping_events.get("declarer_team_loss", 0),
        "sample_1": sample_traces[0] if len(sample_traces) > 0 else "",
        "sample_2": sample_traces[1] if len(sample_traces) > 1 else "",
        "sample_3": sample_traces[2] if len(sample_traces) > 2 else "",
    }
    write_csv_row(csv_path, headers, row)


def save_checkpoint(path: Path, trainer, ep: int, totals: dict):
    """Save full training state so we can resume later."""
    policy_state = model_state_dict(trainer.policy)
    critic_state = model_state_dict(trainer.critic)
    torch.save({
        "episode": ep,
        "policy_state_dict": policy_state,
        "critic_state_dict": critic_state,
        "optimizer_pi_state_dict": trainer.optimizer_pi.state_dict(),
        "optimizer_v_state_dict": trainer.optimizer_v.state_dict(),
        "totals": {
            "team_a": totals["team_a"],
            "team_b": totals["team_b"],
            "illegal": totals["illegal"],
            # Save summary only; the full list can be large.
            "lengths_count": len(totals["lengths"]),
            "lengths_sum": sum(totals["lengths"]),
        },
    }, path)
    torch.save(policy_state, path.with_name("policy_last.pt"))
    torch.save(critic_state, path.with_name("critic_last.pt"))
    print(f"[CHECKPOINT] Saved to {path} (episode {ep})")


def model_state_dict(model):
    """Return a loadable state dict, unwrapping torch.compile modules if needed."""
    return getattr(model, "_orig_mod", model).state_dict()


class TrainingProgress:
    def __init__(self, total_episodes: int, start_episode: int = 0):
        self.total_episodes = max(1, total_episodes)
        self.start_episode = start_episode
        self.start_time = time.time()
        self._last_len = 0

    def _format_eta(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m"
        if minutes:
            return f"{minutes:d}m {secs:02d}s"
        return f"{secs:d}s"

    def update(self, episodes_done: int, totals: dict, losses: Optional[dict] = None) -> None:
        fraction = min(max(episodes_done / self.total_episodes, 0.0), 1.0)
        filled = int(30 * fraction)
        bar = "#" * filled + "-" * (30 - filled)

        elapsed = time.time() - self.start_time
        completed_since_start = episodes_done - self.start_episode
        eps_per_sec = completed_since_start / max(elapsed, 1e-9) if completed_since_start > 0 else 0.0
        remaining = max(self.total_episodes - episodes_done, 0)
        eta = remaining / eps_per_sec if eps_per_sec > 0 else 0.0

        decisive = totals.get("team_a", 0) + totals.get("team_b", 0)
        team_a_rate = (totals.get("team_a", 0) / decisive * 100.0) if decisive else 0.0
        loss_text = ""
        if losses:
            loss_text = (
                f" | pi {losses.get('policy_loss', 0.0):.3f}"
                f" v {losses.get('value_loss', 0.0):.3f}"
                f" H {losses.get('entropy', 0.0):.3f}"
            )

        line = (
            f"\r[{bar}] {fraction * 100:6.2f}% "
            f"{episodes_done:,}/{self.total_episodes:,} eps "
            f"| {eps_per_sec:6.1f} eps/s "
            f"| ETA {self._format_eta(eta)} "
            f"| TeamA {team_a_rate:5.1f}%"
            f"{loss_text}"
        )
        padding = " " * max(0, self._last_len - len(line))
        print(line + padding, end="", flush=True)
        self._last_len = len(line)

    def newline(self) -> None:
        if self._last_len:
            print()
            self._last_len = 0


class AsyncEvalManager:
    """Launch periodic baseline evaluations without blocking the training loop."""

    def __init__(self, cfg: dict, config_path: str, run_dir: Path):
        eval_cfg = cfg.get("eval", {})
        during_cfg = eval_cfg.get("during_training", {})
        self.enabled = bool(during_cfg.get("enabled", False))
        self.interval = max(1, int(during_cfg.get("interval_episodes", during_cfg.get("interval", 10000))))
        self.episodes = int(during_cfg.get("episodes", eval_cfg.get("episodes", 200)))
        self.baseline = during_cfg.get("baseline", eval_cfg.get("baseline", "rule"))
        self.deterministic = bool(during_cfg.get("deterministic", eval_cfg.get("deterministic", True)))
        self.device = during_cfg.get("device", "cpu")
        self.max_parallel = max(1, int(during_cfg.get("max_parallel", 1)))
        self.keep_eval_weights = bool(during_cfg.get("keep_eval_weights", True))
        self.record_match_traces = bool(during_cfg.get("record_match_traces", False))
        self.wait_on_finish = bool(during_cfg.get("wait_on_finish", False))
        self.config_path = config_path
        self.run_dir = run_dir
        self.eval_root = run_dir / "baseline_evals"
        self.aggregate_csv = self.eval_root / "baseline_eval_results.csv"
        self.active = []

    def _prune(self):
        still_running = []
        for episode, proc, log_path, weights_path in self.active:
            return_code = proc.poll()
            if return_code is None:
                still_running.append((episode, proc, log_path, weights_path))
                continue
            status = "finished" if return_code == 0 else f"failed with exit code {return_code}"
            print(f"[ASYNC EVAL] Episode {episode} {status}. Log: {log_path}")
            if return_code == 0 and not self.keep_eval_weights:
                try:
                    weights_path.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[ASYNC EVAL] Could not delete {weights_path}: {exc}")
        self.active = still_running

    def maybe_launch(self, previous_ep: int, current_ep: int, trainer) -> None:
        if not self.enabled:
            return
        self._prune()
        previous_interval = previous_ep // self.interval
        current_interval = current_ep // self.interval
        if current_interval <= previous_interval:
            return
        if len(self.active) >= self.max_parallel:
            print(
                f"[ASYNC EVAL] Skipped episode {current_ep}: "
                f"{len(self.active)} eval process(es) already running."
            )
            return

        eval_dir = self.eval_root / f"eval_ep_{current_ep:09d}"
        ensure_dir(eval_dir)
        weights_path = eval_dir / "policy_snapshot.pt"
        torch.save(model_state_dict(trainer.policy), weights_path)
        log_path = eval_dir / "eval.log"
        cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts" / "eval.py"),
            "--config",
            self.config_path,
            "--weights",
            str(weights_path),
            "--episodes",
            str(self.episodes),
            "--baseline",
            self.baseline,
            "--device",
            self.device,
            "--out-dir",
            str(eval_dir),
            "--aggregate-csv",
            str(self.aggregate_csv),
            "--checkpoint-episode",
            str(current_ep),
        ]
        if self.deterministic:
            cmd.append("--deterministic")
        if not self.record_match_traces:
            cmd.append("--no-match-traces")

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        self.active.append((current_ep, proc, log_path, weights_path))
        print(
            f"[ASYNC EVAL] Started baseline eval at episode {current_ep} "
            f"({self.episodes} games vs {self.baseline}). Log: {log_path}"
        )

    def finish(self):
        if not self.enabled:
            return
        self._prune()
        if not self.active:
            return
        if not self.wait_on_finish:
            print(f"[ASYNC EVAL] {len(self.active)} eval process(es) still running.")
            return
        print(f"[ASYNC EVAL] Waiting for {len(self.active)} eval process(es) to finish...")
        for episode, proc, log_path, weights_path in list(self.active):
            return_code = proc.wait()
            status = "finished" if return_code == 0 else f"failed with exit code {return_code}"
            print(f"[ASYNC EVAL] Episode {episode} {status}. Log: {log_path}")
            if return_code == 0 and not self.keep_eval_weights:
                try:
                    weights_path.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[ASYNC EVAL] Could not delete {weights_path}: {exc}")
        self.active = []


def load_checkpoint(path: Path, trainer):
    """Load training state in-place. Returns (episode, totals) tuple."""
    ckpt = torch.load(path, map_location=trainer.device, weights_only=False)
    trainer.policy.load_state_dict(ckpt["policy_state_dict"])
    trainer.critic.load_state_dict(ckpt["critic_state_dict"])
    trainer.optimizer_pi.load_state_dict(ckpt["optimizer_pi_state_dict"])
    trainer.optimizer_v.load_state_dict(ckpt["optimizer_v_state_dict"])
    ep = ckpt["episode"]
    raw = ckpt["totals"]
    # Rebuild length history from saved summary.
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
    parser.add_argument("--config", type=str, default="configs/new.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint_latest.pt in the run directory if it exists")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override training.episodes from the config")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override training.num_envs from the config")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "gpu"],
                        help="Override device from the config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["training"]["episodes"] = args.episodes
    if args.num_envs is not None:
        cfg["training"]["num_envs"] = args.num_envs
    if args.device is not None:
        cfg["device"] = args.device
    set_seed(cfg["seed"])
    requested_device = cfg.get("device", "cpu").lower()
    prefer_cuda = requested_device in ["cuda", "gpu"]
    device = get_device(prefer_cuda)

    # Use faster float32 matmul when available.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"[DEVICE] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    trainer, env = build_trainer(cfg, device)

    total_episodes = cfg["training"]["episodes"]
    ckpt_1_3 = total_episodes // 3
    ckpt_2_3 = (total_episodes * 2) // 3
    block_size = max(1, int(cfg["training"].get("log_interval", 10)))
    exp_name = cfg["training"].get("exp_name", "default_run")
    run_dir = Path("runs") / exp_name
    ensure_dir(run_dir)
    eval_manager = AsyncEvalManager(cfg, args.config, run_dir)
    csv_path = run_dir / "training_summary.csv"
    match_csv_path = run_dir / "match_traces.csv"
    logging_cfg = cfg.get("logging", {})
    record_matches = logging_cfg.get("record_matches", False)
    record_every = max(1, int(logging_cfg.get("record_every", 1)))
    max_recorded_matches = int(logging_cfg.get("max_recorded_matches", 0))
    recorded_matches = 0

    # Save resumable checkpoints on this interval.
    checkpoint_interval = cfg["training"].get("checkpoint_interval", 1000)
    latest_ckpt_path = run_dir / "checkpoint_latest.pt"

    totals = {"team_a": 0, "team_b": 0, "lengths": [], "illegal": 0}
    block_stats = {"team_a": 0, "team_b": 0, "lengths": [], "illegal": 0, "count": 0, "traces": [],
                   "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0,
                   "shaping_events": {key: 0 for key in SHAPING_EVENT_KEYS}}

    ep = 0
    num_envs = getattr(env, "num_envs", 1)

    # Curriculum setup.
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

    # Resume if requested.
    if latest_ckpt_path.exists() and args.resume:
        ep, totals = load_checkpoint(latest_ckpt_path, trainer)
    elif latest_ckpt_path.exists():
        print(f"[CHECKPOINT] Found {latest_ckpt_path}. Run with --resume to continue from episode {torch.load(latest_ckpt_path, map_location='cpu')['episode']}.")

    progress_bar = TrainingProgress(total_episodes, start_episode=ep)
    progress_bar.update(ep, totals)

    while ep < total_episodes:
        transitions, infos = trainer.collect_episode(env)
        losses = trainer.update(transitions)
        trainer.anneal_lr(ep / total_episodes)

        block_stats["policy_loss"] += losses.get("policy_loss", 0.0)
        block_stats["value_loss"]  += losses.get("value_loss",  0.0)
        block_stats["entropy"]     += losses.get("entropy",     0.0)
        block_stats["updates"]     += 1

        if not isinstance(infos, list):
            infos = [infos]
        remaining_episodes = max(total_episodes - ep, 0)
        if len(infos) > remaining_episodes:
            infos = infos[:remaining_episodes]

        for info_idx, info in enumerate(infos, start=1):
            episodes_done = min(ep + info_idx, total_episodes)
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
            shaping_events = info.get("shaping_events", {})
            for key in SHAPING_EVENT_KEYS:
                block_stats["shaping_events"][key] += int(shaping_events.get(key, 0))

            if (
                record_matches
                and trace
                and episodes_done % record_every == 0
                and (max_recorded_matches <= 0 or recorded_matches < max_recorded_matches)
            ):
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
                    "episode": episodes_done,
                    "winner_team": winner,
                    "final_score": info.get("final_score", ""),
                    "episode_length": length,
                    "illegal_actions": illegal,
                    "partner_save_events": shaping_events.get("partner_save", 0),
                    "trump_cut_events": shaping_events.get("trump_cut", 0),
                    "wasted_trump_events": shaping_events.get("wasted_trump", 0),
                    "late_trick_events": shaping_events.get("late_trick", 0),
                    "declarer_team_win_events": shaping_events.get("declarer_team_win", 0),
                    "declarer_team_loss_events": shaping_events.get("declarer_team_loss", 0),
                    "match_trace": trace,
                })
                recorded_matches += 1

            if block_stats["count"] >= block_size or episodes_done >= total_episodes:
                progress_bar.newline()
                progress = int((episodes_done / total_episodes) * 100)
                n_updates = max(block_stats["updates"], 1)
                avg_losses = {
                    "policy_loss": block_stats["policy_loss"] / n_updates,
                    "value_loss":  block_stats["value_loss"]  / n_updates,
                    "entropy":     block_stats["entropy"]     / n_updates,
                }
                log_block(
                    progress,
                    episodes_done,
                    total_episodes,
                    block_stats["count"],
                    block_stats["team_a"],
                    block_stats["team_b"],
                    block_stats["lengths"],
                    block_stats["illegal"],
                    csv_path,
                    block_stats["traces"],
                    losses=avg_losses,
                    shaping_events=block_stats["shaping_events"],
                )
                block_stats = {"team_a": 0, "team_b": 0, "lengths": [], "illegal": 0, "count": 0, "traces": [],
                               "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0,
                               "shaping_events": {key: 0 for key in SHAPING_EVENT_KEYS}}

        ep_step = len(infos)
        ep += ep_step
        progress_bar.update(ep, totals, losses)

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
            progress_bar.newline()
            torch.save(model_state_dict(trainer.policy), run_dir / "policy_1_3.pt")
            torch.save(model_state_dict(trainer.critic), run_dir / "critic_1_3.pt")
            print(f"Saved 1/3 checkpoint at episode {ep}")
        elif ep >= ckpt_2_3 and (ep - ep_step) < ckpt_2_3:
            progress_bar.newline()
            torch.save(model_state_dict(trainer.policy), run_dir / "policy_2_3.pt")
            torch.save(model_state_dict(trainer.critic), run_dir / "critic_2_3.pt")
            print(f"Saved 2/3 checkpoint at episode {ep}")

        prev_plot = (ep - ep_step) // 5000
        curr_plot = ep // 5000
        if curr_plot > prev_plot:
            progress_bar.newline()
            try:
                from scripts.plot_training import plot_training
                plot_training(csv_path, run_dir)
            except ImportError as exc:
                print(f"[PLOT] Skipped plot update: {exc}")

        prev_interval = (ep - ep_step) // checkpoint_interval
        curr_interval = ep // checkpoint_interval
        if curr_interval > prev_interval:
            progress_bar.newline()
            save_checkpoint(latest_ckpt_path, trainer, ep, totals)

        eval_manager.maybe_launch(ep - ep_step, ep, trainer)

    eval_manager.finish()
    progress_bar.newline()
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

    torch.save(model_state_dict(trainer.policy), run_dir / "policy_last.pt")
    torch.save(model_state_dict(trainer.critic), run_dir / "critic_last.pt")
    
    torch.save(model_state_dict(trainer.policy), run_dir / "policy_3_3.pt")
    torch.save(model_state_dict(trainer.critic), run_dir / "critic_3_3.pt")

if __name__ == "__main__":
    main()
