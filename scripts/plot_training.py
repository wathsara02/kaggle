"""
Generate training progress graphs from a training_summary.csv file.

Can be called:
  - Automatically during training (called by train.py after every log block)
  - Standalone: python scripts/plot_training.py --csv runs/<exp>/training_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for training loops.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _save_figure(fig, out_path: Path):
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved -> {out_path}")
    return out_path


def _plot_win_rate(ax, episodes, df, smooth_window: int):
    a_rate = df["team_a_win_rate"].values
    b_rate = df["team_b_win_rate"].values if "team_b_win_rate" in df.columns else 100.0 - a_rate

    ax.plot(episodes, a_rate, color="steelblue", alpha=0.25, linewidth=1)
    ax.plot(episodes, b_rate, color="tomato", alpha=0.25, linewidth=1)

    if len(df) >= smooth_window:
        sm_ep = episodes[smooth_window - 1:]
        ax.plot(sm_ep, _smooth(a_rate, smooth_window),
                color="steelblue", linewidth=2.5, label="Team A (learned)")
        ax.plot(sm_ep, _smooth(b_rate, smooth_window),
                color="tomato", linewidth=2.5, label="Team B")
    else:
        ax.plot(episodes, a_rate, color="steelblue", linewidth=1.5, label="Team A (learned)")
        ax.plot(episodes, b_rate, color="tomato", linewidth=1.5, label="Team B")

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="50% chance")
    ax.set_title("Win Rate over Training")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_losses(ax, episodes, df, smooth_window: int):
    has_losses = "policy_loss" in df.columns and df["policy_loss"].notna().any()
    if not has_losses:
        ax.text(0.5, 0.5, "Loss data not in CSV.\nRe-run training with updated train.py.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("Training Losses")
        return

    p_loss = df["policy_loss"].values.astype(float)
    v_loss = df["value_loss"].values.astype(float)

    ax.plot(episodes, p_loss, color="royalblue", alpha=0.25, linewidth=1)
    ax.plot(episodes, v_loss, color="darkorange", alpha=0.25, linewidth=1)

    if len(df) >= smooth_window:
        sm_ep = episodes[smooth_window - 1:]
        ax.plot(sm_ep, _smooth(p_loss, smooth_window),
                color="royalblue", linewidth=2.5, label="Policy loss")
        ax.plot(sm_ep, _smooth(v_loss, smooth_window),
                color="darkorange", linewidth=2.5, label="Value loss")
    else:
        ax.plot(episodes, p_loss, color="royalblue", linewidth=1.5, label="Policy loss")
        ax.plot(episodes, v_loss, color="darkorange", linewidth=1.5, label="Value loss")

    ax.set_title("Training Losses")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_entropy(ax, episodes, df, smooth_window: int):
    has_entropy = "entropy" in df.columns and df["entropy"].notna().any()
    if not has_entropy:
        ax.text(0.5, 0.5, "Entropy data not in CSV.\nRe-run training with updated train.py.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("Policy Entropy")
        return

    entropy = df["entropy"].values.astype(float)
    ax.plot(episodes, entropy, color="seagreen", alpha=0.25, linewidth=1)
    if len(df) >= smooth_window:
        ax.plot(episodes[smooth_window - 1:], _smooth(entropy, smooth_window),
                color="seagreen", linewidth=2.5, label="Entropy")
    else:
        ax.plot(episodes, entropy, color="seagreen", linewidth=1.5, label="Entropy")
    ax.set_title("Policy Entropy")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Entropy (nats)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_illegal_actions(ax, episodes, df, smooth_window: int):
    illegal = df["illegal_actions"].values.astype(float)
    block_eps = df["block_episodes"].values.astype(float)
    illegal_rate = np.where(block_eps > 0, illegal / block_eps, 0.0)

    ax.bar(episodes, illegal_rate, width=np.diff(np.append([0], episodes)).clip(1),
           color="salmon", alpha=0.6, label="Illegal actions / episode")

    if len(df) >= smooth_window:
        ax.plot(episodes[smooth_window - 1:], _smooth(illegal_rate, smooth_window),
                color="darkred", linewidth=2.5, label="Smoothed")

    ax.set_title("Illegal Actions per Episode")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Illegal actions / episode")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_shaping_events(ax, episodes, df, smooth_window: int):
    block_eps = df["block_episodes"].values.astype(float)
    event_cols = [
        ("partner_save_events", "Partner saves"),
        ("trump_cut_events", "Trump cuts"),
        ("wasted_trump_events", "Wasted trump"),
        ("late_trick_events", "Late tricks"),
        ("declarer_team_win_events", "Declarer team wins"),
        ("declarer_team_loss_events", "Declarer team losses"),
    ]
    plotted_events = False
    for col, label in event_cols:
        if col in df.columns:
            values = df[col].fillna(0).values.astype(float)
            rates = np.where(block_eps > 0, values / block_eps, 0.0)
            if len(df) >= smooth_window:
                ax.plot(episodes[smooth_window - 1:], _smooth(rates, smooth_window),
                        linewidth=2, label=label)
            else:
                ax.plot(episodes, rates, linewidth=1.5, label=label)
            plotted_events = True

    if plotted_events:
        ax.set_title("Reward-Shaping Events per Episode")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Events / episode")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Reward-shaping event data not in CSV.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("Reward-Shaping Events")


def _save_individual_training_plots(df, episodes, smooth_window: int, out_dir: Path):
    plots = [
        ("training_win_rate.png", _plot_win_rate),
        ("training_losses.png", _plot_losses),
        ("training_entropy.png", _plot_entropy),
        ("training_illegal_actions.png", _plot_illegal_actions),
        ("training_reward_shaping_events.png", _plot_shaping_events),
    ]
    saved = []
    for filename, plotter in plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotter(ax, episodes, df, smooth_window)
        saved.append(_save_figure(fig, out_dir / filename))
    return saved


def plot_training(csv_path, out_dir=None):
    """
    Read *csv_path* and write four-panel training_progress.png to *out_dir*
    (defaults to the same directory as the CSV).

    Returns the path of the saved figure.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[PLOT] CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[PLOT] CSV is empty, skipping plot.")
        return None

    out_dir = Path(out_dir) if out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = df["episodes_completed"].values
    smooth_window = max(3, len(df) // 10)  # adapt window to how much data exists

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Omi MARL — Training Progress", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    a_rate = df["team_a_win_rate"].values
    b_rate = df["team_b_win_rate"].values if "team_b_win_rate" in df.columns else 100.0 - a_rate

    ax1.plot(episodes, a_rate, color="steelblue", alpha=0.25, linewidth=1)
    ax1.plot(episodes, b_rate, color="tomato",    alpha=0.25, linewidth=1)

    if len(df) >= smooth_window:
        sm_ep = episodes[smooth_window - 1:]
        ax1.plot(sm_ep, _smooth(a_rate, smooth_window),
                 color="steelblue", linewidth=2.5, label="Team A (learned)")
        ax1.plot(sm_ep, _smooth(b_rate, smooth_window),
                 color="tomato",    linewidth=2.5, label="Team B")

    ax1.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="50% chance")
    ax1.set_title("Win Rate over Training")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    has_losses = "policy_loss" in df.columns and df["policy_loss"].notna().any()

    if has_losses:
        p_loss = df["policy_loss"].values.astype(float)
        v_loss = df["value_loss"].values.astype(float)

        ax2.plot(episodes, p_loss, color="royalblue", alpha=0.25, linewidth=1)
        ax2.plot(episodes, v_loss, color="darkorange", alpha=0.25, linewidth=1)

        if len(df) >= smooth_window:
            sm_ep = episodes[smooth_window - 1:]
            ax2.plot(sm_ep, _smooth(p_loss, smooth_window),
                     color="royalblue",  linewidth=2.5, label="Policy loss")
            ax2.plot(sm_ep, _smooth(v_loss, smooth_window),
                     color="darkorange", linewidth=2.5, label="Value loss")
        else:
            ax2.plot(episodes, p_loss, color="royalblue", linewidth=1.5, label="Policy loss")
            ax2.plot(episodes, v_loss, color="darkorange", linewidth=1.5, label="Value loss")

        ax2.set_title("Training Losses")
        ax2.set_xlabel("Episodes")
        ax2.set_ylabel("Loss")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Loss data not in CSV.\nRe-run training with updated train.py.",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10, color="gray")
        ax2.set_title("Training Losses")

    ax3 = fig.add_subplot(gs[1, 0])
    has_entropy = "entropy" in df.columns and df["entropy"].notna().any()

    if has_entropy:
        entropy = df["entropy"].values.astype(float)
        ax3.plot(episodes, entropy, color="seagreen", alpha=0.25, linewidth=1)
        if len(df) >= smooth_window:
            ax3.plot(episodes[smooth_window - 1:], _smooth(entropy, smooth_window),
                     color="seagreen", linewidth=2.5, label="Entropy")
        else:
            ax3.plot(episodes, entropy, color="seagreen", linewidth=1.5, label="Entropy")
        ax3.set_title("Policy Entropy  (↓ = more deterministic)")
        ax3.set_xlabel("Episodes")
        ax3.set_ylabel("Entropy (nats)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Entropy data not in CSV.\nRe-run training with updated train.py.",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=10, color="gray")
        ax3.set_title("Policy Entropy")

    ax4 = fig.add_subplot(gs[1, 1])
    illegal = df["illegal_actions"].values.astype(float)
    block_eps = df["block_episodes"].values.astype(float)
    # Normalize by block size.
    illegal_rate = np.where(block_eps > 0, illegal / block_eps, 0.0)

    ax4.bar(episodes, illegal_rate, width=np.diff(np.append([0], episodes)).clip(1),
            color="salmon", alpha=0.6, label="Illegal actions / episode")

    if len(df) >= smooth_window:
        ax4.plot(episodes[smooth_window - 1:], _smooth(illegal_rate, smooth_window),
                 color="darkred", linewidth=2.5, label="Smoothed")

    ax4.set_title("Illegal Actions per Episode  (↓ = better masking learned)")
    ax4.set_xlabel("Episodes")
    ax4.set_ylabel("Illegal actions / episode")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, :])
    event_cols = [
        ("partner_save_events", "Partner saves"),
        ("trump_cut_events", "Trump cuts"),
        ("wasted_trump_events", "Wasted trump"),
        ("late_trick_events", "Late tricks"),
        ("declarer_team_win_events", "Declarer team wins"),
        ("declarer_team_loss_events", "Declarer team losses"),
    ]
    plotted_events = False
    for col, label in event_cols:
        if col in df.columns:
            values = df[col].fillna(0).values.astype(float)
            rates = np.where(block_eps > 0, values / block_eps, 0.0)
            if len(df) >= smooth_window:
                ax5.plot(episodes[smooth_window - 1:], _smooth(rates, smooth_window),
                         linewidth=2, label=label)
            else:
                ax5.plot(episodes, rates, linewidth=1.5, label=label)
            plotted_events = True

    if plotted_events:
        ax5.set_title("Reward-Shaping Events per Episode")
        ax5.set_xlabel("Episodes")
        ax5.set_ylabel("Events / episode")
        ax5.legend(fontsize=8, ncol=3)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "Reward-shaping event data not in CSV.",
                 ha="center", va="center", transform=ax5.transAxes, fontsize=10, color="gray")
        ax5.set_title("Reward-Shaping Events")

    out_path = out_dir / "training_progress.png"
    _save_figure(fig, out_path)
    _save_individual_training_plots(df, episodes, smooth_window, out_dir)
    return out_path


def plot_evaluation(csv_path, out_dir=None):
    """
    Plot one evaluation run from evaluation_summary.csv.

    This is useful for each background checkpoint evaluation directory.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[PLOT] Eval CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[PLOT] Eval CSV is empty, skipping plot.")
        return None

    out_dir = Path(out_dir) if out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = df["episodes_completed"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, df["agent_win_rate"].values.astype(float),
            marker="o", linewidth=2.5, label="Learned agent")
    ax.plot(episodes, df["baseline_win_rate"].values.astype(float),
            marker="o", linewidth=2.0, label="Baseline")
    ax.plot(episodes, df["draw_rate"].values.astype(float),
            marker="o", linewidth=1.8, label="Draws")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Evaluation vs Baseline")
    ax.set_xlabel("Evaluation episodes")
    ax.set_ylabel("Outcome rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out_path = _save_figure(fig, out_dir / "evaluation_win_rates.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    illegal = df["illegal_actions"].values.astype(float)
    block_eps = df["block_episodes"].values.astype(float)
    illegal_rate = np.where(block_eps > 0, illegal / block_eps, 0.0)
    ax.bar(episodes, illegal_rate, width=np.diff(np.append([0], episodes)).clip(1),
           color="salmon", alpha=0.7)
    ax.set_title("Evaluation Illegal Actions per Episode")
    ax.set_xlabel("Evaluation episodes")
    ax.set_ylabel("Illegal actions / episode")
    ax.grid(True, alpha=0.3)
    _save_figure(fig, out_dir / "evaluation_illegal_actions.png")

    return out_path


def plot_baseline_evals(csv_path, out_dir=None):
    """
    Plot aggregate periodic evaluation results written by scripts/eval.py.

    The CSV is expected to contain one row per checkpoint evaluation.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[PLOT] Eval CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[PLOT] Eval CSV is empty, skipping plot.")
        return None

    out_dir = Path(out_dir) if out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    x_col = "checkpoint_episode" if "checkpoint_episode" in df.columns else "eval_index"
    if x_col == "eval_index":
        df = df.copy()
        df["eval_index"] = np.arange(1, len(df) + 1)
    x = df[x_col].values

    fig, ax = plt.subplots(figsize=(10, 6))
    agent_rate = df["agent_win_rate"].values.astype(float)
    baseline_rate = df["baseline_win_rate"].values.astype(float)
    draw_rate = df["draw_rate"].values.astype(float)
    ax.plot(x, agent_rate, marker="o", linewidth=2.5, label="Learned agent")
    ax.plot(x, baseline_rate, marker="o", linewidth=2.0, label="Baseline")
    ax.plot(x, draw_rate, marker="o", linewidth=1.8, label="Draws")
    if "ci_low" in df.columns and "ci_high" in df.columns:
        ax.fill_between(
            x,
            df["ci_low"].values.astype(float) * 100.0,
            df["ci_high"].values.astype(float) * 100.0,
            color="steelblue",
            alpha=0.15,
            label="Learned decisive 95% CI",
        )
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Periodic Evaluation vs Baseline")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Outcome rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out_path = out_dir / "baseline_eval_over_time.png"
    _save_figure(fig, out_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, agent_rate, marker="o", linewidth=2.5, color="steelblue")
    if "ci_low" in df.columns and "ci_high" in df.columns:
        ax.fill_between(
            x,
            df["ci_low"].values.astype(float) * 100.0,
            df["ci_high"].values.astype(float) * 100.0,
            color="steelblue",
            alpha=0.15,
        )
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Learned Agent Win Rate vs Baseline")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    _save_figure(fig, out_dir / "baseline_eval_agent_win_rate.png")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Omi training progress from CSV.")
    parser.add_argument("--csv",  type=str, required=True, help="Path to training_summary.csv")
    parser.add_argument("--out",  type=str, default=None,  help="Output directory (default: CSV directory)")
    parser.add_argument("--eval-csv", type=str, default=None, help="Optional aggregate baseline eval CSV")
    args = parser.parse_args()
    plot_training(args.csv, args.out)
    if args.eval_csv:
        plot_baseline_evals(args.eval_csv, args.out)
