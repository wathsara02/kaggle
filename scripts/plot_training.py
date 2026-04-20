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
matplotlib.use("Agg")  # non-interactive backend — safe for training loops
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


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

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Omi MARL — Training Progress", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── 1. Win Rate ───────────────────────────────────────────────────────────
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

    # ── 2. Policy & Value Loss ────────────────────────────────────────────────
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

        ax2.set_title("Training Losses")
        ax2.set_xlabel("Episodes")
        ax2.set_ylabel("Loss")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Loss data not in CSV.\nRe-run training with updated train.py.",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10, color="gray")
        ax2.set_title("Training Losses")

    # ── 3. Policy Entropy ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    has_entropy = "entropy" in df.columns and df["entropy"].notna().any()

    if has_entropy:
        entropy = df["entropy"].values.astype(float)
        ax3.plot(episodes, entropy, color="seagreen", alpha=0.25, linewidth=1)
        if len(df) >= smooth_window:
            ax3.plot(episodes[smooth_window - 1:], _smooth(entropy, smooth_window),
                     color="seagreen", linewidth=2.5, label="Entropy")
        ax3.set_title("Policy Entropy  (↓ = more deterministic)")
        ax3.set_xlabel("Episodes")
        ax3.set_ylabel("Entropy (nats)")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Entropy data not in CSV.\nRe-run training with updated train.py.",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=10, color="gray")
        ax3.set_title("Policy Entropy")

    # ── 4. Illegal Actions ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    illegal = df["illegal_actions"].values.astype(float)
    block_eps = df["block_episodes"].values.astype(float)
    # Express as rate per episode so blocks of different sizes are comparable
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

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = out_dir / "training_progress.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Omi training progress from CSV.")
    parser.add_argument("--csv",  type=str, required=True, help="Path to training_summary.csv")
    parser.add_argument("--out",  type=str, default=None,  help="Output directory (default: CSV directory)")
    args = parser.parse_args()
    plot_training(args.csv, args.out)
