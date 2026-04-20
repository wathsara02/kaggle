import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from omi_env import rules, encoding
from utils import build_policy, ensure_dir, get_device, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Trained policy weights path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override export output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "cpu") == "cuda")
    policy, obs_dim, history_dim = build_policy(cfg, device)
    policy.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    policy.eval()

    out_dir = Path(args.output_dir or cfg.get("export", {}).get("output_dir", "artifacts"))
    ensure_dir(out_dir)
    torch.save(policy.state_dict(), out_dir / "policy_agent.pt")

    config_dump = {
        "obs_dim": obs_dim,
        "history_dim": history_dim,
        "action_dim": rules.ACTION_DIM,
        "model": cfg["model"],
        "algo": cfg["algo"],
        "env": {"name": "omi_v0"},
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dump, f, indent=2)

    with open(out_dir / "VERSION", "w") as f:
        f.write("1.0.0\n")

    print(f"Exported artifacts to {out_dir}")


if __name__ == "__main__":
    main()
