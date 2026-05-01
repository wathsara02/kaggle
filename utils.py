from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = False) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path):
    """Create *path* (and any missing parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def write_csv_row(path: str, headers: Tuple[str, ...], row: Dict[str, object]):
    file_exists = os.path.exists(path)
    if file_exists:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_headers = next(reader, None)
        if existing_headers != list(headers):
            base = Path(path)
            idx = 1
            backup = base.with_name(f"{base.stem}.legacy_{idx}{base.suffix}")
            while backup.exists():
                idx += 1
                backup = base.with_name(f"{base.stem}.legacy_{idx}{base.suffix}")
            os.replace(path, backup)
            file_exists = False
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def masked_sample(logits: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
    logits = logits + (1.0 - mask) * -1e9
    probs = torch.softmax(logits, dim=-1)
    if deterministic:
        return torch.argmax(probs, dim=-1), probs
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action, probs


def bootstrap_confidence_interval(data, num_bootstrap: int = 2000, alpha: float = 0.05):
    data = np.array(data)
    n = len(data)
    if n == 0:
        return (0.0, 0.0)
    samples = []
    for _ in range(num_bootstrap):
        idx = np.random.randint(0, n, n)
        samples.append(np.mean(data[idx]))
    lower = np.percentile(samples, 100 * (alpha / 2))
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Shared config / model helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, modifying base in-place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> dict:
    """
    Load a YAML config file.

    Always starts with ``configs/default.yaml`` as a base (located relative to
    this file) and deep-merges the target config over it so partial override
    files (e.g. small.yaml, lstm.yaml) never cause KeyErrors.
    """
    # Anchor to this file's directory so the path works regardless of CWD
    _here = Path(__file__).resolve().parent
    default_path = _here / "configs" / "default.yaml"
    with open(default_path, "r") as f:
        cfg = yaml.safe_load(f)

    target = Path(path)
    if not target.is_absolute() and not target.exists():
        target = _here / target
    target = target.resolve()
    if target != default_path:
        with open(target, "r") as f:
            override = yaml.safe_load(f)
        if override:
            cfg = _deep_merge(cfg, override)
    return cfg


def clean_state_dict(state_dict: dict) -> dict:
    """Strip wrappers commonly added by torch.compile or DataParallel."""
    prefixes = ("_orig_mod.", "module.")
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def build_policy(cfg: dict, device: torch.device):
    """
    Construct a PolicyNet from *cfg* without depending on a live environment.

    Returns (policy, obs_dim, history_dim).
    """
    from models.policy import PolicyNet
    from omi_env import encoding, rules

    obs_dim = encoding.observation_length()
    history_dim = encoding.HISTORY_LEN * encoding.HISTORY_FEAT_DIM
    policy = PolicyNet(
        obs_dim=obs_dim,
        history_dim=history_dim,
        action_dim=rules.ACTION_DIM,
        hidden_size=cfg["model"]["recurrent_hidden_size"],
        recurrent_type=cfg["model"]["recurrent_type"],
        hist_feat_dim=encoding.HISTORY_FEAT_DIM,
    ).to(device)
    return policy, obs_dim, history_dim
