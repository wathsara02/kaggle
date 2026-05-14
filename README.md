# Omi Reinforcement Learning

This project trains a multi-agent reinforcement learning policy for Omi, the
Sri Lankan trick-taking card game. The main workflow is:

1. Install Python dependencies.
2. Run a small smoke test.
3. Train with `configs/new.yaml`.
4. Resume training when needed.
5. Evaluate and export the trained policy.

## Project Layout

```text
configs/
  default.yaml      Base config loaded before every other config
  new.yaml          Main training config for this project
  small.yaml        Fast CPU smoke-test config

omi_env/            Omi rules, environment, and observation encoding
models/             Policy and critic neural networks
marl/               MAPPO trainer and vector environment
scripts/
  train.py          Train or resume training
  eval.py           Evaluate a trained policy
  eval_vs_policy.py Compare two trained policies
  plot_training.py  Plot training CSV output
  export.py         Export a trained policy for use elsewhere
tests/              Pytest checks for rules and environment behavior
```

## Requirements

Use Python 3.10 or newer.

Dependencies are listed in `requirements.txt`. It installs the CPU PyTorch
build, which matches the current `configs/new.yaml` setup.

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

```powershell
pip install -r requirements.txt
```

## Verify The Install

Run the tests:

```powershell
pytest
```

Run a short training smoke test:

```powershell
python scripts/train.py --config configs/small.yaml
```

`small.yaml` runs only 20 episodes on CPU. Use it to check that the code,
dependencies, and environment are working before starting a long run.

## Configs

`utils.load_config()` always loads `configs/default.yaml` first. If you pass
another config, that config is merged on top of `default.yaml`.

That means `configs/default.yaml` must stay in the repo even if you normally run
`configs/new.yaml`.

Current configs:

| Config | Use |
| --- | --- |
| `configs/default.yaml` | Base CUDA/T4 settings and shared defaults |
| `configs/new.yaml` | Main project training config |
| `configs/small.yaml` | Fast CPU smoke test |

## Train

Start the main training run:

```powershell
python scripts/train.py --config configs/new.yaml
```

Resume the same run from the latest checkpoint:

```powershell
python scripts/train.py --config configs/new.yaml --resume
```

Useful overrides:

```powershell
python scripts/train.py --config configs/new.yaml --episodes 10000
python scripts/train.py --config configs/new.yaml --num-envs 4
python scripts/train.py --config configs/new.yaml --device cpu
```

The current `new.yaml` writes output under:

```text
runs/local_5600g/
```

Important files in that folder:

| File | Purpose |
| --- | --- |
| `policy_last.pt` | Latest policy weights |
| `checkpoint_latest.pt` | Resume checkpoint |
| `training_summary.csv` | Training metrics |
| `match_traces.csv` | Sampled match records, if enabled |
| `baseline_evals/` | Periodic evaluation results, if enabled |

## Plot Training

After training has written `training_summary.csv`, generate plots:

```powershell
python scripts/plot_training.py --csv runs/local_5600g/training_summary.csv
```

The plot script writes PNG charts next to the CSV file.

## Evaluate

Evaluate the trained policy against the rule-based baseline:

```powershell
python scripts/eval.py --config configs/new.yaml --weights runs/local_5600g/policy_last.pt --episodes 200 --deterministic
```

Evaluate against a random baseline:

```powershell
python scripts/eval.py --config configs/new.yaml --weights runs/local_5600g/policy_last.pt --baseline random --episodes 200
```

Evaluation results are printed in the terminal and saved under the run folder.

## Compare Two Policies

Use this when you have two trained policies and want them to play each other:

```powershell
python scripts/eval_vs_policy.py --weights-a runs/local_5600g/policy_last.pt --weights-b runs/local_5600g/policy_last.pt --episodes 100
```

Change `--weights-b` to the second policy you want to compare.

## Export

Export the trained policy and metadata:

```powershell
python scripts/export.py --config configs/new.yaml --weights runs/local_5600g/policy_last.pt --output_dir artifacts
```

This creates:

```text
artifacts/
  policy_agent.pt
  config.json
  VERSION
```

## Common Commands

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Run tests
pytest

# Quick smoke test
python scripts/train.py --config configs/small.yaml

# Main training
python scripts/train.py --config configs/new.yaml

# Resume training
python scripts/train.py --config configs/new.yaml --resume

# Evaluate latest policy
python scripts/eval.py --config configs/new.yaml --weights runs/local_5600g/policy_last.pt --episodes 200 --deterministic

# Plot training metrics
python scripts/plot_training.py --csv runs/local_5600g/training_summary.csv

# Export policy
python scripts/export.py --config configs/new.yaml --weights runs/local_5600g/policy_last.pt --output_dir artifacts
```

## Notes

- Keep `configs/default.yaml`; `new.yaml` depends on it through config merging.
- Use `configs/small.yaml` before long runs to catch setup problems quickly.
- Use `--resume` for long training sessions so work continues from
  `checkpoint_latest.pt`.
- On Windows with an AMD GPU, normal PyTorch does not use the GPU through CUDA.
