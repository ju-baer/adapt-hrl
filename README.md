# AdaptHRL: Adaptive Recursive Hierarchical Decomposition for Scalable Long-Horizon Decision-Making

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

---

## Overview

AdaptHRL is the first hierarchical reinforcement learning framework that **adapts its hierarchy depth at inference time without retraining**. Existing HRL methods (HAC, HIRO, Options, etc.) fix the depth $L$ before training — meaning an agent trained at horizon $T=1{,}000$ cannot adjust its abstraction level when deployed at $T=20{,}000$. AdaptHRL resolves this through three jointly motivated components:

| Component | What it does | Why it matters |
|-----------|-------------|----------------|
| **Recursive shared policy** $\pi_\theta$ | Single MLP applied at every hierarchy level | Parameter reuse across temporal scales; enables OOD generalisation |
| **Self-supervised gating** $G_\varphi$ | Decides per state-goal pair whether to recurse or act | Domain-agnostic; robust to ±40% label noise |
| **Projection regularisation** $\mathcal{L}_\text{reg}$ | Penalises infeasible sub-goals | Prevents hierarchical divergence; stabilises training |

Structurally, AdaptHRL is the exact RL analogue of **chain-of-thought (CoT) reasoning** in LLMs: a single model recursively decomposes complex inputs into intermediate representations until it decides to emit a terminal output — with depth determined by the model, not the designer.

---

## Results at a Glance

| Method | AntMaze (Reward) | Maze (Succ%) | MT10 (Succ%) | Generalises to T=20k? |
|--------|:---:|:---:|:---:|:---:|
| PPO | 637 ± 64 | 81 ± 5 | 46 ± 7 | 14% |
| HAC | 758 ± 49 | 90 ± 3 | 68 ± 4 | 31% |
| HIRO | 738 ± 53 | 88 ± 3 | 64 ± 4 | 27% |
| **AdaptHRL** | **871 ± 43** | **96 ± 2** | **81 ± 3** | **82%** |

All results: mean ± std over 5 seeds, 100 eval episodes per seed.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/ju-baer/adapt-hrl.git
cd adapt-hrl

# 2. Create environment
conda create -n adapt-hrl python=3.9 -y
conda activate adapt-hrl

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install MuJoCo (required for AntMaze)
#    Follow: https://github.com/openai/mujoco-py#install-mujoco
pip install mujoco==2.3.7

# 5. Install AdaptHRL (editable)
pip install -e .
```

### Verify installation
```bash
python -c "import adapt-hrl; print(adapthrl.__version__)"
python scripts/verify_install.py
```

---

## Quick Start

### Train on AntMaze
```bash
python scripts/train.py \
  --config configs/antmaze.yaml \
  --seed 0
```

### Train on Maze Navigation
```bash
python scripts/train.py \
  --config configs/maze.yaml \
  --seed 0
```

### Train on Meta-World MT10
```bash
python scripts/train.py \
  --config configs/metaworld_mt10.yaml \
  --seed 0
```

### Reproduce the horizon generalisation experiment
```bash
# Train at T=1000
python scripts/train.py --config configs/antmaze.yaml --seed 0

# Evaluate at T=2000, 5000, 10000, 20000 (no retraining)
python scripts/eval_horizon_generalisation.py \
  --checkpoint logs/antmaze/seed_0/best.pt \
  --horizons 2000 5000 10000 20000
```

### Reproduce all paper results
```bash
bash scripts/reproduce_all.sh
```
> Full reproduction requires ~5× A100 days. See `scripts/reproduce_all.sh` for
> per-experiment estimates and how to run subsets.

---

## Repository Structure

```
adapt-hrl/
│
├── adapt-hrl/                   # Core library
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── adapthrl_agent.py   # Main AdaptHRL agent
│   │   ├── baselines/
│   │   │   ├── ppo_agent.py
│   │   │   ├── sac_agent.py
│   │   │   ├── hac_agent.py
│   │   │   └── hiro_agent.py
│   │   └── rollout.py          # Episode rollout logic
│   │
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── policy.py           # Shared policy π_θ (3-layer MLP + spectral norm)
│   │   ├── gating.py           # Gating network G_φ + self-supervised labels
│   │   ├── value.py            # Value function heads
│   │   └── utils.py            # Spectral normalisation, weight init
│   │
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── antmaze.py          # AntMaze-Large-v3 wrapper
│   │   ├── maze.py             # Custom 50×50 maze environment
│   │   ├── metaworld.py        # Meta-World MT10 wrapper
│   │   └── wrappers.py         # Goal-conditioning, horizon wrappers
│   │
│   └── utils/
│       ├── __init__.py
│       ├── buffer.py           # Replay / rollout buffer
│       ├── logger.py           # WandB + CSV logging
│       ├── projection.py       # ProjG feasibility projection
│       └── schedule.py         # Learning rate and threshold schedules
│
├── configs/                    # YAML experiment configs
│   ├── antmaze.yaml
│   ├── maze.yaml
│   ├── metaworld_mt10.yaml
│   ├── extreme_horizon.yaml
│   └── ablations/
│       ├── no_recursion.yaml
│       ├── no_sharing.yaml
│       ├── no_regularisation.yaml
│       └── fixed_depth.yaml
│
├── scripts/
│   ├── train.py                # Main training entry point
│   ├── eval.py                 # Evaluation script
│   ├── eval_horizon_generalisation.py
│   ├── eval_ood.py             # OOD goal evaluation
│   ├── reproduce_all.sh        # Full paper reproduction
│   └── verify_install.py       # Installation check
│
├── tests/
│   ├── test_policy.py
│   ├── test_gating.py
│   ├── test_agent.py
│   ├── test_envs.py
│   └── test_projection.py
│
├── notebooks/
│   ├── 01_quickstart.ipynb
│   ├── 02_visualise_subgoals.ipynb
│   └── 03_reproduce_figures.ipynb
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Configuration

All experiments are controlled by YAML configs. Example (`configs/antmaze.yaml`):

```yaml
env:
  name: antmaze-large-v3
  horizon: 1000
  goal_space: position      # or 'learned'

agent:
  l_max: 10                 # Maximum hierarchy depth
  tau: 0.5                  # Gating threshold
  lambda_reg: 0.1           # Sub-goal regularisation strength
  beta: 0.99                # Self-supervised label momentum

policy:
  hidden_dim: 256
  n_layers: 3
  activation: relu
  spectral_norm: true       # Enforces Lipschitz condition

gating:
  hidden_dim: 128
  n_layers: 2

training:
  total_steps: 1_000_000
  batch_size: 256
  lr: 3.0e-4
  gamma: 0.99
  clip_eps: 0.2             # PPO clip
  gae_lambda: 0.95
  n_seeds: 5
  eval_every: 50_000
  eval_episodes: 100

logging:
  use_wandb: true
  project: adapthrl
  save_checkpoints: true
  checkpoint_every: 100_000
```

Override any field from the command line:
```bash
python scripts/train.py --config configs/antmaze.yaml agent.tau=0.3 training.total_steps=500000
```

---

## Reproducing Paper Results

### Table 1: Main results
```bash
for env in antmaze maze metaworld_mt10; do
  for seed in 0 1 2 3 4; do
    python scripts/train.py --config configs/${env}.yaml --seed ${seed} &
  done
done
wait
python scripts/eval.py --log_dir logs/ --table main
```

### Table 2: Horizon generalisation
```bash
python scripts/eval_horizon_generalisation.py \
  --checkpoint_dir logs/antmaze/ \
  --horizons 1000 2000 5000 10000 20000
```

### Table 3: Ablations
```bash
for cfg in configs/ablations/*.yaml; do
  for seed in 0 1 2 3 4; do
    python scripts/train.py --config ${cfg} --seed ${seed}
  done
done
python scripts/eval.py --log_dir logs/ --table ablations
```

### Figure 2: Depth vs. horizon
```bash
python notebooks/03_reproduce_figures.ipynb  # or run as script:
python scripts/plot_depth_vs_horizon.py --checkpoint_dir logs/antmaze/
```

---

## Extending AdaptHRL

### Add a new environment
```python
# adapt-hrl/envs/my_env.py
from adapt-hrl.envs.wrappers import GoalConditionedWrapper

class MyEnv(GoalConditionedWrapper):
    def __init__(self, horizon: int = 1000):
        super().__init__(base_env=..., goal_space=..., horizon=horizon)

    def project_goal(self, g):
        """ProjG: project a proposed sub-goal onto the feasible set."""
        return np.clip(g, self.observation_space.low, self.observation_space.high)
```

Register it in `adapt-hrl/envs/__init__.py` and create a config YAML.

### Swap the policy backbone
```python
# In configs/custom.yaml
policy:
  backbone: transformer    # Options: mlp (default), transformer, cnn
  hidden_dim: 256
```
---

## Acknowledgements

We thank the authors of [D4RL](https://github.com/Farama-Foundation/d4rl),
[Meta-World](https://github.com/Farama-Foundation/Metaworld), and
[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) for their open-source
environments and baseline implementations.
