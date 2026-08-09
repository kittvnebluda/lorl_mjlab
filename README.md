# Legged Obstacle RL MJLab

Legged locomotion RL tasks for quadrupeds (Unitree Go1, AlienGo), built on
[mjlab](https://github.com/mujocolab/mjlab) — a MuJoCo/MJX-based reimplementation
of Isaac Lab's manager-based RL API.

Adds a **direction-command** locomotion task family (heading + discrete turn,
per Lee et al. 2020) on top of mjlab's own built-in velocity/tracking tasks,
plus PPO symmetry data-augmentation and DAgger teacher→student distillation
support that mjlab's own `rl` module doesn't expose yet.

## Install

```bash
make sync       # GPU (CUDA 13.0 wheels)
make sync-cpu   # CPU only
```

## Tasks

```bash
uv run list-envs
```

## Training

```bash
uv run train Lorl-Direction-Rough-Unitree-Go1
uv run play Lorl-Direction-Rough-Unitree-Go1 --checkpoint <path>

# Distillation needs a trained teacher checkpoint:
uv run train Lorl-Direction-Rough-Unitree-Go1-Distill \
  --agent.resume=True --agent.load-run <teacher_run_dir>
```

Runs log to `logs/rsl_rl/<experiment_name>/`.

## Layout

```
src/lorl_mjlab/
├── robots/unitree_aliengo/   # AlienGo MJCF + actuator/collision config
├── rl/                       # Distillation + PPO-symmetry config/runner extensions
└── tasks/direction/          # Direction command, MDP terms, per-robot configs
```

## Dev

```bash
make format   # ruff format + fix
make type     # ty + pyright
make test     # pytest
```
