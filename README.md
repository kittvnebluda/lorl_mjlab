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

## Teleop

Drive the commands by hand while a policy runs. The controls live in the **Viser** viewer,
so ask for it explicitly — `play` otherwise picks the native viewer whenever `$DISPLAY` is
set, and the native viewer has no GUI panel:

```bash
uv run play Lorl-Direction-Rough-Unitree-Go1 --checkpoint-file <path> --viewer viser
```

Open the printed URL, expand **Commands → Teleop**, and tick `Enable`. Until then the
command terms resample exactly as they always have.

| Widget | Hotkey | Action |
|---|---|---|
| `Enable` | — | Master gate |
| `Move` | `I` / `,` | Head forward (0°) / reverse (180°) |
| `Heading (deg)` | `J` / `L` | Steer left / right, 15° per press |
| `Turn` | `U` / `O` | Turn left / right; steps through {Left, None, Right} |
| `Stand` | `K` | Zero everything back to a stand |

Hotkeys are browser-side, so they only fire while the viewer tab is focused and cannot
collide with the native viewer's own keys. Commands are broadcast to **all** envs.

## Dev

```bash
make format   # ruff format + fix
make type     # ty + pyright
make test     # pytest
```
