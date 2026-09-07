"""Print what get_base_metadata() returns for a task, without training or exporting."""

import json
from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import lorl_mjlab.tasks  # noqa: F401
from lorl_mjlab.rl.exporter_utils import get_base_metadata


@dataclass
class Args:
    task_id: str = "Lorl-Direction-Rough-Unitree-Go1"
    device: str | None = None


def main() -> None:
    import tyro

    args = tyro.cli(Args)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(args.task_id, play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        print(json.dumps(get_base_metadata(env, run_path="local"), indent=4))
    finally:
        env.close()


if __name__ == "__main__":
    main()
