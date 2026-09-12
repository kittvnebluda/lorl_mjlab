from typing import cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlBaseRunnerCfg
from mjlab.tasks.registry import _REGISTRY, _TaskCfg


def register_play_task(
    task_id: str,
    play_env_cfg: ManagerBasedRlEnvCfg,
    rl_cfg: RslRlBaseRunnerCfg,
    runner_cls: type | None = None,
) -> None:
    """Register a task that only exists in play mode.

    For fixed-layout evaluation courses there is nothing to train against.
    ``load_env_cfg(task_id)`` then returns ``None`` and ``train`` fails on it, which is the
    intent; everything that plays, exports or verifies goes through ``play=True``.
    """
    if task_id in _REGISTRY:
        raise ValueError(f"Task '{task_id}' is already registered")
    _REGISTRY[task_id] = _TaskCfg(cast(ManagerBasedRlEnvCfg, None), play_env_cfg, rl_cfg, runner_cls)
