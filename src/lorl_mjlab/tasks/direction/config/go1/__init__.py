from mjlab.tasks.registry import register_mjlab_task

from lorl_mjlab.rl import LorlDistillationRunner, LorlOnPolicyRunner

from .env_cfgs import unitree_go1_direction_env_cfg
from .rl_cfg import (
    unitree_go1_direction_distillation_runner_cfg,
    unitree_go1_direction_ppo_runner_cfg,
)

register_mjlab_task(
    task_id="Lorl-Direction-Rough-Unitree-Go1",
    env_cfg=unitree_go1_direction_env_cfg(),
    play_env_cfg=unitree_go1_direction_env_cfg(play=True),
    rl_cfg=unitree_go1_direction_ppo_runner_cfg(),
    runner_cls=LorlOnPolicyRunner,
)

register_mjlab_task(
    task_id="Lorl-Direction-Rough-Unitree-Go1-Distill",
    env_cfg=unitree_go1_direction_env_cfg(distill=True),
    play_env_cfg=unitree_go1_direction_env_cfg(play=True, distill=True),
    rl_cfg=unitree_go1_direction_distillation_runner_cfg(),
    runner_cls=LorlDistillationRunner,
)
