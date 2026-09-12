from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from lorl_mjlab.rl import LorlDistillationRunner
from lorl_mjlab.tasks.registry import register_play_task

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
    runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
    task_id="Lorl-Direction-Rough-Unitree-Go1-Distill",
    env_cfg=unitree_go1_direction_env_cfg(distill=True),
    play_env_cfg=unitree_go1_direction_env_cfg(play=True, distill=True),
    rl_cfg=unitree_go1_direction_distillation_runner_cfg(),
    runner_cls=LorlDistillationRunner,
)

for _variant in ("flat", "sloped"):
    register_play_task(
        task_id=f"Lorl-Direction-Icra-{_variant.capitalize()}-Unitree-Go1",
        play_env_cfg=unitree_go1_direction_env_cfg(play=True, icra=_variant),
        rl_cfg=unitree_go1_direction_ppo_runner_cfg(),
        runner_cls=MjlabOnPolicyRunner,
    )
    register_play_task(
        task_id=f"Lorl-Direction-Icra-{_variant.capitalize()}-Unitree-Go1-Distill",
        play_env_cfg=unitree_go1_direction_env_cfg(play=True, distill=True, icra=_variant),
        rl_cfg=unitree_go1_direction_distillation_runner_cfg(),
        runner_cls=LorlDistillationRunner,
    )
