"""MDP terms for the direction locomotion task family.

Re-exports everything reusable from ``mjlab.tasks.velocity.mdp`` (which itself
re-exports ``mjlab.envs.mdp``), plus the direction-specific command, rewards,
observations, and curriculum defined locally.
"""

from mjlab.tasks.velocity.mdp import *

from .commands import UniformDirectionCommand, UniformDirectionCommandCfg
from .curriculums import terrain_levels_dir
from .observations import actuator_gains, external_force_b, external_torque_b
from .rewards import (
    base_motion_reward,
    feet_air_time_progress,
    flight_phase,
    lin_vel_z_l2,
    track_direction,
    track_turn,
    undesired_base_contact,
)

__all__ = [
    "UniformDirectionCommand",
    "UniformDirectionCommandCfg",
    "actuator_gains",
    "base_motion_reward",
    "external_force_b",
    "external_torque_b",
    "feet_air_time_progress",
    "flight_phase",
    "lin_vel_z_l2",
    "terrain_levels_dir",
    "track_direction",
    "track_turn",
    "undesired_base_contact",
]
