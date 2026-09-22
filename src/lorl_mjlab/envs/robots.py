"""Per-robot :class:`QuadrupedSetup` instances."""

from __future__ import annotations

from dataclasses import replace

from mjlab.asset_zoo.robots import get_go1_robot_cfg

from lorl_mjlab.envs.robot_setup import QuadrupedSetup
from lorl_mjlab.robots import ALIENGO_STAND_HEIGHT_TARGET, GO1_LEARNED_ACTUATOR_CFGS, get_aliengo_robot_cfg

FOOT_NAMES = ("FR", "FL", "RR", "RL")

# Trunk height above the ray cast terrain height that a standing robot must not sag below.
GO1_STAND_HEIGHT_TARGET: float = 0.223


def go1_setup(learned_actuator: bool = True) -> QuadrupedSetup:
    """Robot-side facts the direction-command tasks need about the Go1.

    Args:
        learned_actuator: Drive the joints with the learned MLP actuator model instead of
            MuJoCo's builtin position actuators. See ``go1_learned_actuator``.
    """
    robot_cfg = get_go1_robot_cfg()
    if learned_actuator:
        assert robot_cfg.articulation is not None
        robot_cfg.articulation = replace(robot_cfg.articulation, actuators=GO1_LEARNED_ACTUATOR_CFGS)
    return QuadrupedSetup(
        robot_cfg=robot_cfg,
        foot_names=FOOT_NAMES,
        thigh_geom_names=tuple(f"{leg}_thigh_collision{i}" for leg in FOOT_NAMES for i in (1, 2, 3)),
        trunk_geom_names=("trunk_collision", "head_collision"),
        stand_height_target=GO1_STAND_HEIGHT_TARGET,
        force_range=(-8.0, 8.0),
        torque_range=(-0.8, 0.8),
    )


def aliengo_setup() -> QuadrupedSetup:
    """Robot-side facts the direction-command tasks need about the AlienGo."""
    return QuadrupedSetup(
        robot_cfg=get_aliengo_robot_cfg(),
        foot_names=FOOT_NAMES,
        thigh_geom_names=tuple(f"{leg}_thigh_collision" for leg in FOOT_NAMES),
        trunk_geom_names=("trunk_collision",),
        stand_height_target=ALIENGO_STAND_HEIGHT_TARGET,
        force_range=(-15.0, 15.0),
        torque_range=(-1.5, 1.5),
    )
