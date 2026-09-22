"""Reset, interval and domain-randomization events

Terms whose ``asset_cfg`` or ranges are left blank here are filled in per robot by
:mod:`lorl_mjlab.envs.robot_setup`.
"""

from __future__ import annotations

from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


def my_events() -> dict[str, EventTermCfg]:
    return {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        ),
        "base_external_force_torque": EventTermCfg(
            func=envs_mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
                "force_range": (0.0, 0.0),  # Set per-robot.
                "torque_range": (0.0, 0.0),  # Set per-robot.
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "operation": "abs",
                "ranges": (0.6, 1.5),
                "shared_random": True,
            },
        ),
        "base_inertial": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
                "alpha_range": (-0.3466, 0.2027),
                "t1_range": (-0.1, 0.1),
                "t2_range": (-0.1, 0.1),
                "t3_range": (-0.1, 0.1),
            },
        ),
        "actuator_gains": EventTermCfg(
            mode="startup",
            func=dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
                "kp_range": (0.7, 1.3),
                "kd_range": (0.7, 1.3),
                "operation": "scale",
                "distribution": "uniform",
            },
        ),
        "joint_friction": EventTermCfg(
            mode="startup",
            func=dr.joint_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "ranges": (0.0, 0.4),
                "operation": "abs",
                "distribution": "uniform",
            },
        ),
        "joint_armature": EventTermCfg(
            mode="startup",
            func=dr.joint_armature,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "ranges": (0.0, 0.05),
                "operation": "abs",
                "distribution": "uniform",
            },
        ),
    }
