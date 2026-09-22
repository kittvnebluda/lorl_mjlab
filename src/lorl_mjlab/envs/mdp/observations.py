from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import foot_contact
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import UniformNoiseCfg as Unoise

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def actuator_gains(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Per-actuator PD gains currently set into the sim, as randomized by ``dr.pd_gains``.

    Returns the concatenation of stiffness and damping over the entity's actuators,
    each divided by its default value (~1.0 dimensionless randomization factor).
    Shape: ``(num_envs, 2 * num_actuators)``.
    """
    asset: Entity = env.scene[asset_cfg.name]
    ctrl_ids = asset.data.indexing.ctrl_ids

    kp = env.sim.model.actuator_gainprm[:, ctrl_ids, 0]
    kd = -env.sim.model.actuator_biasprm[:, ctrl_ids, 2]

    default_gainprm = env.sim.get_default_field("actuator_gainprm")
    default_biasprm = env.sim.get_default_field("actuator_biasprm")
    default_kp = default_gainprm[ctrl_ids, 0]
    default_kd = -default_biasprm[ctrl_ids, 2]

    kp_rel = torch.where(default_kp != 0, kp / default_kp, kp)
    kd_rel = torch.where(default_kd != 0, kd / default_kd, kd)
    return torch.cat([kp_rel, kd_rel], dim=-1)


def external_force_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """External force (body frame) on the selected bodies. Shape: (num_envs, 3*N)."""
    asset: Entity = env.scene[asset_cfg.name]
    force_w = asset.data.body_external_force[:, asset_cfg.body_ids]
    quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    force_b = quat_apply_inverse(quat_w, force_w)
    return force_b.reshape(env.num_envs, -1)


def external_torque_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """External torque (body frame) on the selected bodies. Shape: (num_envs, 3*N)."""
    asset: Entity = env.scene[asset_cfg.name]
    torque_w = asset.data.body_external_torque[:, asset_cfg.body_ids]
    quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    torque_b = quat_apply_inverse(quat_w, torque_w)
    return torque_b.reshape(env.num_envs, -1)


def base_policy_obs_terms(*, command_name: str) -> dict[str, ObservationTermCfg]:
    """Common proprioceptive terms."""
    return {
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "command": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": command_name},
        ),
    }


def base_privileged_obs_terms(*, foot_scan_keys: tuple[str, ...]) -> dict[str, ObservationTermCfg]:
    """Teacher-only terms."""
    return {
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "foot_contacts": ObservationTermCfg(
            func=foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        **{
            f"{foot}_foot_scan": ObservationTermCfg(
                func=envs_mdp.height_scan,
                params={"sensor_name": f"{foot}_foot_scan"},
                noise=Unoise(n_min=-0.1, n_max=0.1),
                clip=(-1.0, 1.0),
            )
            for foot in foot_scan_keys
        },
        "actuator_gains": ObservationTermCfg(func=actuator_gains),
        "forces": ObservationTermCfg(
            func=external_force_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
        ),
        "torques": ObservationTermCfg(
            func=external_torque_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
        ),
    }
