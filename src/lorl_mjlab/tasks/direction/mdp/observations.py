"""Privileged observations exposing domain-randomization events to the critic/teacher.

* ``actuator_gains``    <- ``mjlab.envs.mdp.dr.pd_gains`` (startup)
* ``external_force_b``  <- ``mjlab.envs.mdp.apply_external_force_torque`` (reset)
* ``external_torque_b`` <- ``mjlab.envs.mdp.apply_external_force_torque`` (reset)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

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
