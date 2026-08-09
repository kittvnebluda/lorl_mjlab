"""Direction-specific reward terms.

Everything reusable (dof torques/acc, action rate, joint limits, feet
clearance/swing-height/slip, soft landing, upright, self/thigh-collision cost,
flat orientation, is_alive) comes straight from ``mjlab.envs.mdp`` /
``mjlab.tasks.velocity.mdp`` -- see ``direction_env_cfg.py``. This module only
holds the terms whose semantics are specific to heading+turn commands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def lin_vel_z_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalize vertical base velocity using an L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def track_direction(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Forward-progress reward: linear ramp 0->1 for v_pr in [0, 0.6], 0 while standing.

    v_pr is the base linear velocity projected onto the commanded direction. Unlike a
    Gaussian tracking reward, standing/backward motion (v_pr <= 0) earns nothing, so a
    robot that spins or hops in place cannot farm this term.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    cmd_dir = command[:, :2]
    is_standing_cmd = torch.norm(cmd_dir, dim=1) < 0.1

    v_pr = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * cmd_dir, dim=1)

    rew = torch.clamp(v_pr, min=0.0, max=0.6) / 0.6
    return torch.where(is_standing_cmd, torch.zeros_like(rew), rew)


def track_turn(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Turn-tracking reward that also penalizes yaw spin when no turn is commanded.

    When ``cmd_turn == 0`` the reward peaks at zero yaw rate and decays with any spin,
    so a robot can't spin freely while no turn is commanded.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    cmd_turn = command[:, 2]
    vel_yaw_b = asset.data.root_link_ang_vel_b[:, 2]

    no_turn = cmd_turn.abs() < 0.1
    w_pr = cmd_turn * vel_yaw_b
    turn_rew = torch.where(w_pr >= 0.6, torch.ones_like(w_pr), torch.exp(-1.5 * torch.square(w_pr - 0.6)))
    still_rew = torch.exp(-1.5 * torch.square(vel_yaw_b))

    return torch.where(no_turn, still_rew, turn_rew)


def base_motion_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize lateral drift off the commanded direction and non-yaw base rotation."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    cmd_dir = command[:, :2]

    lv_xy = asset.data.root_link_lin_vel_b[:, :2]
    av_xy = asset.data.root_link_ang_vel_b[:, :2]

    v_pr = torch.sum(lv_xy * cmd_dir, dim=1)
    v_o = torch.norm(lv_xy - v_pr.unsqueeze(1) * cmd_dir, dim=1)

    return torch.exp(-1.5 * torch.square(v_o)) + torch.exp(-1.5 * torch.sum(torch.square(av_xy), dim=1))


def feet_air_time_progress(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Air-time reward gated on real forward progress, with capped flight time.

    The per-foot bonus is clamped (discourages big jumps) and gated on v_pr > 0.1
    (actual forward motion), so hopping in place earns nothing.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    last_air_time = sensor.data.last_air_time
    assert last_air_time is not None
    reward = torch.sum(torch.clamp(last_air_time - threshold, max=0.3) * first_contact, dim=1)

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    cmd_dir = command[:, :2]
    v_pr = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * cmd_dir, dim=1)

    return reward * (v_pr > 0.1).float()


def undesired_base_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize base/belly contact, but only while a movement command is active.

    Counts contact instances whose force exceeds ``force_threshold``, zeroed when the
    command is ~zero (no direction and no turn) so the robot may rest on its belly at
    standstill without being punished.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        violations = (force_mag > force_threshold).any(dim=-1).sum(dim=-1).float()
    else:
        assert data.found is not None
        violations = data.found.sum(dim=-1).float()

    command = env.command_manager.get_command(command_name)
    assert command is not None
    is_moving = (torch.norm(command[:, :2], dim=1) > 0.1) | (command[:, 2].abs() > 0.1)

    return violations * is_moving.float()


def flight_phase(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalty (1.0 per step) when all sensor-tracked feet are simultaneously airborne."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    num_contacts = sensor.data.found.sum(dim=-1)
    return (num_contacts == 0).float()
