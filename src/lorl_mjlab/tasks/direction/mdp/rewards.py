"""Direction-specific reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command_modes(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve the two mutually exclusive command modes.

    Returns ``(command, is_move, is_stand)``, which partition the batch: a zero heading with
    no turn is a stand, anything else is a move.
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None
    is_move = (torch.norm(command[:, :2], dim=1) > 0.1) | (command[:, 2].abs() > 0.1)
    return command, is_move, ~is_move


def lin_vel_z_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalize vertical base velocity using an L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def track_direction(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Forward-progress reward: linear ramp 0->1 for v_pr in [0, 0.6], 0 while standing."""
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
    """Turn-tracking reward that also penalizes yaw spin when no turn is commanded."""
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
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize base/belly contact with the ground."""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        violations = (force_mag > force_threshold).any(dim=-1).sum(dim=-1).float()
    else:
        assert data.found is not None
        violations = data.found.sum(dim=-1).float()

    env.extras["log"]["Metrics/prone"] = (violations > 0).float().mean()
    return violations


def flight_phase_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalty (1.0 per step) when all sensor-tracked feet are simultaneously airborne."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    num_contacts = sensor.data.found.sum(dim=-1)
    return (num_contacts == 0).float()


def stand_height_shortfall(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Trunk-height penalty, active only under a zero (STAND) command.

    Height is measured as trunk-above-mean-foot rather than world z, which makes it
    terrain-relative for free on the rough terrain generator and needs no extra raycast.
    """
    asset: Entity = env.scene[asset_cfg.name]
    *_, is_stand = _command_modes(env, command_name)

    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].mean(dim=1)
    height = asset.data.root_link_pos_w[:, 2] - foot_z

    shortfall = torch.clamp(1.0 - height / target_height, min=0.0)
    return torch.square(shortfall) * is_stand.float()


class radial_progress:
    """Reward getting further from the spawn point than ever before this episode.

    A **max-radius ratchet**, deliberately not a per-step delta. Paying for
    ``clamp(r_t - r_{t-1}, min=0)`` would charge nothing for coming back inward, so
    oscillating out and in farms it without bound. Rewarding only new maxima is
    non-farmable, telescopes to ``r_max - r_start`` over the episode, and -- unlike a signed
    delta -- never punishes moving inward, which matters because the command may legitimately
    point that way.

    Returns a **rate** (m/s): ``RewardManager`` multiplies by ``dt``, so the weight ends up
    denominated in reward per metre traversed.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg  # Unused.
        self._r_max = torch.zeros(env.num_envs, device=env.device)
        # Deliberately lazy: `RewardManager.reset` runs *before* the curriculum updates
        # `env_origins`, so re-baselining eagerly in `reset()` would measure against the old
        # sub-terrain origin and pay a spurious jump on the next step. Instead flag the env
        # and take the baseline from its first post-reset observation, paying nothing then.
        self._needs_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self._needs_init[idx] = True

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]

        terrain = env.scene.terrain
        assert terrain is not None
        terrain_generator = terrain.cfg.terrain_generator
        assert terrain_generator is not None
        # Cap at the promotion radius: past that the curriculum takes over, and an uncapped
        # term could be farmed by running off the terrain patch.
        cap = terrain_generator.size[0] * 0.5

        radius = torch.norm(
            asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2],
            dim=1,
        ).clamp(max=cap)

        # Freshly reset envs take their baseline here and earn nothing this step.
        self._r_max = torch.where(self._needs_init, radius, self._r_max)
        gained = torch.where(
            self._needs_init,
            torch.zeros_like(radius),
            (radius - self._r_max).clamp(min=0.0),
        )
        self._needs_init[:] = False
        self._r_max = torch.maximum(self._r_max, radius)

        return gained / env.step_dt


class stand_posture:
    """Joint-posture reward against the entity's nominal standing pose, active under a zero (STAND) command.

    Implemented as a class so the target pose is captured once at construction rather than
    re-read on every step.

    Currently parked: defined but not wired into any task's reward set. PMTG's trajectory
    generator will define the nominal stance itself. A 1500-iteration A/B found no effect on
    task metrics beyond the run-to-run noise floor -- the only measurable difference was the
    stance itself (thigh 0.83 -> 0.65, calf -1.21 -> -1.06 with the term removed).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.target_joint_pos = default_joint_pos.clone()

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        command_name: str,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        *_, is_stand = _command_modes(env, command_name)

        error = asset.data.joint_pos[:, asset_cfg.joint_ids] - self.target_joint_pos[:, asset_cfg.joint_ids]
        reward = torch.exp(-torch.mean(torch.square(error / std), dim=1))
        return reward * is_stand.float()
