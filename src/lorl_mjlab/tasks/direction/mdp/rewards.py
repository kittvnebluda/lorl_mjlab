"""Direction-specific reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.rewards import feet_slip as _feet_slip
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .rest_command import RestCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _rest_mask(env: ManagerBasedRlEnv, rest_command_name: str | None) -> torch.Tensor | None:
    """Boolean rest mask, or ``None`` when the task is configured without a rest command.

    Every ``rest_command_name`` parameter in this module is optional for the same reason:
    the rest command is a task variant, not a fixture. ``None`` means no env is ever excused,
    which is the correct behavior for a task that never tells the robot to lie down.
    """
    if rest_command_name is None:
        return None
    rest = env.command_manager.get_command(rest_command_name)
    assert rest is not None
    return rest[:, 0] > 0.5


def _command_modes(
    env: ManagerBasedRlEnv,
    command_name: str,
    rest_command_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve the three mutually exclusive command modes.

    Returns ``(command, is_rest, is_move, is_stand)``. The direction term already zeroes its
    command while resting (see ``DirectionWithRestCommand``), so ``is_move`` can never be true
    while resting and the three masks partition the batch. Without a rest command ``is_rest``
    is all-false and the partition degenerates to move/stand.
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None
    is_move = (torch.norm(command[:, :2], dim=1) > 0.1) | (command[:, 2].abs() > 0.1)

    is_rest = _rest_mask(env, rest_command_name)
    if is_rest is None:
        is_rest = torch.zeros_like(is_move)
    return command, is_rest, is_move, ~is_rest & ~is_move


def _release_gate(env: ManagerBasedRlEnv, rest_command_name: str) -> torch.Tensor:
    """Post-rest-release ramp applied to base contact penalties."""
    term = cast(RestCommand, env.command_manager.get_term(rest_command_name))
    return term.release_gate


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
    rest_command_name: str | None = None,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize base/belly contact everywhere except under an active rest command.

    The penalty ramps back in over the rest term's release grace window so the robot is
    not punished for the time it physically takes to stand back up. Without a rest command
    configured there is nothing to excuse and nothing to ramp: the penalty is always on.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        violations = (force_mag > force_threshold).any(dim=-1).sum(dim=-1).float()
    else:
        assert data.found is not None
        violations = data.found.sum(dim=-1).float()

    is_rest = _rest_mask(env, rest_command_name)
    active = torch.ones_like(violations) if is_rest is None else (~is_rest).float()

    grounded_while_active = (violations > 0).float() * active
    env.extras["log"]["Metrics/prone_while_active"] = grounded_while_active.mean()

    if rest_command_name is None:
        return violations
    return violations * active * _release_gate(env, rest_command_name)


def flight_phase_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    rest_command_name: str | None = None,
) -> torch.Tensor:
    """Penalty (1.0 per step) when all sensor-tracked feet are simultaneously airborne.

    Gated off during REST: a folded robot has its feet tucked clear of the ground.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    num_contacts = sensor.data.found.sum(dim=-1)
    penalty = (num_contacts == 0).float()

    is_rest = _rest_mask(env, rest_command_name)
    if is_rest is None:
        return penalty
    return penalty * (~is_rest).float()


def gated_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    rest_command_name: str | None = None,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Rest-aware wrapper around the upstream self-collision cost.

    Mirrors ``mjlab.tasks.velocity.mdp.self_collision_cost`` but excuses contacts under
    an active rest command -- a prone robot necessarily folds its thighs onto the ground.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        violations = (force_mag > force_threshold).any(dim=1).sum(dim=-1).float()
    else:
        assert data.found is not None
        violations = data.found.sum(dim=-1).float()

    is_rest = _rest_mask(env, rest_command_name)
    if is_rest is None:
        return violations
    return violations * (~is_rest).float()


def feet_slip_gated(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    rest_command_name: str | None = None,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Upstream foot-slip cost, silenced while resting."""
    cost = _feet_slip(env, sensor_name, command_name, command_threshold, asset_cfg)
    is_rest = _rest_mask(env, rest_command_name)
    if is_rest is None:
        return cost
    return cost * (~is_rest).float()


def stand_height_shortfall(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_height: float,
    rest_command_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Trunk-height penalty, active only under a zero (STAND) command.

    Height is measured as trunk-above-mean-foot rather than world z, which makes it
    terrain-relative for free on the rough terrain generator and needs no extra raycast.
    """
    asset: Entity = env.scene[asset_cfg.name]
    *_, is_stand = _command_modes(env, command_name, rest_command_name)

    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].mean(dim=1)
    height = asset.data.root_link_pos_w[:, 2] - foot_z

    shortfall = torch.clamp(1.0 - height / target_height, min=0.0)
    return torch.square(shortfall) * is_stand.float()


def rest_effort_cost(
    env: ManagerBasedRlEnv,
    rest_command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize actuator effort while resting, so REST means slack and not just a shape."""
    asset: Entity = env.scene[asset_cfg.name]
    rest = env.command_manager.get_command(rest_command_name)
    assert rest is not None
    effort = torch.sum(torch.square(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1)
    return effort * (rest[:, 0] > 0.5).float()


def rest_descent_rate(
    env: ManagerBasedRlEnv,
    rest_command_name: str,
    max_descent: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize downward base velocity in excess of ``max_descent`` while resting.

    One-sided and dead-banded: descending at up to ``max_descent`` m/s is free, so the term
    does not fight the lie-down motion itself, only the speed of it. Rising costs nothing --
    standing back up is not what this is about.
    """
    asset: Entity = env.scene[asset_cfg.name]
    rest = env.command_manager.get_command(rest_command_name)
    assert rest is not None
    is_rest = rest[:, 0] > 0.5

    descent = torch.clamp(-asset.data.root_link_lin_vel_w[:, 2] - max_descent, min=0.0)

    rest_num = is_rest.float().sum().clamp(min=1.0)
    env.extras["log"]["Metrics/rest_descent_rate"] = (descent * is_rest.float()).sum() / rest_num

    return torch.square(descent) * is_rest.float()


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


class mode_posture:
    """Joint-posture reward against a per-mode target pose.

    Implemented as a class so the target pose (a name -> angle mapping) is resolved into
    a tensor once at construction rather than on every step.

    ``target=None`` means the entity's own default joint positions, i.e. the nominal
    standing pose. ``mode`` selects which command mode the term is active in.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None

        target = cfg.params["target"]
        if cfg.params["mode"] == "rest" and not target:
            raise ValueError(
                "mode_posture(mode='rest') needs an explicit `target` pose. The base task "
                "config leaves it empty on purpose -- set it per-robot."
            )
        if target is None:
            self.target_joint_pos = default_joint_pos.clone()
        else:
            joint_ids, _, values = resolve_matching_names_values(
                data=target,
                list_of_strings=list(asset.joint_names),
            )
            self.target_joint_pos = default_joint_pos.clone()
            self.target_joint_pos[:, joint_ids] = torch.tensor(values, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        mode: Literal["stand", "rest"],
        target: dict[str, float] | None,
        std: float,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        rest_command_name: str | None = None,
    ) -> torch.Tensor:
        del target  # Resolved in __init__
        asset: Entity = env.scene[asset_cfg.name]
        _, is_rest, _, is_stand = _command_modes(env, command_name, rest_command_name)

        error = asset.data.joint_pos[:, asset_cfg.joint_ids] - self.target_joint_pos[:, asset_cfg.joint_ids]
        reward = torch.exp(-torch.mean(torch.square(error / std), dim=1))

        active = {"stand": is_stand, "rest": is_rest}.get(mode, is_stand)
        return reward * active.float()
