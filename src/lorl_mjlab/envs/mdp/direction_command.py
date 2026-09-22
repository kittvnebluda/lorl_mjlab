"""Direction command term for heading + discrete-turn locomotion tasks."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from lorl_mjlab.teleop import build_teleop_gui, heading_vector, teleop_state

if TYPE_CHECKING:
    import viser
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_CMD_DEADBAND = 0.1
"""Magnitude below which a heading or turn command counts as absent."""

_TURN_DEADBAND = 0.1
"""[rad/s]. Yaw rate below which the robot counts as not turning."""


class DirectionCommand(CommandTerm):
    """Command generator producing a directional command vector (Lee et al., 2020).

    The command is a target horizontal heading in the robot's base frame plus a
    discrete turn direction: ``command = <cos(psi), sin(psi), turn>`` with
    ``turn in {-1, 0, 1}``. A standing command is ``<0, 0, 0>``. Unlike
    velocity-tracking commands, this only prescribes heading/turn intent -- the
    policy determines its own speed based on terrain.
    """

    cfg: DirectionCommandCfg

    def __init__(self, cfg: DirectionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]

        self.dir_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Each metric accumulates a per-env sum alongside its own step count,
        # and `reset` flushes the two as a pooled mean.
        self.metrics["cmd_angle_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["turn_sign_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["w_pr"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v_pr"] = torch.zeros(self.num_envs, device=self.device)
        self._metric_counts = {n: torch.zeros(self.num_envs, device=self.device) for n in self.metrics}

        # Episode fraction spent under a turn command. The curriculum exempts
        # turn-dominated episodes from demotion; see `turn_fraction`.
        self._turn_steps = torch.zeros(self.num_envs, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, device=self.device)

        self.teleop = teleop_state

    @property
    def command(self) -> torch.Tensor:
        """The desired direction command in the base frame. Shape is (num_envs, 3)."""
        return self.dir_command_b

    @property
    def turn_fraction(self) -> torch.Tensor:
        """Fraction of the current episode spent under a turn command. Shape (num_envs,)."""
        return self._turn_steps / self._episode_steps.clamp(min=1.0)

    def metric_mean(self, name: str, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor:
        """Pooled per-step mean of an accumulating metric over ``env_ids``."""
        idx = slice(None) if env_ids is None else env_ids
        # `clamp` rather than a branch on the count: an empty selection yields 0/1 = 0
        count = self._metric_counts[name][idx].sum()
        return self.metrics[name][idx].sum() / count.clamp(min=1.0)

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Build the teleop panel."""
        del name, get_env_idx, request_action  # Unused.
        build_teleop_gui(server, self.teleop, on_change)

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Resample/update as usual, then let teleop override the result.

        The override must land *after* ``super().compute(dt, env_ids)``: that is what runs
        ``_resample_command`` (every 10 s and on every reset) and ``_update_command``
        (which zeroes the command for standing envs every single step). Writing earlier --
        or from outside the step -- gets clobbered by one or both. This runs inside
        ``env.step`` before the observations are built, so the policy sees it the same step.
        """
        super().compute(dt, env_ids)
        if not self.teleop.enabled:
            return
        head_x, head_y = heading_vector(self.teleop)
        self.dir_command_b[:, 0] = head_x
        self.dir_command_b[:, 1] = head_y
        self.dir_command_b[:, 2] = self.teleop.turn
        # Otherwise `_update_command` re-zeroes the ~2% of envs sampled as standing.
        self.is_standing_env[:] = False

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        idx = slice(None) if env_ids is None else env_ids
        self._turn_steps[idx] = 0.0
        self._episode_steps[idx] = 0.0

        # Taken before `super().reset` zeroes the accumulators, and overriding what it reports:
        # its mean-of-sums is meaningless now that the metrics hold sums rather than averages.
        # Stacked so the whole flush costs one host sync instead of one per metric.
        names = list(self.metrics)
        pooled = torch.stack([self.metric_mean(name, env_ids) for name in names])

        extras = super().reset(env_ids)
        extras.update(zip(names, pooled.tolist()))
        for count in self._metric_counts.values():
            count[idx] = 0.0
        return extras

    def _accumulate(self, name: str, value: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Fold one step into a metric's per-env sum and step count.

        ``mask`` restricts a term to the commands it is meaningful under; masked-out steps advance
        neither, so they neither bias the sum nor inflate the divisor.
        """
        if mask is None:
            self.metrics[name] += value
            self._metric_counts[name] += 1.0
        else:
            self.metrics[name] += torch.where(mask, value, torch.zeros_like(value))
            self._metric_counts[name] += mask.float()

    def _update_metrics(self) -> None:
        cmd = self.command
        cmd_dir = cmd[:, :2]
        cmd_turn = cmd[:, 2]
        has_dir = torch.norm(cmd_dir, dim=-1) > _CMD_DEADBAND
        is_turn_cmd = cmd_turn.abs() > _CMD_DEADBAND

        vel_xy_b = self.robot.data.root_link_lin_vel_b[:, :2]
        vel_norm = torch.norm(vel_xy_b, dim=-1, keepdim=True)
        vel_dir_b = torch.where(vel_norm > 1e-5, vel_xy_b / vel_norm, torch.zeros_like(vel_xy_b))
        yaw_rate = self.robot.data.root_link_ang_vel_b[:, 2]

        # Direction alignment error (angle between commanded and actual direction). Restricted to
        # envs that have a heading to follow: a stand or pivot command has `cmd_dir == 0`, and the
        # resulting `acos(0) = pi/2` is an artefact of the zero vector, not a tracking failure.
        dot_prod = torch.sum(cmd_dir * vel_dir_b, dim=-1).clamp(-1.0, 1.0)
        self._accumulate("cmd_angle_error", torch.acos(dot_prod), has_dir)

        # Turning alignment error in [0, 2]: 0 correct, 1 not turning, 2 turning the wrong way.
        # Valid under a zero-turn command too, which is what `_TURN_DEADBAND` buys -- see there.
        actual_turn_dir = torch.sign(yaw_rate) * (yaw_rate.abs() > _TURN_DEADBAND)
        self._accumulate("turn_sign_error", torch.abs(cmd_turn - actual_turn_dir))

        # Yaw rate projected onto the commanded turn: the quantity `ang_vel_rew_13` shapes, whose
        # reward saturates at 0.6 rad/s. Sign agreement alone cannot show a robot turning at half
        # that rate, so this is the one to read when judging turn tracking.
        self._accumulate("w_pr", cmd_turn * yaw_rate, is_turn_cmd)

        # Velocity projection (v_pr) onto the commanded direction.
        self._accumulate("v_pr", torch.sum(vel_xy_b * cmd_dir, dim=-1), has_dir)

        # Direct read of the pivot share of the batch, so a change to the sampling weights is
        # visible without waiting for an episode to end and flush the accumulated metrics.
        is_pivot = ~has_dir & is_turn_cmd
        self._env.extras["log"]["Metrics/pivot_command_fraction"] = is_pivot.float().mean()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty(len(env_ids), device=self.device)

        yaw = r.uniform_(*self.cfg.ranges.yaw)
        self.dir_command_b[env_ids, 0] = torch.cos(yaw)
        self.dir_command_b[env_ids, 1] = torch.sin(yaw)

        # Discrete turn direction {-1, 0, 1}: with probability turn_prob, +/-1
        # (equal chance); otherwise 0 (no rotation).
        do_turn = r.uniform_(0.0, 1.0) <= self.cfg.turn_prob
        sign = torch.where(torch.rand(len(env_ids), device=self.device) < 0.5, -1.0, 1.0)
        self.dir_command_b[env_ids, 2] = torch.where(do_turn, sign, torch.zeros_like(sign))

        standing = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # Turn in place: zero heading, turn forced to +/-1
        pivot = (r.uniform_(0.0, 1.0) <= self.cfg.rel_turn_in_place_envs) & ~standing
        self.dir_command_b[env_ids[pivot], :2] = 0.0
        self.dir_command_b[env_ids[pivot], 2] = sign[pivot]

        self.is_standing_env[env_ids] = standing

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.dir_command_b[standing_env_ids, :] = 0.0

        # Skip on resets when `dt=0`
        if env_ids is not None:
            return
        self._turn_steps += (self.command[:, 2].abs() > 0.1).float()
        self._episode_steps += 1.0

    # Visualization.

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw goal-direction (green), current-velocity (blue), and turn (orange) arrows."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        cmds = self.command.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_mat_ws = matrix_from_quat(self.robot.data.root_link_quat_w).cpu().numpy()
        lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()

        scale = self.cfg.viz.scale
        z_offset = self.cfg.viz.z_offset

        for batch in env_indices:
            base_pos_w = base_pos_ws[batch]
            base_mat_w = base_mat_ws[batch]
            cmd = cmds[batch]
            lin_vel_b = lin_vel_bs[batch]

            if np.linalg.norm(base_pos_w) < 1e-6:
                continue

            def local_to_world(
                vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
            ) -> np.ndarray:
                return pos + mat @ vec

            origin = local_to_world(np.array([0.0, 0.0, z_offset]))

            goal_end = local_to_world(np.array([0.0, 0.0, z_offset]) + np.array([cmd[0], cmd[1], 0.0]) * scale)
            visualizer.add_arrow(origin, goal_end, color=(0.1, 0.8, 0.1, 0.8), width=0.02)

            cur_end = local_to_world(
                np.array([0.0, 0.0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0.0]) * scale
            )
            visualizer.add_arrow(origin, cur_end, color=(0.1, 0.4, 0.9, 0.8), width=0.02)

            turn = float(cmd[2])
            if abs(turn) > 0.5:
                turn_end = local_to_world(np.array([0.0, 0.0, z_offset + 0.4 * math.copysign(1.0, turn)]))
                visualizer.add_arrow(origin, turn_end, color=(0.9, 0.6, 0.0, 0.8), width=0.02)


@dataclass(kw_only=True)
class DirectionCommandCfg(CommandTermCfg):
    entity_name: str
    rel_standing_envs: float = 0.02
    """Fraction of environments that should be standing still. Defaults to 0.0."""
    turn_prob: float = 0.3
    """Probability of sampling a non-zero turn command. With probability
    ``turn_prob`` the turn direction is +/-1 (each with equal chance)."""
    rel_turn_in_place_envs: float = 0.1
    """Fraction of environments commanded to pivot in place: zero heading, turn +/-1."""

    @dataclass
    class Ranges:
        yaw: tuple[float, float] = (-math.pi, math.pi)
        """Range for the yaw command (in rad)."""

    ranges: Ranges = field(default_factory=Ranges)

    @dataclass
    class VizCfg:
        z_offset: float = 0.5
        scale: float = 0.5

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> DirectionCommand:
        return DirectionCommand(self, env)
