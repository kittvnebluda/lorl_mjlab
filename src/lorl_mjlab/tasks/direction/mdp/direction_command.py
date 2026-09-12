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

        self.metrics["cmd_angle_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["turn_sign_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v_pr"] = torch.zeros(self.num_envs, device=self.device)

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

    def compute(self, dt: float) -> None:
        """Resample/update as usual, then let teleop override the result.

        The override must land *after* ``super().compute(dt)``: that is what runs
        ``_resample_command`` (every 10 s and on every reset) and ``_update_command``
        (which zeroes the command for standing envs every single step). Writing earlier --
        or from outside the step -- gets clobbered by one or both. This runs inside
        ``env.step`` before the observations are built, so the policy sees it the same step.
        """
        super().compute(dt)
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
        return super().reset(env_ids)

    def _update_metrics(self) -> None:
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        vel_xy_b = self.robot.data.root_link_lin_vel_b[:, :2]
        vel_norm = torch.norm(vel_xy_b, dim=-1, keepdim=True)
        vel_dir_b = torch.where(vel_norm > 1e-5, vel_xy_b / vel_norm, torch.zeros_like(vel_xy_b))

        # Read through ``command``, not ``dir_command_b``: a subclass that masks the command
        # (see ``DirectionWithRestCommand``) must not accrue error against a heading the robot
        # was told to ignore.
        cmd = self.command
        cmd_dir = cmd[:, :2]

        # Direction alignment error (angle between commanded and actual direction).
        dot_prod = torch.sum(cmd_dir * vel_dir_b, dim=-1).clamp(-1.0, 1.0)
        dir_error = torch.acos(dot_prod)
        self.metrics["cmd_angle_error"] += dir_error / max_command_step

        # Turning alignment error: commanded turn dir vs. sign of actual yaw rate.
        actual_turn_dir = torch.sign(self.robot.data.root_link_ang_vel_b[:, 2])
        turn_error = torch.abs(cmd[:, 2] - actual_turn_dir)
        self.metrics["turn_sign_error"] += turn_error / max_command_step

        # Velocity projection (v_pr) onto the commanded direction.
        v_pr = torch.sum(self.robot.data.root_link_lin_vel_b[:, :2] * cmd_dir, dim=-1)
        self.metrics["v_pr"] += v_pr / max_command_step

        # Direct read of the pivot share of the batch, so a change to the sampling weights is
        # visible without waiting for an episode to end and flush the accumulated metrics.
        is_pivot = (torch.norm(cmd_dir, dim=1) < 0.1) & (cmd[:, 2].abs() > 0.1)
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

    def _update_command(self) -> None:
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.dir_command_b[standing_env_ids, :] = 0.0

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


class DirectionWithRestCommand(DirectionCommand):
    """Direction command that goes silent wherever a rest command is active.

    ``dir_command_b`` keeps the raw sampled heading, so the command survives a rest episode
    and comes back unchanged on release instead of waiting for the next resample.
    """

    def __init__(self, cfg: DirectionWithRestCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._rest_command_name = cfg.rest_command_name

    @property
    def command(self) -> torch.Tensor:
        """The direction command, zeroed while resting. Shape is (num_envs, 3)."""
        rest = self._env.command_manager.get_command(self._rest_command_name)
        assert rest is not None
        return self.dir_command_b * (1.0 - rest)


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


@dataclass(kw_only=True)
class DirectionWithRestCommandCfg(DirectionCommandCfg):
    rest_command_name: str = "rest"
    """Name of the rest command term whose active envs zero this command."""

    def build(self, env: ManagerBasedRlEnv) -> DirectionWithRestCommand:
        return DirectionWithRestCommand(self, env)
