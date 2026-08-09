"""Direction command term for heading + discrete-turn locomotion tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformDirectionCommand(CommandTerm):
    r"""Command generator producing a directional command vector (Lee et al., 2020).

    The command is a target horizontal heading in the robot's base frame plus a
    discrete turn direction: ``command = <cos(psi), sin(psi), turn>`` with
    ``turn in {-1, 0, 1}``. A standing command is ``<0, 0, 0>``. Unlike
    velocity-tracking commands, this only prescribes heading/turn intent -- the
    policy determines its own speed based on terrain.
    """

    cfg: UniformDirectionCommandCfg

    def __init__(self, cfg: UniformDirectionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]

        self.dir_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.metrics["cmd_angle_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["turn_sign_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v_pr"] = torch.zeros(self.num_envs, device=self.device)

        # Command-aligned progress integrated per step; read by the terrain curriculum.
        self.command_progress = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The desired direction command in the base frame. Shape is (num_envs, 3)."""
        return self.dir_command_b

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if env_ids is None:
            self.command_progress[:] = 0.0
        else:
            self.command_progress[env_ids] = 0.0
        return super().reset(env_ids)

    def _update_metrics(self) -> None:
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        vel_xy_b = self.robot.data.root_link_lin_vel_b[:, :2]
        vel_norm = torch.norm(vel_xy_b, dim=-1, keepdim=True)
        vel_dir_b = torch.where(vel_norm > 1e-5, vel_xy_b / vel_norm, torch.zeros_like(vel_xy_b))

        cmd_dir = self.dir_command_b[:, :2]

        # Direction alignment error (angle between commanded and actual direction).
        dot_prod = torch.sum(cmd_dir * vel_dir_b, dim=-1).clamp(-1.0, 1.0)
        dir_error = torch.acos(dot_prod)
        self.metrics["cmd_angle_error"] += dir_error / max_command_step

        # Turning alignment error: commanded turn dir vs. sign of actual yaw rate.
        actual_turn_dir = torch.sign(self.robot.data.root_link_ang_vel_b[:, 2])
        turn_error = torch.abs(self.dir_command_b[:, 2] - actual_turn_dir)
        self.metrics["turn_sign_error"] += turn_error / max_command_step

        # Velocity projection (v_pr) onto the commanded direction.
        v_pr = torch.sum(self.robot.data.root_link_lin_vel_b[:, :2] * cmd_dir, dim=-1)
        self.metrics["v_pr"] += v_pr / max_command_step

        self.command_progress += v_pr * self._env.step_dt

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

        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self) -> None:
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.dir_command_b[standing_env_ids, :] = 0.0

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
class UniformDirectionCommandCfg(CommandTermCfg):
    entity_name: str
    rel_standing_envs: float = 0.0
    """Fraction of environments that should be standing still. Defaults to 0.0."""
    turn_prob: float = 0.3
    """Probability of sampling a non-zero turn command. With probability
  ``turn_prob`` the turn direction is +/-1 (each with equal chance);
  otherwise it is 0 (no rotation)."""

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

    def build(self, env: ManagerBasedRlEnv) -> UniformDirectionCommand:
        return UniformDirectionCommand(self, env)
