"""Rest ("lie down and go slack") command term.

Exclusivity with the direction command is enforced by :class:`DirectionWithRestCommand`,
which masks its own output on read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from lorl_mjlab.teleop import teleop_state

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class RestCommand(CommandTerm):
    """Binary "lie down" command with its own resample clock.

    The command is ``<rest>`` with ``rest in {0, 1}``. It is deliberately *not* folded
    into the direction command: resting is a mode, not a heading, and it needs to be
    resampled several times per episode so the policy has to learn to get back up.
    """

    cfg: RestCommandCfg

    def __init__(self, cfg: RestCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.rest_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.is_resting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Ramps 0 -> 1 over ``release_grace_s`` once rest drops, so a robot that is still
        # prone at the moment of release is not punished for the time it physically takes
        # to stand up. Seeded to "fully active" so a fresh episode gets no free pass.
        self._since_release = torch.full((self.num_envs,), cfg.release_grace_s, device=self.device)

        self._rest_steps = torch.zeros(self.num_envs, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, device=self.device)

        self.metrics["rest_fraction"] = torch.zeros(self.num_envs, device=self.device)

        self.teleop = teleop_state

    @property
    def command(self) -> torch.Tensor:
        """The rest command. Shape is (num_envs, 1)."""
        return self.rest_b

    def compute(self, dt: float) -> None:
        """Resample/update as usual, then let teleop override the result."""
        super().compute(dt)
        if not self.teleop.enabled:
            return
        self.is_resting[:] = self.teleop.rest
        self.rest_b[:, 0] = float(self.teleop.rest)

    @property
    def release_gate(self) -> torch.Tensor:
        """Per-env ramp in [0, 1] applied to contact penalties after a rest release."""
        if self.cfg.release_grace_s <= 0.0:
            return torch.ones_like(self._since_release)
        return (self._since_release / self.cfg.release_grace_s).clamp(0.0, 1.0)

    @property
    def rest_fraction(self) -> torch.Tensor:
        """Fraction of the current episode spent resting. Shape is (num_envs,)."""
        return self._rest_steps / self._episode_steps.clamp(min=1.0)

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        idx = slice(None) if env_ids is None else env_ids
        self._rest_steps[idx] = 0.0
        self._episode_steps[idx] = 0.0
        self._since_release[idx] = self.cfg.release_grace_s
        return super().reset(env_ids)

    def _update_metrics(self) -> None:
        self.metrics["rest_fraction"][:] = self.rest_fraction

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        self.is_resting[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rest_prob

    def _update_command(self) -> None:
        self.rest_b[:, 0] = self.is_resting.float()

        # Grace timer: held at zero while resting, accumulating once released.
        self._since_release = torch.where(
            self.is_resting,
            torch.zeros_like(self._since_release),
            self._since_release + self._env.step_dt,
        )

        self._rest_steps += self.is_resting.float()
        self._episode_steps += 1.0

    # Visualization.

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw a downward red arrow over every env currently commanded to rest."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        resting = self.is_resting.cpu().numpy()
        asset = self._env.scene[self.cfg.entity_name]
        base_pos_ws = asset.data.root_link_pos_w.cpu().numpy()

        for batch in env_indices:
            if not resting[batch]:
                continue
            base_pos_w = base_pos_ws[batch]
            if np.linalg.norm(base_pos_w) < 1e-6:
                continue
            start = base_pos_w + np.array([0.0, 0.0, self.cfg.viz.z_offset])
            end = base_pos_w + np.array([0.0, 0.0, self.cfg.viz.z_offset - self.cfg.viz.scale])
            visualizer.add_arrow(start, end, color=(0.9, 0.1, 0.1, 0.8), width=0.02)


@dataclass(kw_only=True)
class RestCommandCfg(CommandTermCfg):
    entity_name: str
    rest_prob: float = 0.05
    """Probability that a given environment is commanded to rest at each resample."""
    release_grace_s: float = 0.5
    """Seconds after a rest release during which base-contact penalties ramp back in."""

    @dataclass
    class VizCfg:
        z_offset: float = 0.5
        scale: float = 0.4

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> RestCommand:
        return RestCommand(self, env)
