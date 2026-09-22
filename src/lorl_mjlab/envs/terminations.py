"""Episode terminations shared by the direction-command tasks."""

from __future__ import annotations

from mjlab.managers.termination_manager import TerminationTermCfg

from lorl_mjlab.envs import mdp


def base_terminations() -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "out_of_terrain_bounds": TerminationTermCfg(func=mdp.out_of_terrain_bounds, time_out=True),
        "flipped": TerminationTermCfg(func=mdp.bad_orientation, time_out=False, params={"limit_angle": 1.4}),
    }
