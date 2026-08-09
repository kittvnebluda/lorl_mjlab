"""Terrain curriculum for the direction command family."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import UniformDirectionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def terrain_levels_dir(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> dict[str, torch.Tensor]:
    """Terrain curriculum driven by command-aligned progress over the whole episode.

    Uses ``UniformDirectionCommand.command_progress`` -- the per-step integral of
    base-frame velocity projected onto the command active that step. Unlike net
    world displacement, this does not cancel when an opposite command is resampled
    mid-episode: a robot that tracks every command keeps accumulating distance.
    Promote when it advanced far along its commands, demote when it barely moved.
    Standing/turning-dominated episodes are guarded against demotion.
    """
    asset: Entity = env.scene[asset_cfg.name]

    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command_term = cast(UniformDirectionCommand, env.command_manager.get_term(command_name))
    assert command_term is not None

    tile = terrain_generator.size[0]

    command = command_term.command[env_ids]
    progress = command_term.command_progress[env_ids]
    v_pr = command_term.metrics["v_pr"][env_ids]

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )

    is_turning = torch.abs(command[:, 2]) >= 0.1
    is_standing = command_term.is_standing_env[env_ids]

    move_up = distance > tile * 0.5
    move_down = (progress < tile * 0.2) | (v_pr < 0.2)
    move_down = move_down & ~is_turning & ~is_standing

    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    return {"min": levels.min(), "mean": levels.mean(), "max": levels.max()}
