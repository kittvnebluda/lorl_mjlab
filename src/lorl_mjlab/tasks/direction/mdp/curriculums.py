"""Terrain curriculum for the direction command family."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .direction_command import DirectionCommand
from .rest_command import RestCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def terrain_levels_dir(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    rest_command_name: str | None = None,
    rest_fraction_threshold: float = 0.25,
    turn_fraction_threshold: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> dict[str, torch.Tensor]:
    """Terrain curriculum that promotes or demotes an env based on how far the robot travelled.

    At every reset we measure ``distance``: the straight-line distance in the world XY plane between the
    robot's final base position and the origin of the sub-terrain it was spawned on.

    - Covered more than half a tile -> promote to a harder level.
    - Covered less than a fifth of a tile -> demote to an easier level.
    - Anything in between -> stay put.

    Demotion is skipped for episodes that were not supposed to cover ground -- standing envs, and episodes
    that spent at least ``rest_fraction_threshold`` / ``turn_fraction_threshold`` of their steps resting or
    turning. Rest and turn are judged over the whole episode, not by the command's value at reset time.

    ``rest_command_name`` is optional: a task configured without a rest command simply has no rest
    exemption to apply.
    """
    asset: Entity = env.scene[asset_cfg.name]

    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command_term = cast(DirectionCommand, env.command_manager.get_term(command_name))
    assert command_term is not None
    rest_term = None
    if rest_command_name is not None:
        rest_term = cast(RestCommand, env.command_manager.get_term(rest_command_name))
        assert rest_term is not None

    tile = terrain_generator.size[0]

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )

    was_turning = command_term.turn_fraction[env_ids] >= turn_fraction_threshold
    is_standing = command_term.is_standing_env[env_ids]

    move_up = distance > tile * 0.5
    move_down = distance < tile * 0.2
    move_down = move_down & ~is_standing & ~was_turning
    if rest_term is not None:
        move_down = move_down & ~(rest_term.rest_fraction[env_ids] >= rest_fraction_threshold)

    # The first reset happens before any episode has been played: levels are still the
    # random `max_init_terrain_level` spread and `distance` is ~0 for everyone, so acting on
    # it would demote every env for not having moved yet. Upstream guards this the same way.
    if env.common_step_counter == 0:
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)

    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "min": levels.min(),
        "mean": levels.mean(),
        "max": levels.max(),
        "distance": distance.mean(),
        "v_pr": command_term.metrics["v_pr"][env_ids].mean(),
    }

    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    terrain_origins = terrain.terrain_origins
    assert terrain_origins is not None
    if terrain_generator.curriculum:
        types = terrain.terrain_types
        for i, name in enumerate(sub_terrain_names):
            mask = types == i
            if mask.any():
                result[f"level/{name}"] = levels[mask].mean()

    return result
