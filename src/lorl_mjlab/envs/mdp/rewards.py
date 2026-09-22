from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def command_modes(
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


def ang_vel_rew_13(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Turn-tracking reward that also penalizes yaw spin when no turn is commanded.

    Lee et al. 2020 eq. 13 verbatim for the turning case; the zero-turn branch is extension,
    as the paper is not specific on this case
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
