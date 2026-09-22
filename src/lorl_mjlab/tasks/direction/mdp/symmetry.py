"""Left-right (sagittal) symmetry augmentation for the direction task.

Only the action mirror is task-specific: the direction task acts in joint space, so its 12-D
action mirrors exactly like a joint vector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction

from lorl_mjlab.envs.mdp.symmetry import augment, lr_perm_from_names, mirror_by_perm

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from tensordict import TensorDict


def _action_perm(env: ManagerBasedRlEnv) -> tuple[list[int], list[int]]:
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    assert isinstance(action_term, JointPositionAction)
    return lr_perm_from_names(list(action_term.target_names))


def _mirror_actions(env: ManagerBasedRlEnv, actions: torch.Tensor) -> torch.Tensor:
    return mirror_by_perm(actions, *_action_perm(env))


def compute_symmetric_states(
    env: ManagerBasedRlEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """rsl_rl symmetry augmentation: append a sagittally-mirrored copy (2x batch)."""
    return augment(env, obs, actions, _mirror_actions, extra_transforms={"actions": lambda x: _mirror_actions(env, x)})
