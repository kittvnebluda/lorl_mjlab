"""Left-right (sagittal) symmetry augmentation for the direction task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

_MIRROR_LEG = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}

# Foot scanner block order [fl, fr, rl, rr] -> [fr, fl, rr, rl].
_FOOT_PERM = [1, 0, 3, 2]
# Per-foot concentric-ring scan ring sizes (must match direction_env_cfg._FOOT_SCAN_PATTERN).
_RING_SIZES = (6, 12, 18)
_FOOT_SCAN_TERMS = ("fl_foot_scan", "fr_foot_scan", "rl_foot_scan", "rr_foot_scan")


def _lr_perm_from_names(names: list[str]) -> tuple[list[int], list[int]]:
    """Build a left<->right index permutation and hip-sign-flip indices from names.

    Names are expected to follow the ``{LEG}_{joint}`` convention (LEG in
    FL/FR/RL/RR), e.g. ``FR_hip_joint``. Returns ``(perm, hip_idx)`` where
    ``perm[i]`` is the source index whose value moves to position ``i`` after
    mirroring, and ``hip_idx`` lists positions that flip sign (ab/adduction).
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    perm = list(range(len(names)))
    hip_idx = []
    for i, name in enumerate(names):
        leg, _, joint = name.partition("_")
        if leg not in _MIRROR_LEG:
            continue
        mirrored_name = f"{_MIRROR_LEG[leg]}_{joint}"
        perm[i] = name_to_idx[mirrored_name]
        if joint.startswith("hip_joint"):
            hip_idx.append(i)
    return perm, hip_idx


def _mirror_by_perm(x: torch.Tensor, perm: list[int], hip_idx: list[int]) -> torch.Tensor:
    """Mirror a length-N vector via a left<->right permutation, flipping hip signs."""
    out = x[..., perm].clone()
    if hip_idx:
        out[..., hip_idx] *= -1.0
    return out


def _scan_ring_perm(ring_sizes=_RING_SIZES) -> list[int]:
    """Index permutation that reflects one foot scan across the sagittal plane."""
    perm = []
    offset = 0
    for n in ring_sizes:
        perm.extend(offset + ((n - i) % n) for i in range(n))
        offset += n
    return perm


_SCAN_PERM = _scan_ring_perm()


def _switch_foot_scans_lr(x: torch.Tensor, scan_len: int) -> torch.Tensor:
    """Mirror the 4 concatenated foot scans: swap fl/fr & rl/rr blocks, reflect each ring."""
    out = x.clone()
    for dst, src in enumerate(_FOOT_PERM):
        d0, s0 = dst * scan_len, src * scan_len
        block = x[..., s0 : s0 + scan_len]
        out[..., d0 : d0 + scan_len] = block[..., _SCAN_PERM]
    return out


def _vec_sign(x: torch.Tensor, signs: list[float]) -> torch.Tensor:
    return x * torch.tensor(signs, device=x.device, dtype=x.dtype)


def _joint_perm(env: ManagerBasedRlEnv) -> tuple[list[int], list[int]]:
    robot = env.unwrapped.scene["robot"]
    return _lr_perm_from_names(list(robot.joint_names))


def _action_perm(env: ManagerBasedRlEnv) -> tuple[list[int], list[int]]:
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    assert isinstance(action_term, JointPositionAction)
    return _lr_perm_from_names(list(action_term.target_names))


def _gains_perm(env: ManagerBasedRlEnv) -> tuple[list[int], list[int]]:
    robot = env.unwrapped.scene["robot"]
    actuator_target = {a.id: a.target.split("/")[-1] for a in robot.spec.actuators}
    ctrl_ids = robot.data.indexing.ctrl_ids
    names = [actuator_target[int(cid)] for cid in ctrl_ids]
    return _lr_perm_from_names(names)


def _t_gains(env: ManagerBasedRlEnv, x: torch.Tensor) -> torch.Tensor:
    # stiffness(N) + damping(N): permute each half, no sign flip.
    perm, _ = _gains_perm(env)
    half = x.shape[-1] // 2
    out = x.clone()
    out[..., :half] = x[..., :half][..., perm]
    out[..., half:] = x[..., half:][..., perm]
    return out


def _t_foot_contacts(x: torch.Tensor) -> torch.Tensor:
    # [fl, fr, rl, rr] -> [fr, fl, rr, rl]
    return x[..., _FOOT_PERM].clone()


def _transform_group(env: ManagerBasedRlEnv, group: str, obs: torch.Tensor) -> torch.Tensor:
    """Apply the left-right mirror to a concatenated observation group tensor."""
    om = env.observation_manager
    term_names = om.active_terms[group]
    term_dims = [int(d[0]) if isinstance(d, (tuple, list)) else int(d) for d in om.group_obs_term_dim[group]]

    joint_perm, hip_idx = _joint_perm(env)

    term_transforms = {
        "joint_pos": lambda x: _mirror_by_perm(x, joint_perm, hip_idx),
        "joint_vel": lambda x: _mirror_by_perm(x, joint_perm, hip_idx),
        "actions": lambda x: _mirror_by_perm(x, *_action_perm(env)),
        "base_ang_vel": lambda x: _vec_sign(x, [-1.0, 1.0, -1.0]),
        "base_lin_vel": lambda x: _vec_sign(x, [1.0, -1.0, 1.0]),
        "projected_gravity": lambda x: _vec_sign(x, [1.0, -1.0, 1.0]),
        "command": lambda x: _vec_sign(x, [1.0, -1.0, -1.0]),
        "rest_command": lambda x: x.clone(),
        "forces": lambda x: _vec_sign(x, [1.0, -1.0, 1.0]),
        "torques": lambda x: _vec_sign(x, [-1.0, 1.0, -1.0]),
        "actuator_gains": lambda x: _t_gains(env, x),
        "foot_contacts": _t_foot_contacts,
    }

    out = obs.clone()
    scan_start: int | None = None
    scan_len: int = 0
    offset = 0
    for name, dim in zip(term_names, term_dims):
        sl = slice(offset, offset + dim)
        if name in _FOOT_SCAN_TERMS:
            if scan_start is None:
                scan_start = offset
                scan_len = dim
        elif name in term_transforms:
            out[..., sl] = term_transforms[name](obs[..., sl])
        # Unknown terms left as-is (identity).
        offset += dim

    if scan_start is not None:
        span = slice(scan_start, scan_start + 4 * scan_len)
        out[..., span] = _switch_foot_scans_lr(obs[..., span], scan_len)
    return out


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRlEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """rsl_rl symmetry augmentation: append a sagittally-mirrored copy (2x batch).

    Returns ``(obs_aug, actions_aug)`` each ``[original, mirrored]`` stacked
    along the batch, or None for whichever input was None.
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2, *(1,) * (obs.ndim - 1))  # type: ignore
        # Keep `.keys()`: TensorDict.__iter__ walks the batch dim, not the keys.
        for group in obs.keys():  # noqa: SIM118
            assert isinstance(group, str)
            group_obs = obs[group]
            assert isinstance(group_obs, torch.Tensor)
            obs_aug[group][:batch_size] = group_obs[:]
            obs_aug[group][batch_size:] = _transform_group(env.unwrapped, group, group_obs)
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        perm, hip_idx = _action_perm(env)
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _mirror_by_perm(actions, perm, hip_idx)
    else:
        actions_aug = None

    return obs_aug, actions_aug
