"""Left-right (sagittal) symmetry machinery shared by the legged locomotion tasks.

The observation mirrors here are task-agnostic: they depend on the robot's joint naming
convention and on the shared observation terms, not on which action space a task uses. Each task
supplies its own action mirror -- the action layout is the one piece that genuinely differs --
and passes it to :func:`augment`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import RingPatternCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from tensordict import TensorDict

MIRROR_LEG = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}

# Foot scanner block order [fl, fr, rl, rr] -> [fr, fl, rr, rl].
FOOT_PERM = [1, 0, 3, 2]
FOOT_SCAN_TERMS = ("fl_foot_scan", "fr_foot_scan", "rl_foot_scan", "rr_foot_scan")


def lr_perm_from_names(names: list[str]) -> tuple[list[int], list[int]]:
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
        if leg not in MIRROR_LEG:
            continue
        mirrored_name = f"{MIRROR_LEG[leg]}_{joint}"
        perm[i] = name_to_idx[mirrored_name]
        if joint.startswith("hip_joint"):
            hip_idx.append(i)
    return perm, hip_idx


def mirror_by_perm(x: torch.Tensor, perm: list[int], hip_idx: list[int]) -> torch.Tensor:
    """Mirror a length-N vector via a left<->right permutation, flipping hip signs."""
    out = x[..., perm].clone()
    if hip_idx:
        out[..., hip_idx] *= -1.0
    return out


def scan_ring_perm(pattern: RingPatternCfg) -> list[int]:
    """Index permutation that reflects one foot scan across the sagittal plane.

    Derived from the pattern rather than hardcoded: tasks pick their own ring counts, and a
    stale constant here silently mirrors the wrong rays (or, if the lengths happen to differ,
    blows up at the assignment). ``RingPatternCfg.generate_rays`` emits the optional centre ray
    first, then each ring counter-clockwise from +x, so reflecting y negates the angle: sample
    ``i`` of an ``n``-sample ring swaps with ``(n - i) % n`` and the centre maps to itself.
    """
    perm = []
    offset = 0
    if pattern.include_center:
        perm.append(offset)
        offset += 1
    for ring in pattern.rings:
        n = ring.num_samples
        perm.extend(offset + ((n - i) % n) for i in range(n))
        offset += n
    return perm


def foot_scan_perm(env: ManagerBasedRlEnv) -> list[int]:
    """Read the live scan pattern off the first foot scanner and build its sagittal mirror."""
    pattern = env.scene[FOOT_SCAN_TERMS[0]].cfg.pattern
    assert isinstance(pattern, RingPatternCfg), f"{FOOT_SCAN_TERMS[0]} is not a ring pattern: {type(pattern)}"
    return scan_ring_perm(pattern)


def switch_foot_scans_lr(x: torch.Tensor, scan_len: int, ring_perm: list[int]) -> torch.Tensor:
    """Mirror the 4 concatenated foot scans: swap fl/fr & rl/rr blocks, reflect each ring."""
    assert len(ring_perm) == scan_len, f"scan block is {scan_len} rays, mirror permutation is {len(ring_perm)}"
    out = x.clone()
    for dst, src in enumerate(FOOT_PERM):
        d0, s0 = dst * scan_len, src * scan_len
        block = x[..., s0 : s0 + scan_len]
        out[..., d0 : d0 + scan_len] = block[..., ring_perm]
    return out


def vec_sign(x: torch.Tensor, signs: list[float]) -> torch.Tensor:
    return x * torch.tensor(signs, device=x.device, dtype=x.dtype)


def joint_perm(env: ManagerBasedRlEnv) -> tuple[list[int], list[int]]:
    robot = env.unwrapped.scene["robot"]
    return lr_perm_from_names(list(robot.joint_names))


def gains_perm(env: ManagerBasedRlEnv) -> tuple[list[int], list[int]]:
    robot = env.unwrapped.scene["robot"]
    actuator_target = {a.id: a.target.split("/")[-1] for a in robot.spec.actuators}
    ctrl_ids = robot.data.indexing.ctrl_ids
    names = [actuator_target[int(cid)] for cid in ctrl_ids]
    return lr_perm_from_names(names)


def mirror_gains(env: ManagerBasedRlEnv, x: torch.Tensor) -> torch.Tensor:
    # stiffness(N) + damping(N): permute each half, no sign flip.
    perm, _ = gains_perm(env)
    half = x.shape[-1] // 2
    out = x.clone()
    out[..., :half] = x[..., :half][..., perm]
    out[..., half:] = x[..., half:][..., perm]
    return out


def mirror_foot_contacts(x: torch.Tensor) -> torch.Tensor:
    """Mirror one scalar per foot: [fl, fr, rl, rr] -> [fr, fl, rr, rl].

    ``FOOT_PERM`` swaps adjacent pairs, so it is equally correct for the scanner-side key order and
    for the robots' own ``("FR", "FL", "RR", "RL")``. Scalars carry no frame, so nothing flips sign.
    """
    return x[..., FOOT_PERM].clone()


def base_term_transforms(env: ManagerBasedRlEnv) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Mirrors for the observation terms every legged task shares.

    Excludes ``actions`` (and any other action-space-dependent term): those are supplied per
    task via ``extra_transforms``.
    """
    perm, hip_idx = joint_perm(env)
    return {
        "joint_pos": lambda x: mirror_by_perm(x, perm, hip_idx),
        "joint_vel": lambda x: mirror_by_perm(x, perm, hip_idx),
        "base_ang_vel": lambda x: vec_sign(x, [-1.0, 1.0, -1.0]),
        "base_lin_vel": lambda x: vec_sign(x, [1.0, -1.0, 1.0]),
        "projected_gravity": lambda x: vec_sign(x, [1.0, -1.0, 1.0]),
        "command": lambda x: vec_sign(x, [1.0, -1.0, -1.0]),
        "forces": lambda x: vec_sign(x, [1.0, -1.0, 1.0]),
        "torques": lambda x: vec_sign(x, [-1.0, 1.0, -1.0]),
        "actuator_gains": lambda x: mirror_gains(env, x),
        "foot_contacts": mirror_foot_contacts,
    }


def _mirror_term(
    x: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor],
    cfg: ObservationTermCfg,
) -> torch.Tensor:
    """Apply a single-frame mirror to a term slice, unpacking a flattened history first.

    A history term is stored term-major and flattened -- ``[x_t0, x_t1, ...]``, oldest first -- so the
    single-frame mirror has to run per frame. Mirroring the flat slice in one go would permute across
    frame boundaries.
    """
    history = cfg.history_length
    if history <= 1:
        return transform(x)
    assert cfg.flatten_history_dim, "unflattened history is not supported by the sagittal mirror"
    frames = x.reshape(*x.shape[:-1], history, x.shape[-1] // history)
    return transform(frames).reshape(x.shape)


def transform_group(
    env: ManagerBasedRlEnv,
    group: str,
    obs: torch.Tensor,
    extra_transforms: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> torch.Tensor:
    """Apply the left-right mirror to a concatenated observation group tensor."""
    om = env.observation_manager
    term_names = om.active_terms[group]
    term_dims = [int(d[0]) if isinstance(d, (tuple, list)) else int(d) for d in om.group_obs_term_dim[group]]

    term_transforms = base_term_transforms(env)
    if extra_transforms is not None:
        term_transforms.update(extra_transforms)

    out = obs.clone()
    scan_start: int | None = None
    scan_len: int = 0
    offset = 0
    for name, dim in zip(term_names, term_dims):
        sl = slice(offset, offset + dim)
        if name in FOOT_SCAN_TERMS:
            if scan_start is None:
                scan_start = offset
                scan_len = dim
        else:
            # An un-mirrored term is invisible to an involution test -- identity *is* an involution --
            # so a forgotten mirror would silently train on a wrong augmentation. Fail instead.
            assert name in term_transforms, f"observation term '{group}.{name}' has no sagittal mirror"
            out[..., sl] = _mirror_term(obs[..., sl], term_transforms[name], om.get_term_cfg(group, name))
        offset += dim

    if scan_start is not None:
        span = slice(scan_start, scan_start + 4 * scan_len)
        out[..., span] = switch_foot_scans_lr(obs[..., span], scan_len, foot_scan_perm(env))
    return out


@torch.no_grad()
def augment(
    env: ManagerBasedRlEnv,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
    mirror_actions: Callable[[ManagerBasedRlEnv, torch.Tensor], torch.Tensor],
    extra_transforms: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
):
    """rsl rl symmetry augmentation: append a sagittally-mirrored copy (2x batch).

    ``mirror_actions`` mirrors one action tensor for the task's action space; a task that also
    *observes* its last action should reuse it for that term via ``extra_transforms``.

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
            obs_aug[group][batch_size:] = transform_group(env.unwrapped, group, group_obs, extra_transforms)
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = mirror_actions(env, actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
