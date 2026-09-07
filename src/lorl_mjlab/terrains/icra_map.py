"""Loading the ICRA2024 QRC course maps."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

import numpy as np
from mujoco import MjSpec

from lorl_mjlab import LORL_SRC_PATH

IcraVariant = Literal["flat", "sloped"]
VARIANTS: tuple[IcraVariant, ...] = get_args(IcraVariant)

ICRA_ASSET_DIR: Path = LORL_SRC_PATH / "terrains" / "assets" / "icra"

# Numeric fields the converter writes and this module reads, named here so the two cannot drift.
EXTENT_FIELD = "course_extent"
"""Bounding-box size of the course, as ``(x, y, z)`` in meters."""
SPAWN_FIELD = "spawn"
"""Robot spawn pad in the map frame, as ``(x, y, yaw)``."""


def map_path(variant: IcraVariant) -> Path:
    return ICRA_ASSET_DIR / f"{variant}.xml"


def load(variant: IcraVariant) -> MjSpec:
    """Parse one course variant into a fresh spec.

    Fresh every call: attaching a spec splices it into the parent, so a shared instance cannot be
    reused across terrain builds.
    """
    path = map_path(variant)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; generate it with `uv run convert-icra-map`")
    return MjSpec.from_file(str(path))


def footprint(variant: IcraVariant) -> np.ndarray:
    """Bounding-box size of the course, in meters, as ``(x, y, z)``."""
    return _field(variant, EXTENT_FIELD, 3)


def start_pose(variant: IcraVariant) -> tuple[np.ndarray, float]:
    """Spawn pad ``(xy, yaw)`` in the map frame."""
    spawn = _field(variant, SPAWN_FIELD, 3)
    return spawn[:2], float(spawn[2])


def _field(variant: IcraVariant, name: str, size: int) -> np.ndarray:
    """Read one numeric field off a map, without compiling it."""
    spec = load(variant)
    if not any(numeric.name == name for numeric in spec.numerics):
        raise ValueError(f"{map_path(variant)} has no '{name}' field; regenerate it with `uv run convert-icra-map`")
    data = np.asarray(spec.numeric(name).data, dtype=np.float64)
    if data.shape != (size,):
        raise ValueError(f"{map_path(variant)} field '{name}' has {data.shape} values, expected {size}")
    return data
