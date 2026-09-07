"""The ICRA2024 QRC competition course as an mjlab sub-terrain.
Run this module to view a course:

    uv run python -m lorl_mjlab.terrains.icra_terrain [sloped]

"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane

from lorl_mjlab.terrains import icra_map
from lorl_mjlab.terrains.icra_map import IcraVariant

FLOOR_RGBA = (0.35, 0.36, 0.38, 1.0)

ATTACH_PREFIX = "icra_"


@dataclass(kw_only=True)
class IcraMapTerrainCfg(SubTerrainCfg):
    """The ICRA2024 QRC course, placed as a single sub-terrain patch."""

    variant: IcraVariant = "flat"
    """Which course layout to build. The sloped variant raises the lanes onto ramps."""
    floor_thickness: float = 1.0
    """Thickness of the solid slab under the course, in meters. The course itself has no ground."""

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del difficulty, rng

        body = spec.body("terrain")
        center = np.array([self.size[0] / 2.0, self.size[1] / 2.0, 0.0])

        geometries = []
        for floor in make_plane(body, self.size, 0.0, center_zero=False, plane_thickness=self.floor_thickness):
            floor.rgba[:] = FLOOR_RGBA
            geometries.append(TerrainGeometry(geom=floor))

        placed = len(body.geoms)
        spec.attach(icra_map.load(self.variant), prefix=ATTACH_PREFIX, frame=body.add_frame(pos=center.tolist()))
        course = body.geoms[placed:]
        if not course:
            raise ValueError(f"the {self.variant} map attached no geoms")
        geometries.extend(TerrainGeometry(geom=geom) for geom in course)

        start_xy, _ = icra_map.start_pose(self.variant)
        return TerrainOutput(
            origin=np.array([start_xy[0] + center[0], start_xy[1] + center[1], 0.0]), geometries=geometries
        )


def icra_patch_size(variant: IcraVariant, margin: float = 1.0) -> tuple[float, float]:
    """Terrain patch size that fits the course footprint plus a walkable margin, in meters."""
    extent = icra_map.footprint(variant)
    return (float(extent[0]) + 2 * margin, float(extent[1]) + 2 * margin)


def icra_start_yaw(variant: IcraVariant) -> float:
    """Heading the robot faces at spawn, in radians, as recorded by the map."""
    return icra_map.start_pose(variant)[1]


if __name__ == "__main__":
    import sys
    import time

    from mjlab.terrains.terrain_generator import TerrainGenerator
    from mujoco import viewer

    from lorl_mjlab.terrains.config import icra_terrains_cfg

    chosen: IcraVariant = "sloped" if "sloped" in sys.argv else "flat"
    viewer_spec = mujoco.MjSpec()
    TerrainGenerator(cfg=icra_terrains_cfg(chosen), device="cpu").compile(viewer_spec)
    model = viewer_spec.compile()
    print(f"[icra] {chosen}: {model.ngeom} geoms, {model.nmesh} meshes")

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with viewer.launch_passive(model, data) as handle:
        handle.cam.lookat[:] = (0.0, 0.0, 0.0)
        handle.cam.distance = 20.0
        handle.cam.azimuth = 90.0
        handle.cam.elevation = -35.0
        handle.sync()
        while handle.is_running():
            time.sleep(0.02)
