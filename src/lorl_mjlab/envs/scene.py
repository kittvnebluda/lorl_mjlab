from __future__ import annotations

from copy import deepcopy

from mjlab.scene import SceneCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from lorl_mjlab.terrains import ROUGH_TERRAINS_CFG

FOOT_SCAN_KEYS = ("fl", "fr", "rl", "rr")
"""Scanner-side leg keys. Distinct from the robots' geom/site naming ("FR", "FL", ...)."""


def foot_scanner(foot: str, pattern: RingPatternCfg) -> RayCastSensorCfg:
    """One downward concentric-ring height scanner attached to a single foot site."""
    return RayCastSensorCfg(
        name=f"{foot}_foot_scan",
        frame=ObjRef(type="site", name="", entity="robot"),  # Set per-robot.
        ray_alignment="yaw",
        pattern=pattern,
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # Terrain only.
        debug_vis=False,
    )


def trunk_height_scanner() -> TerrainHeightSensorCfg:
    """Single downward ray under the trunk, for terrain-relative body height."""
    return TerrainHeightSensorCfg(
        name="trunk_height_scan",
        frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
        ray_alignment="yaw",
        pattern=RingPatternCfg(rings=(), include_center=True),
        max_distance=2.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # Terrain only.
        reduction="mean",
        debug_vis=False,
    )


def make_scene_cfg(*, foot_scan_pattern: RingPatternCfg) -> SceneCfg:
    """Rough-terrain scene with the four foot scanners and the trunk height ray."""
    sensors = tuple(foot_scanner(foot, foot_scan_pattern) for foot in FOOT_SCAN_KEYS) + (trunk_height_scanner(),)
    return SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=deepcopy(ROUGH_TERRAINS_CFG),
            max_init_terrain_level=5,
        ),
        sensors=sensors,
        extent=2.0,
    )


def make_sim_cfg() -> SimulationCfg:
    return SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.005,
            iterations=10,
            ls_iterations=20,
        ),
        nconmax=35,
    )
