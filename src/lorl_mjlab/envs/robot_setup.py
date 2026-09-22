"""Per-robot wiring.

A task's config is robot-agnostic until this module fills in the parts that are not: which
sites the foot scanners ride on, which geoms count as feet/thighs/trunk, and the disturbance
magnitudes the robot's mass can absorb. :class:`QuadrupedSetup` is the data;
:func:`apply_quadruped_setup` and :func:`apply_play_overrides` are the only two steps a
per-robot config needs to call.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg, TerrainHeightSensorCfg

from lorl_mjlab.envs.icra import apply_icra_course
from lorl_mjlab.envs.scene import FOOT_SCAN_KEYS
from lorl_mjlab.terrains import IcraVariant


@dataclass(frozen=True, kw_only=True)
class QuadrupedSetup:
    robot_cfg: EntityCfg
    """The robot entity, already carrying whatever actuator model the caller chose."""

    foot_names: tuple[str, ...]
    """Robot-side leg names, in the robot's own order (e.g. ``("FR", "FL", "RR", "RL")``).

    Foot sites and ``{name}_foot_collision`` geoms are derived from these. Distinct from
    :data:`lorl_mjlab.envs.scene.FOOT_SCAN_KEYS`, which is the scanner-side ordering.
    """

    thigh_geom_names: tuple[str, ...]
    """Thigh collision geoms. Go1 splits each thigh into three, AlienGo uses one."""

    trunk_geom_names: tuple[str, ...]
    """Trunk collision geoms, including the head if the model has a separate one."""

    trunk_body: str = "trunk"

    stand_height_target: float
    """Trunk clearance above the ray-cast terrain that a standing robot must not sag below."""

    force_range: tuple[float, float]
    torque_range: tuple[float, float]

    num_envs: int = 4096

    def foot_geom_names(self) -> tuple[str, ...]:
        return tuple(f"{name}_foot_collision" for name in self.foot_names)

    def scan_key_to_site(self) -> dict[str, str]:
        """Map a scanner key (``"fl"``) to this robot's site name (``"FL"``)."""
        by_upper = {name.upper(): name for name in self.foot_names}
        return {key: by_upper[key.upper()] for key in FOOT_SCAN_KEYS}


def _wire_scanner_frames(cfg: ManagerBasedRlEnvCfg, setup: QuadrupedSetup) -> None:
    scan_key_to_site = setup.scan_key_to_site()
    for sensor in cfg.scene.sensors or ():
        if isinstance(sensor, TerrainHeightSensorCfg) and sensor.name == "trunk_height_scan":
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = setup.trunk_body
        elif isinstance(sensor, RayCastSensorCfg) and sensor.name.endswith("_foot_scan"):
            key = sensor.name.removesuffix("_foot_scan")
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = scan_key_to_site[key]


def _contact_sensors(qs: QuadrupedSetup) -> tuple[ContactSensorCfg, ...]:
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=qs.foot_geom_names(), entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    thigh_ground_cfg = ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=qs.thigh_geom_names),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=qs.trunk_geom_names),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    return (feet_ground_cfg, thigh_ground_cfg, trunk_ground_cfg)


def apply_quadruped_setup(cfg: ManagerBasedRlEnvCfg, qs: QuadrupedSetup) -> None:
    """Bind a robot-agnostic direction-command config to one robot.

    Deliberately does not touch ``cfg.actions``: the action space is the task's business, not
    the robot's, and the PMTG task has no joint-position term to scale.
    """
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.contact_sensor_maxmatch = 500

    cfg.scene.entities = {"robot": qs.robot_cfg}
    cfg.scene.num_envs = qs.num_envs

    _wire_scanner_frames(cfg, qs)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + _contact_sensors(qs)

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    cfg.viewer.body_name = qs.trunk_body
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = qs.foot_geom_names()
    cfg.events["base_inertial"].params["asset_cfg"].body_names = (qs.trunk_body,)
    cfg.events["base_external_force_torque"].params["asset_cfg"].body_names = (qs.trunk_body,)
    cfg.events["base_external_force_torque"].params["force_range"] = qs.force_range
    cfg.events["base_external_force_torque"].params["torque_range"] = qs.torque_range

    privileged = cfg.observations["privileged"].terms
    privileged["forces"].params["asset_cfg"].body_names = (qs.trunk_body,)
    privileged["torques"].params["asset_cfg"].body_names = (qs.trunk_body,)
    # Only the PMTG task observes the randomized friction; the direction task has no such term.
    if "foot_friction" in privileged:
        privileged["foot_friction"].params["asset_cfg"].geom_names = qs.foot_geom_names()


def apply_play_overrides(cfg: ManagerBasedRlEnvCfg, icra: IcraVariant | None = None) -> None:
    """Few envs, endless episodes, no corruption or pushes; optionally the fixed ICRA course."""
    cfg.episode_length_s = int(1e9)
    cfg.scene.num_envs = 15
    cfg.sim.nconmax = None
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(func=envs_mdp.randomize_terrain, mode="reset", params={})

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

    if icra is not None:
        apply_icra_course(cfg, icra)


def non_foot_collision_geoms(qs: QuadrupedSetup) -> tuple[str, ...]:
    """Every named collision geom on the robot except the four feet."""
    spec = qs.robot_cfg.spec_fn()
    feet = set(qs.foot_geom_names())
    names = tuple(geom.name for geom in spec.geoms if geom.name)
    assert all("_collision" in name for name in names), (
        f"expected only collision geoms to be named, found {[n for n in names if '_collision' not in n]}"
    )
    assert feet <= set(names), f"foot geoms missing from the model: {feet - set(names)}"
    return tuple(name for name in names if name not in feet)


def non_foot_contact_sensor(setup: QuadrupedSetup) -> ContactSensorCfg:
    """One sensor counting every non-foot body-terrain contact.

    This is the set Lee et al. eq. 16 penalises, ``|I_c,body \\ I_c,foot|``.
    """
    return ContactSensorCfg(
        name="body_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=non_foot_collision_geoms(setup)),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
