"""Unitree Go1 direction environment configurations."""

from mjlab.asset_zoo.robots import GO1_ACTION_SCALE, get_go1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg

from lorl_mjlab.tasks.direction.direction_env_cfg import make_direction_env_cfg

FOOT_NAMES = ("FR", "FL", "RR", "RL")
_SCAN_KEY_TO_SITE = {"fl": "FL", "fr": "FR", "rl": "RL", "rr": "RR"}


def unitree_go1_direction_env_cfg(
    play: bool = False,
    distill: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go1 direction-command configuration."""
    cfg = make_direction_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.contact_sensor_maxmatch = 500

    cfg.scene.entities = {"robot": get_go1_robot_cfg()}
    cfg.scene.num_envs = 4096

    # Wire foot scan sensors to per-foot sites.
    for sensor in cfg.scene.sensors or ():
        if isinstance(sensor, RayCastSensorCfg) and sensor.name.endswith("_foot_scan"):
            key = sensor.name.removesuffix("_foot_scan")
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = _SCAN_KEY_TO_SITE[key]

    geom_names = tuple(f"{name}_foot_collision" for name in FOOT_NAMES)
    thigh_geom_names = tuple(f"{leg}_thigh_collision{i}" for leg in FOOT_NAMES for i in (1, 2, 3))

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    thigh_ground_cfg = ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=thigh_geom_names),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    trunk_ground_cfg = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=("trunk_collision", "head_collision")),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        thigh_ground_cfg,
        trunk_ground_cfg,
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = GO1_ACTION_SCALE

    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["add_base_mass"].params["asset_cfg"].body_names = ("trunk",)
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)
    cfg.events["base_external_force_torque"].params["asset_cfg"].body_names = ("trunk",)
    cfg.events["base_external_force_torque"].params["force_range"] = (-8.0, 8.0)
    cfg.events["base_external_force_torque"].params["torque_range"] = (-0.8, 0.8)

    cfg.observations["privileged"].terms["forces"].params["asset_cfg"].body_names = ("trunk",)
    cfg.observations["privileged"].terms["torques"].params["asset_cfg"].body_names = ("trunk",)

    cfg.rewards["feet_slide"].params["asset_cfg"].site_names = FOOT_NAMES

    # Apply play mode overrides.
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.scene.num_envs = 15
        cfg.sim.nconmax = None
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5
            cfg.scene.terrain.terrain_generator.border_width = 10.0

    # Distillation variant: corrupt the student's proprioception so the DAgger
    # student learns robustness to real sensor noise. The critic/teacher group
    # stays clean.
    if distill:
        cfg.observations["policy"].enable_corruption = True

    return cfg
