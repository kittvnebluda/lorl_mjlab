"""Direction task configuration.

This module provides a factory function to create a base direction-command
locomotion task config (heading + discrete turn, per Lee et al. 2020).
Robot-specific configurations call the factory and customize as needed --
mirrors ``mjlab.tasks.velocity.velocity_env_cfg.make_velocity_env_cfg``.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from lorl_mjlab.terrains import ROUGH_TERRAINS_CFG, IcraVariant, icra_start_yaw, icra_terrains_generator_cfg

from . import mdp
from .mdp import DirectionCommandCfg

_FOOT_SCAN_PATTERN = RingPatternCfg(
    rings=(
        RingPatternCfg.Ring(radius=0.08, num_samples=6),
        RingPatternCfg.Ring(radius=0.16, num_samples=12),
        RingPatternCfg.Ring(radius=0.24, num_samples=18),
    ),
    include_center=False,
)

_FOOT_NAMES = ("fl", "fr", "rl", "rr")


def apply_icra_course(cfg: ManagerBasedRlEnvCfg, variant: IcraVariant) -> None:
    """Swap the procedural terrain for the fixed ICRA2024 QRC course."""
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_generator = icra_terrains_generator_cfg(variant)
    cfg.scene.num_envs = 1

    cfg.terminations = {}

    cfg.events.pop("randomize_terrain", None)

    yaw = icra_start_yaw(variant)
    cfg.events["reset_base"].params["pose_range"] = {"yaw": (yaw, yaw)}


def _foot_scanner(foot: str) -> RayCastSensorCfg:
    """One downward concentric-ring height scanner attached to a single foot site."""
    return RayCastSensorCfg(
        name=f"{foot}_foot_scan",
        frame=ObjRef(type="site", name="", entity="robot"),  # Set per-robot.
        ray_alignment="yaw",
        pattern=_FOOT_SCAN_PATTERN,
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # Terrain only.
        debug_vis=False,
    )


def _trunk_height_scanner() -> TerrainHeightSensorCfg:
    """Single downward ray under the trunk, for terrain-relative body height.

    One ray suffices because ``stand_height_shortfall`` is STAND-gated and one-sided: a
    1500-iteration seed=1 ablation against a 37-ray reference left the converged penalty
    at zero for 1, 5 and 17 rays.
    """
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


def make_direction_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base direction-command task configuration."""

    ##
    # Sensors
    ##

    foot_scanners = tuple(_foot_scanner(foot) for foot in _FOOT_NAMES)

    ##
    # Observations
    ##

    policy_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "direction"},
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    privileged_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "foot_contacts": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        **{
            f"{foot}_foot_scan": ObservationTermCfg(
                func=envs_mdp.height_scan,
                params={"sensor_name": f"{foot}_foot_scan"},
                noise=Unoise(n_min=-0.1, n_max=0.1),
                clip=(-1.0, 1.0),
            )
            for foot in _FOOT_NAMES
        },
        "actuator_gains": ObservationTermCfg(func=mdp.actuator_gains),
        "forces": ObservationTermCfg(
            func=mdp.external_force_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
        ),
        "torques": ObservationTermCfg(
            func=mdp.external_torque_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
        ),
    }

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False,  # Teacher trains on clean proprio; distill flips this on.
        ),
        "privileged": ObservationGroupCfg(
            terms=privileged_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    ##
    # Metrics
    ##

    metrics = {
        "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        )
    }

    ##
    # Commands
    ##

    commands: dict[str, CommandTermCfg] = {
        "direction": DirectionCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            debug_vis=True,
        ),
    }

    ##
    # Events
    ##

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        ),
        "base_external_force_torque": EventTermCfg(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
                "force_range": (0.0, 0.0),  # Set per-robot.
                "torque_range": (0.0, 0.0),  # Set per-robot.
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "operation": "abs",
                "ranges": (0.6, 1.5),
                "shared_random": True,
            },
        ),
        "base_inertial": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
                "alpha_range": (-0.3466, 0.2027),
                "t1_range": (-0.1, 0.1),
                "t2_range": (-0.1, 0.1),
                "t3_range": (-0.1, 0.1),
            },
        ),
        "actuator_gains": EventTermCfg(
            mode="startup",
            func=dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
                "kp_range": (0.7, 1.3),
                "kd_range": (0.7, 1.3),
                "operation": "scale",
                "distribution": "uniform",
            },
        ),
        "joint_friction": EventTermCfg(
            mode="startup",
            func=dr.joint_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "ranges": (0.0, 0.4),
                "operation": "abs",
                "distribution": "uniform",
            },
        ),
        "joint_armature": EventTermCfg(
            mode="startup",
            func=dr.joint_armature,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "ranges": (0.0, 0.05),
                "operation": "abs",
                "distribution": "uniform",
            },
        ),
    }

    ##
    # Rewards
    ##

    rewards = {
        "alive": RewardTermCfg(func=mdp.is_alive, weight=0.25),
        "track_direction": RewardTermCfg(
            func=mdp.track_direction,
            weight=0.8,
            params={"command_name": "direction"},
        ),
        "track_turn": RewardTermCfg(
            func=mdp.track_turn,
            weight=0.5,
            params={"command_name": "direction"},
        ),
        "base_motion": RewardTermCfg(
            func=mdp.base_motion_reward,
            weight=0.15,
            params={"command_name": "direction"},
        ),
        "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.5),
        "radial_progress": RewardTermCfg(func=mdp.radial_progress, weight=0.5),
        "flight_phase_cost": RewardTermCfg(
            func=mdp.flight_phase_cost,
            weight=-0.5,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "dof_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2.0e-5),
        "dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.0e-7),
        "feet_air_time": RewardTermCfg(
            func=mdp.feet_air_time_progress,
            weight=0.01,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "direction",
                "threshold": 0.5,
            },
        ),
        "undesired_contacts": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-1.0,
            params={
                "sensor_name": "thigh_ground_touch",
                "force_threshold": 1.0,
            },
        ),
        "base_undesired_contact": RewardTermCfg(
            func=mdp.undesired_base_contact,
            weight=-2.0,
            params={
                "sensor_name": "trunk_ground_touch",
                "force_threshold": 1.0,
            },
        ),
        "stand_height_shortfall": RewardTermCfg(
            func=mdp.stand_height_shortfall,
            weight=-2.0,
            params={
                "command_name": "direction",
                "target_height": 0.0,  # Set per-robot.
                "sensor_name": "trunk_height_scan",
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-3.0e-3),
        "feet_slide": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.05,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "direction",
                "command_threshold": 0.01,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        "flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.5),
    }

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "out_of_terrain_bounds": TerminationTermCfg(func=mdp.out_of_terrain_bounds, time_out=True),
        "flipped": TerminationTermCfg(func=mdp.bad_orientation, time_out=False, params={"limit_angle": 1.4}),
    }

    ##
    # Curriculum
    ##

    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_dir,
            params={
                "command_name": "direction",
                "turn_fraction_threshold": 0.25,
            },
        ),
    }

    ##
    # Assemble and return
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=replace(ROUGH_TERRAINS_CFG),
                max_init_terrain_level=5,
            ),
            sensors=foot_scanners + (_trunk_height_scanner(),),
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
            nconmax=35,
        ),
        decimation=4,
        episode_length_s=20.0,
    )
