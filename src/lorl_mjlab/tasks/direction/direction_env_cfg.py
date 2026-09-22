"""Direction-command task configuration.

This module provides a factory function to create a base direction-command
locomotion task config (heading + discrete turn).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RingPatternCfg
from mjlab.viewer import ViewerConfig

from lorl_mjlab.envs.events import my_events
from lorl_mjlab.envs.mdp.observations import base_policy_obs_terms, base_privileged_obs_terms
from lorl_mjlab.envs.scene import FOOT_SCAN_KEYS, make_scene_cfg, make_sim_cfg
from lorl_mjlab.envs.terminations import base_terminations

from . import mdp
from .mdp import DirectionCommandCfg

FOOT_SCAN_PATTERN = RingPatternCfg(
    rings=(
        RingPatternCfg.Ring(radius=0.08, num_samples=6),
        RingPatternCfg.Ring(radius=0.16, num_samples=12),
        RingPatternCfg.Ring(radius=0.24, num_samples=18),
    ),
    include_center=False,
)


def make_direction_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base direction-command task configuration."""

    ##
    # Observations
    ##

    policy_terms = base_policy_obs_terms(command_name="direction")
    policy_terms["actions"] = ObservationTermCfg(func=envs_mdp.last_action)

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False,  # Teacher trains on clean proprio; distill flips this on.
        ),
        "privileged": ObservationGroupCfg(
            terms=base_privileged_obs_terms(foot_scan_keys=FOOT_SCAN_KEYS),
            concatenate_terms=True,
            enable_corruption=False,
        ),
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
            func=mdp.ang_vel_rew_13,
            weight=0.5,
            params={"command_name": "direction"},
        ),
        "base_motion": RewardTermCfg(
            func=mdp.base_motion,
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
        scene=make_scene_cfg(foot_scan_pattern=FOOT_SCAN_PATTERN),
        observations=observations,
        actions=actions,
        commands=commands,
        events=my_events(),
        rewards=rewards,
        terminations=base_terminations(),
        curriculum=curriculum,
        metrics={"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=make_sim_cfg(),
        decimation=4,
        episode_length_s=20.0,
    )
