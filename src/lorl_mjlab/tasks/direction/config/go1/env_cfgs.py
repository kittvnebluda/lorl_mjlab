from mjlab.asset_zoo.robots import GO1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from lorl_mjlab.envs.robot_setup import apply_play_overrides, apply_quadruped_setup
from lorl_mjlab.envs.robots import FOOT_NAMES, go1_setup
from lorl_mjlab.robots import GO1_LEARNED_ACTION_SCALE
from lorl_mjlab.terrains import IcraVariant

from ...direction_env_cfg import make_direction_env_cfg


def unitree_go1_direction_env_cfg(
    play: bool = False,
    distill: bool = False,
    icra: IcraVariant | None = None,
    learned_actuator: bool = True,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go1 direction-command configuration.

    Args:
        play: Play-mode overrides (few envs, endless episodes, no corruption or pushes).
        distill: Corrupt the student's proprioception for DAgger distillation.
        icra: Fixed ICRA2024 QRC course layout. Play mode only.
        learned_actuator: Drive the joints with the learned MLP actuator model instead of
            MuJoCo's builtin position actuators. See ``go1_learned_actuator``.
    """
    assert icra is None or play, "the ICRA course is a fixed layout, only usable in play mode"
    cfg = make_direction_env_cfg()
    qs = go1_setup(learned_actuator)
    apply_quadruped_setup(cfg, qs)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = GO1_LEARNED_ACTION_SCALE if learned_actuator else GO1_ACTION_SCALE

    if learned_actuator:
        # The learned model evaluates its own control law, so randomizing MuJoCo's PD gains
        # would randomize nothing; randomize what it does respect instead.
        cfg.events.pop("actuator_gains")
        cfg.observations["privileged"].terms.pop("actuator_gains")
        cfg.events["actuator_effort"] = EventTermCfg(
            mode="startup",
            func=dr.effort_limits,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
                "effort_limit_range": (0.8, 1.0),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

    cfg.rewards["feet_slide"].params["asset_cfg"].site_names = FOOT_NAMES
    cfg.rewards["stand_height_shortfall"].params["target_height"] = qs.stand_height_target

    if play:
        apply_play_overrides(cfg, icra)

    if distill and not play:
        cfg.observations["policy"].enable_corruption = True

    return cfg
