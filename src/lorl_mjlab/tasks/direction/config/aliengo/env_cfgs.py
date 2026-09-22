from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from lorl_mjlab.envs.robot_setup import apply_play_overrides, apply_quadruped_setup
from lorl_mjlab.envs.robots import FOOT_NAMES, aliengo_setup
from lorl_mjlab.robots import ALIENGO_ACTION_SCALE
from lorl_mjlab.terrains import IcraVariant

from ...direction_env_cfg import make_direction_env_cfg


def unitree_aliengo_direction_env_cfg(
    play: bool = False,
    distill: bool = False,
    icra: IcraVariant | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create Unitree AlienGo direction-command configuration."""
    assert icra is None or play, "the ICRA course is a fixed layout, only usable in play mode"
    cfg = make_direction_env_cfg()
    qs = aliengo_setup()
    apply_quadruped_setup(cfg, qs)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = ALIENGO_ACTION_SCALE

    cfg.rewards["feet_slide"].params["asset_cfg"].site_names = FOOT_NAMES
    cfg.rewards["stand_height_shortfall"].params["target_height"] = qs.stand_height_target

    if play:
        apply_play_overrides(cfg, icra)

    if distill and not play:
        cfg.observations["policy"].enable_corruption = True

    return cfg
