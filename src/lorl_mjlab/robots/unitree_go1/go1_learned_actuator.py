"""Learned (MLP) actuator model for the Unitree Go1."""

from pathlib import Path

from mjlab.actuator import LearnedMlpActuatorCfg
from mjlab.asset_zoo.robots.unitree_go1.go1_constants import HIP_ACTUATOR, KNEE_ACTUATOR
from mjlab.utils.actuator import ElectricActuator

# Walk These Ways (WTW) actuators net: https://github.com/Improbable-AI/walk-these-ways
GO1_ACTUATOR_NET: Path = Path(__file__).parent / "wtw_act_net.pt"

# Coefs with which the net was trained
GO1_NET_KP = 20.0
GO1_NET_KD = 0.5


def _learned_cfg(target_names_expr: tuple[str, ...], motor: ElectricActuator) -> LearnedMlpActuatorCfg:
    """Build a learned-actuator config for one motor group."""
    return LearnedMlpActuatorCfg(
        target_names_expr=target_names_expr,
        effort_limit=motor.effort_limit,
        saturation_effort=motor.effort_limit,
        velocity_limit=motor.velocity_limit,
        armature=motor.reflected_inertia,
        network_file=str(GO1_ACTUATOR_NET),
        pos_scale=-1.0,  # WTW net expects `pos - target`; mjlab computes `target - pos`
        vel_scale=1.0,
        torque_scale=1.0,
        input_order="pos_vel",
        history_length=3,
        stiffness=GO1_NET_KP,
        damping=GO1_NET_KD,
    )


GO1_HIP_LEARNED_ACTUATOR_CFG = _learned_cfg((".*_hip_joint", ".*_thigh_joint"), HIP_ACTUATOR)
GO1_KNEE_LEARNED_ACTUATOR_CFG = _learned_cfg((".*_calf_joint",), KNEE_ACTUATOR)

GO1_LEARNED_ACTUATOR_CFGS = (GO1_HIP_LEARNED_ACTUATOR_CFG, GO1_KNEE_LEARNED_ACTUATOR_CFG)

GO1_LEARNED_ACTION_SCALE: dict[str, float] = {
    name: 0.25 * cfg.effort_limit / GO1_NET_KP for cfg in GO1_LEARNED_ACTUATOR_CFGS for name in cfg.target_names_expr
}
