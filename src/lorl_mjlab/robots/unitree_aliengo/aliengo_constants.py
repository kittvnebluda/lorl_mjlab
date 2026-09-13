from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from lorl_mjlab import LORL_SRC_PATH

##
# MJCF and assets.
##

ALIENGO_XML: Path = LORL_SRC_PATH / "robots" / "unitree_aliengo" / "xmls" / "aliengo.xml"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(ALIENGO_XML))


##
# Actuator config.
##

ALIENGO_LEG_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
    stiffness=40.0,
    damping=2.0,
    effort_limit=44.4,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.5),
    joint_pos={
        ".*R_hip_joint": -0.1,
        ".*L_hip_joint": 0.1,
        "F[LR]_thigh_joint": 0.8,
        "R[LR]_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
    },
    joint_vel={".*": 0.0},
)

# Trunk height above the mean foot height that a standing robot must not sag below.
# ~0.75x the nominal standing height, which is 0.349 m.
# Not to be confused with INIT_STATE's own z (0.5)
ALIENGO_STAND_HEIGHT_TARGET: float = 0.26

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
    solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=1,
    conaffinity=1,
    # Harden all collision geoms.
    solref=(0.01, 1),
    # Configure feet colliders. Other colliders are frictionless (condim=1).
    condim={_foot_regex: 6, ".*_collision": 1},
    priority={_foot_regex: 1, ".*": 0},
    friction={_foot_regex: (1, 5e-3, 5e-4)},
)

##
# Final config.
##

ALIENGO_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(ALIENGO_LEG_ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=0.9,
)


def get_aliengo_robot_cfg() -> EntityCfg:
    """Get a fresh AlienGo robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=ALIENGO_ARTICULATION,
    )


ALIENGO_ACTION_SCALE: dict[str, float] = {}
for a in ALIENGO_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        ALIENGO_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
    from mjlab.entity.entity import Entity
    from mujoco import viewer

    robot = Entity(get_aliengo_robot_cfg())

    viewer.launch(robot.spec.compile())
