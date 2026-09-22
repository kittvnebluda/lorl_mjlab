from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from lorl_mjlab.terrains import IcraVariant, icra_start_yaw, icra_terrains_generator_cfg


def apply_icra_course(cfg: ManagerBasedRlEnvCfg, variant: IcraVariant) -> None:
    """Swap the procedural terrain for the fixed ICRA2024 QRC course."""
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_generator = icra_terrains_generator_cfg(variant)
    cfg.scene.num_envs = 1

    cfg.terminations = {}

    cfg.events.pop("randomize_terrain", None)

    yaw = icra_start_yaw(variant)
    cfg.events["reset_base"].params["pose_range"] = {"yaw": (yaw, yaw)}
