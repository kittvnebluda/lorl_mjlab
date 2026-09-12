import mujoco
import pytest
from mjlab.terrains.terrain_generator import TerrainGenerator

from lorl_mjlab.terrains import icra_map, icra_terrains_generator_cfg


@pytest.fixture(scope="module")
def compiled():
    """Compiled MuJoCo models for both course variants, terrain patch included."""
    models = {}
    for variant in icra_map.VARIANTS:
        spec = mujoco.MjSpec()
        TerrainGenerator(cfg=icra_terrains_generator_cfg(variant), device="cpu").compile(spec)
        models[variant] = spec.compile()
    return models


@pytest.mark.parametrize("variant", icra_map.VARIANTS)
def test_course_geoms_are_visible_to_ray_casting(compiled, variant):
    """Contact sensors match body "terrain" and the foot scanners filter on geom group 0."""
    model = compiled[variant]
    terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    assert terrain != -1

    geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == terrain]
    assert len(geoms) > 400, "the course should contribute hundreds of geoms"
    assert all(model.geom_group[i] == 0 for i in geoms)
