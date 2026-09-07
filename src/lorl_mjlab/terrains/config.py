from mjlab.terrains.config import (
    box_random_grid,
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
    wave_terrain,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "flat": flat(proportion=0.2),
        "pyramid_stairs": pyramid_stairs(proportion=0.2, step_height_range=(0.05, 0.20)),
        "pyramid_stairs_inv": pyramid_stairs_inv(proportion=0.2, step_height_range=(0.05, 0.20)),
        "hf_pyramid_slope": hf_pyramid_slope(proportion=0.1, slope_range=(0.0, 0.5)),
        "hf_pyramid_slope_inv": hf_pyramid_slope_inv(proportion=0.1, slope_range=(0.0, 0.5)),
        "random_rough": random_rough(proportion=0.1),
        "wave_terrain": wave_terrain(proportion=0.1),
        "box_random_grid": box_random_grid(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
            merge_similar_heights=True,
        ),
    },
    add_lights=True,
)
