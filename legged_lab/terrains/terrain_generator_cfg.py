# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.04,
    vertical_scale=0.01,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=1.0
        )
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.04,
    vertical_scale=0.01,
    use_cache=False,
    sub_terrains={
        "inv_pyramid_stairs_25": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.15),
            step_width=0.25,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        "inv_pyramid_stairs_35": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.15),
            step_width=0.35,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_25": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.15),
            step_width=0.25,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_35": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.15),
            step_width=0.35,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.25), border_width=1.0, platform_width=1.5
        ),
        "inv_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.25), border_width=1.0, platform_width=1.5, inverted=True
        ),
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.0, 0.2), platform_width=1.5
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=1.0
        ),
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.1, pit_depth_range=(0.0, 0.3), platform_width=1.5, double_pit=True
        ),
        "star": terrain_gen.MeshStarTerrainCfg(
            proportion=0.2, num_bars=16, bar_width_range=(0.3, 0.5), bar_height_range=(10.0, 10.0), platform_width=1.0
        ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.2, gap_width_range=(0.1, 0.3), platform_width=2.0
        # ), # points in gap are nan
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.05,
            stone_width_range=(0.16, 0.24),
            stone_distance_range=(0.08, 0.08),
            platform_width=1.5,
            border_width=1.0,
        ),
    },
)
