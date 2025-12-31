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


"""Configuration for the ray-cast sensor."""


from isaaclab.sensors.ray_caster import RayCasterCfg as BaseRayCasterCfg
from isaaclab.utils import configclass

from .ray_caster import RayCaster


@configclass
class RayCasterCfg(BaseRayCasterCfg):

    class_type: type = RayCaster
