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

from collections.abc import Sequence

from isaaclab.sensors.ray_caster import RayCaster as BaseRayCaster


class RayCaster(BaseRayCaster):
    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # resample the drift
        self.drift[env_ids, 0] = self.drift[env_ids, 0].uniform_(*self.cfg.drift_range)
        self.drift[env_ids, 1] = self.drift[env_ids, 1].uniform_(*self.cfg.drift_range)
        self.drift[env_ids, 2] = self.drift[env_ids, 2].uniform_(
            *(self.cfg.drift_range[0] * 0.1, self.cfg.drift_range[1] * 0.1)
        )
