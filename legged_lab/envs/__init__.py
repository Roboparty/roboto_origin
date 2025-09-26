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


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.atom01.atom01_config import (
    ATOM01FlatAgentCfg,
    ATOM01FlatEnvCfg,
    ATOM01RoughAgentCfg,
    ATOM01RoughEnvCfg,
)
from legged_lab.envs.atom01.atom01_interrupt_config import (
    ATOM01InterruptAgentCfg,
    ATOM01InterruptEnvCfg
)
from legged_lab.envs.atom01.atom01_interrupt_env import ATOM01InterruptEnv
from legged_lab.envs.atom01.atom01_amp_config import (
    ATOM01AmpAgentCfg,
    ATOM01AmpEnvCfg
)
from legged_lab.envs.atom01.atom01_amp_env import ATOM01AmpEnv
from legged_lab.utils.task_registry import task_registry

task_registry.register("atom01_rough", BaseEnv, ATOM01RoughEnvCfg(), ATOM01RoughAgentCfg())
task_registry.register("atom01_flat", BaseEnv, ATOM01FlatEnvCfg(), ATOM01FlatAgentCfg())
task_registry.register("atom01_interrupt", ATOM01InterruptEnv, ATOM01InterruptEnvCfg(), ATOM01InterruptAgentCfg())
task_registry.register("atom01_amp", ATOM01AmpEnv, ATOM01AmpEnvCfg(), ATOM01AmpAgentCfg())
