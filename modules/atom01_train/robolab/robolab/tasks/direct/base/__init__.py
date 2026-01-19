# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from . import agents
from .base_config import BaseAgentCfg, BaseEnvCfg, RewardCfg, HeightScannerCfg, SceneContextCfg, RobotCfg, ObsScalesCfg, NormalizationCfg, CommandRangesCfg, CommandsCfg, NoiseScalesCfg, NoiseCfg, EventCfg
from .base_env import BaseEnv
from .scene_cfg import SceneCfg
from .terrain_generator_cfg import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG, ROUGH_HARD_TERRAINS_CFG
from . import mdp

import gymnasium as gym

gym.register(
    id="Atom01-Flat",
    entry_point=f"{__name__}.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_env_cfg:ATOM01FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_agent_cfg:ATOM01FlatAgentCfg",
    },
)

gym.register(
    id="Atom01-Rough",
    entry_point=f"{__name__}.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_env_cfg:ATOM01RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_agent_cfg:ATOM01RoughAgentCfg",
    },
)