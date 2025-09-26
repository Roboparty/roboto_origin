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

from typing import TYPE_CHECKING

from rsl_rl.env import VecEnv

if TYPE_CHECKING:
    from legged_lab.envs.base.base_config import BaseEnvCfg, BaseAgentCfg


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: VecEnv, env_cfg: "BaseEnvCfg", train_cfg: "BaseAgentCfg"):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_cfgs(self, name) -> tuple["BaseEnvCfg", "BaseAgentCfg"]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        return env_cfg, train_cfg


# make global task registry
task_registry = TaskRegistry()
