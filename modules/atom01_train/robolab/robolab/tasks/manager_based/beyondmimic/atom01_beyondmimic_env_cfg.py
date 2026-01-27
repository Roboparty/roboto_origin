
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

import os

from robolab.assets.robots import ATOM01_CFG
from robolab.tasks.manager_based.beyondmimic.beyondmimic_env_cfg import BeyondMimicEnvCfg

from isaaclab.utils import configclass
from robolab import ROBOLAB_ROOT_DIR


@configclass
class Atom01BeyondMimicEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ATOM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.motion_file = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "atom01_bm", "yundong1.npz"
        )
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            'left_thigh_yaw_link', 
            'right_thigh_yaw_link', 
            'torso_link', 
            'left_thigh_roll_link', 
            'right_thigh_roll_link', 
            'left_arm_pitch_link', 
            'right_arm_pitch_link', 
            'left_thigh_pitch_link', 
            'right_thigh_pitch_link', 
            'left_arm_roll_link', 
            'right_arm_roll_link', 
            'left_knee_link', 
            'right_knee_link', 
            'left_arm_yaw_link', 
            'right_arm_yaw_link', 
            'left_ankle_pitch_link', 
            'right_ankle_pitch_link', 
            'left_elbow_pitch_link', 
            'right_elbow_pitch_link', 
            'left_ankle_roll_link', 
            'right_ankle_roll_link', 
            'left_elbow_yaw_link', 
            'right_elbow_yaw_link',
        ]

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.episode_length_s = 20.0
