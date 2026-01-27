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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for ANYmal."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
        
        # critic observation group
        # -- original
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        # -- left-right
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env.unwrapped, obs["critic"])

    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)

    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the ANYmal robot. Additionally, if height-scan data is present, it is flipped
    along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # ang vel
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[:, 9:32] = _switch_joints_left_right(obs[:, 9:32])
    # joint vel
    obs[:, 32:55] = _switch_joints_left_right(obs[:, 32:55])
    # last actions
    obs[:, 55:78] = _switch_joints_left_right(obs[:, 55:78])

    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the ANYmal robot. Additionally, if height-scan data is present, it is flipped
    along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # lin vel
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1, -1, 1], device=device)
    # ang vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[:, 12:35] = _switch_joints_left_right(obs[:, 12:35])
    # joint vel
    obs[:, 35:58] = _switch_joints_left_right(obs[:, 35:58])
    # last actions
    obs[:, 58:81] = _switch_joints_left_right(obs[:, 58:81])

    return obs



"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[           
'left_thigh_yaw_joint',   #0
'right_thigh_yaw_joint',  #1
'torso_joint',            #2
'left_thigh_roll_joint',  #3
'right_thigh_roll_joint', #4
'left_arm_pitch_joint',   #5
'right_arm_pitch_joint',  #6
'left_thigh_pitch_joint', #7
'right_thigh_pitch_joint',#8
'left_arm_roll_joint',    #9
'right_arm_roll_joint',   #10
'left_knee_joint',        #11
'right_knee_joint',       #12
'left_arm_yaw_joint',     #13
'right_arm_yaw_joint',    #14
'left_ankle_pitch_joint', #15
'right_ankle_pitch_joint',#16
'left_elbow_pitch_joint', #17
'right_elbow_pitch_joint',#18
'left_ankle_roll_joint',  #19
'right_ankle_roll_joint', #20
'left_elbow_yaw_joint',   #21
'right_elbow_yaw_joint'   #22
]

Correspondingly, the joint ordering for the ANYmal robot is:

* LF = left front --> [0, 4, 8]
* LH = left hind --> [1, 5, 9]
* RF = right front --> [2, 6, 10]
* RH = right hind --> [3, 7, 11]
"""


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = joint_data.clone()
    # left <-- right
    joint_data_switched[..., [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] = joint_data[..., [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]]
    # right <-- left
    joint_data_switched[..., [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] = joint_data[..., [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]]
    
    joint_data_switched[..., [0,1,2,3,4,9,10,13,14,19,20,21,22]] = -1 * joint_data_switched[..., [0,1,2,3,4,9,10,13,14,19,20,21,22]]
    # joint_data_switched[..., [5,6,7,8,11,12,15,16,17,18]] = joint_data[..., [5,6,7,8,11,12,15,16,17,18]]

    # joint_data_switched[..., [0,1,2,3,4,9,10,13,14,19,20,21,22]] *= -1

    return joint_data_switched