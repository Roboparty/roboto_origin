
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

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal
import random

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from robolab.robolab.tasks.manager_based.amp.amp_env import ManagerBasedAmpEnv


def ref_state_init_root(
    env: ManagerBasedAmpEnv, 
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_rsi: bool = True,
):
    """Reference State Initialization (RSI) for the root of the robot.
    Sample from the motion loader and set the root position and orientation.
    Refer to the paper of Adversarial Motion Priors (AMP) for more details.

    Args:
        env (AmpEnv): The manager-based env.
        env_ids (torch.Tensor): The env IDs to reset.
        asset_cfg (SceneEntityCfg, optional): The asset configuration. Defaults to SceneEntityCfg("robot").
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    num_envs = env_ids.shape[0]
    dt = env.cfg.sim.dt * env.cfg.decimation

    if motion_dataset is None:
        # select one dataset randomly by weights
        term_weights = env.motion_data_manager.get_term_weights()
        motion_dataset = random.choices(list(term_weights.keys()), weights=list(term_weights.values()))[0]
    else:
        if motion_dataset not in env.motion_data_manager.active_terms():
            raise ValueError(f"Motion dataset '{motion_dataset}' not found in the active terms.")
    motion_loader = env.motion_data_manager.get_term(motion_dataset)
    motion_ids = motion_loader.sample_motions(num_envs)
    motion_times = motion_loader.sample_times(motion_ids, truncate_time=dt)
    motion_state_dict = motion_loader.get_motion_state(motion_ids, motion_times)
    
    lift_a_little = 0.05
    # lift the root position a little bit to avoid collision with the ground
    motion_state_dict["root_pos_w"][:, 2] += lift_a_little
    
    if not pos_rsi:
        motion_state_dict["root_pos_w"][:, :2] = 0.0    # no offset in x and y
    ref_root_pos_w = motion_state_dict["root_pos_w"] + env.scene.env_origins[env_ids]
    ref_root_quat = motion_state_dict["root_quat"]
    ref_root_vel_w = motion_state_dict["root_vel_w"]
    ref_root_ang_vel_w = motion_state_dict["root_ang_vel_w"]
    
    asset.write_root_pose_to_sim(
        torch.cat([ref_root_pos_w, ref_root_quat], dim=-1),
        env_ids=env_ids,
    )
    asset.write_root_velocity_to_sim(
        torch.cat([ref_root_vel_w, ref_root_ang_vel_w], dim=-1),
        env_ids=env_ids,
    )
    

def ref_state_init_dof(
    env: ManagerBasedAmpEnv, 
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motion_dataset: str | None = None,
):
    """Reference State Initialization (RSI) for the joints (DoF) of the robot.
    Sample from the motion loader and set the joint positions and velocities.
    Refer to the paper of Adversarial Motion Priors (AMP) for more details.

    Args:
        env (AmpEnv): The manager-based env.
        env_ids (torch.Tensor): The env IDs to reset.
        asset_cfg (SceneEntityCfg, optional): The asset configuration. Defaults to SceneEntityCfg("robot").
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    num_envs = env_ids.shape[0]
    dt = env.cfg.sim.dt * env.cfg.decimation
    
    if motion_dataset is None:
        # select one dataset randomly by weights
        term_weights = env.motion_data_manager.get_term_weights()
        motion_dataset = random.choices(list(term_weights.keys()), weights=list(term_weights.values()))[0]
    else:
        if motion_dataset not in env.motion_data_manager.active_terms():
            raise ValueError(f"Motion dataset '{motion_dataset}' not found in the active terms.")
    motion_loader = env.motion_data_manager.get_term(motion_dataset)
    motion_ids = motion_loader.sample_motions(num_envs)
    motion_times = motion_loader.sample_times(motion_ids, truncate_time=dt)
    motion_state_dict = motion_loader.get_motion_state(motion_ids, motion_times)

    joint_pos = motion_state_dict["dof_pos"]
    joint_vel = motion_state_dict["dof_vel"]
    
    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)