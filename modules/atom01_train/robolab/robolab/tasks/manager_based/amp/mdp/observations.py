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

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from robolab.tasks.manager_based.amp.manager_based_animation_env import ManagerBasedAnimationEnv
    from robolab.tasks.manager_based.amp.managers import AnimationTerm
    


def root_local_rot_tan_norm(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    
    root_quat = robot.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)
    
    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)
    
    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def ref_root_local_rot_tan_norm(
    env: ManagerBasedAnimationEnv, 
    animation: str, 
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs
    
    ref_root_quat = animation_term.get_root_quat() # shape: (num_envs, num_steps, 4)
    ref_yaw_quat = math_utils.yaw_quat(ref_root_quat)
    ref_root_quat_local = math_utils.quat_mul(
        math_utils.quat_conjugate(ref_yaw_quat), ref_root_quat
    )  # shape: (num_envs, num_steps, 4)
    ref_root_rotm_local = math_utils.matrix_from_quat(ref_root_quat_local) # shape: (num_envs, num_steps, 3, 3)
    
    tan_vec = ref_root_rotm_local[:, :, :, 0]  # (num_envs, num_steps, 3)
    norm_vec = ref_root_rotm_local[:, :, :, 2]  # (num_envs, num_steps, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (num_envs, num_steps, 6)
    
    if flatten_steps_dim:
        return obs.reshape(num_envs, -1)
    else:
        return obs

def ref_root_projected_gravity(
    env: ManagerBasedAnimationEnv, 
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs
    
    ref_root_quat = animation_term.get_root_quat() # shape: (num_envs, num_steps, 4)
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=ref_root_quat.device).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 3)
    projected_gravity = math_utils.quat_apply_inverse(
        ref_root_quat, gravity_vec.expand(num_envs, -1, -1)
    )  # shape: (num_envs, num_steps, 3)
    
    if flatten_steps_dim:
        return projected_gravity.reshape(num_envs, -1)
    else:
        return projected_gravity
    
def ray_caster(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取激光雷达（RayCaster）传感器的距离数据"""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    origin = sensor.data.pos_w.unsqueeze(1)  # [num_envs, 1, 3]
    hits = sensor.data.ray_hits_w  # [num_envs, num_rays, 3]
    distances = torch.norm(hits - origin, dim=-1).clamp(min=0.2, max=5)  # [num_envs, num_rays]
    return distances




def root_rot_tan_norm(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    
    root_quat = robot.data.root_quat_w
    root_rotm = math_utils.matrix_from_quat(root_quat)
    
    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm[:, :, 0]  # (N, 3)
    norm_vec = root_rotm[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


# def key_body_pos_b(
#     env: ManagerBasedEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=MISSING, preserve_order=True),
# ) -> torch.Tensor:

#     robot: Articulation = env.scene[asset_cfg.name]
    
#     key_body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]  # shape: (num_envs, M, 3)
#     root_pos_w = robot.data.root_pos_w      # shape: (num_envs, 3).
#     root_quat = robot.data.root_quat_w    # shape: (num_envs, 4), w, x, y, z order.
    
#     num_key_bodies = key_body_pos_w.shape[1]
#     num_envs = root_pos_w.shape[0]
    
#     key_body_pos_b = math_utils.quat_apply_inverse(
#         root_quat.unsqueeze(1).expand(-1, num_key_bodies, -1), 
#         key_body_pos_w - root_pos_w.unsqueeze(1).expand(-1, num_key_bodies, -1)
#     )
    
#     return key_body_pos_b.reshape(num_envs, -1)


def ref_root_pos_error(
    env: ManagerBasedAnimationEnv, 
    animation: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    abs_height: bool = True
) -> torch.Tensor:
    """Compute the difference between robot root position and reference motion root position.
    
    The function calculates: reference_root_pos - current_robot_root_pos
    
    Args:
        env: The animation environment.
        animation: Name of the animation term to use as reference.
        asset_cfg: Configuration for the robot asset.
        abs_height: If True, use absolute height from reference motion (returns 3D position).
                   If False, only return horizontal displacement (2D: x, y only).
    
    Returns:
        Flattened tensor with shape:
        - (num_envs, num_steps * 3) if abs_height=True
        - (num_envs, num_steps * 2) if abs_height=False
        
    Note:
        Positive values indicate the reference motion is ahead/above the robot.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    
    ref_root_pos_w = animation_term.get_root_pos_w()  # shape: (num_envs, num_steps, 3)
    root_pos_w = robot.data.root_pos_w - env.scene.env_origins  # shape: (num_envs, 3)
    
    num_envs = root_pos_w.shape[0]
    
    # Compute position difference: ref - current
    # Broadcasting handles the dimension expansion automatically
    pos_diff = ref_root_pos_w - root_pos_w.unsqueeze(1)  # shape: (num_envs, num_steps, 3)
    
    if abs_height:
        # Replace relative z with absolute reference height
        pos_diff[:, :, 2] = ref_root_pos_w[:, :, 2]
        return pos_diff.reshape(num_envs, -1)  # shape: (num_envs, num_steps * 3)
    else:
        # Only return horizontal displacement (x, y)
        return pos_diff[:, :, :2].reshape(num_envs, -1)  # shape: (num_envs, num_steps * 2)


def ref_root_rot_tan_norm(
    env: ManagerBasedAnimationEnv, 
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    
    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_root_rotm = math_utils.matrix_from_quat(ref_root_quat)  # shape: (num_envs, num_steps, 3, 3)
    ref_root_tan_vec = ref_root_rotm[:, :, :, 0]  # shape: (num_envs, num_steps, 3)
    ref_root_norm_vec = ref_root_rotm[:, :, :, 2]  # shape: (num_envs, num_steps, 3)
    obs = torch.cat([ref_root_tan_vec, ref_root_norm_vec], dim=-1)  # shape: (num_envs, num_steps, 6)
    
    if flatten_steps_dim:
        return obs.reshape(env.num_envs, -1)
    else:
        return obs


def ref_root_ang_vel_b(
    env: ManagerBasedAnimationEnv, 
    animation: str, 
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs
    
    ref_root_ang_vel_w = animation_term.get_root_ang_vel_w()  # shape: (num_envs, num_steps, 3)
    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_root_ang_vel = math_utils.quat_apply_inverse(
        ref_root_quat, ref_root_ang_vel_w
    )
    
    if flatten_steps_dim:
        return ref_root_ang_vel.reshape(num_envs, -1)
    else:
        return ref_root_ang_vel
    

def ref_root_lin_vel_b(
    env: ManagerBasedAnimationEnv, 
    animation: str, 
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs
    
    ref_root_lin_vel_w = animation_term.get_root_vel_w()  # shape: (num_envs, num_steps, 3)
    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_root_lin_vel = math_utils.quat_apply_inverse(
        ref_root_quat, ref_root_lin_vel_w
    )
    
    if flatten_steps_dim:
        return ref_root_lin_vel.reshape(num_envs, -1)
    else:
        return ref_root_lin_vel


def ref_joint_pos(
    env: ManagerBasedAnimationEnv, 
    animation: str, 
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    
    ref_dof_pos = animation_term.get_dof_pos()  # shape: (num_envs, num_steps, num_dofs)
    
    if flatten_steps_dim:
        return ref_dof_pos.reshape(env.num_envs, -1)
    else:
        return ref_dof_pos
    
def ref_joint_vel(
    env: ManagerBasedAnimationEnv, 
    animation: str, 
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    
    ref_dof_vel = animation_term.get_dof_vel()  # shape: (num_envs, num_steps, num_dofs)
    
    if flatten_steps_dim:
        return ref_dof_vel.reshape(env.num_envs, -1)
    else:
        return ref_dof_vel

def ref_key_body_pos_b(
    env: ManagerBasedAnimationEnv, 
    animation: str, 
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    
    ref_key_body_pos_b = animation_term.get_key_body_pos_b()  # shape: (num_envs, num_steps, num_key_bodies, 3)
    
    if flatten_steps_dim:
        return ref_key_body_pos_b.reshape(env.num_envs, -1)
    else:
        num_envs = ref_key_body_pos_b.shape[0]
        num_steps = ref_key_body_pos_b.shape[1]
        return ref_key_body_pos_b.reshape(num_envs, num_steps, -1)

