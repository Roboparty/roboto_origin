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

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
    
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    # return torch.exp(-lin_vel_error / std**2)
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    # return torch.exp(-ang_vel_error / std**2)
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def feet_distance_y(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    min: float = 0.2, 
    max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # 只对超过 threshold 的空中时间给予正奖励（防止负值惩罚）
    positive_air = torch.clamp(last_air_time - threshold, min=0.0)
    reward = torch.sum(positive_air * first_contact.float(), dim=1)
    # no reward for zero command
    reward *= (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    asset: Articulation = env.scene["robot"]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize changes in actions
    diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
    diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
    return torch.sum(diff, dim=1)


def feet_orientation_l2(env: ManagerBasedRLEnv, 
                          sensor_cfg: SceneEntityCfg, 
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset:RigidObject = env.scene[asset_cfg.name]
    
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)
    
    num_feet = len(sensor_cfg.body_ids)
    
    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]   # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, 
        asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)
    
    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )
    
def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def joint_energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    return reward

def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def sound_suppression_acc_per_foot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    每只脚单独计算：
    脚接触地面时，z 方向加速度大 → 惩罚
    """

    asset = env.scene["robot"]

    # 1️⃣ 取所有 body 的线加速度 (world)
    # shape: (Nenv, Nbody, 6)
    body_acc = asset.data.body_acc_w

    # 2️⃣ 取“脚”的 z 方向线加速度
    # shape: (Nenv, Nfeet)
    foot_acc_z = body_acc[:, sensor_cfg.body_ids, 2]

    # 3️⃣ 取脚的接触状态
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_force_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    in_contact = torch.abs(contact_force_z) > 1.0  # (Nenv, Nfeet)

    # 4️⃣ 每只脚：加速度平方 × 接触状态
    acc_penalty = (foot_acc_z ** 2) * in_contact.float()

    # 防止数值爆炸（非常重要）
    acc_penalty = torch.clamp(acc_penalty, max=50.0)

    # 5️⃣ 所有脚加起来
    penalty = acc_penalty.sum(dim=1)
    reward = penalty

    # 仅当速度命令较小（小于 1.5）时才启用该奖励
    cmd = env.command_manager.get_command(command_name)
    
    # 使用 xy 分量的速度范数作为速度大小判断
    cmd_speed = torch.norm(cmd[:, :2], dim=1)
    reward = reward * (cmd_speed < 1.5).float()

    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def low_speed_sway_penalty(
    env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize linear and angular velocities when command velocity is below threshold.
    
    This function penalizes the robot for moving (both linear and angular) when the command
    speed is very small, encouraging the robot to remain still during low-speed commands.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get command velocity
    command = env.command_manager.get_command(command_name)
    command_speed = torch.norm(command[:, :2], dim=1)
    
    # Penalize linear velocity in xy plane
    lin_vel_penalty = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    
    # Penalize angular velocity
    ang_vel_penalty = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    
    # Total velocity penalty
    vel_penalty = lin_vel_penalty + ang_vel_penalty
    
    # Apply penalty only when command speed is below threshold
    return vel_penalty * (command_speed < command_threshold).float()


def staged_navigation_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster"),
    heading_threshold: float = 0.78,  # 45°，朝向误差阈值
    distance_threshold: float = 0.5,   #  距离目标的阈值
    near_goal_threshold: float = 2.0,  # 接近目标的距离阈值
    obstacle_threshold: float = 0.8,  # 前方障碍物的距离阈值
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ray_caster: RayCaster = env.scene[sensor_cfg.name]
    
    command = env.command_manager.get_command(command_name)
    des_pos = command[:, :2] # 目标位置
    des_heading = command[:, 2] # 目标朝向
    distance = torch.norm(des_pos, dim=1) # 机器人到目标位置的距离
    
    vx = asset.data.root_lin_vel_b[:, 0] # 机器人在base frame下的前向速度
    vy = asset.data.root_lin_vel_b[:, 1] # 机器人在base frame下的侧向速度
    speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1) # 机器人在水平面的速度大小 
    ang_speed = torch.abs(asset.data.root_ang_vel_b[:, 2]) # 机器人绕垂直轴的角速度

    # 当前移动方向与期望朝向误差（规范化到 [-pi, pi]）
    move_dir_angle = torch.atan2(vy, vx)
    raw_diff = move_dir_angle - des_heading
    diff_wrapped = torch.remainder(raw_diff + torch.pi, 2 * torch.pi) - torch.pi
    move_heading_error = diff_wrapped.abs()
    
    # 雷达前方最近障碍物距离（已在外部被 clamp）
    origin = ray_caster.data.pos_w.unsqueeze(1)  # [num_envs, 1, 3]
    hits = ray_caster.data.ray_hits_w  # [num_envs, num_rays, 3]
    distances = torch.norm(hits - origin, dim=-1).clamp(min=0.2, max=5.0)  # [num_envs, num_rays]
    front_min_dist = torch.min(distances, dim=1).values  # [num_envs]
    
    # 1) 朝向匹配奖励：误差越小奖励越高
    heading_reward = 1.0 / (1.0 + (move_heading_error / (heading_threshold + 1e-6))**2)

    # 2) 沿期望朝向的速度（越朝向目标前进越好），只奖励正向分量
    proj_vel = vx * torch.cos(des_heading) + vy * torch.sin(des_heading)
    progress_reward = torch.tanh(2.0 * proj_vel.clamp(min=0.0, max=1.0))  # 正向速度越大奖励越高，最大值接近1.0

    # 3) 障碍物清除奖励：鼓励与障碍物保持距离
    # 当 front_min_dist < obstacle_threshold 时，距离越大奖励越高（在 safe_min..obstacle_threshold 区间归一化到 0..1）
    safe_min = 0.5
    denom = max(obstacle_threshold - safe_min, 1e-6)
    obs_clearance = torch.clamp(front_min_dist - safe_min, min=0.0, max=obstacle_threshold - safe_min) / denom  # 0..1 when front_min_dist within [safe_min, obstacle_threshold]
    # 保持变量名兼容下游使用
    obs_approach_raw = obs_clearance
    
    # 对于 front_min_dist < safe_min 给出负向惩罚（碰撞/过近）
    collision_penalty = torch.clamp(safe_min - front_min_dist, min=0.0) / safe_min  # 0..1

    # 距离目标的奖励：距离越小越好，使用 near_goal_threshold 归一化尺度
    dist_reward = 1.0 / (1.0 + (distance / (near_goal_threshold + 1e-6))**2)

    # 分阶段加权：远离目标优先前进与保持清除，接近目标优先朝向精确并靠近目标
    is_far = distance > near_goal_threshold
    is_near = torch.logical_and(distance <= near_goal_threshold, distance > distance_threshold)
    is_at_goal = distance <= distance_threshold

    # 权重已调整并规范化（每阶段权重之和约为1）：
    # - far: 优先前进/清除障碍
    # - near: 优先朝向准确
    # - goal: 优先朝向与靠近目标（含站姿保持项）
    far_reward = 0.50 * progress_reward + 0.15 * heading_reward + 0.25 * obs_approach_raw + 0.10 * dist_reward
    near_reward = 0.20 * progress_reward + 0.45 * heading_reward + 0.15 * obs_approach_raw + 0.20 * dist_reward
    goal_reward = 0.05 * progress_reward + 0.60 * heading_reward + 0.15 * torch.exp(-torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)) + 0.20 * dist_reward

    reward = torch.zeros_like(distance)
    reward = torch.where(is_far, far_reward, reward)
    reward = torch.where(is_near, near_reward, reward)
    reward = torch.where(is_at_goal, goal_reward, reward)

    # 减去碰撞/过近惩罚及不良行为惩罚（横向速度、角速度）
    lateral_speed = torch.abs(vy)
    reward = reward - collision_penalty #- 0.08 * lateral_speed - 0.05 * ang_speed

    # 限幅防止数值爆炸
    reward = torch.clamp(reward, min=-1.0, max=2.0)

    return reward