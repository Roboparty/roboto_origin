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

from isaaclab.markers import VisualizationMarkers
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
import torch
import numpy as np

from robolab.tasks.direct.base import (  # noqa:F401
    BaseEnv,
    BaseEnvCfg
)

class InterruptEnv(BaseEnv):
    def __init__(self, cfg, render_mode, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.cfg: BaseEnvCfg

    def init_buffers(self):
        self.extras = {}

        self.episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = CircularBuffer(
            max_len=self.cfg.robot.action_history_length, batch_size=self.num_envs, device=self.device
        )
        self.action_buffer.append(torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        if self.cfg.interrupt.use_interrupt:
            self.interrupt_joint_cfg = SceneEntityCfg(name="robot", joint_names=self.cfg.interrupt.interrupt_joint_names, preserve_order=True)
            self.interrupt_joint_cfg.resolve(self.scene)
            all_indices = torch.arange(self.num_envs, device=self.device)
            perm = torch.randperm(self.num_envs, device=self.device)
            self.interrupt_indices = all_indices[perm[:int(self.num_envs * self.cfg.interrupt.interrupt_ratio)]]
            self.interrupt_mode_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
            self.interrupt_mode_mask[self.interrupt_indices] = True
            self.interrupt_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
            self.interrupt_actions = torch.zeros(self.num_envs, len(self.cfg.interrupt.interrupt_joint_names), dtype=torch.float, device=self.device, requires_grad=False)
            self.interrupt_scale = torch.tensor(self.cfg.interrupt.interrupt_scale, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
            self.interrupt_lower_bound = torch.tensor(self.cfg.interrupt.interrupt_lower_bound, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
            self.interrupt_rad_curriculum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.interrupt_vis = VisualizationMarkers(self.cfg.interrupt_vis)
            self.interrupt_vis.set_visibility(True)
        
        self.init_obs_buffer()

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer.buffer[:, -1, :]
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
            ],
            dim=-1,
        )
        if self.cfg.interrupt.use_interrupt:
            current_actor_obs = torch.cat([current_actor_obs, self.interrupt_mask.unsqueeze(1)], dim=-1)

        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
        feet_contact_force = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :]
        feet_air_time = self.contact_sensor.data.current_air_time[:, self.feet_cfg.body_ids]
        feet_height = torch.stack(
        [
            self.scene[sensor_cfg.name].data.pos_w[:, 2]
            - self.scene[sensor_cfg.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor_cfg in [SceneEntityCfg("left_feet_scanner"), SceneEntityCfg("right_feet_scanner")]
            if sensor_cfg is not None
        ],
        dim=-1,
        )
        feet_height = torch.clamp(feet_height - 0.04, min=0.0, max=1.0)
        feet_height = torch.nan_to_num(feet_height, nan=1.0, posinf=1.0, neginf=0)
        joint_torque = robot.data.applied_torque
        joint_acc = robot.data.joint_acc
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact.float(), feet_contact_force.flatten(1), feet_air_time.flatten(1), feet_height.flatten(1), joint_acc, joint_torque], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def _get_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        if self.cfg.scene_context.height_scanner.enable_height_scan:
            height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                )
            height_scan = torch.clamp(height_scan - self.cfg.normalization.height_scan_offset, min=-1.0, max=1.0)
            height_scan = torch.nan_to_num(height_scan, nan=1.0, posinf=1.0, neginf=-1.0)
            height_scan *= self.obs_scales.height_scan
            current_critic_obs = torch.cat([current_critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            if self.cfg.scene_context.height_scanner.enable_height_scan_actor:
                current_actor_obs = torch.cat([current_actor_obs, height_scan], dim=-1)

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        observations = {"policy": actor_obs, "critic":critic_obs}
        return observations

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene_context.terrain_generator is not None:
            if self.cfg.scene_context.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)
        
        if self.cfg.interrupt.use_interrupt:
            tmp_interrupt_mask = self.interrupt_mask[env_ids]
            tmp_interrupt_mode_mask = self.interrupt_mode_mask[env_ids]
            self.update_interrupt_levels(env_ids[tmp_interrupt_mode_mask], env_ids[tmp_interrupt_mask])
            self.extras["log"].update({"Curriculum/interrupt_levels":  torch.mean(self.interrupt_rad_curriculum[self.interrupt_indices])})

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self._sim_step_counter // self.cfg.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.reset_time_outs

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device)

        self._pre_physics_step(actions)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        if self.cfg.interrupt.use_interrupt:
            if self.common_step_counter % self.cfg.interrupt.interrupt_update_step == 0:
                self.interrupt_actions = self.uniform_interrupt_resample() / self.cfg.robot.action_scale
                self.common_step_counter %= self.cfg.interrupt.interrupt_update_step
            if is_rendering:
                robot_positions = self.robot.data.root_pos_w
                marker_positions = robot_positions + torch.tensor([0.0, 0.0, 0.8], device=self.device)
                interrupt_list = (~self.interrupt_mask).int().cpu().numpy().tolist()
                self.interrupt_vis.visualize(marker_indices=interrupt_list, translations=marker_positions)

        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        self.obs_buf = self._get_observations()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def update_interrupt_levels(self, env_ids, interrupt_env_ids):
        if len(env_ids) == 0:
            return
        distance = torch.norm(self.robot.data.root_pos_w[interrupt_env_ids, :2] - self.scene.env_origins[interrupt_env_ids, :2], dim=1)
        curr_is_pass = distance > self.cfg.scene_context.terrain_generator.size[0] / 2
        curr_is_down = (
            distance < torch.norm(self.command_generator.command[interrupt_env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        curr_is_down *= ~curr_is_pass

        self.interrupt_rad_curriculum[interrupt_env_ids] = torch.where(
            curr_is_down,
            (self.interrupt_rad_curriculum[interrupt_env_ids] - 0.05).clip(min=0),
            torch.where(
                curr_is_pass,
                (self.interrupt_rad_curriculum[interrupt_env_ids] + 0.05).clip(max=self.cfg.interrupt.max_curriculum),
                self.interrupt_rad_curriculum[interrupt_env_ids]
            )
        )

        self.interrupt_mask[env_ids] = (torch.rand(len(env_ids))<=0.5).to(self.device) # Reset with half with interrupt.
        self.interrupt_actions[env_ids] = self.robot.data.joint_pos[env_ids][:, self.interrupt_joint_cfg.joint_ids] - self.robot.data.default_joint_pos[env_ids][:, self.interrupt_joint_cfg.joint_ids]

    def uniform_interrupt_resample(self):
        '''Sample Noise from Uniform interruption'''
        targets = self.interrupt_scale * torch.rand((self.num_envs, len(self.cfg.interrupt.interrupt_joint_names)), device=self.device) + self.interrupt_lower_bound

        # clip interrupt
        left_env_mask1 = targets[:, 1] < 0.5
        targets[left_env_mask1][:, 2] = torch.clamp(targets[left_env_mask1][:, 2], min=-1.57, max=0.85)
        left_env_mask2 = targets[:, 1] < 0
        targets[left_env_mask2][:, [2, 3]] = 0
        right_env_mask1 =  targets[:, 5] > -0.5
        targets[right_env_mask1][:, 6] = torch.clamp(targets[right_env_mask1][:, 2], min=-0.85, max=1.57)
        right_env_mask2 =  targets[:, 5] > 0
        targets[right_env_mask2][:, [6, 7]] = 0

        return torch.clamp(
            targets - self.robot.data.default_joint_pos[:, self.interrupt_joint_cfg.joint_ids], 
            self.robot.data.default_joint_pos_limits[:, self.interrupt_joint_cfg.joint_ids, 0] - self.robot.data.default_joint_pos[:, self.interrupt_joint_cfg.joint_ids],
            self.robot.data.default_joint_pos_limits[:, self.interrupt_joint_cfg.joint_ids, 1] - self.robot.data.default_joint_pos[:, self.interrupt_joint_cfg.joint_ids]
        )

    def curriculum_interrupt_clipping_mean_rad(self):
        # clipping mean with curriculum
        noise_mean = self.interrupt_rad_curriculum.unsqueeze(-1) * (self.robot.data.joint_pos[:, self.interrupt_joint_cfg.joint_ids] - self.robot.data.default_joint_pos[:, self.interrupt_joint_cfg.joint_ids])

        # clipping action rate with curriculum by rad.
        interrupt_actions = torch.clamp(
            self.interrupt_actions,
            (- self.cfg.interrupt.interrupt_init_range * self.interrupt_rad_curriculum.unsqueeze(-1) + noise_mean)/self.cfg.robot.action_scale,
            (self.cfg.interrupt.interrupt_init_range * self.interrupt_rad_curriculum.unsqueeze(-1) + noise_mean)/self.cfg.robot.action_scale
        )
        return interrupt_actions
    
    def random_switch_interrupt(self):
        switch_rand = torch.rand(self.num_envs, device=self.device)
        switch = switch_rand < self.cfg.interrupt.switch_prob
        self.interrupt_mask = torch.where(torch.logical_and(switch, self.interrupt_mode_mask), ~self.interrupt_mask, self.interrupt_mask)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.action_buffer.append(actions)
        self.actions = actions.clone()
        self.actions = torch.clip(self.actions, -self.clip_actions, self.clip_actions).to(self.device)
        if self.cfg.interrupt.use_interrupt:
            self.random_switch_interrupt() 
            interrupt_action_clip = self.curriculum_interrupt_clipping_mean_rad()
            self.actions[:, self.interrupt_joint_cfg.joint_ids] = torch.where(
                self.interrupt_mask.unsqueeze(-1).expand(-1, len(self.interrupt_joint_cfg.joint_ids)),
                interrupt_action_clip,
                self.actions[:, self.interrupt_joint_cfg.joint_ids]
            )
            self.actions = torch.clip(self.actions, -self.clip_actions, self.clip_actions).to(self.device)

        self.actions = self.actions * self.action_scale + self.robot.data.default_joint_pos