import torch
import numpy as np

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.math import quat_apply_inverse,quat_apply_yaw, quat_inv 
from legged_lab.envs.base.base_env import (  # noqa:F401
    BaseEnv,
)
from .atom01_attn_enc_config import ATOM01AttnEncStage1EnvCfg, ATOM01AttnEncStage2EnvCfg

class ATOM01AttnEncEnv(BaseEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.cfg: ATOM01AttnEncStage1EnvCfg | ATOM01AttnEncStage2EnvCfg

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        lin_vel = robot.data.root_lin_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        if self.cfg.attn_enc.vel_in_obs:
            current_actor_obs = torch.cat(
                [
                    ang_vel * self.obs_scales.ang_vel,
                    lin_vel * self.obs_scales.lin_vel,
                    projected_gravity * self.obs_scales.projected_gravity,
                    command * self.obs_scales.commands,
                    joint_pos * self.obs_scales.joint_pos,
                    joint_vel * self.obs_scales.joint_vel,
                    action * self.obs_scales.actions,
                ],
                dim=-1,
            )
        else:
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

        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
        feet_contact_force = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :]
        feet_air_time = self.contact_sensor.data.current_air_time[:, self.feet_cfg.body_ids]
        feet_height = torch.stack(
        [
            self.scene[sensor_cfg.name].data.pos_w[:, 2]
            - self.scene[sensor_cfg.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor_cfg in [self.left_feet_scanner_cfg, self.right_feet_scanner_cfg]
            if sensor_cfg is not None
        ],
        dim=-1,
        )
        feet_height = torch.clamp(feet_height - 0.04, min=0.0, max=1.0)
        feet_height = torch.nan_to_num(feet_height, nan=1.0, posinf=1.0, neginf=0)
        joint_torque = robot.data.applied_torque
        joint_acc = robot.data.joint_acc
        action_delay = self.action_buffer.time_lags.to(self.device).unsqueeze(1)
        if self.cfg.attn_enc.vel_in_obs:
            current_critic_obs = torch.cat(
                [current_actor_obs, feet_contact.float(), feet_contact_force.flatten(1), feet_air_time.flatten(1), feet_height.flatten(1), joint_acc, joint_torque, action_delay.float()], dim=-1
            )
        else:
            current_critic_obs = torch.cat(
                [current_actor_obs, lin_vel * self.obs_scales.lin_vel, feet_contact.float(), feet_contact_force.flatten(1), feet_air_time.flatten(1), feet_height.flatten(1), joint_acc, joint_torque, action_delay.float()], dim=-1
            )

        return current_actor_obs, current_critic_obs

    def init_obs_buffer(self):
        if self.add_noise:
            if self.cfg.attn_enc.vel_in_obs:
                actor_obs, _ = self.compute_current_observations()
                noise_vec = torch.zeros_like(actor_obs[0])
                noise_scales = self.cfg.noise.noise_scales
                noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
                noise_vec[3:6] = noise_scales.lin_vel * self.obs_scales.lin_vel
                noise_vec[6:9] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
                noise_vec[9:12] = 0
                noise_vec[12 : 12 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
                noise_vec[12 + self.num_actions : 12 + self.num_actions * 2] = (
                    noise_scales.joint_vel * self.obs_scales.joint_vel
                )
                noise_vec[12 + self.num_actions * 2 : 12 + self.num_actions * 3] = 0.0
            else:
                actor_obs, _ = self.compute_current_observations()
                noise_vec = torch.zeros_like(actor_obs[0])
                noise_scales = self.cfg.noise.noise_scales
                noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
                noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
                noise_vec[6:9] = 0
                noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
                noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                    noise_scales.joint_vel * self.obs_scales.joint_vel
                )
                noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                )
                height_scan = torch.clamp(height_scan - self.cfg.normalization.height_scan_offset, min=-1.0, max=1.0)
                height_scan = torch.nan_to_num(height_scan, nan=1.0, posinf=1.0, neginf=-1.0)
                height_scan *= self.obs_scales.height_scan
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        # The ray pattern is generated by iterating through x first, then y.
        # This means the flattened (L*W) dimension is ordered as [p(x0,y0), p(x1,y0), ..., p(xL,y0), p(x0,y1), ...].
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
            )
            height_scan = torch.clamp(height_scan - self.cfg.normalization.height_scan_offset, min=-1.0, max=1.0)
            height_scan = torch.nan_to_num(height_scan, nan=1.0, posinf=1.0, neginf=-1.0)
            height_scan *= self.obs_scales.height_scan
            if not self.cfg.attn_enc.critic_encoder:
                current_critic_obs = torch.cat([current_critic_obs, height_scan], dim=-1)
            if self.cfg.scene.height_scanner.enable_height_scan_actor:
                height_scan_actor = height_scan.clone()
                if self.add_noise:
                    height_scan_actor += (2 * torch.rand_like(height_scan_actor) - 1) * self.height_scan_noise_vec
                if not self.cfg.attn_enc.use_attn_enc:
                    current_actor_obs = torch.cat([current_actor_obs, height_scan_actor], dim=-1)

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        if self.cfg.attn_enc.use_attn_enc:
            return actor_obs, critic_obs, height_scan, height_scan_actor
        else:
            return actor_obs, critic_obs

    def get_observations(self):
        if self.cfg.attn_enc.use_attn_enc:
            actor_obs, critic_obs, perception_obs, perception_actor_obs = self.compute_observations()
        else:
            actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        if self.cfg.attn_enc.use_attn_enc:
            self.extras["observations"]["perception"] = perception_obs
            self.extras["observations"]["perception_actor"] = perception_actor_obs
        return actor_obs, self.extras

    def step(self, actions: torch.Tensor):

        delayed_actions = self.action_buffer.compute(actions)
        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        if self.cfg.attn_enc.use_attn_enc:
            actor_obs, critic_obs, perception_obs, perception_actor_obs = self.compute_observations()
        else:
            actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        if self.cfg.attn_enc.use_attn_enc:
            self.extras["observations"]["perception"] = perception_obs
            self.extras["observations"]["perception_actor"] = perception_actor_obs

        return actor_obs, reward_buf, self.reset_buf, self.extras