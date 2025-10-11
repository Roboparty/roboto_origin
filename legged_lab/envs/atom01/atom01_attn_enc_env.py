import torch
import numpy as np

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse,quat_apply_yaw, quat_inv 
from legged_lab.envs.base.base_env import (  # noqa:F401
    BaseEnv,
)
from .atom01_attn_enc_config import ATOM01AttnEncEnvCfg

class ATOM01AttnEncEnv(BaseEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.cfg: ATOM01AttnEncEnvCfg

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        if self.cfg.scene.height_scanner.enable_height_scan:
            if self.cfg.attn_enc.use_attn_enc:
                perception_obs = self.map_scan_base()
            else:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                )
                height_scan = torch.clamp(height_scan - self.cfg.normalization.height_scan_offset, min=-self.cfg.normalization.height_scan_offset, max=self.cfg.normalization.height_scan_offset)
                height_scan = torch.nan_to_num(height_scan, nan=0, posinf=self.cfg.normalization.height_scan_offset, neginf=-self.cfg.normalization.height_scan_offset)
                height_scan *= self.obs_scales.height_scan
                current_critic_obs = torch.cat([current_critic_obs, height_scan], dim=-1)
                if self.add_noise:
                    height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
                if self.cfg.scene.height_scanner.enable_height_scan_actor:
                    current_actor_obs = torch.cat([current_actor_obs, height_scan], dim=-1)

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        if self.cfg.attn_enc.use_attn_enc:
            return actor_obs, critic_obs, perception_obs
        else:
            return actor_obs, critic_obs

    def get_observations(self):
        actor_obs, critic_obs, perception_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        if self.cfg.attn_enc.use_attn_enc:
            self.extras["observations"]["perception"] = perception_obs
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

        actor_obs, critic_obs, perception_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        if self.cfg.attn_enc.use_attn_enc:
            self.extras["observations"]["perception"] = perception_obs

        return actor_obs, reward_buf, self.reset_buf, self.extras
    
    def map_scan_base(self):
        grid_size = self.height_scanner.cfg.pattern_cfg.size  # [L,W] (m)
        resolution = self.height_scanner.cfg.pattern_cfg.resolution
        grid_shape = (int(grid_size[0] / resolution) + 1, int(grid_size[1] / resolution) + 1)

        height_scan = (
            self.height_scanner.data.pos_w[:, :3].unsqueeze(1) - self.height_scanner.data.ray_hits_w[..., :3]
        )
        # shape: [B, L*W, 3]
        L = grid_shape[0]
        W = grid_shape[1]
        B = height_scan.shape[0]

        # convert to base frame
        if self.height_scanner.cfg.ray_alignment == "yaw":
            quat_w2b = quat_inv(self.height_scanner.data.quat_w).unsqueeze(1)  # [B, 1, 4]
            quat_w2b = quat_w2b.expand(-1, L * W, -1).contiguous()  # [B, L*W, 4]
            height_scan = quat_apply_yaw(quat_w2b, height_scan)
        elif self.height_scanner.cfg.ray_alignment == "base":
            quat_w = self.height_scanner.data.quat_w.unsqueeze(1)  # [B, 1, 4]
            quat_w = quat_w.expand(-1, L * W, -1).contiguous()  # [B, L*W, 4]
            height_scan = quat_apply_inverse(quat_w, height_scan)

        height_scan[..., 2] = torch.clamp(height_scan[..., 2] - self.cfg.normalization.height_scan_offset, min=-self.cfg.normalization.height_scan_offset, max=self.cfg.normalization.height_scan_offset)
        height_scan = torch.nan_to_num(height_scan, nan=0, posinf=self.cfg.normalization.height_scan_offset, neginf=-self.cfg.normalization.height_scan_offset)
        height_scan *= self.obs_scales.height_scan

        # The ray pattern is generated by iterating through x first, then y.
        # This means the flattened (L*W) dimension is ordered as [p(x0,y0), p(x1,y0), ..., p(xL,y0), p(x0,y1), ...].
        return height_scan.view(B, W, L, 3)