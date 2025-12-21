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

import argparse
import os

import torch
from isaaclab.app import AppLauncher
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner, AttnEncOnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--plane", action='store_true', help="Use plane terrain")
parser.add_argument("--push_robot", action='store_true', help="Push robot during playing")
parser.add_argument("--keyboard", action='store_true', help="Keyboard control mode")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.devices import Se2Keyboard
from isaaclab.markers import VisualizationMarkers

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

import copy

class TorchAttnEncPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        # copy policy parameters
        self.actor = copy.deepcopy(policy.actor)
        self.encoder = copy.deepcopy(policy.encoder)
        self.num_actor_obs = policy.num_actor_obs
        self.velocity_estimation = policy.velocity_estimation
        self.single_obs_dim = policy.single_obs_dim
        if self.velocity_estimation:
            self.estimator = copy.deepcopy(policy.estimator)
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        prop_obs = self.normalizer(x[:, :self.num_actor_obs])
        perception_obs = x[:, self.num_actor_obs:]
        if self.velocity_estimation:
            velocity = self.estimator(prop_obs)
            obs = torch.cat([prop_obs[:, -self.single_obs_dim:], velocity], dim=1) 
            embedding, attention, *_ = self.encoder(perception_obs, obs, embedding_only=False)
        else:
            embedding, attention, *_ = self.encoder(perception_obs, prop_obs[:, -self.single_obs_dim:], embedding_only=True)
            embedding = torch.cat([embedding, prop_obs], dim=-1)
        return self.actor(embedding)

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class OnnxAttnEncPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        # copy policy parameters
        self.actor = copy.deepcopy(policy.actor)
        self.encoder = copy.deepcopy(policy.encoder)
        self.num_actor_obs = policy.num_actor_obs
        self.velocity_estimation = policy.velocity_estimation
        self.single_obs_dim = policy.single_obs_dim
        self.map_size = policy.map_size
        if self.velocity_estimation:
            self.estimator = copy.deepcopy(policy.estimator)
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        prop_obs = self.normalizer(x[:, :self.num_actor_obs])
        perception_obs = x[:, self.num_actor_obs:]
        if self.velocity_estimation:
            velocity = self.estimator(prop_obs)
            obs = torch.cat([prop_obs[:, -self.single_obs_dim:], velocity], dim=1) 
            embedding, attention, *_ = self.encoder(perception_obs, obs, embedding_only=False)
        else:
            embedding, attention, *_ = self.encoder(perception_obs, prop_obs[:, -self.single_obs_dim:], embedding_only=True)
            embedding = torch.cat([embedding, prop_obs], dim=-1)
        return self.actor(embedding)

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.
        obs = torch.zeros(1, self.num_actor_obs + self.map_size[0]*self.map_size[1])
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )

def visualize_attention(map_scan, root_pose, output_attn, visualizer):
    root_pos = root_pose[:, :3]  # shape (B, 3)
    map_scan_world = -map_scan + root_pos.unsqueeze(1).unsqueeze(1)  # shape (B, W, L, 3)
    B = root_pos.shape[0]

    max_attn_per_env, _ = torch.max(output_attn.view(B, -1), dim=1)
    max_attn_per_env[max_attn_per_env == 0] = 1.0
    normalized_attn = output_attn / max_attn_per_env.view(B, 1, 1)
    normalized_attn = normalized_attn.view(-1)
    # print(normalized_attn)
    attention_indices = torch.zeros_like(normalized_attn, dtype=torch.int)
    for i in range(10):
        color_mask = (normalized_attn > 0.1 * i)
        color_mask = torch.bitwise_and(color_mask, normalized_attn < 0.1 * (i + 1))
        attention_indices[color_mask] = i
    visualizer.visualize(translations=map_scan_world.view(-1, 3), marker_indices=attention_indices)

def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    if not args_cli.push_robot:
        env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5

    kbd_control = None
    if args_cli.keyboard:
        env_cfg.commands.heading_command=False
        env_cfg.commands.rel_standing_envs = 0.0
        kbd_control = Se2Keyboard(
            v_x_sensitivity=0.8, 
            v_y_sensitivity=0.4, 
            omega_z_sensitivity=1.0
        )
    else:
        env_cfg.commands.ranges.lin_vel_x = (0.6, 0.6)
        env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.ranges.heading = (0.0, 0.0)
        env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    if args_cli.plane:
       env_cfg.scene.terrain_generator = None
       env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (1.0, 1.0)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if hasattr(env_cfg, "attn_enc"):
        visualizer = VisualizationMarkers(env_cfg.attn_enc.marker_cfg)

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    if hasattr(env_cfg, 'interrupt') and env_cfg.interrupt.use_interrupt:
        env_cfg.interrupt.interrupt_ratio = 1.0
        env.interrupt_rad_curriculum = torch.ones(env_cfg.scene.num_envs, dtype=torch.float, device=env_cfg.device, requires_grad=False)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner | AttnEncOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if not os.path.exists(export_model_dir):
        os.makedirs(export_model_dir, exist_ok=True)
    if hasattr(env_cfg, "attn_enc"):
        torch_policy_exporter = TorchAttnEncPolicyExporter(runner.alg.policy, runner.obs_normalizer)
        torch_policy_exporter.export(path=export_model_dir, filename="policy.pt")
        onnx_policy_exporter = OnnxAttnEncPolicyExporter(runner.alg.policy, runner.obs_normalizer, verbose=False)
        onnx_policy_exporter.export(path=export_model_dir, filename="policy.onnx")
    else:
        export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, extras = env.get_observations()

    while simulation_app.is_running():
        if kbd_control is not None:
            vel_command = kbd_control.advance()
            if hasattr(env, 'command_generator'):
                env.command_generator.command[:, 0] = vel_command[0]  # lin_vel_x
                env.command_generator.command[:, 1] = vel_command[1]  # lin_vel_y
                env.command_generator.command[:, 2] = vel_command[2]  # ang_vel_z

        with torch.inference_mode():

            if hasattr(env_cfg, "attn_enc"):
                perception_obs = extras["observations"]["perception"]
                if agent_cfg.policy.velocity_estimation:
                    actions, output_attn, velocity = policy(perception_obs, obs)
                else: 
                    actions, output_attn = policy(perception_obs, obs)
                height_scan = (
                    env.height_scanner.data.pos_w[:, :3].unsqueeze(1) - env.height_scanner.data.ray_hits_w[..., :3]
                )
                grid_size = env.height_scanner.cfg.pattern_cfg.size  # [L,W] (m)
                resolution = env.height_scanner.cfg.pattern_cfg.resolution
                grid_shape = (int(grid_size[0] / resolution) + 1, int(grid_size[1] / resolution) + 1)
                L = grid_shape[0]
                W = grid_shape[1]
                B = height_scan.shape[0]
                height_scan = height_scan.view(B, W, L, 3)
                height_scan[..., 2] = torch.clamp(height_scan[..., 2], min=-1.0+env.cfg.normalization.height_scan_offset, max=1.0+env.cfg.normalization.height_scan_offset)
                height_scan[..., :2] = torch.nan_to_num(height_scan[..., :2], nan=0.0, posinf=0.0, neginf=-0.0)
                height_scan[..., 2] = torch.nan_to_num(height_scan[..., 2], nan=1.0+env.cfg.normalization.height_scan_offset, posinf=1.0+env.cfg.normalization.height_scan_offset, neginf=-1.0+env.cfg.normalization.height_scan_offset)
                root_pose = env.robot.data.root_pos_w
                visualize_attention(height_scan, root_pose, output_attn, visualizer)
            else:
                actions = policy(obs)
            obs, _, _, extras = env.step(actions)


if __name__ == "__main__":
    play()
    simulation_app.close()
