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

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--plane", action='store_true', help="Use plane terrain")
parser.add_argument("--push_robot", action='store_true', help="Push robot during playing")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import time
import torch
import copy

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.markers import VisualizationMarkers

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robolab.tasks

class TorchAttnEncPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        # copy policy parameters
        self.actor = copy.deepcopy(policy.actor)
        self.encoder = copy.deepcopy(policy.encoder)
        self.num_actor_obs = policy.num_actor_obs
        self.critic_estimation = policy.critic_estimation
        self.single_obs_dim = policy.single_obs_dim
        if self.critic_estimation:
            self.estimator = copy.deepcopy(policy.estimator)
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        prop_obs = self.normalizer(x[:, :self.num_actor_obs])
        perception_obs = x[:, self.num_actor_obs:]
        if self.critic_estimation:
            critic_pred = self.estimator(prop_obs)
            obs = torch.cat([prop_obs[:, -self.single_obs_dim:], critic_pred], dim=1) 
            embedding, *_ = self.encoder(obs, perception_obs)
        else:
            obs = prop_obs[:, -self.single_obs_dim:]
            embedding, *_ = self.encoder(obs, perception_obs)
        embedding = torch.cat([obs, embedding], dim=-1)
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
        self.critic_estimation = policy.critic_estimation
        self.single_obs_dim = policy.single_obs_dim
        self.map_size = policy.map_size
        if self.critic_estimation:
            self.estimator = copy.deepcopy(policy.estimator)
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        prop_obs = self.normalizer(x[:, :self.num_actor_obs])
        perception_obs = x[:, self.num_actor_obs:]
        if self.critic_estimation:
            critic_pred = self.estimator(prop_obs)
            obs = torch.cat([prop_obs[:, -self.single_obs_dim:], critic_pred], dim=1) 
            embedding, *_ = self.encoder(obs, perception_obs)
        else:
            obs = prop_obs[:, -self.single_obs_dim:]
            embedding, *_ = self.encoder(obs, perception_obs)
        embedding = torch.cat([obs, embedding], dim=-1)
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

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.scene.env_spacing = 2.5

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.noise.add_noise = False
    if not args_cli.push_robot:
        env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 40.0
    env_cfg.commands.heading_command=False
    env_cfg.commands.rel_standing_envs = 0.0

    if args_cli.plane:
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.terrain_type = "plane"

    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)

    if hasattr(env_cfg, "attn_enc"):
        visualizer = VisualizationMarkers(env_cfg.attn_enc.marker_cfg)

    if hasattr(env_cfg, 'interrupt') and env_cfg.interrupt.use_interrupt:
        env_cfg.interrupt.interrupt_ratio = 1.0

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    env.unwrapped.command_generator.command[:, 0] = 0.0
    env.unwrapped.command_generator.command[:, 1] = 0.0
    env.unwrapped.command_generator.command[:, 2] = 0.0

    if hasattr(env_cfg, 'interrupt') and env_cfg.interrupt.use_interrupt:
        env.unwrapped.interrupt_rad_curriculum = torch.ones(env_cfg.scene.num_envs, dtype=torch.float, device=env.unwrapped.device, requires_grad=False)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if not os.path.exists(export_model_dir):
        os.makedirs(export_model_dir, exist_ok=True)
    if hasattr(env_cfg, "attn_enc"):
        torch_policy_exporter = TorchAttnEncPolicyExporter(policy_nn, normalizer)
        torch_policy_exporter.export(path=export_model_dir, filename="policy.pt")
        onnx_policy_exporter = OnnxAttnEncPolicyExporter(policy_nn, normalizer, verbose=False)
        onnx_policy_exporter.export(path=export_model_dir, filename="policy.onnx")
    else:
        export_policy_as_jit(policy_nn, normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
        )

    if not args_cli.headless:
        from robolab.utils.keyboard import Keyboard
        keyboard = Keyboard(env)  # noqa:F841

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if hasattr(env_cfg, "attn_enc"):
                actions, output_attn = policy(obs, return_attention=True)
                height_scan = (
                    env.unwrapped.height_scanner.data.pos_w[:, :3].unsqueeze(1) - env.unwrapped.height_scanner.data.ray_hits_w[..., :3]
                )
                grid_size = env.unwrapped.height_scanner.cfg.pattern_cfg.size  # [L,W] (m)
                resolution = env.unwrapped.height_scanner.cfg.pattern_cfg.resolution
                grid_shape = (int(grid_size[0] / resolution) + 1, int(grid_size[1] / resolution) + 1)
                L = grid_shape[0]
                W = grid_shape[1]
                B = height_scan.shape[0]
                height_scan = height_scan.view(B, W, L, 3)
                height_scan[..., 2] = torch.clamp(height_scan[..., 2], min=-1.0+env.unwrapped.cfg.normalization.height_scan_offset, max=1.0+env.unwrapped.cfg.normalization.height_scan_offset)
                height_scan[..., :2] = torch.nan_to_num(height_scan[..., :2], nan=0.0, posinf=0.0, neginf=-0.0)
                height_scan[..., 2] = torch.nan_to_num(height_scan[..., 2], nan=1.0+env.unwrapped.cfg.normalization.height_scan_offset, posinf=1.0+env.unwrapped.cfg.normalization.height_scan_offset, neginf=-1.0+env.cfg.normalization.height_scan_offset)
                root_pose = env.unwrapped.robot.data.root_pos_w
                visualize_attention(height_scan, root_pose, output_attn, visualizer)
            else:
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
