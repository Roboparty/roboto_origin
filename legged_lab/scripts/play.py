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

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


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
            v_x_sensitivity=env_cfg.commands.ranges.lin_vel_x[1], 
            v_y_sensitivity=env_cfg.commands.ranges.lin_vel_y[1], 
            omega_z_sensitivity=env_cfg.commands.ranges.ang_vel_z[1]
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
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

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

    if hasattr(env_cfg, "attn_enc"):
        pass
    else:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
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
                actions = policy(perception_obs, obs)
            else:
                actions = policy(obs)
            obs, _, _, extras = env.step(actions)


if __name__ == "__main__":
    play()
    simulation_app.close()
