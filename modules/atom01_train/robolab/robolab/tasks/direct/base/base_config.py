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

import math
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)
from isaaclab.sim import SimulationCfg, PhysxCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from .scene_cfg import SceneCfg

from . import mdp


@configclass
class RewardCfg:
    pass


@configclass
class HeightScannerCfg:
    enable_height_scan: bool = False
    enable_height_scan_actor: bool = False
    prim_body_name: str = MISSING
    resolution: float = 0.1
    size: tuple = (1.6, 1.0)
    debug_vis: bool = False
    drift_range: tuple = (0.0, 0.0)
    offset: tuple = (0.0, 0.0, 20.0)


@configclass
class SceneContextCfg:
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5
    height_scanner: HeightScannerCfg = HeightScannerCfg()


@configclass
class RobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_history_length: int = 3
    action_scale: float = 0.25
    terminate_contacts_body_names: list = None
    terminate_base_height: float = None
    terminate_base_orientation: float = None
    feet_body_names: list = MISSING


@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    commands: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    height_scan: float = 1.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5


@configclass
class CommandRangesCfg:
    lin_vel_x: tuple = (-0.6, 1.0)
    lin_vel_y: tuple = (-0.5, 0.5)
    ang_vel_z: tuple = (-1.57, 1.57)
    heading: tuple = (-math.pi, math.pi)


@configclass
class CommandsCfg:
    resampling_time_range: tuple = (10.0, 10.0)
    rel_standing_envs: float = 0.2
    rel_heading_envs: float = 1.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = True
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class NoiseScalesCfg:
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    height_scan: float = 0.1


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True, 
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.05, 0.05)},
        },
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=MISSING
            ),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            }
        },
    )


@configclass
class BaseEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg()
    decimation: int = 4
    sim: SimulationCfg = SimulationCfg(
        dt=0.005, 
        render_interval=decimation, 
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            )
        )
    episode_length_s: float =20.0
    scene_context: SceneContextCfg = SceneContextCfg()
    scene: SceneCfg = MISSING
    robot: RobotCfg = RobotCfg()
    reward: RewardCfg = RewardCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    commands: CommandsCfg = CommandsCfg()
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(),
    )
    events: EventCfg = EventCfg()

    def __post_init__(self):
        pass


@configclass
class BaseAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 12001
    runner_class_name = "OnPolicyRunner"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.994,
        lam=0.9,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 500
    experiment_name = ""
    run_name = ""
    logger = "wandb"
    neptune_project = "robolab"
    wandb_project = "robolab"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    def __post_init__(self):
        pass
