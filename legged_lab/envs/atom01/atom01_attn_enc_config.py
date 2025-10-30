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

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
import matplotlib as mpl
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)
import torch
import numpy as np
from functools import lru_cache

import legged_lab.mdp as mdp
from legged_lab.assets.roboparty import ATOM01_CFG
from legged_lab.envs.base.base_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG, ROUGH_HARD_TERRAINS_CFG


@configclass
class ATOM01RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=4.0, params={"std": 1.0})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 1.0})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-4)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-2e-2)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle_roll.*).*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.16, "max": 0.50},
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.18, "max": 0.35},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_thigh_yaw.*", ".*_thigh_roll.*"]
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*torso.*", ".*_arm_roll.*", ".*_arm_yaw.*", ".*_elbow_pitch.*", ".*_elbow_yaw.*"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.06,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm_pitch.*"],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_pitch.*", ".*_knee.*", ".*_ankle_pitch.*", ".*_ankle_roll.*"])},
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    upward = RewTerm(func=mdp.upward, weight=0.4)
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.2, params={"pos_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]),
                                                                     "vel_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]), 
                                                                     "pos_weight": 1.0, "vel_weight": 0.04})
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "ankle_height":0.04,"threshold":0.02})

def generate_height_scan_mirror(start_idx=140, rows=11, cols=17):
    mirror_indices = []
    for row in range(rows):
        mirror_row = rows - 1 - row
        for col in range(cols):
            mirror_idx = start_idx + col + mirror_row * cols
            mirror_indices.append(mirror_idx)
    mirror_signs = [1] * (rows * cols)
    return mirror_indices, mirror_signs

def generate_joint_mirror(start_idx):
    mirror_indices = []
    mirror_indices.extend([start_idx + 1, start_idx])    
    mirror_indices.append(start_idx + 2)
    for i in range(start_idx + 3, start_idx + 23, 2):
        mirror_indices.extend([i + 1, i])
    mirror_signs = [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
    return mirror_indices, mirror_signs

joint_pos_mirror_indices, joint_pos_mirror_signs = generate_joint_mirror(12)
joint_vel_mirror_indices, joint_vel_mirror_signs = generate_joint_mirror(35)
action_mirror_indices, action_mirror_signs = generate_joint_mirror(58)
policy_obs_mirror_indices = [0, 1, 2,\
                             3, 4, 5,\
                             6, 7, 8,\
                             9, 10, 11]\
                            + joint_pos_mirror_indices + joint_vel_mirror_indices + action_mirror_indices
policy_obs_mirror_signs = [-1, 1, -1,\
                           1, -1, 1,\
                           1, -1, 1,\
                           1, -1, -1] + joint_pos_mirror_signs + joint_vel_mirror_signs + action_mirror_signs
joint_acc_mirror_indices, joint_acc_mirror_signs = generate_joint_mirror(93)
joint_torques_mirror_indices, joint_torques_mirror_signs = generate_joint_mirror(116)
critic_obs_mirror_indices = policy_obs_mirror_indices +\
                            [82, 81,\
                             86, 87, 88, 83, 84, 85,\
                             90, 89,\
                             92, 91]\
                            + joint_acc_mirror_indices + joint_torques_mirror_indices +\
                            [139]
height_scan_mirror_indices, height_scan_mirror_signs = generate_height_scan_mirror(140, 11, 17)
critic_obs_mirror_signs = policy_obs_mirror_signs +\
                           [1, 1,\
                            1, -1, 1, 1, -1, 1,\
                            1, 1,\
                            1, 1]\
                            + joint_acc_mirror_signs + joint_torques_mirror_signs +\
                            [1]
act_mirror_indices = [1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21]
act_mirror_signs = [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
map_scan_mirror_indices, map_scan_mirror_signs = generate_height_scan_mirror(0, 11, 17)

policy_obs_mirror_indices_expanded = []
for i in range(1):
    offset = i * 81
    for idx in policy_obs_mirror_indices:
        policy_obs_mirror_indices_expanded.append(idx + offset)
policy_obs_mirror_signs_expanded = policy_obs_mirror_signs * 1

critic_obs_mirror_indices_expanded = []
for i in range(1):
    offset = i * 140
    for idx in critic_obs_mirror_indices:
        critic_obs_mirror_indices_expanded.append(idx + offset)
critic_obs_mirror_signs_expanded = critic_obs_mirror_signs * 1

@lru_cache(maxsize=None)
def get_policy_obs_mirror_signs_tensor(device, dtype):
    return torch.tensor(policy_obs_mirror_signs_expanded, device=device, dtype=dtype)

def mirror_policy_observation(policy_obs):
    mirrored_policy_obs = policy_obs[..., policy_obs_mirror_indices_expanded]
    signs = get_policy_obs_mirror_signs_tensor(device=policy_obs.device, dtype=policy_obs.dtype)
    mirrored_policy_obs *= signs
    return mirrored_policy_obs

@lru_cache(maxsize=None)
def get_critic_obs_mirror_signs_tensor(device, dtype):
    return torch.tensor(critic_obs_mirror_signs_expanded, device=device, dtype=dtype)

def mirror_critic_observation(critic_obs):
    mirrored_critic_obs = critic_obs[..., critic_obs_mirror_indices_expanded]
    signs = get_critic_obs_mirror_signs_tensor(device=critic_obs.device, dtype=critic_obs.dtype)
    mirrored_critic_obs *= signs
    return mirrored_critic_obs

@lru_cache(maxsize=None)
def get_act_mirror_signs_tensor(device, dtype):
    return torch.tensor(act_mirror_signs, device=device, dtype=dtype)

def mirror_actions(actions):
    mirrored_actions = actions[..., act_mirror_indices]
    signs = get_act_mirror_signs_tensor(device=actions.device, dtype=actions.dtype)
    mirrored_actions *= signs
    return mirrored_actions

@lru_cache(maxsize=None)
def get_map_scan_mirror_signs_tensor(device, dtype):
    return torch.tensor(map_scan_mirror_signs, device=device, dtype=dtype)

def mirror_perception_observation(perception_obs):
    mirrored_obs = perception_obs[..., map_scan_mirror_indices]
    signs = get_map_scan_mirror_signs_tensor(device=perception_obs.device, dtype=perception_obs.dtype)
    mirrored_obs *= signs
    return mirrored_obs

def data_augmentation_func(env, obs, actions, obs_type):
    if obs is None:
        obs_aug = None
    else:
        if obs_type == 'policy':
            obs_aug = torch.cat((obs, mirror_policy_observation(obs)), dim=0)
        elif obs_type == 'critic':
            obs_aug = torch.cat((obs, mirror_critic_observation(obs)), dim=0)
        elif obs_type == 'perception':
            obs_aug = torch.cat((obs, mirror_perception_observation(obs)), dim=0)
        else:
            raise ValueError(f"Mirror logic for observation type '{obs_type}' not implemented")
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug

color = [mpl.colormaps['viridis'](i/9.0)[:-1] for i in range(10)]
markers = {}
for i in range(10):
    markers[f"hit_{i}"] = sim_utils.SphereCfg(
        radius=0.02,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color[i])
    )
@configclass
class AttnEncCfg:
    use_attn_enc: bool = False
    vel_in_obs: bool = False
    critic_encoder: bool = False
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Attention",
        markers=markers,
    )   


@configclass
class ATOM01AttnEncStage2EnvCfg(BaseEnvCfg):

    reward = ATOM01RewardCfg()
    attn_enc = AttnEncCfg(
            use_attn_enc=True,
            vel_in_obs=True,
            critic_encoder=True,
        )

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "base_link"
        self.scene.robot = ATOM01_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = ROUGH_HARD_TERRAINS_CFG
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.enable_height_scan_actor = True
        self.scene.height_scanner.resolution = 0.08
        self.scene.height_scanner.size = (1.28, 0.8)
        self.robot.terminate_contacts_body_names = ["torso_link", ".*_thigh_yaw_link", ".*_thigh_roll_link", ".*_elbow_.*_link", ".*_arm_.*_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.noise.add_noise = True
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link", "base_link"]
        self.domain_rand.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["torso_link", "base_link"]
        self.domain_rand.events.scale_link_mass.params["asset_cfg"].body_names = ["left_.*_link", "right_.*_link"]
        self.domain_rand.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.domain_rand.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.robot.action_scale = 0.25
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.domain_rand.action_delay.params["max_delay"] = 4
        self.normalization.height_scan_offset = 0.75
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03
        self.noise.noise_scales.lin_vel = 0.2
        self.noise.noise_scales.height_scan = 0.025


@configclass
class ATOM01AttnEncStage1EnvCfg(ATOM01AttnEncStage2EnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.noise.add_noise = False
        self.domain_rand.events.add_base_mass = None
        self.domain_rand.events.randomize_rigid_body_com = None
        self.domain_rand.events.scale_link_mass = None
        self.domain_rand.events.scale_actuator_gains = None
        self.domain_rand.events.scale_joint_parameters = None
        self.domain_rand.events.push_robot = None
        self.commands.ranges.lin_vel_x = (0.0, 1.0)


@configclass
class RslRlPpoEncActorCriticCfg(RslRlPpoActorCriticCfg):
    embedding_dim:int = 64
    head_num:int = 8
    map_size:tuple = (17, 11)
    map_resolution:float = 0.08
    single_obs_dim:int = 78
    velocity_estimation:bool = False
    critic_encoder:bool = False

@configclass
class RslRlPpoEncAlgorithmCfg(RslRlPpoAlgorithmCfg):
    velocity_estimation:bool = False
    velocity_slice:slice = slice(78, 81)
    velocity_loss_coef:float = 1.0


@configclass
class ATOM01AttnEncStage2AgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name: str = "atom01_attn_enc"
        self.wandb_project: str = "atom01_attn_enc"
        self.seed = 42
        self.num_steps_per_env = 24
        self.max_iterations = 9001
        self.save_interval = 1000
        self.runner_class_name = "AttnEncOnPolicyRunner"
        self.empirical_normalization = True
        self.policy = RslRlPpoEncActorCriticCfg(
            class_name="AttnEncActorCritic",
            init_noise_std=1.0,
            noise_std_type="scalar",
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            embedding_dim=64,
            head_num=8,
            map_size=(17, 11),
            map_resolution=0.08,
            single_obs_dim=81,
            velocity_estimation=False,
            critic_encoder=True,
        )
        self.algorithm = RslRlPpoEncAlgorithmCfg(
            class_name="AttnEncPPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            velocity_estimation=False,
            velocity_slice=slice(78, 81),
            velocity_loss_coef=0.1,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True, 
                use_mirror_loss=True,
                mirror_loss_coeff=0.2, 
                data_augmentation_func=data_augmentation_func
            ),
            rnd_cfg=None,  # RslRlRndCfg()
        )
        self.clip_actions = 100.0


@configclass
class ATOM01AttnEncStage1AgentCfg(ATOM01AttnEncStage2AgentCfg):
    def __post_init__(self):
        super().__post_init__()