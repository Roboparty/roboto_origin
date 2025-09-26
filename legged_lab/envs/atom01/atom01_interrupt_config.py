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

from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
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
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class ATOM01RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
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
        weight=0.5,
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
            "threshold": 400,
            "max_reward": 400,
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.18, "max": 0.50},
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.19, "max": 0.35},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.06,
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
                "robot", joint_names=[".*torso.*", ".*_elbow_yaw.*"]
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_pitch.*", ".*_knee.*", ".*_ankle_pitch.*", ".*_ankle_roll.*"])},
    )
    joint_deviation_interrupt = RewTerm(
        func=mdp.joint_deviation_interrupt,
        weight=-1.0,
        params={
            "asset_cfg1": SceneEntityCfg(
                "robot", joint_names=[".*_arm_roll.*", ".*_arm_yaw.*", ".*_elbow_pitch.*"]
            ),
            "asset_cfg2": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm_pitch.*"],
            ),
            "weight1": 1.0, "weight2": 0.06
        }
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    upward = RewTerm(func=mdp.upward, weight=0.4)
    stand_still = RewTerm(func=mdp.stand_still_interrupt, weight=-0.3, params={"pos_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]),
                                                                               "vel_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]), 
                                                                               "interrupt_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow_pitch.*"]),
                                                                               "pos_weight": 1.0, "vel_weight": 0.04})
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "ankle_height":0.035,"threshold":0.025})

def generate_height_scan_mirror(start_idx=140, rows=17, cols=11):
    mirror_indices = []
    for row in range(rows):
        for col in range(cols):
            mirror_col = cols - 1 - col
            mirror_idx = start_idx + row * cols + mirror_col
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

joint_pos_mirror_indices, joint_pos_mirror_signs = generate_joint_mirror(9)
joint_vel_mirror_indices, joint_vel_mirror_signs = generate_joint_mirror(32)
action_mirror_indices, action_mirror_signs = generate_joint_mirror(55)
policy_obs_mirror_indices = [0, 1, 2,\
                             3, 4, 5,\
                             6, 7, 8]\
                            + joint_pos_mirror_indices + joint_vel_mirror_indices + action_mirror_indices\
                            + [78]
policy_obs_mirror_signs = [-1, 1, -1,\
                           1, -1, 1,\
                           1, -1, -1] + joint_pos_mirror_signs + joint_vel_mirror_signs + action_mirror_signs\
                           + [1]
joint_acc_mirror_indices, joint_acc_mirror_signs = generate_joint_mirror(94)
joint_torques_mirror_indices, joint_torques_mirror_signs = generate_joint_mirror(117)
critic_obs_mirror_indices = policy_obs_mirror_indices +\
                            [79, 80, 81,\
                             83, 82,\
                             87, 88, 89, 84, 85, 86,\
                             91, 90,\
                             93, 92]\
                            + joint_acc_mirror_indices + joint_torques_mirror_indices +\
                            [140]
height_scan_mirror_indices, height_scan_mirror_signs = generate_height_scan_mirror(141, 17, 11)
critic_obs_mirror_indices += height_scan_mirror_indices
critic_obs_mirror_signs = policy_obs_mirror_signs +\
                           [1, -1, 1,\
                            1, 1,\
                            1, -1, 1, 1, -1, 1,\
                            1, 1,\
                            1, 1]\
                            + joint_acc_mirror_signs + joint_torques_mirror_signs +\
                            [1]
critic_obs_mirror_signs += height_scan_mirror_signs
act_mirror_indices = [1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21]
act_mirror_signs = [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
policy_obs_mirror_indices_expanded = []
for i in range(10):
    offset = i * 79
    for idx in policy_obs_mirror_indices:
        policy_obs_mirror_indices_expanded.append(idx + offset)
policy_obs_mirror_signs_expanded = policy_obs_mirror_signs * 10

critic_obs_mirror_indices_expanded = []
for i in range(10):
    offset = i * 328
    for idx in critic_obs_mirror_indices:
        critic_obs_mirror_indices_expanded.append(idx + offset)
critic_obs_mirror_signs_expanded = critic_obs_mirror_signs * 10

def mirror_policy_observation(policy_obs):
    mirrored_policy_obs = policy_obs[..., policy_obs_mirror_indices_expanded]
    policy_obs_mirror_signs_tensor_expanded = torch.tensor(policy_obs_mirror_signs_expanded, 
                                                           dtype=policy_obs.dtype, 
                                                           device=policy_obs.device)
    mirrored_policy_obs = mirrored_policy_obs * policy_obs_mirror_signs_tensor_expanded
    return mirrored_policy_obs

def mirror_critic_observation(critic_obs):
    mirrored_critic_obs = critic_obs[..., critic_obs_mirror_indices_expanded]
    critic_obs_mirror_signs_tensor_expanded = torch.tensor(critic_obs_mirror_signs_expanded, 
                                                           dtype=critic_obs.dtype, 
                                                           device=critic_obs.device)
    mirrored_critic_obs = mirrored_critic_obs * critic_obs_mirror_signs_tensor_expanded
    return mirrored_critic_obs

def mirror_actions(actions):
    mirrored_actions = actions[..., act_mirror_indices]
    act_mirror_signs_tensor = torch.tensor(act_mirror_signs, dtype=actions.dtype, device=actions.device)
    mirrored_actions = mirrored_actions * act_mirror_signs_tensor
    return mirrored_actions

def data_augmentation_func(env, obs, actions, obs_type):
    if obs is None:
        obs_aug = None
    else:
        if obs_type == 'policy':
            obs_aug = torch.cat((obs, mirror_policy_observation(obs)), dim=0)
        elif obs_type == 'critic':
            obs_aug = torch.cat((obs, mirror_critic_observation(obs)), dim=0)
        else:
            raise ValueError(f"Mirror logic for observation type '{obs_type}' not implemented")
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug

@configclass
class ATOM01InterruptEnvCfg(BaseEnvCfg):

    reward = ATOM01RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = ATOM01_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.scene.height_scanner.enable_height_scan = True
        self.robot.terminate_contacts_body_names = ["torso_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link", "base_link"]
        self.domain_rand.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["torso_link", "base_link"]
        self.domain_rand.events.scale_link_mass.params["asset_cfg"].body_names = ["left_.*_link", "right_.*_link"]
        self.domain_rand.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.domain_rand.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.robot.action_scale = 0.25
        self.domain_rand.action_delay.params["max_delay"] = 5
        self.noise.noise_scales.ang_vel = 0.35
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03

        self.interrupt: InterruptCfg = InterruptCfg(
        use_interrupt = True,
        max_curriculum = 1.0,
        interrupt_ratio = 0.5,
        interrupt_joint_names = [
            "left_arm_pitch_joint",
            "left_arm_roll_joint",
            "left_arm_yaw_joint",
            "left_elbow_pitch_joint",
            "right_arm_pitch_joint",
            "right_arm_roll_joint",
            "right_arm_yaw_joint",
            "right_elbow_pitch_joint",
        ],
    interrupt_scale = [
            3.14, # Arm Pitch -1.57~1.57
            1.25, # Arm Roll, -0.25~1.0
            3.14, # Arm Yaw,  -1.57~1.57
            2.07, # Elbow Pitch, -0.5~1.57
            3.14, # Arm Pitch -1.57~1.57
            1.25, # Arm Roll, -1.0~0.25
            3.14, # Arm Yaw,  -1.57~1.57
            2.07, # Elbow Pitch, -0.5~1.57
        ], # Uniform Distribution Noise for each joint.
    interrupt_lower_bound = [
            -1.57,
            -0.25, 
            -1.57, 
            -0.5, 
            -1.57, 
            -1.0, 
            -1.57,
            -0.5,
        ],
        interrupt_init_range = 0.2,
        interrupt_update_step = 30,
        switch_prob = 0.005,
    )
    interrupt_vis_cfg = VisualizationMarkersCfg(
        markers={
            "interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "no_interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
        prim_path="/Visuals/Command/interrupt",
    )


@configclass
class ATOM01InterruptAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name: str = "atom01_interrupt"
        self.wandb_project: str = "atom01_interrupt"
        self.seed = 42
        self.num_steps_per_env = 24
        self.max_iterations = 9001
        self.save_interval = 1000
        self.empirical_normalization = True
        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
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
class InterruptCfg:
    use_interrupt: bool = False
    max_curriculum: float = 1.0
    interrupt_ratio: float = 0.5
    interrupt_joint_names: list = []
    interrupt_scale : list = []
    interrupt_lower_bound: list = []
    interrupt_init_range: float = 0.2
    interrupt_update_step: int = 30
    switch_prob: float = 0.005
    interrupt_vis_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        markers={
            "interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "no_interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
        prim_path="/Visuals/Command/interrupt",
    )