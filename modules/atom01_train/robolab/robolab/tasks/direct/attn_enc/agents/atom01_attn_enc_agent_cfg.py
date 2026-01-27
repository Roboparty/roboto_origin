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

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)
import torch
from tensordict import TensorDict
from functools import lru_cache

from robolab.tasks.direct.base import (  # noqa:F401
    BaseAgentCfg,
)


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

map_scan_mirror_indices, map_scan_mirror_signs = generate_height_scan_mirror(0, 11, 17)
joint_pos_mirror_indices, joint_pos_mirror_signs = generate_joint_mirror(9)
joint_vel_mirror_indices, joint_vel_mirror_signs = generate_joint_mirror(32)
action_mirror_indices, action_mirror_signs = generate_joint_mirror(55)
policy_obs_mirror_indices = [0, 1, 2,\
                             3, 4, 5,\
                             6, 7, 8]\
                            + joint_pos_mirror_indices + joint_vel_mirror_indices + action_mirror_indices
policy_obs_mirror_signs = [-1, 1, -1,\
                           1, -1, 1,\
                           1, -1, -1] + joint_pos_mirror_signs + joint_vel_mirror_signs + action_mirror_signs
joint_acc_mirror_indices, joint_acc_mirror_signs = generate_joint_mirror(93)
joint_torques_mirror_indices, joint_torques_mirror_signs = generate_joint_mirror(116)
critic_obs_mirror_indices = policy_obs_mirror_indices +\
                            [78, 79, 80,\
                             82, 81,\
                             86, 87, 88, 83, 84, 85,\
                             90, 89,\
                             92, 91]\
                            + joint_acc_mirror_indices + joint_torques_mirror_indices +\
                            [142, 143, 144, 139, 140, 141]
critic_obs_mirror_signs = policy_obs_mirror_signs +\
                           [1, -1, 1,\
                            1, 1,\
                            1, -1, 1, 1, -1, 1,\
                            1, 1,\
                            1, 1]\
                            + joint_acc_mirror_signs + joint_torques_mirror_signs +\
                            [1, -1, 1, 1, -1, 1]
act_mirror_indices = [1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21]
act_mirror_signs = [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
policy_obs_mirror_indices_expanded = []
for i in range(5):
    offset = i * 78
    for idx in policy_obs_mirror_indices:
        policy_obs_mirror_indices_expanded.append(idx + offset)
policy_obs_mirror_signs_expanded = policy_obs_mirror_signs * 5

critic_obs_mirror_indices_expanded = []
for i in range(5):
    offset = i * 145
    for idx in critic_obs_mirror_indices:
        critic_obs_mirror_indices_expanded.append(idx + offset)
critic_obs_mirror_signs_expanded = critic_obs_mirror_signs * 5

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


def data_augmentation_func(env, obs, actions):
    if obs is None:
        obs_aug = None
    else:
        obs_mirror = obs.clone()
        obs_mirror["policy"] = mirror_policy_observation(obs["policy"])
        if "critic" in obs.keys():
            obs_mirror["critic"] = mirror_critic_observation(obs["critic"])
        if "perception_a" in obs.keys():
            obs_mirror["perception_a"] = mirror_perception_observation(obs["perception_a"])
        if "perception_c" in obs.keys():
            obs_mirror["perception_c"] = mirror_perception_observation(obs["perception_c"])
        obs_aug = torch.cat([obs, obs_mirror], dim=0)
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug

@configclass
class RslRlPpoEncActorCriticCfg(RslRlPpoActorCriticCfg):
    embedding_dim:int = 64
    head_num:int = 8
    map_size:tuple = (17, 11)
    map_resolution:float = 0.1
    actor_history_length:int = 5
    critic_history_length:int = 1
    enable_critic_estimation:bool = False
    estimation_slice:list = [78, 79, 80]
    estimaiton_hidden_dims:list = [256, 64]
    enable_obs_encoder:bool = False
    obs_encoder_hidden_dims:list = [256, 64]
    latent_dim:int = 16

@configclass
class RslRlPpoEncAlgorithmCfg(RslRlPpoAlgorithmCfg):
    enable_aux_loss:bool = False
    aux_loss_coef:float = 1.0


@configclass
class ATOM01AttnEncAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name: str = "atom01_attn_enc"
        self.wandb_project: str = "atom01_attn_enc"
        self.seed = 42
        self.obs_groups= {"policy": ["policy"], "critic": ["critic"], "perception":["perception_a", "perception_c"]}
        self.num_steps_per_env = 24
        self.max_iterations = 9001
        self.save_interval = 1000
        self.actor_obs_normalization: True
        self.critic_obs_normalization: True
        self.policy = RslRlPpoEncActorCriticCfg(
            class_name="ActorCriticAttnEnc",
            init_noise_std=1.0,
            noise_std_type="scalar",
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            embedding_dim=32,
            head_num=4,
            map_size=(17, 11),
            map_resolution=0.1,
            actor_history_length=5,
            critic_history_length=5,
            enable_critic_estimation=True,
            estimation_slice=[78, 79, 80],
            estimaiton_hidden_dims=[256, 64],
            enable_obs_encoder=True,
            latent_dim=32,
            obs_encoder_hidden_dims=[256, 128],
        )
        self.algorithm = RslRlPpoEncAlgorithmCfg(
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
            enable_aux_loss=True,
            aux_loss_coef=0.05,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True, 
                use_mirror_loss=True,
                mirror_loss_coeff=0.1, 
                data_augmentation_func=data_augmentation_func
            ),
            rnd_cfg=None,  # RslRlRndCfg()
        )
        self.clip_actions = 100.0