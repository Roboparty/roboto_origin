# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.networks import AttentionEncoder

from rsl_rl.utils import resolve_nn_activation


class AttnEncActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        embedding_dim=64,
        head_num=8,
        map_size=(17, 11),
        map_resolution=0.1,
        single_obs_dim=78,
        critic_estimation=False,
        estimation_slice=[78, 79, 80],
        critic_encoder=False,
        recon_map=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "AttnEncActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.num_actor_obs = num_actor_obs
        self.map_size = map_size
        self.single_obs_dim = single_obs_dim
        self.critic_estimation = critic_estimation
        self.estimation_slice = estimation_slice
        self.critic_encoder = critic_encoder
        self.recon_map = recon_map

        if self.critic_estimation:
            self.last_critic_pred: torch.Tensor = None
        if self.recon_map:
            self.last_map_recon: torch.Tensor = None
            self.recon = nn.Sequential(
                nn.Conv2d((embedding_dim - 3), 32, kernel_size=3, padding="same", padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=1),
            )
            print(f"Reconstructor : {self.recon}")

        # Encoder
        if self.critic_estimation:
            self.encoder =  AttentionEncoder(single_obs_dim+len(self.estimation_slice), embedding_dim, head_num, self.map_size, map_resolution)
        else:
            self.encoder =  AttentionEncoder(single_obs_dim, embedding_dim, head_num, self.map_size, map_resolution)
        print(f"Encoder : {self.encoder}")

        if self.critic_estimation:
            self.estimator = nn.Sequential(
                nn.Linear(self.num_actor_obs, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, len(self.estimation_slice)),
            )
            mlp_input_dim_a = embedding_dim + single_obs_dim + len(self.estimation_slice)
            print(f"Estimator : {self.estimator}")
        else:
            mlp_input_dim_a = embedding_dim + self.num_actor_obs
        if self.critic_encoder:
            mlp_input_dim_c = embedding_dim + num_critic_obs
        else:
            mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, perception_obs, prop_obs):
        if self.critic_estimation:
            self.last_critic_pred = self.estimator(prop_obs)
            obs = torch.cat([prop_obs[:, -self.single_obs_dim:], self.last_critic_pred], dim=1)
            embedding, _, cnn_features = self.encoder(perception_obs, obs, embedding_only=False)
            if self.recon_map:
                self.last_map_recon = self.recon(cnn_features).flatten(1)
        else:
            embedding, _, cnn_features = self.encoder(perception_obs, prop_obs[:, -self.single_obs_dim:], embedding_only=True)
            if self.recon_map:
                self.last_map_recon = self.recon(cnn_features).flatten(1)
            embedding = torch.cat([embedding, prop_obs], dim=-1)
        # compute mean
        mean = self.actor(embedding)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, perception_obs, prop_obs, **kwargs):
        self.update_distribution(perception_obs, prop_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, perception_obs, prop_obs):
        if self.critic_estimation:
            critic_pred = self.estimator(prop_obs)
            obs = torch.cat([prop_obs[:, -self.single_obs_dim:], critic_pred], dim=1)
            embedding, attention, *_ = self.encoder(perception_obs, obs, embedding_only=False)
        else:
            embedding, attention, *_ = self.encoder(perception_obs, prop_obs[:, -self.single_obs_dim:], embedding_only=True)
            embedding = torch.cat([embedding, prop_obs], dim=-1)
        actions_mean = self.actor(embedding)
        if self.critic_estimation:
            return actions_mean, attention, critic_pred
        else:
            return actions_mean, attention

    def evaluate(self, critic_obs, perception_obs=None, **kwargs):
        if self.critic_encoder:
            if self.critic_estimation:
                embedding, *_ = self.encoder(perception_obs, torch.cat([critic_obs[:, :self.single_obs_dim], critic_obs[:, self.estimation_slice]], dim=1), embedding_only=True)
            else:
                embedding, *_ = self.encoder(perception_obs, critic_obs[:, :self.single_obs_dim], embedding_only=True)
            combined = torch.cat([embedding, critic_obs], dim=-1)
            value = self.critic(combined)
        else:
            value = self.critic(critic_obs)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True

    def get_estimation(self):
        if self.critic_estimation and self.last_critic_pred is not None:
            return self.last_critic_pred
        else:
            return None

    def get_map_recon(self):
        if self.last_map_recon is not None:
            return self.last_map_recon
        else:
            return None
