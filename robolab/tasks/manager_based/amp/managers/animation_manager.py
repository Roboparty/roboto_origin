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

from __future__ import annotations

import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerBase, ManagerTermBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from .animation_manager_cfg import AnimationTermCfg
from .motion_data_manager import MotionDataTerm

if TYPE_CHECKING:
    from robolab.tasks.manager_based.amp.manager_based_animation_env import ManagerBasedAnimationEnv

class AnimationTerm(ManagerTermBase):
    cfg: AnimationTermCfg
    _env: ManagerBasedAnimationEnv
    
    def __init__(self, cfg: AnimationTermCfg, env: ManagerBasedAnimationEnv):
        super().__init__(cfg, env)
        
        if cfg.num_steps_to_use > 0:
            self.step_indices = torch.arange(0, cfg.num_steps_to_use, dtype=torch.long, device=env.device)
        elif cfg.num_steps_to_use < 0:
            self.step_indices = torch.arange(cfg.num_steps_to_use + 1, 1, dtype=torch.long, device=env.device)
        else:
            raise ValueError("num_steps_to_use cannot be zero.")
        
        # Get motion data term
        self.motion_data_term: MotionDataTerm = env.motion_data_manager.get_term(cfg.motion_data_term)
        
        # create buffers
        self.num_steps = len(self.step_indices)
        for component in cfg.motion_data_components:
            buffer_shape = (self.num_envs, self.num_steps)
            if component in ["root_pos_w", "root_vel_w", "root_vel_b", "root_ang_vel_w", "root_ang_vel_b"]:
                buffer_shape += (3,)
            elif component == "root_quat":
                buffer_shape += (4,)
            elif component in ["dof_pos", "dof_vel"]:
                num_dofs = self.motion_data_term.num_dofs
                buffer_shape += (num_dofs,)
            elif component == "key_body_pos_b":
                num_key_bodies = self.motion_data_term.num_key_bodies
                buffer_shape += (num_key_bodies, 3)
            else:
                raise ValueError(f"Unknown motion data component: {component}")
            setattr(self, f"{component}_buffer", torch.zeros(buffer_shape, device=env.device, dtype=torch.float32))

        # motion ids for each env
        self.motion_ids = torch.zeros(self.num_envs, device=env.device, dtype=torch.long)
        self.motion_fetch_time = torch.zeros((self.num_envs, self.num_steps), device=env.device, dtype=torch.float32)
        self.motion_durations = torch.zeros(self.num_envs, device=env.device, dtype=torch.float32)

        self.reset(torch.arange(self.num_envs))
        self._fetch_motion_data()
        
        if self.cfg.enable_visualization:
            self.vis_root_offset = torch.tensor(self.cfg.vis_root_offset, device=env.device, dtype=torch.float32).unsqueeze(0)  # (1, 3)
            
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/KeyBodyVisualizerFromTerm",
                markers={
                    "red_sphere": sim_utils.SphereCfg(
                        radius=0.03, 
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                    ),
                }
            )
            self.key_body_marker: VisualizationMarkers = VisualizationMarkers(marker_cfg)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            return
        
        # resample motion ids for the reset envs
        self.motion_ids[env_ids] = self.motion_data_term.sample_motions(len(env_ids))
        self.motion_durations[env_ids] = self.motion_data_term.get_motion_durations(self.motion_ids[env_ids])
        
        truncate_time = self.num_steps * self._env.step_dt
        if self.cfg.random_initialize:
            # random start time
            if self.cfg.num_steps_to_use > 0:
                self.motion_fetch_time[env_ids, 0] = self.motion_data_term.sample_times(self.motion_ids[env_ids], truncate_time_end=truncate_time)
            else:
                self.motion_fetch_time[env_ids, 0] = self.motion_data_term.sample_times(self.motion_ids[env_ids], truncate_time_start=truncate_time)
        else:
            # start from beginning
            self.motion_fetch_time[env_ids, 0] = 0.0
        if self.num_steps > 1:
            self.motion_fetch_time[env_ids, 1:] = self.motion_fetch_time[env_ids, 0:1] + self.step_indices[1:].float() * self._env.step_dt

        self._fetch_motion_data(env_ids)

    def update(self, dt: float):
        if self.cfg.random_fetch:
            if self.cfg.num_steps_to_use > 0:
                self.motion_fetch_time[:, 0] = self.motion_data_term.sample_times(self.motion_ids, truncate_time_end=self.num_steps * dt)
            else: 
                self.motion_fetch_time[:, 0] = self.motion_data_term.sample_times(self.motion_ids, truncate_time_start=self.num_steps * dt)
            
            if self.num_steps > 1:
                self.motion_fetch_time[:, 1:] = self.motion_fetch_time[:, 0:1] + self.step_indices[1:].float() * dt
            
        self._fetch_motion_data()
        
        # Advance time
        if not self.cfg.random_fetch:
            self.motion_fetch_time += dt # only effective when random_fetch is False
            
        if self.cfg.enable_visualization:
            self._visualize()
            
    def _fetch_motion_data(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._env.device)
        
        # Flatten for batch processing
        num_queries = len(env_ids) * self.num_steps
        motion_times_flat = self.motion_fetch_time[env_ids].reshape(-1)
        motion_ids_flat = self.motion_ids[env_ids].repeat_interleave(self.num_steps)
        
        # Sanity check
        assert motion_times_flat.shape[0] == num_queries
        assert motion_ids_flat.shape[0] == num_queries
        
        # Fetch motion data
        motion_data_dict = self.motion_data_term.get_motion_state(
            motion_ids=motion_ids_flat,
            motion_times=motion_times_flat
        )
        
        # Reshape and store in buffers
        for component in self.cfg.motion_data_components:
            if component in motion_data_dict:
                buffer_name = f"{component}_buffer"
                data = motion_data_dict[component]
                # Reshape: (num_envs * num_steps, *) -> (num_envs, num_steps, *)
                data_reshaped = data.view(len(env_ids), self.num_steps, *data.shape[1:])
                getattr(self, buffer_name)[env_ids, :, :] = data_reshaped
            
    def _visualize(self):
        # try to get the 'robot_anim' articulation
        robot_anim: Articulation = self._env.scene["robot_anim"]
        if robot_anim is None:
            print("[WARNING] AnimationTerm visualization: 'robot_anim' not found in the scene.")
            return
        
        # check if some necessary buffers are available
        if not hasattr(self, "root_pos_w_buffer") or not hasattr(self, "root_quat_buffer") or not hasattr(self, "dof_pos_buffer"):
            print("[WARNING] AnimationTerm visualization: 'root_pos_w_buffer' or 'root_quat_buffer' or 'dof_pos_buffer' not found.")
            return
        
        root_pos_w = self.root_pos_w_buffer[:, 0, :]  # (num_envs, 3)
        root_quat = self.root_quat_buffer[:, 0, :]    # (num_envs, 4)
        dof_pos = self.dof_pos_buffer[:, 0, :]        # (num_envs, num_dofs)
        
        root_states = robot_anim.data.default_root_state.clone()
        root_states[:, :3] = root_pos_w + self._env.scene.env_origins[:, :3] + self.vis_root_offset
        root_states[:, 3:7] = root_quat
        root_states[:, 7:10] = 0.0  # zero linear velocity
        root_states[:, 10:13] = 0.0  # zero angular velocity
        robot_anim.write_root_state_to_sim(root_states)
        
        joint_pos = robot_anim.data.default_joint_pos.clone()
        joint_pos[:, :] = dof_pos
        joint_vel = torch.zeros_like(robot_anim.data.default_joint_vel)
        robot_anim.write_joint_state_to_sim(joint_pos, joint_vel)
            
        key_body_pos_b = self.key_body_pos_b_buffer[:, 0, :, :]  # (num_envs, num_key_bodies, 3)
        num_key_bodies = key_body_pos_b.shape[1]
        key_body_pos_w = root_states[:, :3].unsqueeze(1) + math_utils.quat_apply(
            root_quat.unsqueeze(1).expand(-1, num_key_bodies, -1),
            key_body_pos_b
        )
        self.key_body_marker.visualize(
            translations=key_body_pos_w.reshape(-1, 3)
        )
            
    # Some helper functions
    def get_root_pos_w(self, env_ids: Sequence[int] = None) -> torch.Tensor:
        if not hasattr(self, "root_pos_w_buffer"):
            raise AttributeError("root_pos_w_buffer not found in AnimationTerm.")
        if env_ids is None:
            return self.root_pos_w_buffer
        return self.root_pos_w_buffer[env_ids, :, :]
    
    def get_root_quat(self, env_ids: Sequence[int] = None) -> torch.Tensor:
        if not hasattr(self, "root_quat_buffer"):
            raise AttributeError("root_quat_buffer not found in AnimationTerm.")
        if env_ids is None:
            return self.root_quat_buffer
        return self.root_quat_buffer[env_ids, :, :]
    
    def get_dof_pos(self, env_ids: Sequence[int] = None) -> torch.Tensor:
        if not hasattr(self, "dof_pos_buffer"):
            raise AttributeError("dof_pos_buffer not found in AnimationTerm.")
        if env_ids is None:
            return self.dof_pos_buffer
        return self.dof_pos_buffer[env_ids, :, :]
    
    def get_dof_vel(self, env_ids: Sequence[int] = None) -> torch.Tensor:
        if not hasattr(self, "dof_vel_buffer"):
            raise AttributeError("dof_vel_buffer not found in AnimationTerm.")
        if env_ids is None:
            return self.dof_vel_buffer
        return self.dof_vel_buffer[env_ids, :, :]
    
    def get_key_body_pos_b(self, env_ids: Sequence[int] = None) -> torch.Tensor:
        if not hasattr(self, "key_body_pos_b_buffer"):
            raise AttributeError("key_body_pos_b_buffer not found in AnimationTerm.")
        if env_ids is None:
            return self.key_body_pos_b_buffer
        return self.key_body_pos_b_buffer[env_ids, :, :]
    
    def get_root_vel_w(self, env_ids: Sequence[int] = None) -> torch.Tensor:
        if not hasattr(self, "root_vel_w_buffer"):
            raise AttributeError("root_vel_w_buffer not found in AnimationTerm.")
        if env_ids is None:
            return self.root_vel_w_buffer
        return self.root_vel_w_buffer[env_ids, :, :]
    
    def get_root_ang_vel_w(self, env_ids: Sequence[int] = None) -> torch.Tensor:
        if not hasattr(self, "root_ang_vel_w_buffer"):
            raise AttributeError("root_ang_vel_w_buffer not found in AnimationTerm.")
        if env_ids is None:
            return self.root_ang_vel_w_buffer
        return self.root_ang_vel_w_buffer[env_ids, :, :]
            
            
class AnimationManager(ManagerBase):
    
    def __init__(self, cfg: object, env: ManagerBasedAnimationEnv):
        if cfg is None:
            raise ValueError("AnimationManager configuration is required.")
        
        self._terms: dict[str, AnimationTerm] = {}
        self._term_cfgs: dict[str, AnimationTermCfg] = {}
        
        super().__init__(cfg, env)
        
    def __str__(self) -> str:
        """Returns: A string representation for animation data manager."""
        msg = f"<AnimationManager> contains {len(self._terms)} active terms.\n"
        
        # create table for term information
        table = PrettyTable()
        table.title = "Animation Manager Terms"
        table.field_names = ["Index", "Term Name", "Motion Data Term", "Num Steps to Use", "Random Init", "Random Fetch"]
        for index, (term_name, term_cfg) in enumerate(self._term_cfgs.items()):
            table.add_row([
                index, 
                term_name,
                term_cfg.motion_data_term,
                term_cfg.num_steps_to_use,
                term_cfg.random_initialize,
                term_cfg.random_fetch
            ])
        msg += table.get_string()
        msg += "\n"
        
        return msg

    def update(self, dt: float):
        """Update all animation terms.

        Args:
            dt: The time step to update the animation terms.
        """
        for term in self._terms.values():
            term.update(dt)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            return {}
        # reset all terms
        for term in self._terms.values():
            term.reset(env_ids)
        return {}
        
    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active command terms."""
        return list(self._terms.keys())
    
    def get_term(self, term_name: str) -> AnimationTerm:
        """Get the animation data term by name."""
        if term_name not in self._terms:
            raise KeyError(f"Animation data term '{term_name}' not found.")
        return self._terms[term_name]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, AnimationTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type AnimationTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = AnimationTerm(term_cfg, self._env)
            # add class to dict
            self._terms[term_name] = term
            self._term_cfgs[term_name] = term_cfg





