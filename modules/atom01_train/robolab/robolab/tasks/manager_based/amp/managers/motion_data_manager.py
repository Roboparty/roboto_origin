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

import os
import numpy as np
import enum
import joblib
import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation

from isaaclab.managers import ManagerBase, ManagerTermBase
from .motion_data_term_cfg import MotionDataTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from robolab.utils.math import vel_forward_diff, ang_vel_from_quat_diff, quat_slerp, linear_interpolate, calc_frame_blend


class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1


class MotionDataTerm(ManagerTermBase):
    
    cfg: MotionDataTermCfg
    _env: ManagerBasedEnv

    def __init__(self, cfg: MotionDataTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        assert os.path.exists(cfg.motion_data_dir), \
            f"Motion data directory {cfg.motion_data_dir} does not exist."
            
        self._load_motion_data()
        
    def _load_motion_data(self):
        # list the motion data files in the directory
        motion_files = [f for f in os.listdir(self.cfg.motion_data_dir) if f.endswith('.pkl')]
        if not motion_files:
            raise ValueError(f"No motion data files with .pkl extension found in {self.cfg.motion_data_dir}.")
        
        self.motion_weights_dict = self.cfg.motion_data_weights

        self.motion_durations = []
        self.motion_fps = []
        self.motion_dt = []
        self.motion_num_frames = []
        self.motion_weights = []
        self.motion_loop_modes = []
        
        self.root_pos_w = []
        self.root_quat = []
        self.root_vel_w = []
        self.root_ang_vel_w = []
        self.dof_pos = []
        self.dof_vel = []
        self.key_body_pos_w = []

        # only load the motion data files that are in the motion weights dict
        for motion_name, motion_weight in self.motion_weights_dict.items():
            # check if the motion file name is valid
            motion_file = f"{motion_name}.pkl"
            if motion_file not in motion_files:
                raise ValueError(f"Motion name {motion_name} defined in motion weights not found in motion data directory {self.cfg.motion_data_dir}. Available files: {motion_files}")

            # load the motion data file
            motion_path = os.path.join(self.cfg.motion_data_dir, motion_file)
            print(f"[Motion Data Manager] Loading motion data from {motion_path}...")
            motion_raw_data = joblib.load(motion_path)
            if not isinstance(motion_raw_data, dict):
                raise ValueError(f"Motion data file {motion_file} does not contain a valid dictionary.")
            
            # Some info about the motion
            fps = motion_raw_data["fps"]
            dt = 1.0 / fps
            num_frames = len(motion_raw_data["root_pos"])
            if num_frames < 2:
                raise ValueError(f"[MotionLoader] Motion has only {num_frames} frames, cannot compute velocity.")
            duration = dt * (num_frames - 1)
            loop_mode = motion_raw_data["loop_mode"]
            
            self.motion_durations.append(duration)
            self.motion_fps.append(fps)
            self.motion_dt.append(dt)
            self.motion_num_frames.append(num_frames)
            self.motion_loop_modes.append(loop_mode)
            self.motion_weights.append(motion_weight)
            
            # Get the motion data
            
            # root position in world frame, shape (num_frames, 3)
            root_pos_w = torch.from_numpy(motion_raw_data["root_pos"]).to(self.device).float()
            root_pos_w.requires_grad_(False)
            # root rotation (quaternion) from world frame to body frame, shape (num_frames, 4), in (w, x, y, z) format
            root_quat = torch.from_numpy(motion_raw_data["root_rot"]).to(self.device).float()
            root_quat.requires_grad_(False)
            
            # root velocity in world frame, shape (num_frames, 3)
            root_vel_w = vel_forward_diff(root_pos_w, dt)
            root_vel_w.requires_grad_(False)
            
            # root angular velocity in world frame, shape (num_frames, 3)
            root_ang_vel_w = ang_vel_from_quat_diff(root_quat, dt, in_frame="world")
            root_ang_vel_w.requires_grad_(False)
            
            # dof position, shape (num_frames, num_dofs)
            dof_pos = torch.from_numpy(motion_raw_data["dof_pos"]).to(self.device).float()
            dof_pos.requires_grad_(False)
            
            # dof velocity, shape (num_frames, num_dofs)
            dof_vel = vel_forward_diff(dof_pos, dt)
            dof_vel.requires_grad_(False)
            
            # key body position in world frame, shape (num_frames, num_key_bodies, 3)
            key_body_pos_w = torch.from_numpy(motion_raw_data["key_body_pos"]).to(self.device).float()
            key_body_pos_w.requires_grad_(False)
            
            self.root_pos_w.append(root_pos_w)
            self.root_quat.append(root_quat)
            self.root_vel_w.append(root_vel_w)
            self.root_ang_vel_w.append(root_ang_vel_w)
            self.dof_pos.append(dof_pos)
            self.dof_vel.append(dof_vel)
            self.key_body_pos_w.append(key_body_pos_w)
        
        self.motion_fps = torch.tensor(self.motion_fps, dtype=torch.float32, device=self.device)
        self.motion_dt = torch.tensor(self.motion_dt, dtype=torch.float32, device=self.device)
        self.motion_durations = torch.tensor(self.motion_durations, dtype=torch.float32, device=self.device)
        self.motion_num_frames = torch.tensor(self.motion_num_frames, dtype=torch.int32, device=self.device)
        self.motion_loop_modes = torch.tensor(self.motion_loop_modes, dtype=torch.int32, device=self.device)
        # Get the normalized motion weights
        self.motion_weights = torch.tensor(self.motion_weights, dtype=torch.float32, device=self.device)
        self.motion_weights = self.motion_weights / torch.sum(self.motion_weights)
        
        # Some other information
        self.num_dofs = self.dof_pos[0].shape[1]
        self.num_key_bodies = self.key_body_pos_w[0].shape[1]
        
        # Concatenate all motion data along the first dimension
        self.root_pos_w = torch.cat(self.root_pos_w, dim=0)
        self.root_quat = torch.cat(self.root_quat, dim=0)
        self.root_vel_w = torch.cat(self.root_vel_w, dim=0)
        self.root_ang_vel_w = torch.cat(self.root_ang_vel_w, dim=0)
        self.dof_pos = torch.cat(self.dof_pos, dim=0)
        self.dof_vel = torch.cat(self.dof_vel, dim=0)
        self.key_body_pos_w = torch.cat(self.key_body_pos_w, dim=0)
        
        num_motions = self.get_num_motions()
        self.motion_ids = torch.arange(num_motions, dtype=torch.long, device=self.device)
        
        lengths_shifted = self.motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.motion_start_indices = torch.cumsum(lengths_shifted, dim=0)
        
        return
         
    # Some helper functions
    
    def get_num_motions(self) -> int:
        """Get the number of motions loaded."""
        return self.motion_num_frames.shape[0]
    
    def get_total_duration(self) -> float:
        """Get the total duration of all motions."""
        return torch.sum(self.motion_durations).item()

    def get_motion_durations(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Get the duration of a specific motion.

        Args:
            motion_id (torch.Tensor): A tensor of motion IDs for which to get the duration.

        Returns:
            float: The duration of the motion in seconds.
        """
        return self.motion_durations[motion_ids]
        
    def get_motion_loop_modes(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Get the loop mode of a specific motion.

        Args:
            motion_id (torch.Tensor): A tensor of motion IDs for which to get the loop mode.

        Returns:
            int: The loop mode of the motion.
        """
        return self.motion_loop_modes[motion_ids]
        
    def sample_motions(self, n: int) -> torch.Tensor:
        """Sample a batch of motion IDs.

        Args:
            n (int): The number of motion IDs to sample.

        Returns:
            torch.Tensor: A tensor of sampled motion IDs, shape (n,).
        """
        motion_ids = torch.multinomial(self.motion_weights, num_samples=n, replacement=True)
        return motion_ids
        
    def sample_times(self, motion_ids: torch.Tensor, truncate_time_start: float = None, truncate_time_end: float = None):
        """Sample time within the duration of the given motions.
        
        Args:
            motion_ids (torch.Tensor): A tensor of motion IDs, shape (batch_size,).
            truncate_time_start (float | None): If provided, the sampled time will be truncated
                from the start, i.e., sampled in [truncate_time_start, duration]. Default is None.
            truncate_time_end (float | None): If provided, the sampled time will be truncated
                from the end, i.e., sampled in [0, duration - truncate_time_end]. Default is None.
                
        Returns:
            torch.Tensor: A tensor of sampled times, shape (batch_size,).
        """
        motion_durations = self.motion_durations[motion_ids]
        
        # Calculate valid time range
        time_start = torch.zeros_like(motion_durations)
        time_end = motion_durations.clone()
        
        if truncate_time_start is not None:
            assert truncate_time_start >= 0, f"[MotionLoader] truncate_time_start must be non-negative, but got {truncate_time_start}."
            time_start = torch.clamp(time_start + truncate_time_start, min=0.0, max=motion_durations)
        
        if truncate_time_end is not None:
            assert truncate_time_end >= 0, f"[MotionLoader] truncate_time_end must be non-negative, but got {truncate_time_end}."
            time_end = torch.clamp(time_end - truncate_time_end, min=0.0)
        
        # Check if valid range exists
        valid_range = time_end - time_start
        if torch.any(valid_range <= 0.0):
            print("[Warning] Some motions have invalid time range after truncation (start >= end).")
            valid_range = torch.clamp(valid_range, min=1e-6)  # Prevent division by zero
        
        # Sample time within the valid range
        phase = torch.rand(motion_ids.shape, device=self.device)
        sample_times = time_start + phase * valid_range
        
        return sample_times
        
    def calc_motion_phase(self, motion_ids, times):
        motion_durations = self.motion_durations[motion_ids]
        loop_modes = self.motion_loop_modes[motion_ids]
        phase = calc_phase(times, motion_durations, loop_modes)
        return phase
    
    def _calc_frame_blend(self, motion_ids: torch.Tensor, times: torch.Tensor):
        num_frames = self.motion_num_frames[motion_ids]
        dt = self.motion_dt[motion_ids]
        motion_start_indices = self.motion_start_indices[motion_ids]
        
        phase = self.calc_motion_phase(motion_ids, times)
        
        frame_idx0 = (phase * (num_frames - 1).float()).long()
        frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1).float() - frame_idx0.float()
        
        frame_idx0 = frame_idx0 + motion_start_indices
        frame_idx1 = frame_idx1 + motion_start_indices
        
        return frame_idx0, frame_idx1, blend
        
    
    # def _allocate_temp_tensors(self, n):
    #     """Allocate temporary tensors for motion state computation."""
    #     root_pos_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_pos_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_quat_0 = torch.empty([n, 4], dtype=torch.float32, device=self.device)
    #     root_quat_1 = torch.empty([n, 4], dtype=torch.float32, device=self.device)
    #     root_vel_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_vel_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_ang_vel_w_0 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     root_ang_vel_w_1 = torch.empty([n, 3], dtype=torch.float32, device=self.device)
    #     dof_pos_0 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     dof_pos_1 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     dof_vel_0 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     dof_vel_1 = torch.empty([n, self.num_dofs], dtype=torch.float32, device=self.device)
    #     key_body_pos_w_0 = torch.empty([n, self.num_key_bodies, 3], dtype=torch.float32, device=self.device)
    #     key_body_pos_w_1 = torch.empty([n, self.num_key_bodies, 3], dtype=torch.float32, device=self.device)

    #     return (root_pos_w_0, root_pos_w_1,
    #             root_quat_0, root_quat_1,
    #             root_vel_w_0, root_vel_w_1,
    #             root_ang_vel_w_0, root_ang_vel_w_1,
    #             dof_pos_0, dof_pos_1,
    #             dof_vel_0, dof_vel_1,
    #             key_body_pos_w_0, key_body_pos_w_1)
    
    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> dict[str, torch.Tensor]:

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
    
        root_pos_w_0 = self.root_pos_w[frame_idx0]
        root_pos_w_1 = self.root_pos_w[frame_idx1]
        root_quat_0 = self.root_quat[frame_idx0]
        root_quat_1 = self.root_quat[frame_idx1]
        root_vel_w_0 = self.root_vel_w[frame_idx0]
        root_vel_w_1 = self.root_vel_w[frame_idx1]
        root_ang_vel_w_0 = self.root_ang_vel_w[frame_idx0]
        root_ang_vel_w_1 = self.root_ang_vel_w[frame_idx1]
        dof_pos_0 = self.dof_pos[frame_idx0]
        dof_pos_1 = self.dof_pos[frame_idx1]
        dof_vel_0 = self.dof_vel[frame_idx0]
        dof_vel_1 = self.dof_vel[frame_idx1]
        key_body_pos_w_0 = self.key_body_pos_w[frame_idx0]
        key_body_pos_w_1 = self.key_body_pos_w[frame_idx1]
        
        # interpolate the values

        root_quat = quat_slerp(q0=root_quat_0, q1=root_quat_1, blend=blend)

        blend = blend.unsqueeze(-1)  # make it (n, 1) for broadcasting
        root_pos_w = torch.lerp(root_pos_w_0, root_pos_w_1, blend)
        root_vel_w = torch.lerp(root_vel_w_0, root_vel_w_1, blend)
        root_vel_b = math_utils.quat_apply_inverse(root_quat, root_vel_w)
        root_ang_vel_w = torch.lerp(root_ang_vel_w_0, root_ang_vel_w_1, blend)
        root_ang_vel_b = math_utils.quat_apply_inverse(root_quat, root_ang_vel_w)
        dof_pos = torch.lerp(dof_pos_0, dof_pos_1, blend)
        dof_vel = torch.lerp(dof_vel_0, dof_vel_1, blend)
        key_body_pos_w = torch.lerp(key_body_pos_w_0, key_body_pos_w_1, blend.unsqueeze(1))
        key_body_pos_b = math_utils.quat_apply_inverse(
            root_quat.unsqueeze(1).expand(-1, self.num_key_bodies, -1),
            key_body_pos_w - root_pos_w.unsqueeze(1)
        )

        return {
            "root_pos_w": root_pos_w,
            "root_quat": root_quat,
            "root_vel_w": root_vel_w,
            "root_vel_b": root_vel_b,
            "root_ang_vel_w": root_ang_vel_w,
            "root_ang_vel_b": root_ang_vel_b,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "key_body_pos_b": key_body_pos_b,
        }
        
        
class MotionDataManager(ManagerBase):
    """Manager for motion data.
    
    This manager is responsible for loading and managing motion data terms.
    Each motion data term is responsible for managing a group of data.
    """
    
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        
        # check that cfg is not None
        if cfg is None:
            raise ValueError("MotionDataManager requires a valid configuration object.")
        
        self._terms: dict[str, MotionDataTerm] = {}
        self._term_cfgs: dict[str, MotionDataTermCfg] = {}
        
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for motion data manager."""
        msg = f"<MotionDataManager> contains {len(self._terms)} active terms.\n"
        
        # create table for term information
        table = PrettyTable()
        table.title = "Motion Data Manager Terms"
        table.field_names = ["Index", "Motion Dataset", "Total Duration"]
        # set alignment of table columns
        table.align["Motion Dataset"] = "l"
        table.align["Total Duration"] = "r"
        # add info on each term
        for index, (term_name, term) in enumerate(self._terms.items()):
            table.add_row([index, term_name, term.get_total_duration()])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg
    
    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active command terms."""
        return list(self._terms.keys())
    
    def get_term(self, term_name: str) -> MotionDataTerm:
        """Get the motion data term by name."""
        if term_name not in self._terms:
            raise KeyError(f"Motion data term '{term_name}' not found.")
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
            if not isinstance(term_cfg, MotionDataTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type MotionDataTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = MotionDataTerm(term_cfg, self._env)
            # add class to dict
            self._terms[term_name] = term
            self._term_cfgs[term_name] = term_cfg


@torch.jit.script
def calc_phase(times: torch.Tensor, motion_duration: torch.Tensor, loop_mode: torch.Tensor) -> torch.Tensor:
    phase = times / motion_duration
        
    loop_wrap_mask = (loop_mode == int(LoopMode.WRAP.value))
    phase_wrap = phase[loop_wrap_mask]
    phase_wrap = phase_wrap - torch.floor(phase_wrap)
    phase[loop_wrap_mask] = phase_wrap
        
    phase = torch.clip(phase, 0.0, 1.0)

    return phase

        
        