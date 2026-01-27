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

import numpy as np
import torch
from typing import Optional

import isaaclab.utils.math as math_utils

@torch.jit.script
def vel_forward_diff(data: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute the forward differences of the input data

    Args:
        data (torch.Tensor): The input data tensor of shape (N, dim).
        dt (float): The time step duration.
    """
    N = data.shape[0]
    if N < 2:
        raise RuntimeError(f"Input data has only {N} frames, cannot compute velocity.")
    vel = torch.zeros_like(data)
    vel[:-1] = (data[1:] - data[:-1]) / dt
    vel[-1] = vel[-2]  # use the last value as the same as the second last value
    return vel


@torch.jit.script
def ang_vel_from_quat_diff(quat: torch.Tensor, dt: float, in_frame:str = "body") -> torch.Tensor:
    """Compute the angular velocity from quaternion differences.

    Args:
        quat (torch.Tensor): The input quaternion tensor of shape (N, 4), 
                            representing the rotation from world to body frame.
        dt (float): The time step duration.
        in_frame (str): The frame in which the angular velocity is expressed, either "body" or "world".
    """
    if in_frame not in ["body", "world"]:
        raise ValueError(f"Invalid in_frame value: {in_frame}. Must be 'body' or 'world'.")
    
    N = quat.shape[0]
    if N < 2:
        raise RuntimeError(f"Input quaternion has only {N} frames, cannot compute angular velocity.")
    
    ang_vel = torch.zeros((N, 3), dtype=torch.float32, device=quat.device)
    for i in range(N-1):
        q1 = quat[i].unsqueeze(0)  # from world frame to body, shape (1, 4)
        q2 = quat[i + 1].unsqueeze(0)  # from world frame to body (at next time), shape (1, 4)

        diff_quat = math_utils.quat_mul(math_utils.quat_conjugate(q1), q2)
        diff_angle_axis = math_utils.axis_angle_from_quat(diff_quat)
        if in_frame == "world":
            diff_angle_axis = math_utils.quat_apply(q1, diff_angle_axis)
        ang_vel[i, :] = diff_angle_axis.squeeze() / dt  # convert to angular velocity

    ang_vel[-1, :] = ang_vel[-2, :]  # use the last value as the same as the second last value
    
    return ang_vel


def quat_slerp(
    q0: torch.Tensor,
    *,
    q1: Optional[torch.Tensor] = None,
    blend: Optional[torch.Tensor] = None,
    start: Optional[np.ndarray] = None,
    end: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Interpolation between consecutive rotations (Spherical Linear Interpolation).

    Args:
        q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
        q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
        blend: Interpolation coefficient between 0 (q0) and 1 (q1). Shape is (N,) or (N, M).
        start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
            the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
        end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
            the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

    Returns:
        Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
    """
    if start is not None and end is not None:
        return quat_slerp(q0=q0[start], q1=q0[end], blend=blend)
    if q0.ndim >= 2:
        blend = blend.unsqueeze(-1) # type: ignore
    if q0.ndim >= 3:
        blend = blend.unsqueeze(-1) # type: ignore

    qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
    cos_half_theta = (
        q0[..., qw] * q1[..., qw]
        + q0[..., qx] * q1[..., qx]
        + q0[..., qy] * q1[..., qy]
        + q0[..., qz] * q1[..., qz]
    )

    neg_mask = cos_half_theta < 0
    q1 = q1.clone() # type: ignore
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
    ratio_b = torch.sin(blend * half_theta) / sin_half_theta

    new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
    new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
    new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
    new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

    new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
    return new_q

@torch.jit.script
def linear_interpolate(x0: torch.Tensor, x1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Linear interpolate between two tensors.

    Args:
        x0 (torch.Tensor): shape (N, M)
        x1 (torch.Tensor): shape (N, M)
        blend (torch.Tensor): shape(N, 1)
    """
    return x0 * (1 - blend) + x1 * blend


@torch.jit.script
def calc_frame_blend(time:torch.Tensor, duration:torch.Tensor, num_frames:torch.Tensor, dt:torch.Tensor):

    phase = time / duration
    phase = torch.clamp(phase, min=0.0, max=1.0)
    
    frame_idx0 = (phase * (num_frames - 1).float()).long()
    frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
    blend = (time - frame_idx0.float() * dt) / dt
    
    return frame_idx0, frame_idx1, blend