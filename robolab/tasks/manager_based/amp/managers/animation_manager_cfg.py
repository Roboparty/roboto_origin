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

from dataclasses import MISSING

from isaaclab.utils import configclass

@configclass
class AnimationTermCfg:
    """"Configuration for an animation"""
    
    motion_data_term: str = MISSING
    """The motion data term to use for this animation term."""
    
    motion_data_components: list[str] = MISSING
    """The components of the motion data to use for this animation term."""
    
    num_steps_to_use: int = 1
    """Number of steps of motion data to extract from the motion data term.
        If positive, extracts current and future steps.
        If negative, extracts current and past steps.
        1 and -1 both extract only the current step.
        0 is invalid.
    """
    
    random_initialize: bool = False
    """Whether to randomly initialize the starting point in the motion data term."""
    
    random_fetch: bool = False
    """Whether to randomly fetch the motion data at each step."""
    
    enable_visualization: bool = True
    """Whether to enable visualization for this animation term."""
    
    vis_root_offset: list[float] = (0.0, 0.0, 0.0)
    """Root position offset for visualization (x, y, z)."""