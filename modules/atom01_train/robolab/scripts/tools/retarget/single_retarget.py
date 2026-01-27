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

"""
This module provides functionality to convert motion data from GMR format to Legged Lab format.

Ref: 
    - https://github.com/xbpeng/MimicKit/blob/main/tools/gmr_to_mimickit/gmr_to_mimickit.py
    - https://github.com/HybridRobotics/whole_body_tracking/blob/main/scripts/csv_to_npz.py

What does this script do?
    - Reorder the dof data from GMR (in mujoco order) to Legged Lab (in Isaac Lab order).
    - Add loop mode to the motion data.

Usage: 
    Command line:
        python scripts/tools/retarget/gmr_to_lab.py \
            --robot g1 \
            --input_file <path_to_gmr_file> \
            --output_file <path_to_output_file> \
            --config_file <path_to_config_file> \
            [--loop {wrap,clamp}] \
            [--frame_range START END]
    
    Required arguments:
        --robot: Robot name to be used (default: g1).
        --input_file: Path to the input GMR motion file (pickle format).
        --output_file: Path to save the converted motion data (pickle format).
        --config_file: Path to the configuration file (yaml format).
                      Should contain: gmr_dof_names, lab_dof_names, lab_key_body_names
    
    Optional arguments:
        --loop {wrap,clamp}     Loop mode for the motion (default: clamp).
        --frame_range START END Frame range to extract.
                               If not provided, all frames will be processed.
                               Example: --frame_range 10 100
    
    AppLauncher arguments:
        --headless              Run without GUI.
        --device {cpu,cuda:0}   Device to use for simulation (default: cuda:0).
    
    Example:
        # Convert full motion with GUI
        python scripts/tools/retarget/gmr_to_lab.py \
            --robot g1 \
            --input_file data/gmr/walk.pkl \
            --output_file data/lab/walk.pkl \
            --config_file scripts/tools/retarget/g1_29dof.yaml
        
        # Convert specific frame range without GUI
        python scripts/tools/retarget/gmr_to_lab.py \
            --robot g1 \
            --input_file data/gmr/walk.pkl \
            --output_file data/lab/walk_clip.pkl \
            --config_file scripts/tools/retarget/g1_29dof.yaml \
            --frame_range 10 100 \
            --loop wrap \
            --headless

GMR Format:
    The input GMR format should be a pickle file containing a dictionary with keys:
    - 'fps': Frame rate (int)
    - 'root_pos': Root position array, shape (num_frames, 3)
    - 'root_rot': Root rotation quaternions, shape (num_frames, 4), format (x, y, z, w)
    - 'dof_pos': Degrees of freedom positions, shape (num_frames, num_dofs)
    - 'local_body_pos': Currently unused (can be None)
    - 'link_body_list': Currently unused (can be None)

"""



import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualization of retargeted data.")
parser.add_argument(
    "--robot", 
    type=str,
    default="atom01",
    choices=["atom01"],
    help="The robot name to be used.",
)
parser.add_argument(
    "--input_file",
    type=str,
    default="source/legged_lab/legged_lab/data/MotionData/ACCAD/mixed_walk_run_lab/xiaobu.pkl",
    help="Path to the input GMR motion file (pickle format).",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="source/legged_lab/legged_lab/data/MotionData/ACCAD/mixed_walk_run_lab/xiaobu.pkl",
    help="Path to save the converted motion data (pickle format).",
)
parser.add_argument(
    "--config_file",
    type=str,
    default="scripts/tools/config/atom01.yaml",
    help="Path to YAML config containing gmr_dof_names, lab_dof_names",
)
parser.add_argument(
    "--loop",
    type=str,
    choices=["wrap", "clamp"],
    default="clamp",
    help="Loop mode for the motion (default: clamp).",
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). If not provided, all frames will be"
        " loaded."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
import numpy as np
import pickle
import argparse
import enum
import yaml
import torch
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

##
# Pre-defined configs
##
if args_cli.robot == "atom01":
    from legged_lab.assets.roboparty import ATOM01_CFG as ROBOT_CFG
else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")

# Import functions and classes from gmr_to_lab.py
# Add the script directory to path to allow imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from gmr_to_lab import (
        LoopMode,
        extract_gmr_data,
        run_simulator,
        ReplayMotionsSceneCfg,
    )
except ImportError as e:
    print(f"Error importing from gmr_to_lab.py: {e}")
    print("Make sure gmr_to_lab.py is in the same directory as this script.")
    raise



if __name__ == "__main__":
    with open(args_cli.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    gmr_dof_names = config['gmr_dof_names']
    lab_dof_names = config['lab_dof_names']
    lab_key_body_names = config['lab_key_body_names']
    
    loop_mode = LoopMode.CLAMP if args_cli.loop == "clamp" else LoopMode.WRAP
    
    motion_data_dict = extract_gmr_data(
        gmr_file_path=args_cli.input_file,
        gmr_dof_names=gmr_dof_names,
        lab_dof_names=lab_dof_names,
        loop_mode=loop_mode,
        start_frame=args_cli.frame_range[0] if args_cli.frame_range else 0,
        end_frame=args_cli.frame_range[1] if args_cli.frame_range else -1,
    )
    
    fps = motion_data_dict['fps']
    dt = 1.0 / fps
    
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args_cli.device))
    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=1, 
        env_spacing=3.0,
        robot=ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")    
    )
    scene = InteractiveScene(scene_cfg)
    
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    
    sim.reset()
    print("Simulation starting ...")
    
    motion_data_dicts = [motion_data_dict]
    motion_data_dicts = run_simulator(simulation_app, sim, scene, motion_data_dicts, lab_key_body_names)
    
    motion_data_dict = motion_data_dicts[0]
    
    print("\n" + "="*60)
    print("💾 SAVING CONVERTED DATA")
    print("="*60)
    print(f"📁 Output File: {args_cli.output_file}")
    print(f"🧮 Number of Frames: {args_cli.frame_range[1] - args_cli.frame_range[0] if args_cli.frame_range else 'All'}")
    print(f"🔁 Loop Mode: {loop_mode.name}")
    print("="*60 + "\n")
    
    with open(args_cli.output_file, 'wb') as f:
        pickle.dump(motion_data_dict, f)
    print("✅ Data saved successfully.")
    print("="*60 + "\n")
    
    print("Closing simulation app...")
    simulation_app.close()
    print(f"✅ Simulation app closed.")