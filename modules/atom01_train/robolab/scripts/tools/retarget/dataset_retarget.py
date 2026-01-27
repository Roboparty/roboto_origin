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
Batch retargeting tool: convert multiple GMR motion files to Legged Lab format.

Behavior:
 - Reads all .pkl files from the input directory (sorted).
 - For each file, loads the GMR pickle and uses extract_gmr_data to convert it (entire motion).
 - Runs the simulator once with all motions (num_envs = number of motions) and collects key body positions.
 - Saves each converted motion dict to the output directory with the same filename.

Usage example:
    python scripts/tools/retarget/dataset_retarget.py \
        --robot g1 \
        --input_dir data/gmr/ \
        --output_dir data/lab/ \
        --config_file scripts/tools/retarget/config/g1_29dof.yaml \
        --loop clamp

This script intentionally does NOT support start/end frame clipping; it converts full motions.
"""

import argparse
import os
from pathlib import Path
import yaml
import pickle

from isaaclab.app import AppLauncher

# append AppLauncher cli args
parser = argparse.ArgumentParser(description="Batch retarget GMR -> Legged Lab (multiple files).")
parser.add_argument(
    "--robot",
    type=str,
    default="atom01", 
    help="Robot name to use (default: atom01)",
)
parser.add_argument(
    "--input_dir",
    type=str,
    default="source/legged_lab/legged_lab/data/MotionData/atom01_gmr",
    help="Directory containing input GMR .pkl files",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="source/legged_lab/legged_lab/data/MotionData/atom01_lab",
    help="Directory to write converted .pkl files",
)
parser.add_argument(
    "--config_file",
    type=str,
    default="scripts/tools/retarget/config/atom01.yaml",
    help="Path to YAML config containing gmr_dof_names, lab_dof_names, lab_key_body_names",
)
parser.add_argument(
    "--loop",
    type=str,
    choices=["wrap", "clamp"],
    default="clamp",
    help="Loop mode for motion (default: clamp)",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

"""Launch Omniverse App"""
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import sys
import numpy as np
import torch
import warnings
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

# load robot cfg as single_retarget does
if args_cli.robot == "atom01":
    from legged_lab.assets.roboparty import ATOM01_CFG as ROBOT_CFG
else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")

# import functions from gmr_to_lab (must be in same directory)
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
try:
    from gmr_to_lab import LoopMode, extract_gmr_data, run_simulator, ReplayMotionsSceneCfg
except ImportError as e:
    print(f"Error importing from gmr_to_lab.py: {e}")
    raise


def list_input_files(input_dir: str):
    p = Path(input_dir)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix == ".pkl"])
    return files


def main():
    # read config
    with open(args_cli.config_file, 'r') as f:
        config = yaml.safe_load(f)

    gmr_dof_names = config['gmr_dof_names']
    lab_dof_names = config['lab_dof_names']
    lab_key_body_names = config['lab_key_body_names']

    loop_mode = LoopMode.CLAMP if args_cli.loop == "clamp" else LoopMode.WRAP

    input_files = list_input_files(args_cli.input_dir)
    if len(input_files) == 0:
        print(f"No .pkl files found in input directory: {args_cli.input_dir}")
        return

    Path(args_cli.output_dir).mkdir(parents=True, exist_ok=True)

    # load and convert all gmr files (entire motion)
    motion_data_dicts = []
    input_names = []
    fps_values = []

    print(f"Found {len(input_files)} files to convert.")
    for p in input_files:
        print(f"Loading and converting: {p.name}")
        motion = extract_gmr_data(
            gmr_file_path=str(p),
            gmr_dof_names=gmr_dof_names,
            lab_dof_names=lab_dof_names,
            loop_mode=loop_mode,
            start_frame=0,
            end_frame=-1,
        )
        motion_data_dicts.append(motion)
        input_names.append(p.name)
        fps_values.append(motion['fps'])

    # check fps consistency
    if not all(f == fps_values[0] for f in fps_values):
        print(fps_values)
        warnings.warn("Motions have different fps. Using fps from first motion.")

    fps = fps_values[0]
    dt = 1.0 / fps

    # start simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args_cli.device))
    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=len(motion_data_dicts),
        env_spacing=3.0,
        robot=ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    scene = InteractiveScene(scene_cfg)

    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    sim.reset()
    print("Simulation starting ...")

    # run simulator with all motions
    motion_data_dicts = run_simulator(simulation_app, sim, scene, motion_data_dicts, lab_key_body_names)

    # save outputs
    print("Saving converted motions to output directory...")
    for name, motion in zip(input_names, motion_data_dicts):
        out_path = Path(args_cli.output_dir) / name
        with open(out_path, 'wb') as f:
            pickle.dump(motion, f)
        print(f"Saved: {out_path}")

    print("Closing simulation app...")
    simulation_app.close()
    print("Done.")


if __name__ == '__main__':
    main()
