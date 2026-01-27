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

import mujoco, mujoco_viewer
from tqdm import tqdm
from robolab.assets import ISAAC_DATA_DIR
from pynput import keyboard
import os
import csv
import time

class cmd:
    reset_requested = False

def on_press(key):
    try:
        if key.char == '0':
            cmd.reset_requested = True
            print('Reset')
    except AttributeError:
        pass

def on_release(key):
    pass

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener

def load_qpos_from_csv(csv_file):
    qpos_list = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            q = [float(val) for val in row]
            qpos=q.copy()
            qpos[3:7]=[q[6],q[3],q[4],q[5]]
            qpos_list.append(qpos)
    return qpos_list

def run_mujoco(cfg,loop=False,motion_file=None):
    print("=" * 60)
    keyboard_listener = start_keyboard_listener()

    qpos_list=load_qpos_from_csv(motion_file)

    num_frames = len(qpos_list)
    print(len(qpos_list))
    def frame_idx(t):
        if loop and num_frames > 0:
            return t % num_frames
        return t if t < num_frames else num_frames - 1

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos=qpos_list[0]
    mujoco.mj_fwdPosition(model, data)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    os.environ['MUJOCO_GL'] = 'glfw'
    
    mode = 'window'
    viewer = mujoco_viewer.MujocoViewer(model, data,width=1920, height=1080)
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0
    viewer.cam.lookat = [0, 0, 1]
    viewer.render()

    count_lowlevel = 0
    start_time = time.time()
    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        if cmd.reset_requested:
            print('Performing reset: restoring qpos/qvel and zeroing commands')
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            # clear commands and history
            cmd.reset()
            mujoco.mj_forward(model, data)
            cmd.reset_requested = False

        idx=frame_idx(count_lowlevel)
        data.qpos[:] = qpos_list[idx]
        mujoco.mj_fwdPosition(model, data)
        viewer.render()

        elapsed_real_time = time.time() - start_time
        target_sim_time = (step + 1) * cfg.sim_config.dt
        if elapsed_real_time < target_sim_time:
            time.sleep(target_sim_time - elapsed_real_time)

        count_lowlevel += 1

    else:
        viewer.close()
    keyboard_listener.stop()
    print("Simulation finished.")

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--motion_file',type=str,help='path to motion file(csv)')
    parser.add_argument('--loop',action="store_true",help='loop the policy')
    args = parser.parse_args()

    class Sim2simCfg():

        class sim_config:
            mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/atom01/mjcf/atom01.xml'
            sim_duration = 1000.0
            dt = 0.02
            
    run_mujoco(Sim2simCfg(),args.loop,args.motion_file)
