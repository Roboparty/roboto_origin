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

import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from robolab.assets import ISAAC_DATA_DIR
import torch
import os
import cv2
import matplotlib.pyplot as plt # Import matplotlib

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg, headless=False):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
        headless: If True, run without GUI and save video.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos[-cfg.robot_config.num_actions:] = cfg.robot_config.default_pos
    mujoco.mj_step(model, data)
    
    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    os.environ['MUJOCO_GL'] = 'glfw'
    # 根据 headless 参数选择渲染模式
    if headless:
        renderer = mujoco.Renderer(model, width=1920, height=1080)
        # 设置视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 创建并配置相机
        cam = mujoco.MjvCamera()
        cam.distance = 4.0      # 增加距离以获得更好的视角
        cam.azimuth = 45.0     # 水平旋转角度
        cam.elevation = -20.0   # 垂直俯仰角度
        cam.lookat = [0, 0, 1]  # 观察点位置
        out = cv2.VideoWriter('simulation.mp4', fourcc, 1.0/cfg.sim_config.dt/cfg.sim_config.decimation, (1920, 1080))
    else:
        mode = 'window'
        viewer = mujoco_viewer.MujocoViewer(model, data, mode=mode, width=1920, height=1080)
        # 设置窗口模式下的相机参数
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat = [0, 0, 1]


    target_pos = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
    action = np.zeros((cfg.robot_config.num_actions), dtype=np.double)

    hist_obs = np.zeros((cfg.robot_config.frame_stack, cfg.robot_config.num_single_obs), dtype=np.double)
    hist_obs.fill(0.0)

    count_lowlevel = 0

    # --- Data collection lists for plotting (LOW FREQUENCY ONLY) ---
    time_data = []
    commanded_joint_pos_data = []
    actual_joint_pos_data = []
    tau = np.zeros((cfg.robot_config.num_actions), dtype=np.double)  # Initialize tau
    tau_data = []
    commanded_lin_vel_x_data = []
    commanded_lin_vel_y_data = []
    commanded_ang_vel_z_data = []
    actual_lin_vel_data = [] # Store [vx, vy] at low freq
    actual_ang_vel_data = [] # Store [wz] at low freq
    # -------------------------------------------------------------
    is_first_frame = True
    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.robot_config.num_actions:]
        dq = dq[-cfg.robot_config.num_actions:]

        # 1000hz -> 100hz/50hz
        if count_lowlevel % cfg.sim_config.decimation == 0:
            q_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
            dq_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
            q_ = q - cfg.robot_config.default_pos
            for i in range(len(cfg.robot_config.usd2urdf)):
                q_obs[i] = q_[cfg.robot_config.usd2urdf[i]]
                dq_obs[i] = dq[cfg.robot_config.usd2urdf[i]]

            obs = np.zeros([1, cfg.robot_config.num_single_obs], dtype=np.float32)
            
            obs[0, 0:3] = omega
            obs[0, 3:6] = gvec
            obs[0, 6] = cmd.vx 
            obs[0, 7] = cmd.vy 
            obs[0, 8] = cmd.dyaw 
            obs[0, 9:32] = q_obs
            obs[0, 32:55] = dq_obs
            obs[0, 55:78] = action

            if is_first_frame:
                hist_obs = np.tile(obs, (cfg.robot_config.frame_stack, 1))
                is_first_frame = False
            else:
                hist_obs = np.concatenate((hist_obs[1:], obs.reshape(1, -1)), axis=0)

            policy_input = hist_obs.reshape(1, -1).astype(np.float32)
            with torch.inference_mode():
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()

            target_q = action * cfg.robot_config.action_scale
            for i in range(len(cfg.robot_config.usd2urdf)):
                target_pos[cfg.robot_config.usd2urdf[i]] = target_q[i]
            target_pos = target_pos + cfg.robot_config.default_pos

            # --- Capture actual state at this low-frequency step ---
            # Note: q, v, omega were just computed by get_obs() for the current simulation step
            q_low_freq = q.copy()
            v_low_freq = v[:2].copy() # Capture x and y linear velocity
            omega_low_freq = omega[2].copy() # Capture z angular velocity
            # -----------------------------------------------------

            # --- Collect low-frequency data for plotting ---
            # Use the exact simulation time at this low-freq step
            time_data.append(step * cfg.sim_config.dt)
            commanded_joint_pos_data.append(target_pos.copy())
            actual_joint_pos_data.append(q_low_freq) # Use the captured actual joint pos
            tau_data.append(tau.copy())
            commanded_lin_vel_x_data.append(cmd.vx)
            commanded_lin_vel_y_data.append(cmd.vy)
            commanded_ang_vel_z_data.append(cmd.dyaw)
            actual_lin_vel_data.append(v_low_freq) # Use the captured actual lin vel
            actual_ang_vel_data.append(omega_low_freq) # Use the captured actual ang vel
            # ----------------------------------------------

            if headless:
                renderer.update_scene(data, camera=cam)
                img = renderer.render()  # 直接获取RGB图像
                out.write(img)
            else:
                viewer.render()
            
        target_vel = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_pos, q, cfg.robot_config.kps,
                        target_vel, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau
        mujoco.mj_step(model, data)

        count_lowlevel += 1

    if headless:
        out.release()
    else:
        viewer.close()

     # --- Plotting Section (Using only low-frequency data) ---

    print("Simulation finished. Generating plots...")

    # Convert collected data to numpy arrays
    time_data = np.array(time_data)
    commanded_joint_pos_data = np.array(commanded_joint_pos_data)
    actual_joint_pos_data = np.array(actual_joint_pos_data)
    tau_data = np.array(tau_data)
    commanded_lin_vel_x_data = np.array(commanded_lin_vel_x_data)
    commanded_lin_vel_y_data = np.array(commanded_lin_vel_y_data)
    commanded_ang_vel_z_data = np.array(commanded_ang_vel_z_data)
    actual_lin_vel_data = np.array(actual_lin_vel_data)
    actual_ang_vel_data = np.array(actual_ang_vel_data)


    # Plot 1: Commanded vs Actual Joint Positions
    num_joints = cfg.robot_config.num_actions
    n_cols = 4 # Or adjust based on num_joints
    n_rows = (num_joints + n_cols - 1) // n_cols

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes1 = axes1.flatten()

    joint_names = [f'Joint {i+1}' for i in range(num_joints)] # Generic names (consider using specific robot joint names if available)

    for i in range(num_joints):
        ax = axes1[i]
        # Plotting low-frequency commanded and actual joint positions
        ax.plot(time_data, commanded_joint_pos_data[:, i], label='Commanded', linestyle='--')
        ax.plot(time_data, actual_joint_pos_data[:, i], label='Actual')
        ax.set_title(joint_names[i])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [rad]")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for i in range(num_joints, len(axes1)):
        fig1.delaxes(axes1[i])

    fig1.suptitle("Commanded vs Actual Joint Positions", fontsize=16)
    plt.tight_layout()


    # Plot 2: Commanded vs Actual Base Velocities
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Linear Velocity X
    # Plotting low-frequency commanded and actual velocities
    axes2[0].plot(time_data, commanded_lin_vel_x_data, label='Commanded Vx', linestyle='--')
    axes2[0].plot(time_data, actual_lin_vel_data[:, 0], label='Actual Vx')
    axes2[0].set_title("Base Linear Velocity X")
    axes2[0].set_xlabel("Time [s]")
    axes2[0].set_ylabel("Velocity [m/s]")
    axes2[0].legend()
    axes2[0].grid(True)

    # Linear Velocity Y
    axes2[1].plot(time_data, commanded_lin_vel_y_data, label='Commanded Vy', linestyle='--')
    axes2[1].plot(time_data, actual_lin_vel_data[:, 1], label='Actual Vy')
    axes2[1].set_title("Base Linear Velocity Y")
    axes2[1].set_xlabel("Time [s]")
    axes2[1].set_ylabel("Velocity [m/s]")
    axes2[1].legend()
    axes2[1].grid(True)

    # Angular Velocity Z
    axes2[2].plot(time_data, commanded_ang_vel_z_data, label='Commanded Dyaw', linestyle='--')
    axes2[2].plot(time_data, actual_ang_vel_data, label='Actual Dyaw') # actual_ang_vel_data is already 1D
    axes2[2].set_title("Base Angular Velocity Z (Dyaw)")
    axes2[2].set_xlabel("Time [s]")
    axes2[2].set_ylabel("Angular Velocity [rad/s]")
    axes2[2].legend()
    axes2[2].grid(True)

    fig2.suptitle("Commanded vs Actual Base Velocities", fontsize=16)
    plt.tight_layout()

    # plt.show()
    fig1.savefig("joint_positions.png")
    fig2.savefig("base_velocities.png")

    print("Plots finished.")
    # --- End Plotting Section ---

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    parser.add_argument('--headless', action='store_true',
                      help='Run without GUI and save video')
    args = parser.parse_args()

    class Sim2simCfg():

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/atom01/mjcf/atom01_terrain.xml'
            else:
                mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/atom01/mjcf/atom01.xml'
            sim_duration = 10.0
            dt = 0.001
            decimation = 20

        class robot_config:
            kps = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 150, 40, 40, 40, 30, 20, 40, 40, 40, 30, 20], dtype=np.double)
            kds = np.array([3.3, 3.3, 3.3, 5.0, 2.0, 2.0, 3.3, 3.3, 3.3, 5.0, 2.0, 2.0, 5.0, 2.0, 2.0, 2.0, 1.5, 1.0, 2.0, 2.0, 2.0, 1.5, 1.0], dtype=np.double)
            default_pos = np.array([0, 0, -0.1, 0.3, -0.2, 0, 0, 0, -0.1, 0.3, -0.2, 0, 0, 0.18, 0.06, 0, 0.78, 0, 0.18, -0.06, 0, 0.78, 0], dtype=np.double)
            tau_limit = 200. * np.ones(23, dtype=np.double)
            frame_stack = 10
            num_single_obs = 78
            num_observations = 780
            num_actions = 23
            action_scale = 0.25
            # 'left_thigh_yaw_joint', 'right_thigh_yaw_joint', 'torso_joint', 'left_thigh_roll_joint', 'right_thigh_roll_joint', 'left_arm_pitch_joint', 'right_arm_pitch_joint', 'left_thigh_pitch_joint', 'right_thigh_pitch_joint', 'left_arm_roll_joint', 'right_arm_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_arm_yaw_joint', 'right_arm_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_pitch_joint', 'right_elbow_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_elbow_yaw_joint', 'right_elbow_yaw_joint'
            usd2urdf = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg(), args.headless)
