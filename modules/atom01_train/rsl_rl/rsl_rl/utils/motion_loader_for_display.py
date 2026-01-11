# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

import glob
import json

import numpy as np
import torch


class AMPLoaderDisplay:
    JOINT_POS_SIZE = 26

    JOINT_VEL_SIZE = 26

    JOINT_POSE_START_IDX = 0
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    ROOT_STATES_NUM = 6
    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    def __init__(
        self,
        device,
        time_between_frames,
        motion_files=glob.glob("datasets/motion_amp_expert/*"),
    ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames

        # Values to store for each trajectory.
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_frame_durations = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split(".")[0])
            with open(motion_file) as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                self.trajectories_full.append(
                    torch.tensor(
                        motion_data[:, : AMPLoaderDisplay.JOINT_VEL_END_IDX], dtype=torch.float32, device=device
                    )
                )
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                print(f"traj_len:{traj_len}")
                self.trajectory_lens.append(traj_len)

            print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_lens = np.array(self.trajectory_lens)

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def slerp(self, frame1, frame2, blend):
        return (1.0 - blend) * frame1 + blend * frame2

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """
        joints0, joints1 = AMPLoaderDisplay.get_joint_pose(frame0), AMPLoaderDisplay.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoaderDisplay.get_joint_vel(frame0), AMPLoaderDisplay.get_joint_vel(frame1)

        blend_joint_q = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([blend_joint_q, blend_joints_vel])


    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_joint_pose(pose):
        return pose[AMPLoaderDisplay.JOINT_POSE_START_IDX : AMPLoaderDisplay.JOINT_POSE_END_IDX]

    def get_joint_vel(pose):
        return pose[AMPLoaderDisplay.JOINT_VEL_START_IDX : AMPLoaderDisplay.JOINT_VEL_END_IDX]
