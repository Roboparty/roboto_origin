from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
import torch
import numpy as np

from rsl_rl.utils import AMPLoaderDisplay
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate

from legged_lab.envs.base.base_env import (  # noqa:F401
    BaseEnv,
)

from .atom01_amp_config import ATOM01AmpEnvCfg

class ATOM01AmpEnv(BaseEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.cfg: ATOM01AmpEnvCfg
        if self.cfg.amp.use_amp:
            self.amp_loader_display = AMPLoaderDisplay(
                motion_files=self.cfg.amp.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
            )
            self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

    def init_buffers(self):
        super().init_buffers()
        if self.cfg.amp.use_amp:
            self.feet_body_ids, _ = self.robot.find_bodies(
                name_keys=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            )
            self.elbow_body_ids, _ = self.robot.find_bodies(
                name_keys=["left_elbow_pitch_link", "right_elbow_pitch_link"], preserve_order=True
            )
            self.left_leg_ids, _ = self.robot.find_joints(
                name_keys=[
                    "left_thigh_roll_joint",
                    "left_thigh_pitch_joint",
                    "left_thigh_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                ],
                preserve_order=True,
            )
            self.right_leg_ids, _ = self.robot.find_joints(
                name_keys=[
                    "right_thigh_roll_joint",
                    "right_thigh_pitch_joint",
                    "right_thigh_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                ],
                preserve_order=True,
            )
            self.left_arm_ids, _ = self.robot.find_joints(
                name_keys=[
                    "left_arm_pitch_joint",
                    "left_arm_roll_joint",
                    "left_arm_yaw_joint",
                    "left_elbow_pitch_joint",
                ],
                preserve_order=True,
            )
            self.right_arm_ids, _ = self.robot.find_joints(
                name_keys=[
                    "right_arm_pitch_joint",
                    "right_arm_roll_joint",
                    "right_arm_yaw_joint",
                    "right_elbow_pitch_joint",
                ],
                preserve_order=True,
            )
            self.ankle_joint_ids, _ = self.robot.find_joints(
                name_keys=["left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"],
                preserve_order=True,
            )


    def visualize_motion(self, time):
            """
            Update the robot simulation state based on the AMP motion capture data at a given time.

            This function sets the joint positions and velocities, root position and orientation,
            and linear/angular velocities according to the AMP motion frame at the specified time,
            then steps the simulation and updates the scene.

            Args:
                time (float): The time (in seconds) at which to fetch the AMP motion frame.

            Returns:
                None
            """
            visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
            device = self.device

            dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
            dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)

            dof_pos[:, self.left_leg_ids] = visual_motion_frame[6:12]
            dof_pos[:, self.right_leg_ids] = visual_motion_frame[12:18]
            dof_pos[:, self.left_arm_ids] = visual_motion_frame[18:22]
            dof_pos[:, self.right_arm_ids] = visual_motion_frame[22:26]

            dof_vel[:, self.left_leg_ids] = visual_motion_frame[32:38]
            dof_vel[:, self.right_leg_ids] = visual_motion_frame[38:44]
            dof_vel[:, self.left_arm_ids] = visual_motion_frame[44:48]
            dof_vel[:, self.right_arm_ids] = visual_motion_frame[48:52]

            self.robot.write_joint_position_to_sim(dof_pos)
            self.robot.write_joint_velocity_to_sim(dof_vel)

            env_ids = torch.arange(self.num_envs, device=device)

            root_pos = visual_motion_frame[:3].clone()
            root_pos[2] += 0.3

            euler = visual_motion_frame[3:6].cpu().numpy()
            quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
            quat_wxyz = torch.tensor(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=device
            )

            lin_vel = visual_motion_frame[27:30].clone()
            ang_vel = torch.zeros_like(lin_vel)

            # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            root_state = torch.zeros((self.num_envs, 13), device=device)
            root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
            root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
            root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
            root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

            self.robot.write_root_state_to_sim(root_state, env_ids)
            self.sim.render()
            self.sim.step()
            self.scene.update(dt=self.step_dt)

    def get_amp_obs_for_expert_trans(self):
        """Gets amp obs from policy"""
        self.left_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))
        self.right_arm_local_vec = torch.tensor([0.0, 0.0, -0.3], device=self.device).repeat((self.num_envs, 1))

        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_rotate(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        self.left_leg_dof_pos = self.robot.data.joint_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = self.robot.data.joint_pos[:, self.right_leg_ids]
        self.left_leg_dof_vel = self.robot.data.joint_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = self.robot.data.joint_vel[:, self.right_leg_ids]
        self.left_arm_dof_pos = self.robot.data.joint_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = self.robot.data.joint_pos[:, self.right_arm_ids]
        self.left_arm_dof_vel = self.robot.data.joint_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = self.robot.data.joint_vel[:, self.right_arm_ids]
        return torch.cat(
            (
                self.right_arm_dof_pos,
                self.left_arm_dof_pos,
                self.right_leg_dof_pos,
                self.left_leg_dof_pos,
                self.right_arm_dof_vel,
                self.left_arm_dof_vel,
                self.right_leg_dof_vel,
                self.left_leg_dof_vel,
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )