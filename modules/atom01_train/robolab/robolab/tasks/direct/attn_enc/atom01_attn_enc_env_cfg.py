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

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
import matplotlib as mpl
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
import math

from robolab.tasks.direct.base import mdp
from robolab.assets.robots import ATOM01_CFG
from robolab.tasks.direct.base import (  # noqa:F401
    BaseAgentCfg, 
    BaseEnvCfg, 
    RewardCfg, 
    HeightScannerCfg, 
    SceneContextCfg, 
    RobotCfg, 
    ObsScalesCfg, 
    NormalizationCfg, 
    CommandRangesCfg, 
    CommandsCfg, 
    NoiseScalesCfg, 
    NoiseCfg, 
    EventCfg,
    GRAVEL_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG,
    ROUGH_HARD_TERRAINS_CFG,
    SceneCfg
)


@configclass
class ATOM01RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.25, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-4)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-2e-2)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle_roll.*).*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.10, "max": 0.50},
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.13, "max": 0.41},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_thigh_yaw.*", ".*_thigh_roll.*"]
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*torso.*", ".*_arm_roll.*", ".*_arm_yaw.*", ".*_elbow_pitch.*", ".*_elbow_yaw.*"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.09,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm_pitch.*"],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_pitch.*", ".*_knee.*", ".*_ankle_pitch.*", ".*_ankle_roll.*"])},
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    upward = RewTerm(func=mdp.upward, weight=0.4)
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.2, params={"pos_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]),
                                                                     "vel_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]), 
                                                                     "pos_weight": 0.0, "vel_weight": 0.04})
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "ankle_height":0.04,"threshold":0.02})
    undesired_foothold = RewTerm(
        func=mdp.undesired_foothold, 
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "ankle_height":0.04})


color = [tuple(float(c) for c in mpl.colormaps["viridis"](i / 9.0)[:-1]) for i in range(10)]
markers = {}
for i in range(10):
    markers[f"hit_{i}"] = sim_utils.SphereCfg(
        radius=0.02,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color[i])
    )
@configclass
class AttnEncCfg:
    use_attn_enc: bool = False
    vel_in_obs: bool = False
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Attention",
        markers=markers,
    )   


@configclass
class ATOM01AttnEncEnvCfg(BaseEnvCfg):

    reward = ATOM01RewardCfg()
    attn_enc = AttnEncCfg(
            use_attn_enc=True,
            vel_in_obs=False,
        )

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 23
        self.observation_space = 78
        self.state_space = 145
        self.scene_context.robot = ATOM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.height_scanner.prim_body_name = "base_link"
        self.scene_context.terrain_type = "generator"
        self.scene_context.terrain_generator = ROUGH_HARD_TERRAINS_CFG
        self.scene_context.height_scanner.enable_height_scan = True
        self.scene_context.height_scanner.enable_height_scan_actor = True
        self.scene_context.height_scanner.resolution = 0.1
        self.scene_context.height_scanner.size = (1.6, 1.0)
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt = self.sim.dt,
            step_dt = self.decimation * self.sim.dt
        )
        self.robot.terminate_contacts_body_names = ["torso_link", ".*_thigh_yaw_link", ".*_thigh_roll_link", ".*_elbow_.*_link", ".*_arm_.*_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.noise.add_noise = True
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["torso_link"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["left_.*_link", "right_.*_link"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.robot.action_scale = 0.25
        self.robot.actor_obs_history_length = 5
        self.robot.critic_obs_history_length = 5
        self.normalization.height_scan_offset = 0.75
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03
        self.noise.noise_scales.lin_vel = 0.2
        self.noise.noise_scales.height_scan = 0.025
        self.commands.ranges = CommandRangesCfg(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        )