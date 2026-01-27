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

from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

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
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
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
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
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
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.16, "max": 0.50},
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.18, "max": 0.35},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"])},
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
                "robot", joint_names=[".*torso.*", ".*_elbow_yaw.*"]
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_pitch.*", ".*_knee.*", ".*_ankle_pitch.*", ".*_ankle_roll.*"])},
    )
    joint_deviation_interrupt = RewTerm(
        func=mdp.joint_deviation_interrupt,
        weight=-1.0,
        params={
            "asset_cfg1": SceneEntityCfg(
                "robot", joint_names=[".*_arm_roll.*", ".*_arm_yaw.*", ".*_elbow_pitch.*"]
            ),
            "asset_cfg2": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm_pitch.*"],
            ),
            "weight1": 1.0, "weight2": 0.06
        }
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    upward = RewTerm(func=mdp.upward, weight=0.4)
    stand_still = RewTerm(func=mdp.stand_still_interrupt, weight=-0.2, params={"pos_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]),
                                                                               "vel_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow.*", ".*torso.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*"]), 
                                                                               "interrupt_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow_pitch.*"]),
                                                                               "pos_weight": 1.0, "vel_weight": 0.04})
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "ankle_height":0.04,"threshold":0.02})
    action_penalty = RewTerm(func=mdp.action_penalty_interrupt, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow_pitch.*"])})


@configclass
class InterruptCfg:
    use_interrupt: bool = False
    max_curriculum: float = 1.0
    interrupt_ratio: float = 0.5
    interrupt_joint_names: list = []
    interrupt_scale : list = []
    interrupt_lower_bound: list = []
    interrupt_init_range: float = 0.2
    interrupt_update_step: int = 30
    switch_prob: float = 0.005
    interrupt_vis: VisualizationMarkersCfg = VisualizationMarkersCfg(
        markers={
            "interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "no_interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
        prim_path="/Visuals/Command/interrupt",
    )

@configclass
class ATOM01InterruptEnvCfg(BaseEnvCfg):

    reward = ATOM01RewardCfg()
    interrupt = InterruptCfg(
        use_interrupt = True,
        max_curriculum = 1.0,
        interrupt_ratio = 0.5,
        interrupt_joint_names = [
            "left_arm_pitch_joint",
            "left_arm_roll_joint",
            "left_arm_yaw_joint",
            "left_elbow_pitch_joint",
            "right_arm_pitch_joint",
            "right_arm_roll_joint",
            "right_arm_yaw_joint",
            "right_elbow_pitch_joint",
        ],
    interrupt_scale = [
            3.14, # Arm Pitch -1.57~1.57
            1.82, # Arm Roll, -0.25~1.57
            3.14, # Arm Yaw,  -1.57~1.57
            2.07, # Elbow Pitch, -0.5~1.57
            3.14, # Arm Pitch -1.57~1.57
            1.82, # Arm Roll, -1.57~0.25
            3.14, # Arm Yaw,  -1.57~1.57
            2.07, # Elbow Pitch, -0.5~1.57
        ], # Uniform Distribution Noise for each joint.
    interrupt_lower_bound = [
            -1.57,
            -0.25, 
            -1.57, 
            -0.5, 
            -1.57, 
            -1.57, 
            -1.57,
            -0.5,
        ],
        interrupt_init_range = 0.2,
        interrupt_update_step = 30,
        switch_prob = 0.005,
    )
    interrupt_vis = VisualizationMarkersCfg(
        markers={
            "interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "no_interrupt": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
        prim_path="/Visuals/Command/interrupt",
    )

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 23
        self.observation_space = 79
        self.state_space = 140
        self.scene_context.robot = ATOM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.height_scanner.prim_body_name = "base_link"
        self.scene_context.terrain_type = "generator"
        self.scene_context.terrain_generator = GRAVEL_TERRAINS_CFG
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt = self.sim.dt,
            step_dt = self.decimation * self.sim.dt
        )
        self.robot.terminate_contacts_body_names = ["torso_link", ".*_thigh_yaw_link", ".*_thigh_roll_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link", "base_link"]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["torso_link", "base_link"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["left_.*_link", "right_.*_link"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.robot.action_scale = 0.25
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03