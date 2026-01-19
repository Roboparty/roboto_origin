#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
import argparse


class MotionLoader:
    def __init__(self, motion_file: str, logger, usd2urdf: bool = False):
        try:
            data = np.load(motion_file)
            joint_default_angle = np.array([0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0, 0.0, -0.1, 0.3, -0.2, 0.0, 0.0, 0.18, 0.06, 0.0, 0.78, 0.0, 0.18, -0.06, 0.0, 0.78, 0.0])
            self.fps = int(data['fps'].item())
            pos_usd = data['joint_pos']
            vel_usd = data['joint_vel']
            self.joint_pos = pos_usd.copy()
            self.joint_vel = vel_usd.copy()
            if usd2urdf:
                logger.info("Converting joint order from USD to URDF")
                joint_map = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
                self.joint_pos[:, joint_map] = pos_usd
                self.joint_vel[:, joint_map] = vel_usd
            self.joint_pos -= joint_default_angle
            
            self.num_frames = self.joint_pos.shape[0]
            self.num_joints = self.joint_pos.shape[1]
            
            logger.info(f"Loaded motion file: {motion_file}")
            logger.info(f"FPS: {self.fps}, Frames: {self.num_frames}, Joints: {self.num_joints}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load motion file: {e}")
    
    def get_pos(self, frame: int) -> np.ndarray:
        return self.joint_pos[frame]
    
    def get_vel(self, frame: int) -> np.ndarray:
        return self.joint_vel[frame]


class MotionPlayer(Node): 
    def __init__(self, motion_file: str, speed: float = 1.0, usd2urdf: bool = False):
        super().__init__('motion_player_node')
        self.motion_loader = MotionLoader(motion_file, self.get_logger(), usd2urdf)
        self.positions = [0.0] * self.motion_loader.num_joints
        
        # 创建发布者
        self.ll_pub = self.create_publisher(JointState, "/joint_command_left_leg", 1)
        self.rl_pub = self.create_publisher(JointState, "/joint_command_right_leg", 1)
        self.la_pub = self.create_publisher(JointState, "/joint_command_left_arm", 1)
        self.ra_pub = self.create_publisher(JointState, "/joint_command_right_arm", 1)

        # 播放控制
        self.is_playing = False
        self.speed = min(max(0.1, speed), 1.0)
        self.get_logger().info(f"Playback speed: {self.speed}x")

        self.step = int(200 / (self.motion_loader.fps * self.speed))
        self.i = 0
        self.is_playing = True
        self.timer = self.create_timer(1.0/200, self.timer_cbk)
    
    def publish_joint_state(self):
        ll_msg = JointState()
        ll_msg.header.stamp = self.get_clock().now().to_msg()
        ll_msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        ll_msg.position = self.positions[:6]
        ll_msg.velocity = [0.0] * 6
        ll_msg.effort = [0.0] * 6
        self.ll_pub.publish(ll_msg)

        rl_msg = JointState()
        rl_msg.header.stamp = self.get_clock().now().to_msg()
        rl_msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        rl_msg.position = self.positions[6:13]
        rl_msg.velocity = [0.0] * 7
        rl_msg.effort = [0.0] * 7
        self.rl_pub.publish(rl_msg)

        la_msg = JointState()
        la_msg.header.stamp = self.get_clock().now().to_msg()
        la_msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        la_msg.position = self.positions[13:18]
        la_msg.velocity = [0.0] * 5
        la_msg.effort = [0.0] * 5
        self.la_pub.publish(la_msg)

        ra_msg = JointState()
        ra_msg.header.stamp = self.get_clock().now().to_msg()
        ra_msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        ra_msg.position = self.positions[18:23]
        ra_msg.velocity = [0.0] * 5
        ra_msg.effort = [0.0] * 5
        self.ra_pub.publish(ra_msg)

    def update_frame(self, frame_idx: int):
        if frame_idx >= self.motion_loader.num_frames:
            self.get_logger().warn(f"Frame index {frame_idx} out of range")
            return
        positions = self.motion_loader.get_pos(frame_idx)
        self.positions = positions.tolist()
    
    def timer_cbk(self):
        if self.is_playing:
            if self.i % self.step == 0:
                frame_idx = self.i // self.step
                if frame_idx >= self.motion_loader.num_frames:
                    self.is_playing = False
                    self.get_logger().info("Motion playback finished")
                    return
                self.update_frame(frame_idx)
            self.publish_joint_state()
            self.i += 1
    
    def stop(self):
        self.is_playing = False

def parse_args():
    parser = argparse.ArgumentParser(description='Motion Player - 播放 motion 数据并发布关节状态')
    parser.add_argument('--motion_file', type=str, required=True, help='Motion 文件路径')
    parser.add_argument('--speed', type=float, default=1.0, help='播放速度倍率')
    parser.add_argument('--usd2urdf', action='store_true',  help='是否将 USD 关节顺序转换为 URDF 关节顺序，默认关闭')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    rclpy.init()
    try:
        player = MotionPlayer(args.motion_file, args.speed, args.usd2urdf)
        rclpy.spin(player)
    except KeyboardInterrupt:
        player.is_playing = False
        player.get_logger().info("Interrupted by user")
    except Exception as e:
        player.is_playing = False
        if 'player' in locals():
            player.get_logger().error(f"Error: {e}")
        else:
            print(f"Error: {e}")
        return 1
    finally:
        if 'player' in locals():
            player.destroy_node()
        rclpy.shutdown()
    return 0

if __name__ == '__main__':
    sys.exit(main())