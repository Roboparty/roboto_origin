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


"""Keyboard controller for SE(2) control."""

import weakref
from collections.abc import Callable

import carb
import omni
import torch
from isaaclab.devices.device_base import DeviceBase

from robolab.tasks.direct.base.base_env import BaseEnv


class Keyboard(DeviceBase):

    def __init__(self, env: BaseEnv, lin_vel_step: float = 0.05, ang_vel_step: float = 0.05):
        """Initialize the keyboard layer.
        
        Args:
            env: The environment to control.
            lin_vel_step: Step size for linear velocity control.
            ang_vel_step: Step size for angular velocity control.
        """
        self.env = env
        self.lin_vel_step = lin_vel_step
        self.ang_vel_step = ang_vel_step
        
        # velocity command state
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel = 0.0
        
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        
        print("[Keyboard] Velocity control initialized:")
        print("  W/S: Forward/Backward")
        print("  A/D: Left/Right turn")
        print("  Q/E: Strafe left/right")
        print("  X: Stop (zero velocity)")
        print("  R: Reset environment")

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for ManagerBasedRLEnv: {self.__class__.__name__}\n"
        return msg

    """
    Operations
    """

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        pass

    def advance(self):
        pass

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            if event.input.name in self._INPUT_KEY_MAPPING:
                key = event.input.name
                
                # Reset environment
                if key == "R":
                    self.env.episode_length_buf = torch.ones_like(self.env.episode_length_buf) * 1e6
                    print("[Keyboard] Environment reset triggered")
                # Forward/Backward (lin_vel_x)
                elif key == "W":
                    self.lin_vel_x += self.lin_vel_step
                    self._update_commands()
                elif key == "S":
                    self.lin_vel_x -= self.lin_vel_step
                    self._update_commands()
                # Turn left/right (ang_vel)
                elif key == "Q":
                    self.ang_vel += self.ang_vel_step
                    self._update_commands()
                elif key == "E":
                    self.ang_vel -= self.ang_vel_step
                    self._update_commands()
                # Strafe left/right (lin_vel_y)
                elif key == "A":
                    self.lin_vel_y += self.lin_vel_step
                    self._update_commands()
                elif key == "D":
                    self.lin_vel_y -= self.lin_vel_step
                    self._update_commands()
                # Stop (zero velocity)
                elif key == "X":
                    self.lin_vel_x = 0.0
                    self.lin_vel_y = 0.0
                    self.ang_vel = 0.0
                    self._update_commands()
                    print("[Keyboard] Stopped - all velocities set to zero")

        # since no error, we are fine :)
        return True

    def _update_commands(self):
        """Update the velocity commands in the environment."""
        env = self.env.unwrapped
        commands = env.command_generator.command
        commands[:, 0] = self.lin_vel_x  # lin_vel_x
        commands[:, 1] = self.lin_vel_y  # lin_vel_y
        commands[:, 2] = self.ang_vel    # ang_vel
        print(f"[Keyboard] Vel: vx={self.lin_vel_x:.2f}, vy={self.lin_vel_y:.2f}, ang_vel={self.ang_vel:.2f}")

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            "W": "forward",
            "S": "backward", 
            "Q": "turn_left",
            "E": "turn_right",
            "A": "strafe_left",
            "D": "strafe_right",
            "X": "stop",
            "R": "reset_envs",
        }
