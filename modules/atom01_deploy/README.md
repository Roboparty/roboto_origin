# ATOM01 ROS2 Deploy

[![ROS2](https://img.shields.io/badge/ROS2-Humble-silver)](https://docs.ros.org/en/humble/index.html)
![C++](https://img.shields.io/badge/C++-17-blue)
[![Linux platform](https://img.shields.io/badge/platform-linux--x86_64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Linux platform](https://img.shields.io/badge/platform-linux--aarch64-orange.svg)](https://releases.ubuntu.com/22.04/)

[English](README.md) | [中文](README_CN.md)

## Overview

This repository provides a deployment framework using ROS2 as middleware with a modular architecture for seamless customization and extension.

**Maintainer**: Zhihao Liu
**Contact**: <ZhihaoLiu_hit@163.com>

**Key Features:**

- `Easy to Use` Provides complete detailed code for learning and code modification.
- `Isolation` Different functions are implemented by different packages, supporting custom function packages.
- `Long-term Support` This repository will be updated with the training repository code and will be supported long-term.

## Environment Setup

The deployment code runs on Orange Pi 5 Plus with Ubuntu 22.04 system and kernel version 5.10. We perform the environment configuration for deployment on Orange Pi 5 Plus.

First install ROS2 humble, refer to [ROS official](https://docs.ros.org/en/humble/Installation.html) for installation.

The deployment also depends on libraries such as ccache fmt spdlog eigen3. Execute the following command on the host computer for installation:

```bash
sudo apt update && sudo apt install -y ccache libfmt-dev libspdlog-dev libeigen3-dev
```

Then install the 5.10 real-time kernel for Orange Pi 5 Plus:

```bash
git clone https://github.com/Roboparty/atom01_deploy.git
cd atom01_deploy
sudo apt install ./*.deb
```

Next, grant the user permission to set real-time priorities:

```bash
sudo nano /etc/security/limits.conf
```

Add the following two lines at the end of the file, replacing orangepi with your actual username:

```bash
# Allow user 'orangepi' to set real-time priorities
orangepi   -   rtprio   98
orangepi   -   memlock  unlimited
```

Save, exit, and then reboot the Orange Pi for the settings to take effect.

## Hardware Connection

In the motor driver, can0 corresponds to the left leg, can1 corresponds to the right leg and waist, can2 corresponds to the left hand, and can3 corresponds to the right hand. By default, they are numbered according to the order of USB-to-CAN insertion into the host computer, with the first inserted being can0. It is recommended to plug the USB-to-CAN into the 3.0 interface of the host computer. If using a USB hub, please also use a 3.0 interface USB hub and plug it into the 3.0 interface. IMU and gamepad can be plugged into USB2.0 interfaces.

Write udev rules to bind USB ports with USB-to-CAN devices, so you don't need to care about the order of device insertion into the host computer. An example is in 99-auto-up-devs.rules. What needs to be modified is the KERNELS item, changing it to the KERNELS attribute item of the USB interface that the USB-to-CAN device wants to bind to. Input the following command on the host computer to monitor USB events:

```bash
sudo udevadm monitor
```

When inserting a device into the USB port, the KERNELS attribute item of that USB interface will be displayed, such as /devices/pci0000:00/0000:00:14.0/usb3/3-8. When matching the KERNELS attribute item, we can use 3-8. If you want to bind to a USB port on a hub connected to that USB interface, 3-8-x will appear, then use 3-8-x to match the USB port on the hub.

After writing, execute the following on the host computer:

```bash
sudo cp 99-auto-up-devs.rules /etc/udev/rules.d/
sudo udevadm control --reload
sudo udevadm trigger
```

After that, unplug all CAN devices and plug them into the specified USB interfaces to take effect.

The udev rules also include IMU serial port configuration, which also needs to modify the KERNELS item. The method is not repeated here. If the rules take effect normally, all CAN should be automatically configured and enabled. You can input the `ip a` command on the host computer to check the results.

If you don't configure udev rules, you need to insert USB-to-CAN devices in the order mentioned above and manually configure CAN and IMU serial ports:

```bash
sudo ip link set canX up type can bitrate 1000000
sudo ip link set canX txqueuelen 1000
# canX is can0 can1 can2 can3, you need to input the above two commands for each CAN

sudo chmod 666 /dev/ttyUSB0
```

## Software Usage

### IMU

Compile:

```bash
cd imu
colcon build --symlink-install --cmake-args -G Ninja
```

Launch:

```bash
source install/setup.bash
ros2 launch hipnuc_imu imu_spec_msg.launch.py
```

If IMU serial port permissions are normal, you can see IMU data using PlotJuggler or `ros2 topic echo`.

### MOTORS

First modify the parameter file to ensure parameters are correct!!!

Compile:

```bash
cd motors
colcon build --symlink-install --cmake-args -G Ninja
```

Launch:

```bash
source install/setup.bash
ros2 launch motors motors_spec_msg.launch.py
```

If CAN configuration is normal, all motors' green lights should turn on at this time. If motor zero points haven't been configured yet, insert the calibration tool and position the motor to zero point, then input:

```bash
# If already zeroed, please don't execute this command again unless the motor significantly loses zero point!!!
ros2 service call /set_zeros motors/srv/SetZeros
```

Observe that the motor green lights turn off one by one, indicating zeroing is in progress.

After zeroing is complete, restart motors and open PlotJuggler, subscribe to motor state topics and input:

```bash
ros2 service call /read_motors motors/srv/ReadMotors
```

At this time, you can see the current position of each motor in PlotJuggler. Ensure there are no reversed joints and all are near zero point, then input:

```bash
ros2 service call /reset_motors motors/srv/ResetMotors
```

To perform motor reset.

### INFERENCE

First modify the parameter file to ensure parameters are correct!!! Put the trained ONNX model into the models folder.

Compile:

```bash
cd inference
colcon build --symlink-install --cmake-args -G Ninja
```

After ensuring IMU is started, motors are started and properly reset, launch:

```bash
source install/setup.bash
ros2 launch inference inference.launch.py
```

### Joy

After connecting the Logitech gamepad, launch:

```bash
ros2 run joy joy_node
```

Then you can use the gamepad. Right joystick controls forward/backward/left/right, LT RT controls left/right turn.
