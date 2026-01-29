# ATOM01 ROS2 Deploy

[![ROS2](https://img.shields.io/badge/ROS2-Humble-silver)](https://docs.ros.org/en/humble/index.html)
![C++](https://img.shields.io/badge/C++-17-blue)
[![Linux platform](https://img.shields.io/badge/platform-linux--x86_64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Linux platform](https://img.shields.io/badge/platform-linux--aarch64-orange.svg)](https://releases.ubuntu.com/22.04/)

[English](README.md) | [中文](README_CN.md)

## 概述

本仓库提供了使用ROS2作为中间件的部署框架，并具有模块化架构，便于无缝定制和扩展。

**维护者**: 刘志浩
**联系方式**: <ZhihaoLiu_hit@163.com>

**主要特性:**

- `易于上手` 提供全部细节代码，便于学习并允许修改代码。
- `隔离性` 不同功能由不同包实现，支持加入自定义功能包。
- `长期支持` 本仓库将随着训练仓库代码的更新而更新，并将长期支持。

## 环境配置

部署代码在香橙派5plus上运行，系统为ubuntu22.04，内核版本为5.10，我们在香橙派5plus上进行部署的环境配置。

首先安装ROS2 humble，参考[ROS官方](https://docs.ros.org/en/humble/Installation.html)进行安装。

部署还依赖ccache fmt spdlog eigen3等库，在上位机中执行指令进行安装：

```bash
sudo apt update && sudo apt install -y ccache libfmt-dev libspdlog-dev libeigen3-dev
```

然后安装香橙派5plus的5.10实时内核

```bash
git clone https://github.com/Roboparty/atom01_deploy.git
cd atom01_deploy
sudo apt install ./*.deb
```

接下来为用户授予实时优先级设置权限：

```bash
sudo nano /etc/security/limits.conf
```

在文件末尾添加以下两行，将 orangepi 替换为你的实际用户名：

```bash
# Allow user 'orangepi' to set real-time priorities
orangepi   -   rtprio   98
orangepi   -   memlock  unlimited
```

保存退出后重启香橙派使设置生效。

## 硬件连接

电机驱动中can0对应左腿，can1对应右腿加腰，can2对应左手，can3对应右手，默认按照usb转can插入上位机顺序编号，先插入的是can0。建议将USB转CAN插在上位机的3.0接口上，如果使用USB扩展坞也请使用3.0接口的USB扩展坞并插在3.0接口上，IMU和手柄插在USB2.0接口即可。

编写udev规则用来将USB口与usb转can绑定，即可不需要再管设备插入上位机顺序，示例在99-auto-up-devs.rules中，需要修改的是KERNELS项，将其修改为该usb转can想要对应绑定的USB接口的KERNELS属性项。在上位机输入指令监视USB事件：

```bash
sudo udevadm monitor
```

在USB上插入设备时就会显示该USB接口的KERNELS属性项，如/devices/pci0000:00/0000:00:14.0/usb3/3-8，我们在匹配KERNELS属性项时使用3-8即可。如果想要绑定在该USB接口上的扩展坞上的USB口则会有3-8.x出现，此时使用3-8.x进行匹配扩展坞上的USB口。

编写完成后在上位机中执行：

```bash
sudo cp 99-auto-up-devs.rules /etc/udev/rules.d/
sudo udevadm control --reload
sudo udevadm trigger
```

之后拔下所有can设备再插入指定USB接口即可生效。

该udev规则还包括IMU串口配置，也需要修改KERNELS项，方法不再赘述。如果规则正常生效，can应该全部自动配置完毕并使能，可以在上位机中输入ip a指令查看结果。

如果不配置udev规则，则需要按照上文顺序插入usb转can，并手动配置can和IMU串口：

```bash
sudo ip link set canX up type can bitrate 1000000
sudo ip link set canX txqueuelen 1000
# canX 为 can0 can1 can2 can3，需要为每个can都输入一遍上面两个指令

sudo chmod 666 /dev/ttyUSB0
```

## 软件使用

### IMU

编译：

```bash
cd imu
colcon build --symlink-install --cmake-args -G Ninja
```

启动：

```bash
source install/setup.bash
ros2 launch hipnuc_imu imu_spec_msg.launch.py
```

如果IMU串口权限正常，使用 PlotJuggler 或者 `ros2 topic echo` 可以看到IMU数据。

### MOTORS

首先修改参数文件，保证参数正确！！！

编译：

```bash
cd motors
colcon build --symlink-install --cmake-args -G Ninja
```

启动：

```bash
source install/setup.bash
ros2 launch motors motors_spec_msg.launch.py
```

如果can配置正常，此时所有电机绿灯亮起。如果还未配置电机零点，将标定件插入把电机摆至零点后输入:

```bash
# 如果标零过请不要再执行这一条，除非电机显著丢失零点！！！
ros2 service call /set_zeros motors/srv/SetZeros
```

观察到电机绿灯一个个灭下说明正在标零。

标零完成后重新启动 motors 并打开 PlotJuggler，订阅电机 state 话题后输入：

```bash
ros2 service call /read_motors motors/srv/ReadMotors
```

此时在 PlotJuggler 中可以看到各个电机此时位置，保证没有反向关节且都在零点附近后输入：

```bash
ros2 service call /reset_motors motors/srv/ResetMotors
```

进行电机归零。

### INFERENCE

首先修改参数文件，保证参数正确！！！将训练得到的onnx模型放到models文件夹中。

编译：

```bash
cd inference
colcon build --symlink-install --cmake-args -G Ninja
```

保证IMU启动、电机启动且正常归零后，启动：

```bash
source install/setup.bash
ros2 launch inference inference.launch.py
```

### Joy

连接上logitech手柄后，启动：

```bash
ros2 run joy joy_node
```

即可使用手柄，右摇杆控制前后左右，LT RT控制左转右转。
