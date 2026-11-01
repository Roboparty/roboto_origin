# 萝博头_原型机——全开源手搓级人形机器人

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![ROS2](https://img.shields.io/badge/ROS2-Humble-silver)](https://docs.ros.org/en/humble/index.html) [![Isaac Sim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) [![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.1-silver)](https://isaac-sim.github.io/IsaacLab) [![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html) [![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/C++17) [![Platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://ubuntu.com/) [![Robotics](https://img.shields.io/badge/Robotics-Humanoid-green.svg)](https://github.com/Roboparty) [![Reinforcement Learning](https://img.shields.io/badge/RL-IsaacLab-red.svg)](https://github.com/leggedrobotics/rsl_rl)

---

![机器人概览](assets/1280X1280.JPEG)

**[English](README.md)**

## 关于我们

我们是 **RoboParty**，上海萝博派对科技有限公司，成立于2025年2月21日。四月份我们开始正式做人形机器人，经历四个月完成了原型机的研发，也就是这台萝博头原型机。我们一直秉持着开源的理念，ROBOTO_ORIGIN的全研发流程和全部的结构、电气、训练、部署都进行了开源。

随着我们逐步推进新机器人的研发，我们发现一台高性能的机器人远远不是手搓所能达到的，所以我们决定将这台能跑能跳的原型机正式开源，以记录我们的来时路。

萝博头原型机可以完全通过淘宝采购和嘉立创打样（一键复刻）完成，真正仅靠自己手搓便能完全组装。依靠我们提供的开源训练代码和部署代码，能轻松实现行走跑步的基础功能。后续我们也将逐步把我们在这台机器人上实现的更多算法添加到开源仓库中。在此之上，作为一台全开源机器人，它的功能可以由广大开发者和使用者定义，因此，创意工坊也会稍后上线。

---

## 文档

**[人形机器人Know-How文档](https://roboparty.com/documents/party-os?doc=motion-control-know-how&lang=zh)** - 包含原型机全研发流程介绍

---

## 交流群

**QQ群：** 546376843

<table>
  <tr>
    <td><img src="assets/qqqrcode.jpg" alt="QQ群二维码" width="200"/></td>
  </tr>
  <tr>
    <td align="center"><b>QQ群</b></td>
  </tr>
</table>

---

## 关于本仓库

本仓库为萝博头原型机的核心子仓库的快照，将会每日更新以方便一次拉取全部资源。

**重要说明：** 本仓库 `roboto_origin` 仅作为快照聚合仓库。模块级问题反馈和代码贡献应前往对应的子仓库中进行；本仓库的 Issues 和 Pull Requests 仅用于快照同步问题、根目录文档、链接、仓库元数据或同步流程修复等聚合层工作。

### 贡献方式

如果您希望为项目做出贡献，请根据贡献内容选择对应的子仓库：

| 子仓库                                                                    | 贡献方向                                           |
| ------------------------------------------------------------------------- | -------------------------------------------------- |
| **[rpo_hardware](https://github.com/Roboparty/rpo_hardware)**             | RPO 机械结构设计、CAD图纸、PCB设计、BOM改进        |
| **[roboparty_deploy](https://github.com/Roboparty/roboparty_deploy)**     | ROS2驱动开发、中间件模块、部署配置、IMU/电机集成   |
| **[roboparty_train](https://github.com/Roboparty/roboparty_train)**       | 强化学习算法、训练环境、仿真配置、Sim2Sim迁移      |
| **[rpo_description](https://github.com/Roboparty/rpo_description)**       | RPO URDF/MJCF运动学/动力学描述、视觉/碰撞网格      |
| **[roboparty_firmware](https://github.com/Roboparty/roboparty_firmware)** | 固件开发、嵌入式软件、USB2CAN、OrangePi/RDK构建工具、系统守护进程管理 |
| **[rpo_appearance](https://github.com/Roboparty/rpo_appearance)**         | RPO 外观件设计、外壳结构、视觉参考、表面处理与装配说明，仅供静态展示，不建议打印 |
| **[roboparty_navigation](https://github.com/Roboparty/roboparty_navigation)** | RPO 实验相关导航与定位应用模块 |
| **[roboparty_xr_teleop](https://github.com/Roboparty/roboparty_xr_teleop)** | RPO 实验相关 XR 遥操作应用。 |

**兼容说明：** 快照目录 `modules/...` 使用目标公开仓库名。部分内部包名、文件名和兼容路径仍可能保留历史 Atom01/atom01 名称。

**详细贡献指南请点击这里：** [CONTRIBUTING_CN.md](CONTRIBUTING_CN.md)

**[机器人BOM表点这里](./assets/BOM.md)**

---

<table>
  <tr>
    <td><img src="assets/atom01-01.gif" alt="机器人演示GIF 1" width="100%"/></td>
    <td><img src="assets/atom01-02.gif" alt="机器人演示GIF 2" width="100%"/></td>
  </tr>
</table>

![机器人细节](assets/1280X1280.PNG)

---

## 资源指引

### 仓库模块

| 模块名称               | 介绍                                                                                              | 仓库链接                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **rpo_hardware**        | RPO 机器人的硬件设计文件，包含结构图纸和设计资料                                                  | https://github.com/Roboparty/rpo_hardware       |
| **roboparty_deploy**    | ROS2部署框架，提供模块化架构的中间件，用于机器人部署和控制，支持IMU、电机驱动、推理等功能         | https://github.com/Roboparty/roboparty_deploy   |
| **roboparty_train**     | 基于IsaacLab的直接训练工作流，提供高透明度和低重构难度的强化学习训练环境，支持Sim2Sim迁移到MuJoCo | https://github.com/Roboparty/roboparty_train    |
| **rpo_description**     | RPO 机器人的URDF/MJCF模型文件，包含机器人运动学和动力学描述，用于仿真和可视化                    | https://github.com/Roboparty/rpo_description    |
| **roboparty_firmware**  | 固件模块，提供嵌入式软件支持，包括USB2CAN、OrangePi/RDK构建工具和系统守护进程管理                 | https://github.com/Roboparty/roboparty_firmware |
| **rpo_appearance**      | RPO 机器人的外观设计文件，包含外观件、外壳设计资料和原型机外观说明，仅供静态展示，不建议打印      | https://github.com/Roboparty/rpo_appearance     |
| **roboparty_navigation** | RPO 实验相关导航与定位应用模块                                                                  | https://github.com/Roboparty/roboparty_navigation |
| **roboparty_xr_teleop** | RPO 实验相关 XR 遥操作应用。                                                                   | https://github.com/Roboparty/roboparty_xr_teleop |

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/Roboparty/roboto_origin.git

# 更新仓库
git pull
```

进入各模块目录 `modules/...`，按照模块内 README 指引即可。

---

## 常见问题

### 是否需要机器人经验？

具备 Linux 和 ROS2 基础会有帮助，但文档面向入门用户设计。建议从[文档站](https://roboparty.com/roboto_origin/doc)开始。

---

## 行为准则

本项目已采用[行为准则](CODE_OF_CONDUCT_CN.md)来营造欢迎和包容的社区环境。所有贡献者和用户都应遵守这些准则。

**[Code of Conduct](CODE_OF_CONDUCT.md)**


---

## 免责声明

### 开源软件授权与免责协议

本软件按"原样"提供，不提供任何形式的担保，无论是明示还是暗示，包括但不限于对适销性、特定用途适用性和非侵权性的担保。

**硬件耦合免责：** 本软件输出的指令在物理硬件上执行，由此在物理世界产生的任何后果均由硬件操作者全权负责。本软件仅作为"辅助工具"，操作权完全在用户手中。在任何情况下，作者或版权持有者均不对任何索赔、损害或其他责任承担责任，无论是在合同诉讼、侵权诉讼还是其他诉讼中，因软件或与软件的使用或其他交易而产生、引起或与之相关的。

### 二次开发用户协议

对于用户使用、修改、改编、改进、二次开发、运行、分发或以其他任何方式使用开源组件的行为，用户同意承担全部责任。**萝博派对不承担任何责任。**

使用、修改或分发本软件，即表示您确认理解并同意承担与此类行为相关的所有风险，包括但不限于：
- 机器人运行造成的人身伤害或财产损失
- 因参数配置不当导致的设备损坏
- 因安全测试不足导致的事故

---

**许可证：** 除非另有说明，由 RoboParty 创作的聚合层内容按 GNU GPL 第 3 版提供。`modules/` 下的快照及第三方内容继续适用其各自的许可证，不会被根目录 `LICENSE` 重新许可。适用范围详见 [LICENSES.md](LICENSES.md)。

---

[![Star History Chart](https://raw.githubusercontent.com/Roboparty/roboto_origin/star-history/star-history.svg)](https://github.com/Roboparty/roboto_origin)

---

## 贡献者

感谢所有为项目做出贡献的人员！

<a href="https://github.com/Roboparty/roboto_origin/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Roboparty/roboto_origin" alt="贡献者"/>
</a>
