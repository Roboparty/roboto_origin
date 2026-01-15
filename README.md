# ROBOTO_ORIGIN - Fully Open-Source DIY Humanoid Robot

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![ROS2](https://img.shields.io/badge/ROS2-Humble-silver)](https://docs.ros.org/en/humble/index.html) [![Isaac Sim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) [![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.1-silver)](https://isaac-sim.github.io/IsaacLab) [![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html) [![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/C++17) [![Platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://ubuntu.com/) [![Robotics](https://img.shields.io/badge/Robotics-Humanoid-green.svg)](https://github.com/Roboparty) [![Reinforcement Learning](https://img.shields.io/badge/RL-IsaacLab-red.svg)](https://github.com/leggedrobotics/rsl_rl)

---

![Robot Overview](assets/1280X1280.JPEG)

**[中文说明点这里](README_cn.md)**

## About Us

We are **RoboParty**, founded on February 21, 2025. We started developing humanoid robots in April and completed the prototype **ROBOTO_ORIGIN** in just four months. We have always upheld the philosophy of open source. ROBOTO_ORIGIN's entire R&D process, including all structures, electronics, training, and deployment, has been open-sourced.

As we advance the development of new robots, we realize that a high-performance robot cannot be achieved through DIY alone. Therefore, we have decided to officially open-source this running and jumping prototype to document our journey.

> This robot can be completely assembled through Taobao procurement and Jialichuang prototyping. It can be truly assembled independently through DIY. With our open-source training and deployment code, you can easily achieve walking and running.

In the future, we will gradually add the motion control algorithms implemented on this robot to the open-source repository. However, as a fully open-source robot, its functionality can be defined by the vast number of developers and users, so a creative workshop will also be launched soon.

### Contributing

**Important:** The `roboto_origin` repository serves as a snapshot aggregation only. All issue reporting and code contributions should be made to the individual sub-repositories.

If you wish to contribute to the project, please select the appropriate sub-repository based on your contribution:

| Sub-Repository                                                            | Contribution Areas                                                                         |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **[Atom01_hardware](https://github.com/Roboparty/Atom01_hardware)**       | Mechanical structure design, CAD drawings, PCB design, BOM improvements                    |
| **[atom01_deploy](https://github.com/Roboparty/atom01_deploy)**           | ROS2 driver development, middleware modules, deployment configs, IMU/motor integration     |
| **[atom01_train](https://github.com/Roboparty/atom01_train)**             | RL algorithms, training environments, simulation configs, Sim2Sim transfer                 |
| **[atom01_description](https://github.com/Roboparty/atom01_description)** | URDF kinematic/dynamic descriptions, visual/collision meshes, joint parameter optimization |

**For detailed contribution guidelines:** [CONTRIBUTING.md](CONTRIBUTING.md)

**[BOM table](./assets/BOM_EN.md)**

---

<table>
  <tr>
    <td><img src="assets/atom01-01.gif" alt="Robot Demo GIF 1" width="100%"/></td>
    <td><img src="assets/atom01-02.gif" alt="Robot Demo GIF 2" width="100%"/></td>
  </tr>
</table>

![Robot Details](assets/1280X1280.PNG)

## Resource Guide

### Repository Modules

| Module Name            | Description                                                                                                                                               | Repository Link                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Atom01_hardware**    | Hardware design files for Atom01 robot, including structural drawings and design materials                                                                | https://github.com/Roboparty/Atom01_hardware    |
| **atom01_deploy**      | ROS2 deployment framework with modular architecture middleware for robot deployment and control, supporting IMU, motor drivers, inference, etc.           | https://github.com/Roboparty/atom01_deploy      |
| **atom01_train**       | Direct IsaacLab training workflow providing high transparency and low refactoring difficulty RL training environment, supports Sim2Sim transfer to MuJoCo | https://github.com/Roboparty/atom01_train       |
| **atom01_description** | URDF model files for Atom01 robot, containing kinematic and dynamic descriptions for simulation and visualization                                         | https://github.com/Roboparty/atom01_description |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/Roboparty/roboto_origin.git

# Update repository
git pull
```

Navigate to each module directory `modules/...` and follow the README inside each module to continue.

---

## Code of Conduct

This project has adopted the [Code of Conduct](CODE_OF_CONDUCT.md) to foster a welcoming and inclusive community. All contributors and users are expected to adhere to these guidelines.

**[中文版行为准则](CODE_OF_CONDUCT_CN.md)**

---

**This project is licensed under the GNU General Public License Version 3 (GPLv3). See [LICENSE](LICENSE) for details.**
