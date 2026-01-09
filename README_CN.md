# ATOM01-Lab: ATOM01 的 IsaacLab 工作流

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-2.3.3-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

[English](README.md) | [中文](README_CN.md)

## 概述

本仓库提供了使用 IsaacLab 训练足式机器人的直接工作流程。它提供了直接环境的高透明度和低重构难度，并使用 isaaclab 组件简化工作流程。代码库基于 IsaacLab 构建，支持 Sim2Sim 传输到 MuJoCo，并具有模块化架构，便于无缝定制和扩展。

**维护者**: 刘志浩
**联系方式**: ZhihaoLiu_hit@163.com

**主要特性:**

- `易于重组` 提供直接工作流程，允许精细定义环境逻辑。
- `隔离性` 在核心 Isaac Lab 仓库之外工作，确保开发工作保持自包含。
- `长期支持` 本仓库将随着 isaac sim 和 isaac lab 的更新而更新，并将长期支持。



## 安装

ATOM01-Lab 是基于最新版本的 Isaacsim/IsaacLab 构建的。建议使用ATOM01-Lab 的最新版本。

- 按照[安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装 Isaac Lab。我们推荐使用 conda 安装，因为它简化了从终端调用 Python 脚本的过程。

- 独立于 Isaac Lab 安装克隆此仓库（即在 `IsaacLab` 目录之外）：

```bash
git clone https://github.com/Roboparty/atom01_train.git
```

- 使用已安装 Isaac Lab 的 python 解释器，安装库

```bash
cd atom01_train
pip install -e .
```

- 通过运行以下命令验证扩展是否正确安装：

```bash
python robolab/scripts/rsl_rl/train.py --task=atom01_flat --headless --logger=tensorboard --num_envs=64
```

## 用法

### 训练
```bash
python robolab/scripts/rsl_rl/train.py --task=atom01_flat --headless --logger=tensorboard --num_envs=8192
```

### 测试
```bash
python robolab/scripts/rsl_rl/play.py --task=atom01_flat --num_envs=1
```

### Sim2Sim
```bash
python robolab/scripts/mujoco/sim2sim_atom01.py --load_model "{exported/policy.pt model full path here}"
```

## 多GPU和多节点训练

ATOM01-Lab 支持使用 rsl_rl 进行多GPU和多节点强化学习，用法与 IsaacLab 完全相同。[详细信息](https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html)


## 参考和致谢
本项目仓库建立在巨人的肩膀上。
* [IsaacLab](https://github.com/isaac-sim/IsaacLab) IsaacLab 中的各种可重用实用组件极大地简化了 robolab 的复杂性。
* [legged_gym](https://github.com/leggedrobotics/legged_gym) 我们借用了 legged_gym 的代码组织和环境定义逻辑，并尽可能简化了它。
