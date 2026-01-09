# ATOM01-Lab: Direct IsaacLab Workflow for ATOM01

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-2.3.3-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

[English](README.md) | [中文](README_CN.md)

## Overview

This repository provides a direct workflow for training a legged robot using IsaacLab. It provides high transparency and low refactoring difficulty of the direct environment, and uses isaaclab components to simplify the workflow. The codebase is built on IsaacLab, supports Sim2Sim transfer to MuJoCo, and features a modular architecture for seamless customization and extension. 

**Maintainer**: Zhihao Liu
**Contact**: ZhihaoLiu_hit@163.com

**Key Features:**

- `Easy to Reorganize` Provides a direct workflow, allowing for fine-grained definition of environment logic.
- `Isolation` Work outside the core Isaac Lab repository, ensuring that the development efforts remain self-contained.
- `Long-term support` This repository will be updated with the updates of isaac sim and isaac lab, and will be supported for a long time.



## Installation

ATOM01-Lab is built against the latest version of Isaacsim/IsaacLab. It is recommended to follow the latest updates of ATOM01-Lab.

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
git clone https://github.com/Roboparty/atom01_train.git
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
cd atom01_train
pip install -e .
```

- Verify that the extension is correctly installed by running the following command:

```bash
python robolab/scripts/rsl_rl/train.py --task=Atom01-Flat --headless --logger=tensorboard --num_envs=64
```

## Usage

### Train
```bash
python robolab/scripts/rsl_rl/train.py --task=Atom01-Flat --headless --logger=tensorboard --num_envs=8192
```

### Play
```bash
python robolab/scripts/rsl_rl/play.py --task=Atom01-Flat --num_envs=1
```

### Sim2Sim
```bash
python robolab/scripts/mujoco/sim2sim_atom01.py --load_model "{exported/policy.pt model full path here}"
```

## Multi-GPU and Multi-Node Training

ATOM01-Lab supports multi-GPU and multi-node reinforcement learning using rsl_rl, the usage is exactly the same as IsaacLab. [Detailed information](https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html)


## References and Thanks
This project repository builds upon the shoulders of giants.
* [IsaacLab](https://github.com/isaac-sim/IsaacLab)   The various reusable practical components in IsaacLab greatly simplify the complexity of robolab.
* [legged_gym](https://github.com/leggedrobotics/legged_gym)   We borrowed the code organization and environment definition logic of legged_gym and simplified it as much as possible.
