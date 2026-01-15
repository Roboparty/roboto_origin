# Roboto_Power_V1.0 - 机器人核心分电板

**[中文](./README_cn.md)** | [English](./README.md)

![Manufacturer](https://img.shields.io/badge/Manufacturer-RoboParty-blue)
![Hardware](https://img.shields.io/badge/Hardware-V1.0-green)
![Voltage](https://img.shields.io/badge/Max_Voltage-48V-red)

![Render](00_Docs/Images/power_render.png)

## 📖 概述 (Overview)

**Roboto_Power_v1.0** 是 Roboto 原型机电气系统的核心组件。

作为机器人的能量枢纽，它负责将电池输入的总电源分配至各个电机及控制模块，实现了“集中控制与分布式供电”的架构。该分电板与以下模块共同构成了完整的电气系统：
* 上位机 (Orange Pi 5 Plus)
* 48V 转 5V 降压模块
* USB-to-CAN 通讯板

## 📂 仓库目录 (File Structure)

本仓库包含制造该分电板所需的所有文件：

- **PCB 制造文件 (Gerber):** `01_Gerber/`
- **BOM 与坐标文件:** `02_Assembly/`
- **详细说明文档:** `00_Docs/`

## 🔌 接口定义 (Interfaces)

### 1. 正面接口布局
![Interfaces Top](00_Docs/Images/power_interface_top.png)

| 编号 | 接口类型 | 说明 | 备注 |
| :--- | :--- | :--- | :--- |
| **①** | **电源输入 (Main In)** | XT60/XT90 (公) | 连接电池主电源 |
| **②** | **电源分流输出** | XT30 (母) | 分配给各个关节电机供电 (6路) |
| **③** | **CAN 信号集线器** | GH1.25 | 4路 CAN 总线信号分发 (正面) |

### 2. 背面接口布局
![Interfaces Bottom](00_Docs/Images/power_interface_bottom.png)

| 编号 | 接口类型 | 说明 |
| :--- | :--- | :--- |
| **④** | **CAN 信号扩展** | GH1.25 连接器，用于扩展更多 CAN 节点或连接 USB转CAN 模块 |

> **提示：** 本板集成了 CAN 总线集线器功能，方便走线布局。

## ⚠️ 极重要注意事项 (Critical Precautions)

> 🛑 **请在操作前仔细阅读！错误操作将导致设备烧毁！**

1.  **电源极性 (Polarity):**
    * **接线一定注意正负极！！**
    * **接线一定注意正负极！！**
    * **接线一定注意正负极！！**
    * 插上电池前，请务必使用万用表检查输入端正负极是否短路。

2.  **CAN 线序:**
    * 请严格核对板上丝印 `H` (High) 和 `L` (Low)。
    * CAN_H 接 CAN_H，CAN_L 接 CAN_L，**切勿接反**，否则会导致通讯中断。

## 🏭 制造文件下载 (Downloads)

如果您需要自行制造该模块，请下载以下文件：

* **Gerber 文件 (打板):** [POWER_BOARD_GERBER_V1.0.zip](01_Gerber/POWER_BOARD_GERBER_V1.0.zip)
* **物料清单 (BOM):** [BOM_POWER_BOARD-V1.0.xlsx](02_Assembly/BOM_POWER_BOARD-V1.0.xlsx)
* **贴片坐标 (CPL):** [PickAndPlace_POWER_BOARD_V1_0.xlsx](02_Assembly/PickAndPlace_POWER_BOARD_V1_0.xlsx)

---
**技术支持:** 如遇到技术问题，请联系 RoboParty 技术团队。