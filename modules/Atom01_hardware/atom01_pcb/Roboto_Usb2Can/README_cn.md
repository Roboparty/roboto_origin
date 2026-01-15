# Roboto_Usb2Can_V1.0 - USB 转 4路 CAN 通讯模块

**[中文](./README_cn.md)** | [English](./README.md)

![Manufacturer](https://img.shields.io/badge/Manufacturer-RoboParty-blue)
![Hardware](https://img.shields.io/badge/Hardware-V1.0-green)
![OS](https://img.shields.io/badge/OS-Linux_Only-orange)

![Render](00_Docs/Images/render.png)

## 📖 概述 (Overview)

**Roboto_Usb2Can_V1.0** 是由 **萝博派对 (RoboParty)** 研发的高集成度通讯转换设备。

该模块主要用于机器人控制系统，能够将上位机（如 Orange Pi 5 Plus 等 Linux 设备）的 USB 信号转换为 **4 路独立的 CAN 总线信号**，从而实现对伺服电机等执行机构的高效控制。

> ⚠️ **注意：** 本设备目前仅支持在 **Linux 系统** 下识别和使用。

## 📂 仓库目录 (File Structure)

本仓库包含制造和使用该模块所需的所有文件：

- **PCB 制造文件 (Gerber):** `01_Gerber/`
- **BOM 与坐标文件:** `02_Assembly/`
- **固件文件:** `03_Firmware/`
- **详细说明文档:** `00_Docs/`

## 🔌 接口定义 (Interfaces)

### 1. 硬件接口布局
![Interfaces](00_Docs/Images/interface_top.png)

| 编号 | 名称 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| **①/④** | **固定孔** | 机械 | 标准安装孔位 |
| **②** | **烧录指示灯** | LED | 对应背部 4 路烧录状态 |
| **③** | **USB-C** | 数据/电源 | 连接上位机 (Linux)，输入电压 5V |
| **④** | **CAN 接口** | GH1.25 | 4路 CAN 总线输出 (4L, 4H, 3L, 3H...) |

### 2. 执行机构连接推荐
为方便管理，建议按照以下对应关系连接机器人关节：

- **CAN 1:** 左腿 (含腰关节)
- **CAN 2:** 右腿
- **CAN 3:** 左臂
- **CAN 4:** 右臂

## 🛠️ 烧录指南 (Firmware Flashing)

本设备背面预留了 4 组 SWD 烧录接口，需要使用 **1.25mm 烧录针** 配合 ST-Link 或类似工具进行烧录。

![Flashing Ports](00_Docs/Images/interface_bottom.png)

### 烧录步骤：
1. **固件下载：** 获取 [RoboParty-HubUsb2Can_1.00.bin](03_Firmware/RoboParty-HubUsb2Can_1.00.bin)。
2. **连接烧录器：** 接口定义从左至右依次为：`GND`, `3V3`, `CLK`, `DIO`。
3. **顺序烧录：** 请依次对背面的 **4 个烧录口** 进行固件烧录（每个接口对应一路 CAN 控制芯片）。

## ⚠️ 注意事项 (Precautions)

1.  **电源安全：**
    * 仅限 **5V** 供电，严禁接入更高电压。
    * 请勿反向暴力插拔 Type-C 线缆。
2.  **终端电阻：**
    * CAN 总线规范要求每路总线末端需接 **120Ω 终端电阻**。
    * 如果您的伺服电机内部已集成电阻，则无需额外添加；否则必须在末端并联电阻。
3.  **环境要求：**
    * 避免接触水、导电灰尘或金属碎屑，防止短路。

## 🏭 制造文件下载 (Downloads)

如果您需要自行制造该模块，请下载以下文件：

* **Gerber 文件 (打板):** [Roboto_Usb2Can_V1.0.rar](01_Gerber/Roboto_Usb2Can_V1.0.rar)
* **物料清单 (BOM):** [HUB-USB2CAN_Bom.xlsx](02_Assembly/HUB-USB2CAN_Bom.xlsx)
* **贴片坐标 (CPL):** [PickAndPlace.xlsx](02_Assembly/PickAndPlace_HUB_USBTOCAN_V1_0.xlsx)

---
**技术支持:** 如遇到技术问题，请联系 RoboParty 技术团队。