This is the central hardware repository for the **ROBOTO Bipedal Robot**. We aim to lower the barrier to entry by providing a full-stack open-source solution, from mechanical structures to core circuitry.

**[🇨🇳 中文文档点这里](./README_cn.md)**

This repository consists of three core modules. Please click the links below for detailed documentation:

### 📂 Core Modules Index

| Module Name | Description | Key Features | Documentation |
| :--- | :--- | :--- | :--- |
| **🤖 Robot Body**<br>(Mechanical) | Structure & Assembly | • Beginner-friendly Guide<br>• URDF Simulation Files | [View Docs](./atom01_mechnaic/README.md) |
| **⚡ Power Board**<br>(Distribution) | Energy Hub | • 48V Max Input<br>• XT30/XT60 Connectors<br>• Centralized Power | [View Docs](./atom01_pcb/Roboto_Power/README.md) |
| **📡 Comm Module**<br>(USB-to-CAN) | Signal Gateway | • USB to 4-Ch CAN<br>• Linux Only<br>• 120Ω Termination | [View Docs](./atom01_pcb/Roboto_Usb2Can/README.md) |

### 🚀 Quick Start Roadmap

1.  **Preparation**: Download BOMs for each module, purchase parts, and order PCBs.
2.  **Mechanical Build**: Follow the `Assembly_Guide` in the **Robot Body** folder.
3.  **Electronics Integration**:
    * Manufacture and install the **Power Board**.
    * Manufacture and flash firmware for the **USB-to-CAN Module**.
4.  **Wiring & Debug**: Connect the battery following the Power Board guide (⚠️ Check Polarity!), and connect to the host computer via the Comm Module.

---

## 📂 Repository Structure

```text
├── Roboto_Mechanical/       # [Main] Robot structure, URDF, Assembly Guides
├── Roboto_Power/            # [Circuit] Power Distribution Board (48V)
├── Roboto_Usb2Can/          # [Circuit] Communication Module (USB -> 4xCAN)