# Roboto_Power_V1.0 - Robot Core Power Distribution Board

[English](./README.md) | **[🇨🇳 中文](./README_cn.md)**

![Manufacturer](https://img.shields.io/badge/Manufacturer-RoboParty-blue)
![Hardware](https://img.shields.io/badge/Hardware-V1.0-green)
![Voltage](https://img.shields.io/badge/Max_Voltage-48V-red)

![Render](00_Docs/Images/power_render.png)

## 📖 Overview

**Roboto_Power_v1.0** is the core component of the Roboto prototype's electrical system.

Acting as the robot's energy hub, it is responsible for distributing the main battery power to various motors and control modules, implementing a "centralized control & distributed power" architecture. This power distribution board (PDB) works in conjunction with the following modules to form the complete electrical system:
* Host Computer (Orange Pi 5 Plus)
* 48V to 5V Buck Converter Module
* USB-to-CAN Communication Board

## 📂 Repository Structure

This repository contains all files required to manufacture this power board:

- **PCB Manufacturing Files (Gerber):** `01_Gerber/`
- **BOM & Coordinate Files:** `02_Assembly/`
- **Detailed Documentation:** `00_Docs/`

## 🔌 Interface Definitions

### 1. Top Layer Layout
![Interfaces Top](00_Docs/Images/power_interface_top.png)

| No. | Interface Type | Description | Notes |
| :--- | :--- | :--- | :--- |
| **①** | **Main Power Input** | XT60/XT90 (Male) | Connects to the main battery |
| **②** | **Power Split Output** | XT30 (Female) | Power supply for joint motors (6 channels) |
| **③** | **CAN Signal Hub** | GH1.25 | 4-channel CAN bus signal distribution (Top) |

### 2. Bottom Layer Layout
![Interfaces Bottom](00_Docs/Images/power_interface_bottom.png)

| No. | Interface Type | Description |
| :--- | :--- | :--- |
| **④** | **CAN Signal Extension** | GH1.25 Connector, used for extending more CAN nodes or connecting the USB-to-CAN module |

> **Note:** This board integrates a CAN bus hub function to simplify wiring layouts.

## ⚠️ Critical Precautions

> 🛑 **Please read carefully before operation! Incorrect operation will result in equipment damage!**

1.  **Power Polarity:**
    * **PAY ATTENTION TO POSITIVE AND NEGATIVE POLARITY!!**
    * **PAY ATTENTION TO POSITIVE AND NEGATIVE POLARITY!!**
    * **PAY ATTENTION TO POSITIVE AND NEGATIVE POLARITY!!**
    * Before connecting the battery, be sure to use a multimeter to check for short circuits at the input terminals.

2.  **CAN Wiring Sequence:**
    * Strictly check the silkscreen markings `H` (High) and `L` (Low) on the board.
    * Connect CAN_H to CAN_H, and CAN_L to CAN_L. **Do not reverse them**, otherwise communication will fail.

## 🏭 Manufacturing Downloads

If you need to manufacture this module yourself, please download the following files:

* **Gerber Files (PCB Fab):** [POWER_BOARD_GERBER_V1.0.zip](01_Gerber/POWER_BOARD_GERBER_V1.0.zip)
* **Bill of Materials (BOM):** [BOM_POWER_BOARD-V1.0.xlsx](02_Assembly/BOM_POWER_BOARD-V1.0.xlsx)
* **Pick & Place Coordinates (CPL):** [PickAndPlace_POWER_BOARD_V1_0.xlsx](02_Assembly/PickAndPlace_POWER_BOARD_V1_0.xlsx)

---
**Tech Support:** If you encounter technical issues, please contact the RoboParty technical team.