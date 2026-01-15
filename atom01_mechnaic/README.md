# ROBOTO

[English](./README.md) | **[🇨🇳 中文](./README_cn.md)**

![Hardware](https://img.shields.io/badge/Hardware-v1.0-blue)
![Docs](https://img.shields.io/badge/Guide-Beginner_Friendly-green)
![License](https://img.shields.io/badge/License-MIT-orange)

> **Project Introduction:** This is an open-source bipedal robot project. We are dedicated to lowering the barrier to entry for assembly by providing comprehensive, "beginner-friendly" tutorials covering everything from mechanical structure to circuit connections.

---

## 📚 Core Documentation

We have prepared detailed PDF manuals for beginners. Please **be sure to read them in the following order**:

| Order | Document Name | Description | Location |
| :---: | :--- | :--- | :--- |
| 1️⃣ | **[Assembly SOP](00_Docs/)** | Detailed robot manufacturing process | `00_Docs/` |
| 2️⃣ | **[Mechanical Assembly Guide](00_Docs/Assembly_Guide_v1.14.pdf)** | Detailed steps for assembling the mechanical structure | `00_Docs/` |


---

## 🛠️ Assembly Roadmap


### Phase 1: Preparation
- [ ] **Tools Preparation:** - [ ] **Inventory/BOM Check:** - [ ] **PCB Ordering:** ### Phase 2: Mechanical Assembly
> See `Assembly_Guide_v1.14.pdf` for details
- [ ] **Leg Assembly:**
- [ ] **Arm Assembly:** - [ ] **Torso Integration:** ### Phase 3: Electronics & Wiring
> ⚠️ **CRITICAL WARNING: Always check polarity (positive/negative) before powering on!**
- [ ] **Power Board Installation:** - [ ] **Wiring Layout:** - [ ] **Communication Connections:** ## 📂 Project File Structure

This repository uses a modular structure, organized as follows:
```text
├── 00_docs/                         # [Docs] Core Manuals
│   ├── BOM_Mechanical.xlsx          # Bill of Materials
│   ├── Assembly_Guide_v1.14.pdf     # Assembly Guide
│   └── Standard Operating Procedure.pdf # SOP
│
├── 01_SW_Project/                   # [Source Files] Mechanical Design Projects
│   
├── 02_Fabrication/                  # [Fabrication] Production Files
│   
└── 03_URDF/                         # [Simulation] Robot Description Files