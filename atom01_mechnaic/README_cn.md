# ROBOTO

**[中文](./README_cn.md)** | [English](./README.md)

![Hardware](https://img.shields.io/badge/Hardware-v1.0-blue)
![Docs](https://img.shields.io/badge/Guide-Beginner_Friendly-green)
![License](https://img.shields.io/badge/License-MIT-orange)

> **项目简介：** 本项目是一个开源的双足机器人。本项目致力于降低组装门槛，提供了从机械结构到电路连接的“保姆级”装配教程。

---

## 📚 核心文档导航 (Documentation)

我们为初学者准备了详细的 PDF 手册，请**务必按照以下顺序阅读**：

| 顺序 | 文档名称 | 内容说明 | 存放位置 |
| :---: | :--- | :--- | :--- |
| 1️⃣ | **[装配作业指导书](00_Docs/)** | 详细机器人的制造流程 | `00_Docs/` |
| 2️⃣ | **[机械装配指导书 ](00_Docs/Assembly_Guide_v1.14.pdf)** | 机械结构的详细组装步骤 | `00_Docs/` |


---

## 🛠️ 装配路线图 (Assembly Roadmap)


### 第一阶段：准备工作 (Preparation)
- [ ] **工具准备：** 
- [ ] **物料清点：** 
- [ ] **PCB 打样：** 

### 第二阶段：机械组装 (Mechanical)
> 详见 `Assembly_Guide_v1.14.pdf`
- [ ] **腿部总成：**
- [ ] **手臂总成：** 
- [ ] **躯干集成：** 

### 第三阶段：电子与接线 (Electronics & Wiring)
> ⚠️ **高能预警：上电前务必测量正负极！**
- [ ] **电源板安装：** 
- [ ] **走线布局：** 
- [ ] **通讯连接：** 


## 📂 项目文件架构 (File Structure)

本仓库采用模块化结构，文件组织如下：
```text
├── 00_docs/                         # [文档] 核心说明书
│   ├── BOM_Mechanical.xlsx          # 采购清单
│   ├── Assembly_Guide_v1.14.pdf     # 组装教程
│   └── Standard Operating Procedure.pdf # SOP 标准作业书
│
├── 01_SW_Project/                   # [源文件] 机械设计工程
│   
├── 02_Fabrication/                  # [制造] 生产加工文件
│   
└── 03_URDF/                         # [仿真] 机器人描述文件
    