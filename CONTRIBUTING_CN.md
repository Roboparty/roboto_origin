# 贡献指南

感谢您对萝博头原型机项目的关注！本文档提供了贡献项目的相关指南和说明。

## 重要说明：仓库结构

**`roboto_origin` 主仓库是一个快照仓库。**

本仓库作为所有子仓库的每日聚合快照，旨在为用户提供一个完整的、开箱即用的代码库，无需额外的子模块初始化操作。

### 对贡献者意味着什么

- **不要**向 `roboto_origin` 提交 Pull Request 或 Issue
- **请**将贡献提交到对应的子仓库
- 主仓库会自动从子仓库更新快照

## 如何贡献

### 1. 确定正确的子仓库

查看下表中的模块描述，确定您的贡献应该针对哪个子仓库：

| 子仓库                                                                    | 用途               | 贡献内容                                      |
| ------------------------------------------------------------------------- | ------------------ | --------------------------------------------- |
| **[Atom01_hardware](https://github.com/Roboparty/Atom01_hardware)**       | 硬件设计文件       | 机械结构、CAD图纸、PCB设计、BOM改进           |
| **[atom01_deploy](https://github.com/Roboparty/atom01_deploy)**           | ROS2部署框架       | 驱动开发、中间件模块、部署配置、IMU/电机集成  |
| **[atom01_train](https://github.com/Roboparty/atom01_train)**             | IsaacLab训练工作流 | 强化学习算法、训练环境、仿真配置、Sim2Sim迁移 |
| **[atom01_description](https://github.com/Roboparty/atom01_description)** | URDF机器人模型     | 运动学/动力学描述、视觉/碰撞网格、关节参数    |

### 2. Fork 并克隆目标子仓库

```bash
# 在 GitHub 上 fork 子仓库，然后克隆它
git clone https://github.com/YOUR_USERNAME/<sub-repo-name>.git
cd <sub-repo-name>

# 添加上游远程仓库
git remote add upstream https://github.com/Roboparty/<sub-repo-name>.git
```

### 3. 创建功能分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/bug-fix-name
```

### 4. 进行修改

- 编写清晰的、符合仓库现有风格的代码
- 如适用，添加测试
- 更新相关文档
- 提交清晰的、描述性的 commit 信息

### 5. 提交 Pull Request

```bash
git push origin feature/your-feature-name
```

然后在子仓库的 GitHub 页面上打开 Pull Request。

## 贡献指南

### 代码质量

- 遵循各子仓库中现有的代码风格和约定
- 编写有意义的 commit 信息
- 为复杂逻辑添加注释
- 提交前彻底测试您的更改

### 文档

- 修改时更新相关文档
- 为新功能包含使用示例
- 在代码注释中记录 API 更改

### 问题报告

报告 Bug 或请求功能时：

1. 导航到相应子仓库的 Issues 标签页
2. 搜索现有 issue 以避免重复
3. 使用清晰、描述性的标题
4. 提供详细信息：
   - 环境详情（操作系统、软件版本）
   - 复现步骤（针对 Bug）
   - 预期行为 vs 实际行为
   - 相关日志或截图

### 许可证

所有贡献均受 GPLv3 许可证约束。通过贡献，您同意您的贡献将按照与项目相同的条款进行许可。

## 开发工作流程

### 对于使用者

如果您想使用或基于 ROBOTO_ORIGIN 项目进行开发：

1. 克隆本仓库：
   ```bash
   git clone https://github.com/Roboparty/roboto_origin.git
   ```

2. 所有代码立即可用 - 无需初始化子模块

3. 导航到 `modules/` 目录中的各个模块

4. 按照各模块中的 README 说明操作

### 对于维护者

使用 `sync_subtrees.sh` 脚本更新快照：

```bash
./sync_subtrees.sh
```

该脚本会：
- 从所有子仓库拉取最新更改
- 更新 `modules/` 目录中的子树快照
- 将所有子模块代码展平为可追踪文件
- 提交更新的快照

脚本每日运行以保持主仓库同步。

## 社区准则

- 在所有互动中保持尊重和建设性
- 欢迎新贡献者并帮助他们入门
- 关注什么对社区最有利
- 对其他社区成员表示同理心

详细的社区准则请参阅我们的[行为准则](CODE_OF_CONDUCT_CN.md)。

## 获取帮助

- **QQ群：** 1078670917
- **邮箱：** zhangbaoxin@roboparty.com
- **GitHub Issues：** 在相应的子仓库中提问

## 致谢

做出重大改进的贡献者将在项目文档中获得认可。感谢您帮助改进 ROBOTO_ORIGIN！

---

**请记住：** 所有贡献必须提交到相应的子仓库，而不是这个主快照仓库。
