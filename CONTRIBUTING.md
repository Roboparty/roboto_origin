# Contributing to ROBOTO_ORIGIN

Thank you for your interest in contributing to ROBOTO_ORIGIN! This document provides guidelines and instructions for contributing to the project.

## Important Note About Repository Structure

**The `roboto_origin` repository is a snapshot-only repository.**

This repository serves as a daily-updated aggregation snapshot of all sub-repositories. It is designed to provide users with a complete, ready-to-use codebase without requiring additional submodule initialization.

### What This Means for Contributors

- **DO NOT** submit pull requests or issues to `roboto_origin`
- **DO** submit contributions to the specific sub-repository where your changes belong
- The main repository will automatically update its snapshot from sub-repositories

## How to Contribute

### 1. Identify the Correct Sub-Repository

Review the module descriptions below to determine which sub-repository your contribution should target:

| Sub-Repository                                                            | Purpose                    | Contribution Topics                                                                      |
| ------------------------------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------- |
| **[Atom01_hardware](https://github.com/Roboparty/Atom01_hardware)**       | Hardware design files      | Mechanical structures, CAD drawings, PCB designs, BOM improvements                       |
| **[atom01_deploy](https://github.com/Roboparty/atom01_deploy)**           | ROS2 deployment framework  | Driver development, middleware modules, deployment configurations, IMU/motor integration |
| **[atom01_train](https://github.com/Roboparty/atom01_train)**             | IsaacLab training workflow | RL algorithms, training environments, simulation configs, Sim2Sim transfer               |
| **[atom01_description](https://github.com/Roboparty/atom01_description)** | URDF robot models          | Kinematic/dynamic descriptions, visual/collision meshes, joint parameters                |

### 2. Fork and Clone the Target Sub-Repository

```bash
# Fork the sub-repository on GitHub, then clone it
git clone https://github.com/YOUR_USERNAME/<sub-repo-name>.git
cd <sub-repo-name>

# Add upstream remote
git remote add upstream https://github.com/Roboparty/<sub-repo-name>.git
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-fix-name
```

### 4. Make Your Changes

- Write clean, documented code following the repository's existing style
- Add tests if applicable
- Update documentation as needed
- Commit your changes with clear, descriptive messages

### 5. Submit a Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a pull request on the sub-repository's GitHub page.

## Contribution Guidelines

### Code Quality

- Follow existing code style and conventions in each sub-repository
- Write meaningful commit messages
- Add comments for complex logic
- Test your changes thoroughly before submitting

### Documentation

- Update relevant documentation when making changes
- Include usage examples for new features
- Document API changes in code comments

### Issue Reporting

When reporting bugs or requesting features:

1. Navigate to the appropriate sub-repository's Issues tab
2. Search existing issues to avoid duplicates
3. Use clear, descriptive titles
4. Provide detailed information:
   - Environment details (OS, software versions)
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Relevant logs or screenshots

### License

All contributions are subject to the GPLv3 license. By contributing, you agree that your contributions will be licensed under the same terms as the project.

## Development Workflow

### For Users

If you want to use or build upon the ROBOTO_ORIGIN project:

1. Clone this repository:
   ```bash
   git clone https://github.com/Roboparty/roboto_origin.git
   ```

2. All code is immediately available - no submodule initialization needed

3. Navigate to individual modules in `modules/` directory

4. Follow README instructions in each module

### For Maintainers

The `sync_subtrees.sh` script is used to update the snapshot:

```bash
./sync_subtrees.sh
```

This script:
- Pulls latest changes from all sub-repositories
- Updates subtree snapshots in `modules/` directory
- Flattens all submodule code into trackable files
- Commits the updated snapshot

The script runs daily to keep the main repository synchronized.

## Community Guidelines

- Be respectful and constructive in all interactions
- Welcome new contributors and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

For detailed community guidelines, please refer to our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Help

- **QQ Group:** 1078670917
- **Email:** zhangbaoxin@roboparty.com
- **GitHub Issues:** Post in the appropriate sub-repository

## Recognition

Contributors who make significant improvements will be recognized in the project documentation. Thank you for helping make ROBOTO_ORIGIN better!

---

**Remember:** All contributions must be made to the specific sub-repositories, not to this main snapshot repository.
