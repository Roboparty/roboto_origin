# Hipnuc IMU ROS 2 Driver

这是一个用于 Hipnuc IMU 的 ROS 2 驱动程序包 (ROS 2 Humble/Foxy)。它支持串口 (Serial) 和 CAN 总线接口，并发布标准的 ROS 传感器消息。

## 功能特点

- **多接口支持**：支持串口 (UART) 和 CAN 总线通信（支持 J1939 协议解析）。
- **标准消息**：发布标准的 `sensor_msgs/Imu` 等消息。
- **丰富的数据**：不仅包含 IMU 数据，还支持磁力计、欧拉角、温度、气压、GPS 等数据的发布（具体取决于设备和协议）。
- **参数配置**：通过 YAML 配置文件轻松配置端口、波特率和发布频率。

## 安装与构建

1. **克隆仓库**

    将此仓库克隆到你的 ROS 2 工作空间的 `src` 目录下：

    ```bash
    cd ~/ros2_ws/src
    git clone <repository_url> hipnuc_imu
    ```

2. **构建**

    回到工作空间根目录并编译：

    ```bash
    cd ~/ros2_ws
    colcon build --packages-select hipnuc_imu hipnuc_lib_package
    ```

3. **配置环境**

    ```bash
    source install/setup.bash
    ```

## 配置

配置文件位于 `src/hipnuc_imu/config/hipnuc_config.yaml`。你可以根据你的硬件连接修改以下参数：

```yaml
imu_node:
    ros__parameters:
        # 协议选择: "serial" 或 "can"
        protocol: "can"

        # 串口配置（protocol="serial" 时使用）
        serial_port: "/dev/ttyUSB0"
        baud_rate: 115200

        # CAN 配置（protocol="can" 时使用）
        can_interface: "can4"

        # 通用配置
        frame_id: "base_link"
        imu_topic: "/IMU_data"
        publish_rate: 500  # 发布频率 (Hz)
```

## 运行

使用提供的 launch 文件启动驱动节点：

```bash
ros2 launch hipnuc_imu imu_spec_msg.launch.py
```

## 发布的话题 (Published Topics)

| 话题名称 | 消息类型 | 描述 | 仅 CAN 模式 |
| :--- | :--- | :--- | :---: |
| `/IMU_data` | `sensor_msgs/msg/Imu` | 加速度和角速度数据 | 否 |
| `/imu/mag` | `sensor_msgs/msg/MagneticField` | 磁力计数据 | 是 |
| `/imu/euler` | `geometry_msgs/msg/Vector3Stamped` | 欧拉角 (Roll, Pitch, Yaw) | 是 |
| `/imu/temperature` | `sensor_msgs/msg/Temperature` | 内部温度 | 是 |
| `/imu/pressure` | `sensor_msgs/msg/FluidPressure` | 气压数据 | 是 |
| `/imu/gps/fix` | `sensor_msgs/msg/NavSatFix` | GPS 定位数据 | 是 |
| `/imu/gps/velocity` | `geometry_msgs/msg/TwistStamped` | GPS 速度数据 | 是 |

> **注意**: 串口模式目前主要发布 `/IMU_data` 话题。CAN 模式下支持更多传感器数据的解析和发布。

## 硬件设置与故障排除 (CAN J1939)

如果你使用的是 CAN 接口 (J1939 协议) 且遇到通信问题，请参考以下步骤进行设备配置。

### 修改波特率与节点 ID

以下指南假设你的设备 ID 为 `08` (默认可能是其它值，请根据实际抓包确认)。

1. **确认设备状态**：使用 `candump` 查看数据。如果数据 ID 以 `0CFF` 开头（例如 `0CFF3708`），则末尾两位 `08` 为设备源地址。

2. **修改波特率命令** (示例：将设备从 500k 改为 1M)

    确保你的 CAN 接口当前配置为设备能通信的波特率 (例如 500k)，然后发送以下指令：

    - **步骤 1：发送“修改波特率为 1M”指令**

        - ID: `0CEF0800` (发送给 08，来自 00)
        - 数据: `9A 00 06 00 00 00 00 00` (地址 009A, 写命令 06, 数据 0 = 1000K)

        ```bash
        cansend can4 0CEF0800#9A00060000000000
        ```

    - **步骤 2：发送“保存配置”指令** (至关重要)

        - 数据: `00 00 06 00 00 00 00 00` (地址 0000, 写命令 06, 数据 0 = 保存)

        ```bash
        cansend can4 0CEF0800#0000060000000000
        ```

    - **步骤 3：发送“复位”指令** (重启生效)

        - 数据: `00 00 06 00 FF 00 00 00` (地址 0000, 写命令 06, 数据 FF = 复位)

        ```bash
        cansend can4 0CEF0800#00000600FF000000
        ```

3. **验证修改**

    复位后，你需要将主机的 CAN 接口波特率也修改为 1M，才能再次接收到数据：

    ```bash
    sudo ip link set can4 down
    sudo ip link set can4 up type can bitrate 1000000
    candump can4
    ```

    如果这时候看到熟悉的数据 `0CFF3708...` 再次刷屏，恭喜你，修改成功！之后你就可以把 IMU 的线并联到电机所在的任意接口上了。

```python
#!/usr/bin/env python3
import imu_py
import time

def example_serial_imu():
    """使用串口连接的IMU示例"""
    print("=== 串口IMU示例 ===")
    try:
        imu = imu_py.IMUDriver.create_imu(
            imu_id=1,
            interface_type="serial",
            interface="/dev/ttyUSB0",
            imu_type="HIPNUC",
            baudrate=921600
        )
    except Exception as e:
        print(f"创建IMU失败: {e}")
        return
    
    print(f"IMU ID: {imu.get_imu_id()}")
    
    for i in range(1000):
        quat = imu.get_quat()
        print(f"四元数: w={quat[0]:.4f}, x={quat[1]:.4f}, y={quat[2]:.4f}, z={quat[3]:.4f}")
        
        ang_vel = imu.get_ang_vel()
        print(f"角速度: x={ang_vel[0]:.4f}, y={ang_vel[1]:.4f}, z={ang_vel[2]:.4f} rad/s")
        
        lin_acc = imu.get_lin_acc()
        print(f"线加速度: x={lin_acc[0]:.4f}, y={lin_acc[1]:.4f}, z={lin_acc[2]:.4f} m/s^2")
        
        temp = imu.get_temperature()
        print(f"温度: {temp:.2f}°C")
        
        print("-" * 50)
        time.sleep(0.01)

if __name__ == "__main__":
    example_serial_imu()
```
