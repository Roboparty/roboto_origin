#!/bin/bash

# 颜色定义，用于美化输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

# 函数：打印成功消息
print_success() {
    echo -e "${GREEN}$1${NC}"
}

# 函数：打印提示消息
print_info() {
    echo -e "${YELLOW}$1${NC}"
}

# 函数：打印错误消息
print_error() {
    echo -e "${RED}$1${NC}"
}

# 函数：启动组件并检查（先启动ROS节点，再设置实时优先级）
start_component() {
    local session_name=$1
    local launch_cmd=$2
    local node_name=$3
    local sleep_time=$4

    print_info "启动 $session_name ..."
    # 在screen会话中启动ROS命令，并确保传递DDS配置环境变量
    screen -dmS $session_name bash -c "export RMW_IMPLEMENTATION='$RMW_IMPLEMENTATION'; export RMW_FASTRTPS_USE_QOS_FROM_XML='$RMW_FASTRTPS_USE_QOS_FROM_XML'; export FASTRTPS_DEFAULT_PROFILES_FILE='$FASTRTPS_DEFAULT_PROFILES_FILE'; $launch_cmd; exec bash"
    sleep $sleep_time

    if ! ros2 node list | grep -q "$node_name"; then
        print_error "$session_name 启动失败！未检测到 $node_name 节点。"
        cleanup_sessions
        exit 1
    fi
}

# 函数：清理所有会话
cleanup_sessions() {
    screen -S imu_session -X quit 2>/dev/null
    screen -S motor_session -X quit 2>/dev/null
    screen -S inference_session -X quit 2>/dev/null
    screen -S joy_session -X quit 2>/dev/null
}

# 函数：详细验证 DDS 配置是否生效
verify_dds_effectiveness() {
    print_info "详细验证 DDS 配置是否生效..."
    sleep 2
    
    # 1. 检查环境变量
    print_info "检查环境变量..."
    echo "RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
    echo "FASTRTPS_DEFAULT_PROFILES_FILE: $FASTRTPS_DEFAULT_PROFILES_FILE"
    
    # 2. 验证配置文件是否被读取
    print_info "验证配置文件读取..."
    if [ -f "$FASTRTPS_DEFAULT_PROFILES_FILE" ]; then
        print_success "配置文件存在"
        
        # 检查XML语法
        if command -v xmllint &> /dev/null; then
            if xmllint --noout "$FASTRTPS_DEFAULT_PROFILES_FILE" 2>/dev/null; then
                print_success "XML 格式正确"
            else
                print_error "XML 格式错误"
                xmllint "$FASTRTPS_DEFAULT_PROFILES_FILE"
                return 1
            fi
        fi
    else
        print_error "配置文件不存在: $FASTRTPS_DEFAULT_PROFILES_FILE"
        return 1
    fi
    
    # 3. 检查进程是否使用了 Fast DDS
    print_info "检查进程 DDS 实现..."
    for node in "imu_node" "motors_node" "inference_node" "joy_node"; do
        local pid=$(pgrep -x "$node" 2>/dev/null)
        if [ -n "$pid" ]; then
            # 检查进程环境变量
            local env_file="/proc/$pid/environ"
            if [ -f "$env_file" ]; then
                if grep -z "FASTRTPS_DEFAULT_PROFILES_FILE" "$env_file" >/dev/null 2>&1; then
                    print_success "$node 环境变量设置正确"
                else
                    print_error "$node 缺少 FASTRTPS_DEFAULT_PROFILES_FILE 环境变量"
                fi
                
                if grep -z "RMW_IMPLEMENTATION=rmw_fastrtps_cpp" "$env_file" >/dev/null 2>&1; then
                    print_success "$node RMW 实现正确"
                else
                    print_error "$node RMW 实现不正确"
                fi
            fi
        fi
    done
    
    # 4. 检查共享内存传输
    print_info "检查共享内存传输..."
    local shm_files=$(ls /dev/shm/ 2>/dev/null | grep -E "(fastrtps|fast_dds|rmw)" | wc -l)
    if [ "$shm_files" -gt 0 ]; then
        print_success "共享内存传输活跃 ($shm_files 个文件)"
    else
        print_error "共享内存传输未检测到"
    fi
    
    # 5. 测试 DDS 发现性能
    print_info "测试 DDS 发现性能..."
    local start_time=$(date +%s%3N)
    ros2 node list >/dev/null 2>&1
    local end_time=$(date +%s%3N)
    local discovery_time=$((end_time - start_time))
    
    if [ "$discovery_time" -lt 500 ]; then
        print_success "DDS 发现延迟: ${discovery_time}ms (优秀)"
    elif [ "$discovery_time" -lt 1000 ]; then
        print_info "DDS 发现延迟: ${discovery_time}ms (良好)"
    else
        print_error "DDS 发现延迟: ${discovery_time}ms (较慢)"
    fi
}

# 切换到脚本目录
cd "$(dirname "$0")"

# 设置 DDS 配置文件
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
export FASTRTPS_DEFAULT_PROFILES_FILE="$(pwd)/rt_fastdds_profile.xml"
print_info "设置 DDS 配置文件: $FASTRTPS_DEFAULT_PROFILES_FILE"

# 检查 DDS 配置文件是否存在
if [ ! -f "$FASTRTPS_DEFAULT_PROFILES_FILE" ]; then
    print_error "DDS 配置文件不存在: $FASTRTPS_DEFAULT_PROFILES_FILE"
    exit 1
fi

# 检查 colcon 和 ros2
if ! command -v colcon &> /dev/null; then
    print_error "colcon 未安装，请安装 ROS 2 开发工具"
    exit 1
fi
if ! command -v ros2 &> /dev/null; then
    print_error "ros2 未安装"
    exit 1
fi

# 检查是否已安装screen
if ! command -v screen &> /dev/null; then
    print_error "screen 未安装"
    exit 1
fi

# 检查是否已source setup文件
if [ -z "$AMENT_PREFIX_PATH" ]; then
    print_info "未检测到ROS 2环境，正在执行source..."
    source /opt/ros/humble/setup.bash || {
        print_error "无法source /opt/ros/humble/setup.bash，请检查路径是否正确"
        exit 1
    }
fi

# 编译IMU包
print_info "编译IMU包..."
cd imu || {
    print_error "找不到imu目录"
    exit 1
}
colcon build --symlink-install|| {
    print_error "IMU包编译失败"
    exit 1
}
source install/setup.bash
cd ..

# 编译电机包
print_info "编译电机包..."
cd motors || {
    print_error "找不到motors目录"
    exit 1
}
colcon build --symlink-install|| {
    print_error "电机包编译失败"
    exit 1
}
source install/setup.bash
cd ..

# 编译推理包
print_info "编译推理包..."
cd inference || {
    print_error "找不到inference目录"
    exit 1
}
colcon build --symlink-install|| {
    print_error "推理包编译失败"
    exit 1
}
source install/setup.bash
cd ..

# 停止可能正在运行的screen会话
print_info "停止现有相关screen会话..."
cleanup_sessions

start_component "imu_session" "ros2 launch hipnuc_imu imu_spec_msg.launch.py" "imu_node" 2
start_component "motor_session" "ros2 launch motors motors_spec_msg.launch.py" "motors_node" 2
start_component "inference_session" "ros2 launch inference inference.launch.py" "inference_node" 2
start_component "joy_session" "ros2 run joy joy_node" "joy_node" 2

# 验证节点的 DDS 配置
verify_dds_effectiveness

# 所有组件启动完成
print_success "----------------------------------------"
print_success "所有组件已在后台成功启动！"
print_success "使用以下命令查看各组件输出："
print_success "IMU: screen -r imu_session"
print_success "电机: screen -r motor_session"
print_success "推理模块: screen -r inference_session"
print_success "手柄控制: screen -r joy_session"
print_success "----------------------------------------"
print_info "若要退出某个screen会话，按Ctrl+A然后按D"
print_info "使用以下命令停止所有组件："
print_info "screen -S imu_session -X quit"
print_info "screen -S motor_session -X quit"
print_info "screen -S inference_session -X quit"
print_info "screen -S joy_session -X quit"
print_success "----------------------------------------"
print_info "按下X使能/失能电机"
print_info "按下A复位电机"
print_info "按下B开始/暂停推理"
print_info "按下Y切换控制模式"
