#!/bin/bash

# 定义 CAN 接口名称和波特率
CAN_SEND_IFACE="can0"
CAN_RECV_IFACE="can1"
CAN_BAUDRATE="1000000"

# 定义测试参数
# 报文发送频率 (msgs/sec)
CAN_GEN_RATE="2000"
CAN_GEN_INTERVAL=$(awk "BEGIN {printf \"%.1f\", 1000 / $CAN_GEN_RATE}") # 毫秒间隔
# 报文发送持续时间 (秒)
TEST_DURATION="10"

# --- 检查和配置 CAN 接口 ---
echo "--- 正在检查并配置 CAN 接口 $CAN_SEND_IFACE ---"
if ! ip link show $CAN_SEND_IFACE &> /dev/null; then
    echo "错误：找不到 CAN 接口 $CAN_SEND_IFACE。请检查硬件连接和驱动程序。"
    exit 1
fi
sudo ip link set $CAN_SEND_IFACE down
sudo ip link set $CAN_SEND_IFACE type can bitrate $CAN_BAUDRATE
sudo ip link set $CAN_SEND_IFACE up
sudo ip link set $CAN_SEND_IFACE txqueuelen 1000

if ! ip link show $CAN_SEND_IFACE | grep -q "state UP"; then
    echo "错误：无法启动 CAN 接口 $CAN_SEND_IFACE。请检查配置。"
    exit 1
fi
echo "CAN 接口 $CAN_SEND_IFACE 已成功启动，波特率为 $CAN_BAUDRATE。"

echo "--- 正在检查并配置 CAN 接口 $CAN_RECV_IFACE ---"
if ! ip link show $CAN_RECV_IFACE &> /dev/null; then
    echo "错误：找不到 CAN 接口 $CAN_RECV_IFACE。请检查硬件连接和驱动程序。"
    exit 1
fi
sudo ip link set $CAN_RECV_IFACE down
sudo ip link set $CAN_RECV_IFACE type can bitrate $CAN_BAUDRATE
sudo ip link set $CAN_RECV_IFACE up
sudo ip link set $CAN_RECV_IFACE txqueuelen 1000

if ! ip link show $CAN_RECV_IFACE | grep -q "state UP"; then
    echo "错误：无法启动 CAN 接口 $CAN_RECV_IFACE。请检查配置。"
    exit 1
fi
echo "CAN 接口 $CAN_RECV_IFACE 已成功启动，波特率为 $CAN_BAUDRATE。"

# --- 开始高频率通信测试 ---
echo "--- 正在开始高频率 CAN 报文生成 ---"
TOTAL_SENT=$((CAN_GEN_RATE * TEST_DURATION))
echo "频率: $CAN_GEN_RATE 报文/秒, 持续时间: $TEST_DURATION 秒, 计划发送总数: $TOTAL_SENT"

# 使用 timeout 限制 cangen 的运行时间
cangen $CAN_SEND_IFACE -g $CAN_GEN_INTERVAL -i -I 100-200 -L 8 -D i -n $TOTAL_SENT > /dev/null &
CAN_GEN_PID=$!
echo "cangen 进程 PID: $CAN_GEN_PID"
echo "同时监听 CAN 报文..."

# candump 输出保存到文件
CANDUMP_LOG="candump_$(date +%Y%m%d_%H%M%S).log"
CANDUMP_DURATION=$((TEST_DURATION + 2))
stdbuf -oL timeout ${CANDUMP_DURATION}s candump $CAN_RECV_IFACE > "$CANDUMP_LOG" &
CANDUMP_PID=$!

# 等待 cangen 和 candump 进程结束
wait $CAN_GEN_PID
wait $CANDUMP_PID

echo "--- 测试完成 ---"

# --- 结果分析与丢包率计算 ---
echo "--- 结果分析 ---"

if [ -f "$CANDUMP_LOG" ]; then
    # 使用 wc -l 统计接收到的报文行数
    TOTAL_RECV=$(wc -l < "$CANDUMP_LOG")
else
    TOTAL_RECV=0
fi

# 确保接收数不大于发送数
if [ "$TOTAL_RECV" -gt "$TOTAL_SENT" ]; then
    TOTAL_RECV=$TOTAL_SENT
fi

PACKET_LOSS=$((TOTAL_SENT - TOTAL_RECV))

# 使用 awk 进行浮点数计算以得到百分比
if [ $TOTAL_SENT -gt 0 ]; then
    LOSS_RATE=$(awk "BEGIN {printf \"%.2f\", $PACKET_LOSS / $TOTAL_SENT * 100}")
else
    LOSS_RATE="0.00"
fi

echo "日志文件: $CANDUMP_LOG"
echo "理论发送总数: $TOTAL_SENT"
echo "实际接收总数: $TOTAL_RECV"
echo "丢失报文数: $PACKET_LOSS"
echo "丢包率: $LOSS_RATE %"