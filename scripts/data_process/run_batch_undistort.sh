#!/bin/bash

# 批量鱼眼畸变校正快速启动脚本
# 使用方法: ./run_batch_undistort.sh [mode] [parallel] [sequences...]

set -e

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
DEFAULT_ROOT_DIR="/gemini/user/private/3D/data/EgoHumans"
DEFAULT_MODE="exo"
DEFAULT_PARALLEL=1

# 解析命令行参数
MODE=${1:-$DEFAULT_MODE}
PARALLEL=${2:-$DEFAULT_PARALLEL}
SEQUENCES="${@:3}"  # 从第3个参数开始的所有参数

echo "=========================================="
echo "EgoHumans 批量鱼眼畸变校正处理"
echo "=========================================="
echo "根目录: $DEFAULT_ROOT_DIR"
echo "处理模式: $MODE"
echo "并行度: $PARALLEL"

if [ -n "$SEQUENCES" ]; then
    echo "指定序列: $SEQUENCES"
fi

echo "=========================================="

# 检查Python脚本是否存在
if [ ! -f "batch_undistort.py" ]; then
    echo "错误: 未找到 batch_undistort.py 脚本"
    echo "请确保在正确的目录中运行此脚本"
    exit 1
fi

if [ ! -f "standalone_undistort.py" ]; then
    echo "错误: 未找到 standalone_undistort.py 脚本"
    echo "请确保 standalone_undistort.py 在同一目录中"
    exit 1
fi

# 检查根目录是否存在
if [ ! -d "$DEFAULT_ROOT_DIR" ]; then
    echo "错误: 根目录不存在: $DEFAULT_ROOT_DIR"
    echo "请检查路径是否正确"
    exit 1
fi

# 构建命令
CMD="python batch_undistort.py --root_dir \"$DEFAULT_ROOT_DIR\" --mode $MODE --parallel $PARALLEL"

if [ -n "$SEQUENCES" ]; then
    CMD="$CMD --sequences $SEQUENCES"
fi

echo "即将执行: $CMD"
echo ""

# 询问用户确认
read -p "是否继续执行? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "用户取消操作"
    exit 1
fi

echo "开始处理..."
echo "=========================================="

# 执行命令
eval $CMD

echo "=========================================="
echo "处理完成!"
echo "详细日志已保存到: batch_undistort.log"
echo "==========================================" 