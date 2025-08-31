#!/usr/bin/env python3
"""
测试独立去畸变脚本
"""

import os
import sys
import subprocess

def test_undistort():
    """测试去畸变脚本"""
    
    # 数据路径
    sequence_path = "/gemini/user/private/3D/data/EgoHumans/01_tagging/004_tagging"
    
    # 检查数据路径是否存在
    if not os.path.exists(sequence_path):
        print(f"错误: 数据路径不存在: {sequence_path}")
        return False
    
    print(f"测试序列: {sequence_path}")
    
    # 脚本路径
    script_path = "/gemini/user/private/3D/data/EgoHumans/scripts/data_process/standalone_undistort.py"
    
    # 构建命令
    cmd = [
        "python", script_path,
        "--sequence_path", sequence_path,
        "--mode", "exo",
        "--max_frames", "5"  # 只处理前5帧进行测试
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        # 运行脚本
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("标准输出:")
        print(result.stdout)
        
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✓ 脚本运行成功!")
            return True
        else:
            print(f"✗ 脚本运行失败，退出码: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ 脚本运行超时")
        return False
    except Exception as e:
        print(f"✗ 运行脚本时出错: {e}")
        return False

if __name__ == "__main__":
    success = test_undistort()
    sys.exit(0 if success else 1) 