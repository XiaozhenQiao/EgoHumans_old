#!/usr/bin/env python3
"""
批量去畸变处理配置示例
可以复制此文件并修改参数来定制处理流程
"""

import os
import subprocess
import sys

# =============================================================================
# 配置参数
# =============================================================================

# 数据集根目录
ROOT_DIR = "/gemini/user/private/3D/data/EgoHumans"

# 处理配置
PROCESSING_CONFIG = {
    # 处理模式: 'all', 'ego', 'exo'
    'mode': 'exo',
    
    # 并行进程数（建议设置为CPU核心数的50-80%）
    'parallel': 4,
    
    # 最大处理帧数（None表示处理所有帧，用于测试时可设置较小值）
    'max_frames': None,
    
    # 是否执行模拟运行（只显示将处理的序列，不实际处理）
    'dry_run': False,
}

# 序列选择配置
SEQUENCE_CONFIG = {
    # 要处理的序列类别（None表示处理所有）
    'categories': None,  # 例如: ['01_tagging', '02_lego']
    
    # 要处理的具体序列（None表示处理所有）
    'specific_sequences': None,  # 例如: ['01_tagging/001_tagging', '02_lego/001_lego']
    
    # 要排除的序列类别
    'exclude_categories': None,  # 例如: ['07_tennis']
    
    # 要排除的具体序列
    'exclude_sequences': None,  # 例如: ['01_tagging/014_tagging']
}

# =============================================================================
# 预定义配置模板
# =============================================================================

def get_test_config():
    """测试配置：处理少量序列和帧数"""
    return {
        'mode': 'exo',
        'parallel': 1,
        'max_frames': 10,
        'dry_run': False,
        'specific_sequences': ['01_tagging/001_tagging']
    }

def get_exo_only_config():
    """仅处理exo相机配置"""
    return {
        'mode': 'exo',
        'parallel': 4,
        'max_frames': None,
        'dry_run': False,
    }

def get_ego_only_config():
    """仅处理ego相机配置"""
    return {
        'mode': 'ego',
        'parallel': 2,  # ego相机处理通常更耗内存
        'max_frames': None,
        'dry_run': False,
    }

def get_full_config():
    """完整处理配置：所有相机"""
    return {
        'mode': 'all',
        'parallel': 3,
        'max_frames': None,
        'dry_run': False,
    }

def get_category_config(categories):
    """处理特定类别的配置"""
    return {
        'mode': 'exo',
        'parallel': 4,
        'max_frames': None,
        'dry_run': False,
        'categories': categories
    }

# =============================================================================
# 执行函数
# =============================================================================

def build_command(config):
    """根据配置构建命令"""
    script_path = os.path.join(os.path.dirname(__file__), 'batch_undistort.py')
    
    cmd = [
        sys.executable,
        script_path,
        '--root_dir', ROOT_DIR,
        '--mode', config.get('mode', 'exo'),
        '--parallel', str(config.get('parallel', 1))
    ]
    
    if config.get('max_frames'):
        cmd.extend(['--max_frames', str(config['max_frames'])])
    
    if config.get('dry_run'):
        cmd.append('--dry_run')
    
    # 处理序列选择
    sequences = []
    if config.get('categories'):
        sequences.extend(config['categories'])
    
    if config.get('specific_sequences'):
        sequences.extend(config['specific_sequences'])
    
    if sequences:
        cmd.extend(['--sequences'] + sequences)
    
    return cmd

def run_processing(config):
    """执行处理"""
    cmd = build_command(config)
    
    print("即将执行命令:")
    print(' '.join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("处理完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"处理失败: {e}")
        return False

# =============================================================================
# 使用示例
# =============================================================================

def main():
    """主函数 - 选择并执行配置"""
    
    # 可选的配置
    configs = {
        '1': ('测试配置 (处理1个序列的10帧)', get_test_config()),
        '2': ('仅处理exo相机', get_exo_only_config()),
        '3': ('仅处理ego相机', get_ego_only_config()),
        '4': ('处理所有相机', get_full_config()),
        '5': ('处理tagging和lego类别', get_category_config(['01_tagging', '02_lego'])),
        '6': ('自定义配置', PROCESSING_CONFIG),
    }
    
    print("批量去畸变处理配置选择:")
    print("=" * 50)
    
    for key, (desc, _) in configs.items():
        print(f"{key}. {desc}")
    
    print("=" * 50)
    
    choice = input("请选择配置 (1-6): ").strip()
    
    if choice not in configs:
        print("无效选择")
        return
    
    desc, config = configs[choice]
    print(f"\n选择的配置: {desc}")
    print("配置详情:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print()
    confirm = input("确认执行? (y/N): ").strip().lower()
    
    if confirm == 'y':
        run_processing(config)
    else:
        print("取消执行")

if __name__ == "__main__":
    main() 