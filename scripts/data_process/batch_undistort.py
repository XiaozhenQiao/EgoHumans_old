#!/usr/bin/env python3
"""
批量鱼眼畸变校正脚本
调用standalone_undistort.py对整个EgoHumans数据集进行去畸变处理

Usage:
    python batch_undistort.py --root_dir /path/to/EgoHumans --mode exo
    python batch_undistort.py --root_dir /path/to/EgoHumans --mode all --parallel 4
"""

import os
import sys
import subprocess
import argparse
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_undistort.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def find_all_sequences(root_dir):
    """递归查找所有序列目录"""
    sequences = []
    
    for root, dirs, files in os.walk(root_dir):
        # 检查当前目录是否是一个序列目录
        # 序列目录包含：exo, ego, colmap等子目录
        if all(subdir in dirs for subdir in ['exo', 'colmap']):
            # 确保不是根目录本身
            if root != root_dir:
                sequences.append(root)
    
    return sorted(sequences)


def get_sequences_by_pattern(root_dir, patterns=None):
    """根据模式获取序列目录"""
    sequences = []
    
    # 如果没有指定模式，查找所有序列
    if not patterns:
        return find_all_sequences(root_dir)
    
    # 根据模式查找
    for pattern in patterns:
        pattern_path = os.path.join(root_dir, pattern)
        if os.path.isdir(pattern_path):
            # 检查是否是序列目录
            if all(os.path.exists(os.path.join(pattern_path, subdir)) 
                   for subdir in ['exo', 'colmap']):
                sequences.append(pattern_path)
            else:
                # 可能是类别目录，查找其下的序列
                for item in os.listdir(pattern_path):
                    item_path = os.path.join(pattern_path, item)
                    if (os.path.isdir(item_path) and 
                        all(os.path.exists(os.path.join(item_path, subdir)) 
                            for subdir in ['exo', 'colmap'])):
                        sequences.append(item_path)
    
    return sorted(sequences)


def check_sequence_validity(sequence_path):
    """检查序列是否有效且适合去畸变处理"""
    try:
        # 检查必要目录
        required_dirs = ['exo', 'colmap']
        for dir_name in required_dirs:
            if not os.path.exists(os.path.join(sequence_path, dir_name)):
                return False, f"缺少目录: {dir_name}"
        
        # 检查colmap相机参数文件
        cameras_txt = os.path.join(sequence_path, 'colmap', 'workplace', 'cameras.txt')
        if not os.path.exists(cameras_txt):
            return False, "缺少colmap相机参数文件"
        
        # 检查exo相机数据
        exo_dir = os.path.join(sequence_path, 'exo')
        cam_dirs = [d for d in os.listdir(exo_dir) 
                   if d.startswith('cam') and os.path.isdir(os.path.join(exo_dir, d))]
        
        if not cam_dirs:
            return False, "未找到exo相机目录"
        
        # 检查第一个相机是否有图像
        first_cam = sorted(cam_dirs)[0]
        images_dir = os.path.join(exo_dir, first_cam, 'images')
        if not os.path.exists(images_dir):
            return False, f"未找到图像目录: {first_cam}/images"
        
        images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        if len(images) == 0:
            return False, f"图像目录为空: {first_cam}/images"
        
        # 检查ego数据（如果存在）
        ego_dir = os.path.join(sequence_path, 'ego')
        ego_cameras = 0
        if os.path.exists(ego_dir):
            aria_dirs = [d for d in os.listdir(ego_dir) 
                        if d.startswith('aria') and os.path.isdir(os.path.join(ego_dir, d))]
            ego_cameras = len(aria_dirs)
        
        return True, f"序列有效，包含{len(cam_dirs)}个exo相机，{ego_cameras}个ego相机，{len(images)}帧图像"
        
    except Exception as e:
        return False, f"检查序列时发生错误: {e}"


def process_single_sequence(sequence_path, mode, max_frames=None, scale_factor=1.0):
    """处理单个序列的去畸变"""
    sequence_name = os.path.basename(sequence_path)
    
    logger.info(f"开始处理序列: {sequence_name}")
    logger.info(f"序列路径: {sequence_path}")
    
    # 检查序列有效性
    is_valid, message = check_sequence_validity(sequence_path)
    if not is_valid:
        logger.error(f"序列 {sequence_name} 无效: {message}")
        return False, f"序列无效: {message}"
    
    logger.info(f"序列 {sequence_name} 检查通过: {message}")
    
    # 获取standalone_undistort.py脚本路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'standalone_undistort.py')
    
    if not os.path.exists(script_path):
        logger.error(f"未找到standalone_undistort.py脚本: {script_path}")
        return False, "未找到处理脚本"
    
    # 构建命令
    cmd = [
        sys.executable,
        script_path,
        '--sequence_path', sequence_path,
        '--mode', mode
    ]
    
    if max_frames:
        cmd.extend(['--max_frames', str(max_frames)])
    
    if scale_factor != 1.0:
        cmd.extend(['--scale_factor', str(scale_factor)])
    
    # 执行处理
    start_time = time.time()
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=7200  # 2小时超时
        )
        
        processing_time = time.time() - start_time
        logger.info(f"序列 {sequence_name} 处理完成，耗时: {processing_time:.2f}秒")
        
        # 检查输出
        if result.stdout:
            logger.info(f"处理输出: {result.stdout}")
        
        # 验证输出文件
        if verify_undistort_output(sequence_path, mode, scale_factor):
            logger.info(f"序列 {sequence_name} 去畸变输出验证成功")
            return True, f"处理成功，耗时{processing_time:.2f}秒"
        else:
            logger.warning(f"序列 {sequence_name} 输出验证失败，但处理可能已完成")
            return True, f"处理完成，耗时{processing_time:.2f}秒（验证警告）"
            
    except subprocess.TimeoutExpired:
        logger.error(f"序列 {sequence_name} 处理超时")
        return False, "处理超时"
    except subprocess.CalledProcessError as e:
        logger.error(f"序列 {sequence_name} 处理失败: {e}")
        if e.stdout:
            logger.error(f"标准输出: {e.stdout}")
        if e.stderr:
            logger.error(f"错误输出: {e.stderr}")
        return False, f"处理失败: {e}"
    except Exception as e:
        logger.error(f"序列 {sequence_name} 发生未知错误: {e}")
        return False, f"未知错误: {e}"


def verify_undistort_output(sequence_path, mode, scale_factor=1.0):
    """验证去畸变输出文件是否正确生成"""
    try:
        verification_count = 0
        
        # 检查exo相机去畸变输出
        if mode in ['all', 'exo']:
            exo_dir = os.path.join(sequence_path, 'exo')
            if os.path.exists(exo_dir):
                cam_dirs = [d for d in os.listdir(exo_dir) 
                           if d.startswith('cam') and os.path.isdir(os.path.join(exo_dir, d))]
                for cam_dir in cam_dirs:
                    if scale_factor != 1.0:
                        undistort_dir = os.path.join(exo_dir, cam_dir, f'undistorted_images_scale{scale_factor}')
                    else:
                        undistort_dir = os.path.join(exo_dir, cam_dir, 'undistorted_images')
                    
                    if os.path.exists(undistort_dir):
                        undistorted_images = [f for f in os.listdir(undistort_dir) if f.endswith('.jpg')]
                        if undistorted_images:
                            verification_count += 1
                            logger.info(f"找到 {cam_dir} 的 {len(undistorted_images)} 张去畸变图像")
        
        # 检查ego相机去畸变输出
        if mode in ['all', 'ego']:
            ego_dir = os.path.join(sequence_path, 'ego')
            if os.path.exists(ego_dir):
                aria_dirs = [d for d in os.listdir(ego_dir) 
                            if d.startswith('aria') and os.path.isdir(os.path.join(ego_dir, d))]
                for aria_dir in aria_dirs:
                    if scale_factor != 1.0:
                        undistort_dir = os.path.join(ego_dir, aria_dir, 'images', f'undistorted_rgb_scale{scale_factor}')
                    else:
                        undistort_dir = os.path.join(ego_dir, aria_dir, 'images', 'undistorted_rgb')
                    
                    if os.path.exists(undistort_dir):
                        undistorted_images = [f for f in os.listdir(undistort_dir) if f.endswith('.jpg')]
                        if undistorted_images:
                            verification_count += 1
                            logger.info(f"找到 {aria_dir} 的 {len(undistorted_images)} 张去畸变图像")
        
        return verification_count > 0
        
    except Exception as e:
        logger.error(f"验证输出时发生错误: {e}")
        return False


def process_sequences_parallel(sequences, mode, max_frames, max_workers, scale_factor=1.0):
    """并行处理多个序列"""
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_sequence = {
            executor.submit(process_single_sequence, seq, mode, max_frames, scale_factor): seq 
            for seq in sequences
        }
        
        # 收集结果
        for future in as_completed(future_to_sequence):
            sequence = future_to_sequence[future]
            sequence_name = os.path.basename(sequence)
            
            try:
                success, message = future.result()
                results.append((sequence_name, success, message))
                
                if success:
                    logger.info(f"✓ {sequence_name}: {message}")
                else:
                    logger.error(f"✗ {sequence_name}: {message}")
                    
            except Exception as e:
                logger.error(f"✗ {sequence_name}: 处理异常 - {e}")
                results.append((sequence_name, False, f"处理异常: {e}"))
    
    return results


def process_sequences_sequential(sequences, mode, max_frames, scale_factor=1.0):
    """顺序处理多个序列"""
    results = []
    
    for i, sequence in enumerate(sequences, 1):
        sequence_name = os.path.basename(sequence)
        logger.info(f"处理进度: {i}/{len(sequences)} - {sequence_name}")
        
        try:
            success, message = process_single_sequence(sequence, mode, max_frames, scale_factor)
            results.append((sequence_name, success, message))
            
            if success:
                logger.info(f"✓ {sequence_name}: {message}")
            else:
                logger.error(f"✗ {sequence_name}: {message}")
                
        except Exception as e:
            logger.error(f"✗ {sequence_name}: 处理异常 - {e}")
            results.append((sequence_name, False, f"处理异常: {e}"))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='批量鱼眼畸变校正处理')
    parser.add_argument('--root_dir', type=str, 
                       default='/gemini/user/private/3D/data/EgoHumans',
                       help='EgoHumans数据集根目录路径')
    parser.add_argument('--sequences', type=str, nargs='+',
                       help='指定要处理的序列名称或模式（可选）')
    parser.add_argument('--mode', default='exo', choices=['all', 'ego', 'exo'],
                       help='处理模式: all, ego, or exo')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='每个序列最大处理帧数（用于测试）')
    parser.add_argument('--parallel', type=int, default=1,
                       help='并行处理的进程数（默认为1，顺序处理）')
    parser.add_argument('--scale_factor', type=float, default=2.0,
                       help='图像缩放因子，用于降低分辨率 (默认: 1.0, 即不缩放)')
    parser.add_argument('--dry_run', action='store_true',
                       help='模拟运行，只显示将要处理的序列')
    
    args = parser.parse_args()
    
    # 验证根目录
    if not os.path.exists(args.root_dir):
        logger.error(f"根目录不存在: {args.root_dir}")
        sys.exit(1)
    
    logger.info(f"开始批量去畸变处理")
    logger.info(f"根目录: {args.root_dir}")
    logger.info(f"模式: {args.mode}")
    logger.info(f"并行度: {args.parallel}")
    logger.info(f"缩放因子: {args.scale_factor}")
    
    if args.max_frames:
        logger.info(f"最大帧数限制: {args.max_frames}")
    
    # 获取要处理的序列
    sequences = get_sequences_by_pattern(args.root_dir, args.sequences)
    
    if not sequences:
        logger.error("未找到任何序列目录")
        sys.exit(1)
    
    logger.info(f"找到 {len(sequences)} 个序列:")
    for seq in sequences:
        logger.info(f"  - {os.path.basename(seq)}")
    
    # 模拟运行
    if args.dry_run:
        logger.info("模拟运行模式，实际不执行处理")
        return
    
    # 确认处理
    scale_info = f", 缩放因子: {args.scale_factor}" if args.scale_factor != 1.0 else ""
    print(f"\n将要处理 {len(sequences)} 个序列，模式: {args.mode}{scale_info}")
    if input("是否继续? (y/N): ").lower() != 'y':
        logger.info("用户取消处理")
        return
    
    # 开始处理
    start_time = time.time()
    
    if args.parallel > 1:
        # 并行处理
        max_workers = min(args.parallel, len(sequences), multiprocessing.cpu_count())
        logger.info(f"使用 {max_workers} 个进程并行处理")
        results = process_sequences_parallel(sequences, args.mode, args.max_frames, max_workers, args.scale_factor)
    else:
        # 顺序处理
        logger.info("顺序处理序列")
        results = process_sequences_sequential(sequences, args.mode, args.max_frames, args.scale_factor)
    
    # 统计结果
    total_time = time.time() - start_time
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    logger.info(f"\n" + "="*60)
    logger.info(f"批量处理完成!")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"成功: {successful}/{len(results)}")
    logger.info(f"失败: {failed}/{len(results)}")
    
    if failed > 0:
        logger.info(f"\n失败的序列:")
        for name, success, message in results:
            if not success:
                logger.info(f"  - {name}: {message}")
    
    logger.info(f"详细日志已保存到: batch_undistort.log")


if __name__ == "__main__":
    main() 