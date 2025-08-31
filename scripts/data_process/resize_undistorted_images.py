#!/usr/bin/env python3
"""
对已去畸变图像进行缩放处理的脚本
专门用于处理EgoHumans数据集中已经去畸变的图像，将其缩放到指定分辨率

Usage:
    python resize_undistorted_images.py --root_dir /path/to/EgoHumans --scale_factor 2.0
    python resize_undistorted_images.py --root_dir /path/to/EgoHumans --scale_factor 2.0 --parallel 4
"""

import os
import sys
import argparse
import cv2
import logging
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resize_undistorted_images.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def find_sequences(root_dir):
    """查找所有数据序列目录"""
    sequences = []
    
    # 只查找符合数据命名规范的目录
    data_categories = ['01_tagging', '02_lego', '03_fencing', '04_basketball', 
                      '05_volleyball', '06_badminton', '07_tennis']
    
    for category in os.listdir(root_dir):
        # 只处理数据类别目录
        if category not in data_categories:
            continue
            
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            # 查找类别下的所有子序列
            for subseq in os.listdir(category_path):
                subseq_path = os.path.join(category_path, subseq)
                if os.path.isdir(subseq_path):
                    # 验证子序列目录包含exo或ego目录
                    has_exo = os.path.exists(os.path.join(subseq_path, 'exo'))
                    has_ego = os.path.exists(os.path.join(subseq_path, 'ego'))
                    
                    if has_exo or has_ego:
                        sequences.append({
                            'name': f"{category}/{subseq}",
                            'path': subseq_path,
                            'category': category,
                            'subseq': subseq
                        })
    
    return sorted(sequences, key=lambda x: x['name'])


def find_undistorted_directories_in_sequence(seq_path, seq_name):
    """在指定序列中查找所有包含去畸变图像的目录"""
    undistorted_dirs = []
    
    # 查找exo相机目录
    exo_path = os.path.join(seq_path, 'exo')
    if os.path.exists(exo_path):
        for cam_name in os.listdir(exo_path):
            cam_path = os.path.join(exo_path, cam_name)
            if os.path.isdir(cam_path):
                undistorted_path = os.path.join(cam_path, 'undistorted_images')
                if os.path.exists(undistorted_path):
                    # 检查是否包含jpg文件
                    try:
                        if any(f.endswith('.jpg') for f in os.listdir(undistorted_path)):
                            undistorted_dirs.append({
                                'type': 'exo',
                                'path': undistorted_path,
                                'camera_name': cam_name,
                                'sequence': seq_name
                            })
                    except (OSError, PermissionError):
                        logger.warning(f"无法访问目录: {undistorted_path}")
    
    # 查找ego相机目录
    ego_path = os.path.join(seq_path, 'ego')
    if os.path.exists(ego_path):
        for aria_name in os.listdir(ego_path):
            aria_path = os.path.join(ego_path, aria_name)
            if os.path.isdir(aria_path):
                for session in os.listdir(aria_path):
                    session_path = os.path.join(aria_path, session)
                    if os.path.isdir(session_path):
                        undistorted_path = os.path.join(session_path, 'undistorted_rgb')
                        if os.path.exists(undistorted_path):
                            try:
                                if any(f.endswith('.jpg') for f in os.listdir(undistorted_path)):
                                    undistorted_dirs.append({
                                        'type': 'ego',
                                        'path': undistorted_path,
                                        'camera_name': f"{aria_name}_{session}",
                                        'sequence': seq_name
                                    })
                            except (OSError, PermissionError):
                                logger.warning(f"无法访问目录: {undistorted_path}")
    
    return sorted(undistorted_dirs, key=lambda x: (x['type'], x['camera_name']))


def process_sequence(seq_info, scale_factor, target_resolution, camera_types, parallel_workers=1):
    """处理单个序列中的所有去畸变目录"""
    seq_name = seq_info['name']
    seq_path = seq_info['path']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"开始处理序列: {seq_name}")
    logger.info(f"序列路径: {seq_path}")
    
    # 查找该序列中的所有去畸变目录
    undistorted_dirs = find_undistorted_directories_in_sequence(seq_path, seq_name)
    
    if not undistorted_dirs:
        logger.warning(f"序列 {seq_name} 中未找到任何去畸变图像目录")
        return []
    
    # 过滤相机类型
    filtered_dirs = [d for d in undistorted_dirs if d['type'] in camera_types]
    
    if not filtered_dirs:
        logger.warning(f"序列 {seq_name} 中未找到指定类型的相机目录")
        return []
    
    # 按相机类型分组显示
    exo_cameras = [d for d in filtered_dirs if d['type'] == 'exo']
    ego_cameras = [d for d in filtered_dirs if d['type'] == 'ego']
    
    logger.info(f"找到 {len(filtered_dirs)} 个去畸变目录:")
    if exo_cameras:
        logger.info(f"  - EXO相机 ({len(exo_cameras)}个): {', '.join([d['camera_name'] for d in exo_cameras])}")
    if ego_cameras:
        logger.info(f"  - EGO相机 ({len(ego_cameras)}个): {', '.join([d['camera_name'] for d in ego_cameras])}")
    
    # 处理目录
    start_time = time.time()
    
    if parallel_workers > 1 and len(filtered_dirs) > 1:
        # 并行处理
        max_workers = min(parallel_workers, len(filtered_dirs), multiprocessing.cpu_count())
        logger.info(f"使用 {max_workers} 个进程并行处理")
        results = process_directories_parallel(filtered_dirs, scale_factor, max_workers, target_resolution)
    else:
        # 顺序处理
        logger.info("顺序处理目录")
        results = process_directories_sequential(filtered_dirs, scale_factor, target_resolution)
    
    # 统计序列结果
    seq_time = time.time() - start_time
    successful = sum(1 for _, _, success in results if success)
    failed = len(results) - successful
    total_images = sum(count for _, count, success in results if success)
    
    logger.info(f"\n序列 {seq_name} 处理完成:")
    logger.info(f"  耗时: {seq_time:.2f}秒")
    logger.info(f"  成功目录: {successful}/{len(results)}")
    logger.info(f"  失败目录: {failed}")
    logger.info(f"  处理图像数: {total_images}")
    
    if failed > 0:
        logger.warning(f"  失败的目录:")
        for dir_info, _, success in results:
            if not success:
                logger.warning(f"    - {dir_info['camera_name']} ({dir_info['type']})")
    
    return results


def resize_images_in_directory(dir_info, scale_factor, target_resolution=None):
    """对目录中的所有图像进行缩放"""
    input_dir = dir_info['path']
    camera_type = dir_info['type']
    camera_name = dir_info['camera_name']
    sequence = dir_info['sequence']
    
    # 创建输出目录
    parent_dir = os.path.dirname(input_dir)
    if scale_factor != 1.0:
        output_dir_name = f"undistorted_{'rgb' if camera_type == 'ego' else 'images'}_scale{scale_factor}"
    else:
        output_dir_name = f"undistorted_{'rgb' if camera_type == 'ego' else 'images'}_resized"
    
    output_dir = os.path.join(parent_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    if not image_files:
        logger.warning(f"目录 {input_dir} 中没有找到jpg图像")
        return 0
    
    processed_count = 0
    logger.info(f"开始处理 {sequence}/{camera_name} ({camera_type})，共 {len(image_files)} 张图像")
    
    # 读取第一张图像以确定目标尺寸
    first_image_path = os.path.join(input_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        logger.error(f"无法读取图像: {first_image_path}")
        return 0
    
    h, w = first_image.shape[:2]
    
    # 计算目标尺寸
    if target_resolution:
        new_w, new_h = target_resolution
    else:
        new_w, new_h = int(w // scale_factor), int(h // scale_factor)
    
    logger.info(f"原始尺寸: {w}x{h}, 目标尺寸: {new_w}x{new_h}")
    
    # 处理所有图像
    for image_file in tqdm(image_files, desc=f"缩放 {camera_name}", leave=False):
        try:
            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, image_file)
            
            # 如果输出文件已存在，跳过
            if os.path.exists(output_path):
                processed_count += 1
                continue
            
            # 读取、缩放、保存图像
            image = cv2.imread(input_path)
            if image is not None:
                resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(output_path, resized_image)
                processed_count += 1
            else:
                logger.error(f"无法读取图像: {input_path}")
                
        except Exception as e:
            logger.error(f"处理图像 {input_path} 时发生错误: {e}")
    
    logger.info(f"完成 {sequence}/{camera_name}，处理了 {processed_count}/{len(image_files)} 张图像")
    return processed_count


def process_directory_worker(args):
    """工作进程函数"""
    dir_info, scale_factor, target_resolution = args
    return resize_images_in_directory(dir_info, scale_factor, target_resolution)


def process_directories_parallel(undistorted_dirs, scale_factor, max_workers, target_resolution=None):
    """并行处理多个目录"""
    results = []
    
    # 准备参数
    worker_args = [(dir_info, scale_factor, target_resolution) for dir_info in undistorted_dirs]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_dir = {
            executor.submit(process_directory_worker, args): args[0] 
            for args in worker_args
        }
        
        # 收集结果
        for future in as_completed(future_to_dir):
            dir_info = future_to_dir[future]
            
            try:
                processed_count = future.result()
                results.append((dir_info, processed_count, True))
                
            except Exception as e:
                logger.error(f"处理目录 {dir_info['path']} 时发生异常: {e}")
                results.append((dir_info, 0, False))
    
    return results


def process_directories_sequential(undistorted_dirs, scale_factor, target_resolution=None):
    """顺序处理多个目录"""
    results = []
    
    for i, dir_info in enumerate(undistorted_dirs, 1):
        logger.info(f"处理进度: {i}/{len(undistorted_dirs)} - {dir_info['sequence']}/{dir_info['camera_name']}")
        
        try:
            processed_count = resize_images_in_directory(dir_info, scale_factor, target_resolution)
            results.append((dir_info, processed_count, True))
            
        except Exception as e:
            logger.error(f"处理目录 {dir_info['path']} 时发生异常: {e}")
            results.append((dir_info, 0, False))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='对已去畸变图像进行缩放处理')
    parser.add_argument('--root_dir', type=str, default="/gemini/user/private/3D/data/EgoHumans",
                       help='EgoHumans数据集根目录路径')
    parser.add_argument('--scale_factor', type=float, default=2.0,
                       help='缩放因子 (默认: 2.0, 即缩小到1/2)')
    parser.add_argument('--target_resolution', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                       help='目标分辨率，例如: --target_resolution 1920 1080')
    parser.add_argument('--parallel', type=int, default=1,
                       help='并行处理的进程数（默认为1，顺序处理）')
    parser.add_argument('--sequences', type=str, nargs='+',
                       help='指定要处理的序列名称（可选）')
    parser.add_argument('--camera_types', type=str, nargs='+', choices=['exo', 'ego'],
                       default='exo', help='指定要处理的相机类型')
    parser.add_argument('--dry_run', action='store_true',
                       help='模拟运行，只显示将要处理的目录')
    
    args = parser.parse_args()
    
    # 验证根目录
    if not os.path.exists(args.root_dir):
        logger.error(f"根目录不存在: {args.root_dir}")
        sys.exit(1)
    
    # 处理目标分辨率
    target_resolution = None
    if args.target_resolution:
        target_resolution = tuple(args.target_resolution)
        logger.info(f"使用目标分辨率: {target_resolution[0]}x{target_resolution[1]}")
    else:
        logger.info(f"使用缩放因子: {args.scale_factor}")
    
    logger.info(f"开始查找序列目录")
    logger.info(f"根目录: {args.root_dir}")
    logger.info(f"并行度: {args.parallel}")
    logger.info(f"处理相机类型: {args.camera_types}")
    
    # 查找所有序列
    all_sequences = find_sequences(args.root_dir)
    
    if not all_sequences:
        logger.error("未找到任何序列目录")
        sys.exit(1)
    
    # 过滤序列
    if args.sequences:
        filtered_sequences = [seq for seq in all_sequences if seq['name'] in args.sequences]
    else:
        filtered_sequences = all_sequences
    
    if not filtered_sequences:
        logger.error("根据过滤条件未找到任何序列")
        sys.exit(1)
    
    logger.info(f"找到 {len(filtered_sequences)} 个序列待处理:")
    for seq in filtered_sequences:
        logger.info(f"  - {seq['name']}")
    
    # 模拟运行 - 显示每个序列的详细信息
    if args.dry_run:
        logger.info("\n=== 模拟运行模式 ===")
        for seq in filtered_sequences:
            undistorted_dirs = find_undistorted_directories_in_sequence(seq['path'], seq['name'])
            filtered_dirs = [d for d in undistorted_dirs if d['type'] in args.camera_types]
            
            exo_cameras = [d for d in filtered_dirs if d['type'] == 'exo']
            ego_cameras = [d for d in filtered_dirs if d['type'] == 'ego']
            
            logger.info(f"\n序列 {seq['name']}:")
            logger.info(f"  路径: {seq['path']}")
            if exo_cameras:
                logger.info(f"  EXO相机 ({len(exo_cameras)}个): {', '.join([d['camera_name'] for d in exo_cameras])}")
            if ego_cameras:
                logger.info(f"  EGO相机 ({len(ego_cameras)}个): {', '.join([d['camera_name'] for d in ego_cameras])}")
            if not filtered_dirs:
                logger.info(f"  无匹配的相机目录")
        
        logger.info("\n模拟运行完成，实际不执行处理")
        return
    
    # 确认处理
    resolution_info = f"目标分辨率: {target_resolution[0]}x{target_resolution[1]}" if target_resolution else f"缩放因子: {args.scale_factor}"
    print(f"\n将要处理 {len(filtered_sequences)} 个序列")
    print(f"处理方式: {resolution_info}")
    print(f"相机类型: {', '.join(args.camera_types)}")
    if input("是否继续? (y/N): ").lower() != 'y':
        logger.info("用户取消处理")
        return
    
    # 开始逐序列处理
    start_time = time.time()
    all_results = []
    total_sequences_processed = 0
    total_sequences_failed = 0
    
    for i, seq in enumerate(filtered_sequences, 1):
        logger.info(f"\n{'*'*80}")
        logger.info(f"处理序列进度: {i}/{len(filtered_sequences)}")
        
        try:
            seq_results = process_sequence(
                seq, args.scale_factor, target_resolution, 
                args.camera_types, args.parallel
            )
            all_results.extend(seq_results)
            total_sequences_processed += 1
            
        except Exception as e:
            logger.error(f"处理序列 {seq['name']} 时发生异常: {e}")
            total_sequences_failed += 1
    
    # 统计总体结果
    total_time = time.time() - start_time
    successful_dirs = sum(1 for _, _, success in all_results if success)
    failed_dirs = len(all_results) - successful_dirs
    total_images = sum(count for _, count, success in all_results if success)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"批量缩放处理完成!")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"成功处理序列: {total_sequences_processed}/{len(filtered_sequences)}")
    logger.info(f"失败序列: {total_sequences_failed}")
    logger.info(f"成功处理目录: {successful_dirs}/{len(all_results)}")
    logger.info(f"失败目录: {failed_dirs}")
    logger.info(f"总处理图像数: {total_images}")
    
    if failed_dirs > 0:
        logger.info(f"\n失败的目录:")
        for dir_info, _, success in all_results:
            if not success:
                logger.info(f"  - {dir_info['sequence']}/{dir_info['camera_name']} ({dir_info['type']})")
    
    logger.info(f"\n详细日志已保存到: resize_undistorted_images.log")

if __name__ == "__main__":
    main() 