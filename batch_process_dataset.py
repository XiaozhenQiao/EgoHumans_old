#!/usr/bin/env python3
"""
批处理脚本：对整个EgoHumans数据集进行数据整理
使用merge_temporal_data.py处理所有序列
"""

import os
import subprocess
import sys
import time
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import yaml


def load_config(sequence_path, category, sequence_name):
    """加载序列对应的config文件"""
    # 构建config文件路径
    config_dir = os.path.join(os.path.dirname(__file__), 'egohumans', 'configs')
    category_name = category.split('_', 1)[1]  # 从 "06_badminton" 提取 "badminton"
    config_file = os.path.join(config_dir, category_name, f"{sequence_name}.yaml")
    
    if not os.path.exists(config_file):
        return None, f"配置文件不存在: {config_file}"
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config, "配置文件加载成功"
    except Exception as e:
        return None, f"配置文件加载失败: {e}"


def get_valid_cameras(config):
    """从config中获取有效的相机列表"""
    # 获取无效的aria和exo相机
    invalid_arias = config.get('INVALID_ARIAS', [])
    invalid_exos = config.get('INVALID_EXOS', [])
    
    # 获取手动标注的相机
    manual_exo_cameras = []
    if 'CALIBRATION' in config and 'MANUAL_EXO_CAMERAS' in config['CALIBRATION']:
        manual_exo_cameras = config['CALIBRATION']['MANUAL_EXO_CAMERAS']
    
    return {
        'invalid_arias': invalid_arias,
        'invalid_exos': invalid_exos,
        'manual_exo_cameras': manual_exo_cameras
    }


def check_sequence_validity(sequence_path, category, sequence_name):
    """检查序列是否有效（包含必要的目录结构和config配置）"""
    # 首先加载config文件
    config, config_msg = load_config(sequence_path, category, sequence_name)
    if config is None:
        return False, config_msg
    
    # 获取有效相机信息
    camera_info = get_valid_cameras(config)
    
    # 检查基本目录结构
    required_dirs = [
        'processed_data',
        'colmap/workplace',
        'exo'
    ]
    
    for dir_name in required_dirs:
        if not os.path.exists(os.path.join(sequence_path, dir_name)):
            return False, f"缺少目录: {dir_name}"
    
    # 检查是否有图像数据（排除无效相机）
    exo_dir = os.path.join(sequence_path, 'exo')
    all_cam_dirs = [d for d in os.listdir(exo_dir) if d.startswith('cam') and os.path.isdir(os.path.join(exo_dir, d))]
    
    # 过滤掉无效的exo相机
    valid_cam_dirs = [d for d in all_cam_dirs if d not in camera_info['invalid_exos']]
    
    if not valid_cam_dirs:
        return False, f"未找到有效的相机目录，所有相机都被标记为无效: {camera_info['invalid_exos']}"
    
    # 检查第一个有效相机是否有去畸变图像
    first_cam = sorted(valid_cam_dirs)[0]
    undistorted_images_dir = os.path.join(exo_dir, first_cam, 'undistorted_images_scale2.0')
    if not os.path.exists(undistorted_images_dir):
        return False, f"未找到去畸变图像目录: {first_cam}/undistorted_images_scale2.0"
    
    images = os.listdir(undistorted_images_dir)
    if len(images) == 0:
        return False, f"去畸变图像目录为空: {first_cam}/undistorted_images_scale2.0"
    
    # 检查COLMAP文件
    colmap_files = ['cameras.txt', 'images.txt', 'colmap_from_aria_transforms.pkl']
    workplace_dir = os.path.join(sequence_path, 'colmap', 'workplace')
    for file_name in colmap_files:
        if not os.path.exists(os.path.join(workplace_dir, file_name)):
            return False, f"缺少COLMAP文件: {file_name}"
    
    # 构建详细的有效性信息
    info_parts = [
        f"包含{len(images)}帧去畸变图像",
        f"有效相机: {len(valid_cam_dirs)}/{len(all_cam_dirs)}"
    ]
    
    if camera_info['invalid_arias']:
        info_parts.append(f"无效aria: {camera_info['invalid_arias']}")
    
    if camera_info['invalid_exos']:
        info_parts.append(f"无效exo: {camera_info['invalid_exos']}")
    
    if camera_info['manual_exo_cameras']:
        info_parts.append(f"手动标注相机: {camera_info['manual_exo_cameras']}")
    
    return True, f"序列有效，{', '.join(info_parts)}"


def process_single_sequence(sequence_info, scale_factor=2.0):
    """处理单个序列"""
    category, sequence_name, sequence_path = sequence_info
    output_path = os.path.join(sequence_path, 'data_para')
    
    print(f"开始处理序列: {category}/{sequence_name}")
    print(f"  缩放系数: {scale_factor}")
    
    # 检查序列有效性
    is_valid, message = check_sequence_validity(sequence_path, category, sequence_name)
    if not is_valid:
        print(f"序列 {category}/{sequence_name} 无效: {message}")
        return False, f"序列无效: {message}"
    
    print(f"序列 {category}/{sequence_name} 检查通过: {message}")
    
    # 加载config获取相机配置信息
    config, _ = load_config(sequence_path, category, sequence_name)
    camera_info = get_valid_cameras(config)
    
    # 如果输出目录已存在，清空以确保新数据完全替换旧数据
    if os.path.exists(output_path):
        existing_files = os.listdir(output_path)
        if existing_files:
            print(f"序列 {category}/{sequence_name} 已存在处理结果，将用新数据替换")
        shutil.rmtree(output_path)
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 构建命令，传递config信息
    script_path = os.path.join(os.path.dirname(__file__), 'merge_temporal_data.py')
    cmd = [
        sys.executable,
        script_path,
        '--root_path', sequence_path,
        '--output_path', output_path,
        '--scale_factor', str(scale_factor)
    ]
    
    # 添加config参数
    if camera_info['invalid_arias']:
        cmd.extend(['--invalid_arias'] + camera_info['invalid_arias'])
    
    if camera_info['invalid_exos']:
        cmd.extend(['--invalid_exos'] + camera_info['invalid_exos'])
    
    if camera_info['manual_exo_cameras']:
        cmd.extend(['--manual_exo_cameras'] + camera_info['manual_exo_cameras'])
    
    # 执行处理
    start_time = time.time()
    try:
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=3600  # 1小时超时
        )
        
        processing_time = time.time() - start_time
        print(f"序列 {category}/{sequence_name} 处理完成，耗时: {processing_time:.2f}秒")
        
        # 验证输出文件
        if verify_output(output_path, sequence_path, category, sequence_name):
            print(f"序列 {category}/{sequence_name} 输出验证成功")
            return True, f"处理成功，耗时{processing_time:.2f}秒"
        else:
            print(f"序列 {category}/{sequence_name} 输出验证失败")
            return False, "输出验证失败"
            
    except subprocess.TimeoutExpired:
        print(f"序列 {category}/{sequence_name} 处理超时")
        return False, "处理超时"
    except subprocess.CalledProcessError as e:
        print(f"序列 {category}/{sequence_name} 处理失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False, f"处理失败: {e}"
    except Exception as e:
        print(f"序列 {category}/{sequence_name} 发生未知错误: {e}")
        return False, f"未知错误: {e}"


def verify_output(output_path, sequence_path, category, sequence_name):
    """验证输出文件是否正确生成（考虑config中的无效相机）"""
    try:
        files = os.listdir(output_path)
        
        # 加载config获取有效相机信息
        config, _ = load_config(sequence_path, category, sequence_name)
        if config is None:
            print("无法加载config文件进行输出验证")
            return False
        
        camera_info = get_valid_cameras(config)
        
        # 检查相机文件
        cam_files = [f for f in files if f.startswith('cam') and f.endswith('.npz')]
        if len(cam_files) == 0:
            print("未找到相机文件")
            return False
        
        # 检查SMPL文件（排除无效的aria）
        smpl_files = [f for f in files if f.startswith('smpl_') and f.endswith('.npz')]
        expected_arias = ['aria01', 'aria02', 'aria03', 'aria04']
        valid_arias = [aria for aria in expected_arias if aria not in camera_info['invalid_arias']]
        
        for aria in valid_arias:
            expected_file = f"smpl_{aria}.npz"
            if expected_file not in smpl_files:
                print(f"未找到SMPL文件: {expected_file}")
                return False
        
        # 检查是否生成了无效aria的文件（这些不应该存在）
        for invalid_aria in camera_info['invalid_arias']:
            invalid_file = f"smpl_{invalid_aria}.npz"
            if invalid_file in smpl_files:
                print(f"发现不应该存在的无效aria文件: {invalid_file}")
                return False
        
        info_parts = [
            f"{len(cam_files)}个相机文件",
            f"{len(smpl_files)}个SMPL文件"
        ]
        
        if camera_info['invalid_arias']:
            info_parts.append(f"已排除无效aria: {camera_info['invalid_arias']}")
        
        print(f"输出验证成功: {', '.join(info_parts)}")
        return True
        
    except Exception as e:
        print(f"验证输出时发生错误: {e}")
        return False


def scan_dataset(root_dir, include_categories=None, exclude_categories=None, include_sequences=None, exclude_sequences=None):
    """扫描整个数据集，返回所有有效序列的信息"""
    sequences = []
    
    # 扫描所有分类目录
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        # 只处理符合格式的分类目录（如：01_tagging, 02_lego等）
        if not (item.startswith(('01_tagging', '02_lego', '03_fencing', '04_basketball', '05_volleyball', '06_badminton', '07_tennis')) and '_' in item):
            continue
        
        category = item
        
        # 检查分类包含/排除模式
        if include_categories:
            if not any(pattern in category for pattern in include_categories):
                continue
        
        if exclude_categories:
            if any(pattern in category for pattern in exclude_categories):
                continue
        
        print(f"扫描分类目录: {category}")
        
        # 扫描该分类下的所有序列
        category_path = item_path
        for seq_item in os.listdir(category_path):
            seq_path = os.path.join(category_path, seq_item)
            if not os.path.isdir(seq_path):
                continue
            
            # 检查序列包含/排除模式
            if include_sequences:
                if not any(pattern in seq_item for pattern in include_sequences):
                    continue
            
            if exclude_sequences:
                if any(pattern in seq_item for pattern in exclude_sequences):
                    continue
            
            sequences.append((category, seq_item, seq_path))
    
    return sorted(sequences)


def main():
    parser = argparse.ArgumentParser(description='批处理EgoHumans整个数据集')
    parser.add_argument('--root_dir', type=str, 
                       default='/gemini/user/private/3D/data/EgoHumans',
                       help='数据集根目录路径')
    parser.add_argument('--include_categories', type=str, nargs='+',
                       help='包含的分类模式（如: tagging lego fencing）')
    parser.add_argument('--exclude_categories', type=str, nargs='+',
                       help='排除的分类模式')
    parser.add_argument('--include_sequences', type=str, nargs='+',
                       help='包含的序列模式（如: 001 002）')
    parser.add_argument('--exclude_sequences', type=str, nargs='+',
                       help='排除的序列模式')
    parser.add_argument('--parallel', type=int, default=2,
                       help='并行处理的进程数（默认为2）')
    parser.add_argument('--scale_factor', type=float, default=2.0,
                       help='图像缩放系数（默认: 2.0）')
    parser.add_argument('--dry_run', action='store_true',
                       help='模拟运行，只显示将要处理的序列')
    parser.add_argument('--continue_on_error', action='store_true',
                       help='遇到错误时继续处理其他序列')
    
    args = parser.parse_args()
    
    # 检查根目录
    if not os.path.exists(args.root_dir):
        print(f"根目录不存在: {args.root_dir}")
        sys.exit(1)
    
    # 扫描数据集
    print(f"开始扫描数据集: {args.root_dir}")
    sequences = scan_dataset(
        args.root_dir,
        args.include_categories,
        args.exclude_categories, 
        args.include_sequences,
        args.exclude_sequences
    )
    
    if not sequences:
        print("未找到要处理的序列")
        sys.exit(1)
    
    print(f"找到 {len(sequences)} 个序列需要处理")
    print(f"使用缩放系数: {args.scale_factor}")
    
    # 按分类统计
    categories = {}
    for category, seq_name, seq_path in sequences:
        if category not in categories:
            categories[category] = []
        categories[category].append(seq_name)
    
    print("序列分布:")
    for category, seqs in categories.items():
        print(f"  {category}: {len(seqs)} 个序列")
    
    if args.dry_run:
        print("模拟运行模式，检查序列有效性:")
        for category, seq_name, seq_path in sequences:
            is_valid, message = check_sequence_validity(seq_path, category, seq_name)
            status = "✓" if is_valid else "✗"
            print(f"{status} {category}/{seq_name}: {message}")
        return
    
    # 执行处理
    start_time = time.time()
    results = {}
    
    if args.parallel > 1:
        # 并行处理
        print(f"使用 {args.parallel} 个进程并行处理")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    process_single_sequence, 
                    seq_info,
                    args.scale_factor
                ): seq_info for seq_info in sequences
            }
            
            for future in as_completed(futures):
                category, seq_name, seq_path = futures[future]
                try:
                    success, message = future.result()
                    results[f"{category}/{seq_name}"] = (success, message)
                except Exception as e:
                    print(f"序列 {category}/{seq_name} 处理时发生异常: {e}")
                    results[f"{category}/{seq_name}"] = (False, f"异常: {e}")
                    
                    if not args.continue_on_error:
                        print("遇到错误，停止处理。使用 --continue_on_error 继续处理其他序列")
                        break
    else:
        # 顺序处理
        print("顺序处理序列")
        for seq_info in sequences:
            category, seq_name, seq_path = seq_info
            success, message = process_single_sequence(seq_info, args.scale_factor)
            results[f"{category}/{seq_name}"] = (success, message)
            
            if not success and not args.continue_on_error:
                print("遇到错误，停止处理。使用 --continue_on_error 继续处理其他序列")
                break
    
    # 统计结果
    total_time = time.time() - start_time
    successful = sum(1 for success, _ in results.values() if success)
    failed = len(results) - successful
    
    print(f"\n{'='*80}")
    print(f"批处理完成！")
    print(f"总序列数: {len(sequences)}")
    print(f"已处理: {len(results)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每序列: {total_time/len(results):.2f}秒" if results else "无")
    print(f"{'='*80}")
    
    # 按分类统计结果
    print("\n分类处理结果:")
    for category in categories.keys():
        cat_results = {k: v for k, v in results.items() if k.startswith(f"{category}/")}
        cat_success = sum(1 for success, _ in cat_results.values() if success)
        cat_total = len(cat_results)
        print(f"  {category}: {cat_success}/{cat_total} 成功")
    
    # 详细结果
    print("\n详细结果:")
    for seq_name, (success, message) in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {seq_name}: {message}")
    
    # 失败的序列
    if failed > 0:
        print("\n失败的序列:")
        for seq_name, (success, message) in results.items():
            if not success:
                print(f"✗ {seq_name}: {message}")
        
        if not args.continue_on_error:
            sys.exit(1)


if __name__ == "__main__":
    main() 