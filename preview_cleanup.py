#!/usr/bin/env python3
"""
预览脚本 - 显示EgoHumans数据集中会被删除的undistorted_images文件夹
不会实际删除任何文件，只是显示会受影响的文件夹
"""

import os
import sys
from pathlib import Path

def preview_undistorted_folders(base_path):
    """
    预览会被删除的undistorted_images和undistorted_images_scale2.0文件夹
    
    Args:
        base_path (str): EgoHumans数据集的根路径
    """
    base_path = Path(base_path)
    
    # 要删除的文件夹名称
    folders_to_delete = ['undistorted_images', 'undistorted_images_scale2.0']
    
    # 统计信息
    found_folders = []
    total_size = 0
    
    print(f"预览扫描数据集: {base_path}")
    print("=" * 60)
    
    # 遍历01-07子数据集
    for dataset_num in range(1, 8):
        dataset_dir = base_path / f"{dataset_num:02d}_*"
        
        # 使用glob查找匹配的目录
        import glob
        dataset_paths = glob.glob(str(dataset_dir))
        
        for dataset_path in dataset_paths:
            dataset_path = Path(dataset_path)
            dataset_name = dataset_path.name
            print(f"\n扫描数据集: {dataset_name}")
            
            # 遍历数据集内的子目录
            if dataset_path.exists() and dataset_path.is_dir():
                for root, dirs, files in os.walk(dataset_path):
                    root_path = Path(root)
                    
                    # 检查当前目录中是否有要删除的文件夹
                    for folder_name in folders_to_delete:
                        folder_path = root_path / folder_name
                        if folder_path.exists() and folder_path.is_dir():
                            try:
                                # 计算文件夹大小
                                folder_size = sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())
                                folder_size_mb = folder_size / (1024 * 1024)
                                total_size += folder_size
                                
                                # 显示要删除的路径
                                relative_path = folder_path.relative_to(base_path)
                                print(f"  发现: {relative_path} ({folder_size_mb:.1f} MB)")
                                found_folders.append((relative_path, folder_size))
                                
                            except Exception as e:
                                print(f"  警告: 无法访问 {folder_path} - {e}")
    
    print("\n" + "=" * 60)
    print("预览结果:")
    print(f"总共发现: {len(found_folders)} 个目标文件夹")
    print(f"总大小: {total_size / (1024 * 1024 * 1024):.2f} GB")
    
    if found_folders:
        print("\n详细列表:")
        for folder_path, size in found_folders:
            size_mb = size / (1024 * 1024)
            print(f"  {folder_path} ({size_mb:.1f} MB)")

def main():
    """主函数"""
    # 获取脚本所在目录作为数据集根路径
    base_path = Path(__file__).parent.absolute()
    
    print("EgoHumans数据集清理预览")
    print(f"数据集路径: {base_path}")
    print("\n此脚本将预览以下文件夹:")
    print("- undistorted_images")
    print("- undistorted_images_scale2.0")
    print("\n在01-07所有子数据集中查找这些文件夹。")
    print("注意: 这只是预览，不会删除任何文件。")
    
    # 执行预览操作
    preview_undistorted_folders(base_path)

if __name__ == "__main__":
    main() 