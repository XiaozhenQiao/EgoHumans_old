#!/usr/bin/env python3
"""
脚本用于删除EgoHumans数据集中的undistorted_images和undistorted_images_scale2.0文件夹
遍历01-07子数据集，查找并删除指定的文件夹
"""

import os
import shutil
import sys
from pathlib import Path

def delete_undistorted_folders(base_path):
    """
    遍历数据集并删除undistorted_images和undistorted_images_scale2.0文件夹
    
    Args:
        base_path (str): EgoHumans数据集的根路径
    """
    base_path = Path(base_path)
    
    # 要删除的文件夹名称
    folders_to_delete = ['undistorted_images', 'undistorted_images_scale2.0']
    
    # 统计信息
    deleted_count = 0
    total_checked = 0
    
    print(f"开始扫描数据集: {base_path}")
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
            print(f"\n正在处理数据集: {dataset_name}")
            
            # 遍历数据集内的子目录
            if dataset_path.exists() and dataset_path.is_dir():
                for root, dirs, files in os.walk(dataset_path):
                    root_path = Path(root)
                    
                    # 检查当前目录中是否有要删除的文件夹
                    for folder_name in folders_to_delete:
                        folder_path = root_path / folder_name
                        if folder_path.exists() and folder_path.is_dir():
                            total_checked += 1
                            try:
                                # 显示要删除的路径
                                relative_path = folder_path.relative_to(base_path)
                                print(f"  删除: {relative_path}")
                                
                                # 删除文件夹及其内容
                                shutil.rmtree(folder_path)
                                deleted_count += 1
                                
                            except Exception as e:
                                print(f"  错误: 无法删除 {folder_path} - {e}")
    
    print("\n" + "=" * 60)
    print(f"扫描完成!")
    print(f"总共发现: {total_checked} 个目标文件夹")
    print(f"成功删除: {deleted_count} 个文件夹")
    
    if total_checked != deleted_count:
        print(f"删除失败: {total_checked - deleted_count} 个文件夹")

def main():
    """主函数"""
    # 获取脚本所在目录作为数据集根路径
    base_path = Path(__file__).parent.absolute()
    
    print("EgoHumans数据集清理脚本")
    print(f"数据集路径: {base_path}")
    print("\n此脚本将删除以下文件夹:")
    print("- undistorted_images")
    print("- undistorted_images_scale2.0")
    print("\n在01-07所有子数据集中查找并删除这些文件夹。")
    
    # 确认操作
    response = input("\n是否继续? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("操作已取消。")
        sys.exit(0)
    
    # 执行删除操作
    delete_undistorted_folders(base_path)

if __name__ == "__main__":
    main() 