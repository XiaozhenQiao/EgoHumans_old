#!/usr/bin/env python3
"""
EgoHumans数据集批量处理 - 快速启动脚本

提供常用的批量处理选项，方便用户快速使用。
"""

import os
import sys

def show_menu():
    print("=" * 60)
    print("🏗️  EgoHumans数据集批量处理 - 快速启动")
    print("=" * 60)
    print()
    print("选择处理方式:")
    print("1. 📊 扫描数据集结构")
    print("2. 🔍 预览处理计划 (前10个序列)")
    print("3. 🚀 处理整个数据集")
    print("4. 🎯 只处理特定类别")
    print("5. 🧪 测试模式 (处理前3个序列)")
    print("6. 📈 排除大型类别 (badminton有61个序列)")
    print("7. 🔧 自定义选项")
    print("0. 🚪 退出")
    print()

def main():
    while True:
        show_menu()
        choice = input("请选择 (0-7): ").strip()
        
        if choice == "0":
            print("👋 再见!")
            break
        elif choice == "1":
            print("📊 扫描数据集结构...")
            os.system("python batch_process_dataset.py --scan_only")
        elif choice == "2":
            print("🔍 预览处理计划...")
            os.system("python batch_process_dataset.py --dry_run --max_sequences 10")
        elif choice == "3":
            print("🚀 开始处理整个数据集...")
            confirm = input("⚠️  这将处理所有133个序列，可能需要数小时。确认？(y/N): ")
            if confirm.lower() in ['y', 'yes']:
                os.system("python batch_process_dataset.py")
            else:
                print("❌ 已取消")
        elif choice == "4":
            print("可用类别:")
            print("  01_tagging (14个序列)")
            print("  02_lego (6个序列)")
            print("  03_fencing (14个序列)")  
            print("  04_basketball (14个序列)")
            print("  05_volleyball (11个序列)")
            print("  06_badminton (61个序列)")
            print("  07_tennis (13个序列)")
            print()
            categories = input("请输入类别 (空格分隔): ").strip()
            if categories:
                cmd = f"python batch_process_dataset.py --include_categories {categories}"
                print(f"🎯 执行: {cmd}")
                os.system(cmd)
            else:
                print("❌ 未选择类别")
        elif choice == "5":
            print("🧪 测试模式 - 处理前3个序列...")
            os.system("python batch_process_dataset.py --max_sequences 3")
        elif choice == "6":
            print("📈 排除badminton类别，处理其他72个序列...")
            confirm = input("确认处理？(y/N): ")
            if confirm.lower() in ['y', 'yes']:
                os.system("python batch_process_dataset.py --exclude_categories 06_badminton")
            else:
                print("❌ 已取消")
        elif choice == "7":
            print("🔧 自定义选项:")
            print("示例命令:")
            print("  python batch_process_dataset.py --help")
            print("  python batch_process_dataset.py --include_categories 01_tagging")
            print("  python batch_process_dataset.py --max_sequences 5")
            print("  python batch_process_dataset.py --force")
            print()
            cmd = input("请输入完整命令 (不含python): ").strip()
            if cmd:
                os.system(f"python {cmd}")
            else:
                print("❌ 命令为空")
        else:
            print("❌ 无效选择，请重试")
        
        print("\n" + "=" * 60)
        input("按回车键继续...")

if __name__ == "__main__":
    main() 