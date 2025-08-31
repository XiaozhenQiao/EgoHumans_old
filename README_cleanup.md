# EgoHumans数据集清理脚本使用说明

本目录包含用于清理EgoHumans数据集中`undistorted_images`和`undistorted_images_scale2.0`文件夹的脚本。

## 文件说明

### 1. preview_cleanup.py
**预览脚本** - 安全的预览工具，不会删除任何文件。
- 扫描01-07所有子数据集
- 显示会被删除的文件夹及其大小
- 提供总计信息

### 2. cleanup_undistorted_folders.py
**删除脚本** - 实际执行删除操作的脚本。
- 删除指定的文件夹
- 提供详细的删除日志
- 包含确认提示

## 使用步骤

### 第一步：预览要删除的内容
```bash
python3 preview_cleanup.py
```

这将显示所有会被删除的文件夹，以及它们的大小。这一步是**完全安全**的，不会修改任何文件。

### 第二步（可选）：执行删除
**警告：以下操作不可逆！**

```bash
python3 cleanup_undistorted_folders.py
```

脚本会要求你确认操作。输入`y`或`yes`继续，其他任何输入将取消操作。

## 预览结果摘要

根据预览脚本的扫描结果：
- 发现大量`undistorted_images`和`undistorted_images_scale2.0`文件夹
- 涵盖01_tagging, 03_fencing, 04_basketball, 05_volleyball, 06_badminton, 07_tennis等数据集
- 总计可能释放数百GB的磁盘空间

## 安全提示

1. **备份重要数据**：在运行删除脚本前，确保重要数据已备份
2. **仔细检查预览结果**：运行预览脚本查看具体会删除哪些文件夹
3. **确认删除范围**：脚本只删除名为`undistorted_images`和`undistorted_images_scale2.0`的文件夹
4. **测试环境**：如果可能，先在测试环境中验证脚本行为

## 脚本特性

- **路径安全**：使用相对路径，在脚本所在目录执行
- **错误处理**：包含异常处理，避免脚本崩溃
- **详细日志**：提供删除过程的详细信息
- **确认机制**：删除前需要用户确认

## 支持的数据集

脚本会遍历以下模式的目录：
- 01_*（例如：01_tagging）
- 02_*（例如：02_lego）
- 03_*（例如：03_fencing）
- 04_*（例如：04_basketball）
- 05_*（例如：05_volleyball）
- 06_*（例如：06_badminton）
- 07_*（例如：07_tennis）

在每个数据集的深层目录结构中查找目标文件夹。 