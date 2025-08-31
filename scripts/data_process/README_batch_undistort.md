# 批量鱼眼畸变校正脚本使用说明

这个脚本 `batch_undistort.py` 用于对整个EgoHumans数据集进行批量鱼眼畸变校正处理，通过调用 `standalone_undistort.py` 来实现。

## 功能特性

- **批量处理**: 自动发现并处理数据集中的所有序列
- **并行处理**: 支持多进程并行处理以提高效率
- **模式选择**: 支持处理exo相机、ego相机或全部相机
- **序列筛选**: 可以指定要处理的特定序列或模式
- **进度监控**: 实时显示处理进度和详细日志
- **错误处理**: 自动跳过无效序列，记录处理失败的原因
- **输出验证**: 验证去畸变图像是否成功生成

## 安装依赖

```bash
pip install opencv-python numpy tqdm
```

## 基本用法

### 1. 处理所有序列的exo相机（默认）

```bash
cd /gemini/user/private/3D/data/EgoHumans/scripts/data_process
python batch_undistort.py
```

### 2. 处理所有序列的所有相机

```bash
python batch_undistort.py --mode all
```

### 3. 仅处理ego相机

```bash
python batch_undistort.py --mode ego
```

### 4. 并行处理（推荐用于大数据集）

```bash
python batch_undistort.py --mode all --parallel 4
```

### 5. 处理特定序列

```bash
# 处理特定的序列类别
python batch_undistort.py --sequences 01_tagging 02_lego

# 处理特定的单个序列
python batch_undistort.py --sequences 01_tagging/001_tagging
```

### 6. 测试模式（限制帧数）

```bash
python batch_undistort.py --max_frames 10 --dry_run
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--root_dir` | str | `/gemini/user/private/3D/data/EgoHumans` | EgoHumans数据集根目录路径 |
| `--sequences` | str+ | None | 指定要处理的序列名称或模式 |
| `--mode` | choice | `exo` | 处理模式: `all`, `ego`, `exo` |
| `--max_frames` | int | None | 每个序列最大处理帧数（用于测试） |
| `--parallel` | int | 1 | 并行处理的进程数 |
| `--dry_run` | flag | False | 模拟运行，只显示将要处理的序列 |

## 处理模式说明

- **`exo`**: 仅处理外部相机（多视角相机）的鱼眼畸变
- **`ego`**: 仅处理第一人称相机（Aria设备）的鱼眼畸变  
- **`all`**: 处理所有相机的鱼眼畸变

## 输出结构

处理完成后，去畸变图像将保存在以下位置：

```
序列目录/
├── exo/
│   ├── cam01/
│   │   └── undistorted_images/     # exo相机去畸变图像
│   │       ├── 00001.jpg
│   │       ├── 00002.jpg
│   │       └── ...
│   └── cam02/
│       └── undistorted_images/
├── ego/
│   ├── aria01/
│   │   └── images/
│   │       └── undistorted_rgb/    # ego相机去畸变图像
│   │           ├── 00001.jpg
│   │           ├── 00002.jpg
│   │           └── ...
│   └── aria02/
└── ...
```

## 日志和监控

- **实时日志**: 处理过程会在终端实时显示进度
- **日志文件**: 详细日志保存到 `batch_undistort.log`
- **进度统计**: 显示成功/失败序列的统计信息

## 性能优化建议

### 1. 并行处理
```bash
# 根据CPU核心数设置并行度，通常设置为核心数的50-80%
python batch_undistort.py --parallel 4
```

### 2. 分批处理
```bash
# 先处理一个序列类别
python batch_undistort.py --sequences 01_tagging

# 再处理另一个类别
python batch_undistort.py --sequences 02_lego
```

### 3. 测试运行
```bash
# 在全量处理前先测试
python batch_undistort.py --max_frames 5 --dry_run
```

## 故障排除

### 1. 序列检查失败
确保序列目录包含必要的子目录：
- `exo/` - 外部相机数据
- `colmap/workplace/cameras.txt` - 相机参数文件

### 2. 内存不足
- 减少并行度：`--parallel 1`
- 分批处理不同的序列类别

### 3. 处理超时
- 脚本设置了2小时超时，大序列可能需要更长时间
- 可以修改脚本中的 `timeout=7200` 参数

### 4. 依赖问题
确保安装了所需的Python包：
```bash
pip install opencv-python numpy tqdm
```

## 示例工作流程

```bash
# 1. 切换到脚本目录
cd /gemini/user/private/3D/data/EgoHumans/scripts/data_process

# 2. 测试运行，检查要处理的序列
python batch_undistort.py --dry_run

# 3. 先处理一个小序列测试
python batch_undistort.py --sequences 01_tagging/001_tagging --max_frames 10

# 4. 批量处理所有exo相机
python batch_undistort.py --mode exo --parallel 4

# 5. 查看处理结果
cat batch_undistort.log | grep "处理完成"
```

## 注意事项

1. **磁盘空间**: 确保有足够的磁盘空间存储去畸变图像
2. **处理时间**: 大数据集的处理可能需要数小时到数天
3. **系统资源**: 并行处理会占用大量CPU和内存
4. **数据备份**: 建议在处理前备份原始数据

## 相关文件

- `standalone_undistort.py` - 核心去畸变处理脚本
- `batch_undistort.py` - 本批量处理脚本
- `batch_undistort.log` - 处理日志文件 