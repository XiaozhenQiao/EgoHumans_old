#!/usr/bin/env python3
"""
Standalone Image Undistortion Script for EgoHumans Dataset
处理exo相机数据的独立去畸变脚本，不依赖mmcv等库

Usage:
    python standalone_undistort.py --sequence_path /path/to/sequence --mode exo
"""

import numpy as np
import os
import argparse
import cv2
from tqdm import tqdm
import json
from pathlib import Path


class CameraIntrinsics:
    """相机内参类，用于存储和处理相机内参数"""
    
    def __init__(self, params):
        self.params = params
        
    def world_to_image(self, point_2d):
        """将归一化坐标转换为图像坐标"""
        fx, fy, cx, cy = self.params[:4]
        k1, k2, k3, k4 = self.params[4:8]
        
        x, y = point_2d[0], point_2d[1]
        
        # 计算畸变
        r2 = x*x + y*y
        r4 = r2 * r2
        r6 = r4 * r2
        
        # Fisheye distortion model
        theta = np.arctan(np.sqrt(r2))
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta6 * theta2
        
        theta_d = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
        
        if r2 > 1e-8:
            inv_r = theta_d / np.sqrt(r2)
            x_distorted = x * inv_r
            y_distorted = y * inv_r
        else:
            x_distorted = x
            y_distorted = y
            
        # 转换到图像坐标
        u = fx * x_distorted + cx
        v = fy * y_distorted + cy
        
        return np.array([u, v])


class ExoCamera:
    """ExoCamera类，处理外部相机的去畸变"""
    
    def __init__(self, camera_name, colmap_camera_id, image_width, image_height, 
                 intrinsics_params, images_path):
        self.camera_name = camera_name
        self.colmap_camera_id = colmap_camera_id
        self.image_width = image_width
        self.image_height = image_height
        self.images_path = images_path
        
        # 设置相机内参
        self.intrinsics = CameraIntrinsics(intrinsics_params)
        
        # OpenCV fisheye参数
        fx, fy, cx, cy = intrinsics_params[:4]
        k1, k2, k3, k4 = intrinsics_params[4:8]
        
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.D_fisheye = np.array([k1, k2, k3, k4])
        self.K_undistorted = None
        
        print(f"初始化相机 {camera_name}: 分辨率 {image_width}x{image_height}")
        
    def init_undistort_map(self):
        """初始化去畸变映射"""
        try:
            self.K_undistorted = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D_fisheye, 
                (self.image_width, self.image_height), 
                np.eye(3), balance=1.0
            )
            print(f"相机 {self.camera_name} 去畸变映射初始化成功")
        except Exception as e:
            print(f"相机 {self.camera_name} 去畸变映射初始化失败: {e}")
            # 使用原始内参作为备选
            self.K_undistorted = self.K.copy()
            
    def get_image_path(self, time_stamp):
        """获取图像路径"""
        image_path = os.path.join(self.images_path, f'{time_stamp:05d}.jpg')
        return image_path
        
    def get_image(self, time_stamp):
        """读取图像"""
        image_path = self.get_image_path(time_stamp)
        if not os.path.exists(image_path):
            return None
        image = cv2.imread(image_path)
        return image
        
    def get_undistorted_image(self, time_stamp):
        """获取去畸变图像"""
        image = self.get_image(time_stamp)
        if image is None:
            return None
            
        try:
            # 使用OpenCV fisheye undistortion，和原始代码保持一致
            undistorted_image = cv2.fisheye.undistortImage(
                image, self.K, self.D_fisheye, None, self.K
            )
            return undistorted_image
        except Exception as e:
            print(f"去畸变处理失败 {self.camera_name} 时间戳 {time_stamp}: {e}")
            return image  # 返回原图作为备选


class AriaCamera:
    """AriaCamera类，处理ego相机的去畸变"""
    
    def __init__(self, camera_name, intrinsics, images_path, calibration_path):
        self.camera_name = camera_name
        self.intrinsics = intrinsics  # 15个参数的数组
        self.images_path = images_path
        self.calibration_path = calibration_path
        
        # 图像尺寸 (旋转后的aria图像)
        self.rotated_image_width = 1408
        self.rotated_image_height = 1408
        self.image_width = 1408
        self.image_height = 1408
        
        # 构建内参矩阵
        self.intrinsic_matrix = np.array([
            [self.intrinsics[0], 0, self.intrinsics[1]],
            [0, self.intrinsics[0], self.intrinsics[2]],
            [0, 0, 1]
        ])
        self.intrinsic_matrix_inv = np.linalg.inv(self.intrinsic_matrix)
        
        # 去畸变映射
        self.x_map = None
        self.y_map = None
        
        print(f"初始化Aria相机 {camera_name}")
        
    def init_undistort_map(self):
        """初始化去畸变映射"""
        cache_dir = os.path.dirname(self.calibration_path)
        map_path = os.path.join(cache_dir, f'{self.camera_name}_undistort_map.npz')
        
        if os.path.exists(map_path):
            print(f'从缓存加载去畸变映射 {self.camera_name}')
            data = np.load(map_path)
            self.x_map = data['x_map']
            self.y_map = data['y_map']
        else:
            print(f'创建去畸变映射 {self.camera_name} (这可能需要一些时间)')
            self.x_map = np.zeros((self.rotated_image_height, self.rotated_image_width), dtype=np.float32)
            self.y_map = np.zeros((self.rotated_image_height, self.rotated_image_width), dtype=np.float32)
            
            for x in tqdm(range(self.rotated_image_width), desc=f"生成 {self.camera_name} 去畸变映射"):
                for y in range(self.rotated_image_height):
                    # 将像素坐标投影到3D相机坐标
                    cam_coords = self.intrinsic_matrix_inv @ np.array([x, y, 1])
                    # 投影到鱼眼图像
                    p = self.image_from_cam(cam_coords)
                    self.x_map[y, x] = p[0]
                    self.y_map[y, x] = p[1]
            
            print(f'保存去畸变映射到缓存 {self.camera_name}')
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(map_path, x_map=self.x_map, y_map=self.y_map)
    
    def image_from_cam(self, point_3d, eps=1e-9):
        """从相机坐标系投影到图像坐标系"""
        if point_3d[2] < 0:  # 点在相机后面
            return np.array([-1, -1])
            
        # 归一化
        inv_z = 1 / point_3d[2]
        ab = point_3d[:2] * inv_z
        
        # 计算径向距离
        ab_squared = ab ** 2
        r_sq = ab_squared[0] + ab_squared[1]
        r = np.sqrt(r_sq)
        
        if r < eps:
            uvDistorted = ab
        else:
            # 鱼眼畸变模型
            theta = np.arctan(r)
            theta_sq = theta ** 2
            
            # 使用内参中的畸变系数
            startK = 3
            theta_distorted = theta * (1 + 
                self.intrinsics[startK] * theta_sq +
                self.intrinsics[startK + 1] * theta_sq ** 2 +
                self.intrinsics[startK + 2] * theta_sq ** 3 +
                self.intrinsics[startK + 3] * theta_sq ** 4 +
                self.intrinsics[startK + 4] * theta_sq ** 5 +
                self.intrinsics[startK + 5] * theta_sq ** 6
            )
            
            # 切向畸变
            startP = startK + 6
            p1, p2 = self.intrinsics[startP], self.intrinsics[startP + 1]
            
            uvDistorted = ab * (theta_distorted / r)
            
            # 应用切向畸变
            uvDistorted[0] += 2 * p1 * ab[0] * ab[1] + p2 * (r_sq + 2 * ab_squared[0])
            uvDistorted[1] += p1 * (r_sq + 2 * ab_squared[1]) + 2 * p2 * ab[0] * ab[1]
        
        # 转换到像素坐标
        point_2d = uvDistorted * self.intrinsics[0] + self.intrinsics[1:3]
        return point_2d
        
    def get_image_path(self, time_stamp):
        """获取图像路径"""
        return os.path.join(self.images_path, f'{time_stamp:05d}.jpg')
        
    def get_image(self, time_stamp):
        """读取图像"""
        image_path = self.get_image_path(time_stamp)
        if not os.path.exists(image_path):
            return None
        return cv2.imread(image_path)
        
    def get_undistorted_image_aria(self, time_stamp):
        """获取去畸变的aria图像"""
        image = self.get_image(time_stamp)
        if image is None:
            return None
            
        # 旋转图像
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # 应用去畸变
        undistorted_rotated_image = cv2.remap(rotated_image, self.x_map, self.y_map, cv2.INTER_CUBIC)
        # 旋转回来
        undistorted_image = cv2.rotate(undistorted_rotated_image, cv2.ROTATE_90_CLOCKWISE)
        return undistorted_image
        
    def get_undistorted_image(self, time_stamp):
        """获取去畸变图像"""
        return self.get_undistorted_image_aria(time_stamp)


def load_colmap_cameras(cameras_txt_path):
    """加载colmap相机参数"""
    cameras = {}
    
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }
    
    return cameras


def get_exo_camera_mapping_from_colmap(colmap_cameras, exo_dir, colmap_dir):
    """从colmap数据中获取exo相机映射"""
    
    camera_mapping = {}
    
    # 简化版本：根据分辨率判断哪些是exo相机
    for camera_id, cam_info in colmap_cameras.items():
        # exo相机的特征：分辨率不是1408x1408
        if not (cam_info['width'] == 1408 and cam_info['height'] == 1408):
            # 尝试从images.txt中找到对应的相机名称
            images_txt_path = os.path.join(colmap_dir, 'images.txt')
            if os.path.exists(images_txt_path):
                with open(images_txt_path, 'r') as f:
                    lines = f.readlines()
                    # 跳过前4行注释
                    for i in range(4, len(lines), 2):  # 每隔一行读取
                        line = lines[i].strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 10:
                                line_camera_id = int(parts[8])
                                if line_camera_id == camera_id:
                                    image_path = parts[9]
                                    camera_name = image_path.split('/')[0]
                                    if camera_name.startswith('cam'):
                                        camera_mapping[camera_name] = camera_id
                                        break
    
    print(f"检测到的exo相机映射: {camera_mapping}")
    return camera_mapping


def main():
    parser = argparse.ArgumentParser(description='独立的图像去畸变脚本')
    parser.add_argument('--sequence_path', required=True, help='序列数据路径')
    parser.add_argument('--output_path', help='输出路径，默认为序列路径')
    parser.add_argument('--mode', default='exo', choices=['all', 'ego', 'exo'], 
                       help='处理模式: all, ego, or exo')
    parser.add_argument('--max_frames', type=int, default=None, 
                       help='最大处理帧数，用于测试')
    parser.add_argument('--scale_factor', type=float, default=2.0,
                       help='图像缩放因子，用于降低分辨率 (默认: 1.0, 即不缩放)')
    
    args = parser.parse_args()
    
    sequence_path = args.sequence_path
    output_path = args.output_path if args.output_path else sequence_path
    
    print(f"处理序列: {sequence_path}")
    print(f"输出路径: {output_path}")
    print(f"模式: {args.mode}")
    print(f"缩放因子: {args.scale_factor}")
    
    # 路径设置
    exo_dir = os.path.join(sequence_path, 'exo')
    ego_dir = os.path.join(sequence_path, 'ego')
    colmap_dir = os.path.join(sequence_path, 'colmap', 'workplace')
    
    # 检查路径是否存在
    if not os.path.exists(sequence_path):
        print(f"错误: 序列路径不存在: {sequence_path}")
        return
        
    # 加载colmap相机参数
    cameras_txt_path = os.path.join(colmap_dir, 'cameras.txt')
    if not os.path.exists(cameras_txt_path):
        print(f"错误: 找不到相机参数文件: {cameras_txt_path}")
        return
        
    colmap_cameras = load_colmap_cameras(cameras_txt_path)
    print(f"加载了 {len(colmap_cameras)} 个相机参数")
    
    # 获取相机映射
    exo_camera_mapping = get_exo_camera_mapping_from_colmap(colmap_cameras, exo_dir, colmap_dir)
    
    # 初始化相机
    cameras = {}
    
    if args.mode in ['all', 'exo']:
        # 处理exo相机
        if os.path.exists(exo_dir):
            exo_camera_names = [name for name in sorted(os.listdir(exo_dir)) 
                              if name.startswith('cam') and os.path.isdir(os.path.join(exo_dir, name))]
            
            for cam_name in exo_camera_names:
                if cam_name in exo_camera_mapping:
                    colmap_id = exo_camera_mapping[cam_name]
                    if colmap_id in colmap_cameras:
                        cam_info = colmap_cameras[colmap_id]
                        images_path = os.path.join(exo_dir, cam_name, 'images')
                        
                        if os.path.exists(images_path):
                            camera = ExoCamera(
                                cam_name, colmap_id,
                                cam_info['width'], cam_info['height'],
                                cam_info['params'], images_path
                            )
                            cameras[(cam_name, 'rgb')] = camera
                            print(f"添加了exo相机: {cam_name}")
    
    if args.mode in ['all', 'ego']:
        # 处理ego相机 (aria相机)
        if os.path.exists(ego_dir):
            aria_names = [name for name in sorted(os.listdir(ego_dir)) 
                         if name.startswith('aria') and os.path.isdir(os.path.join(ego_dir, name))]
            
            for aria_name in aria_names:
                rgb_path = os.path.join(ego_dir, aria_name, 'rgb')
                calib_path = os.path.join(ego_dir, aria_name, 'calib')
                
                if os.path.exists(rgb_path) and os.path.exists(calib_path):
                    # 这里需要加载aria相机的内参，通常存储在calib目录中
                    # 为简化，我们使用默认的aria内参
                    default_intrinsics = [
                        610.28, 704, 704,  # f, cx, cy
                        0.4, -0.56, 0.77, -0.44, 0, 0,  # k1-k6
                        0, 0,  # p1, p2
                        0, 0, 0, 0  # s1-s4
                    ]
                    
                    camera = AriaCamera(aria_name, default_intrinsics, rgb_path, calib_path)
                    cameras[(aria_name, 'rgb')] = camera
                    print(f"添加了aria相机: {aria_name}")
    
    if not cameras:
        print("错误: 没有找到任何相机数据")
        return
    
    # 初始化去畸变映射
    print("初始化去畸变映射...")
    for (camera_name, camera_mode), camera in cameras.items():
        camera.init_undistort_map()
    
    # 确定时间范围
    max_time = 0
    for (camera_name, camera_mode), camera in cameras.items():
        images_path = camera.images_path
        if os.path.exists(images_path):
            image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
            if image_files:
                max_time = max(max_time, len(image_files))
    
    if max_time == 0:
        print("错误: 没有找到任何图像文件")
        return
        
    if args.max_frames:
        max_time = min(max_time, args.max_frames)
    
    print(f"将处理 {max_time} 帧")
    
    # 处理图像
    for t in tqdm(range(1, max_time + 1), desc="处理图像"):
        for (camera_name, camera_mode), camera in cameras.items():
            try:
                # 获取去畸变图像
                undistorted_image = camera.get_undistorted_image(t)
                
                if undistorted_image is not None:
                    # 如果需要缩放，对去畸变后的图像进行缩放
                    if args.scale_factor != 1.0:
                        h, w = undistorted_image.shape[:2]
                        new_h, new_w = int(h // args.scale_factor), int(w // args.scale_factor)
                        undistorted_image = cv2.resize(undistorted_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # 确定输出目录
                    if camera_name.startswith('aria'):
                        if args.scale_factor != 1.0:
                            output_dir = os.path.join(output_path, 'ego', camera_name, 'images', f'undistorted_rgb_scale{args.scale_factor}')
                        else:
                            output_dir = os.path.join(output_path, 'ego', camera_name, 'images', 'undistorted_rgb')
                    else:
                        if args.scale_factor != 1.0:
                            output_dir = os.path.join(output_path, 'exo', camera_name, f'undistorted_images_scale{args.scale_factor}')
                        else:
                            output_dir = os.path.join(output_path, 'exo', camera_name, 'undistorted_images')
                    
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 保存图像
                    output_file = os.path.join(output_dir, f'{t:05d}.jpg')
                    cv2.imwrite(output_file, undistorted_image)
                    
            except Exception as e:
                print(f"处理失败 {camera_name} 帧 {t}: {e}")
    
    print("去畸变处理完成!")


if __name__ == "__main__":
    main() 