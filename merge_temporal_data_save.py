import torch
import numpy as np
import os
import cv2
from smplx import SMPL    
import pickle
import re
import pycolmap
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import argparse
import IPython
import yaml
import shutil
import glob


def cleanup_output_directory(output_path):
    """清理输出目录中的所有文件"""
    if os.path.exists(output_path):
        print(f"清理输出目录: {output_path}")
        try:
            # 删除目录中的所有文件，但保留目录本身
            for file_path in glob.glob(os.path.join(output_path, "*")):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"删除文件: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"删除目录: {file_path}")
            print("输出目录清理完成")
        except Exception as e:
            print(f"警告: 清理输出目录时出错: {e}")
    else:
        print(f"创建输出目录: {output_path}")
        os.makedirs(output_path, exist_ok=True)


def load_config(root_path):
    """从数据集序列对应的config文件中加载配置参数"""
    # 从路径中提取序列名和运动类型
    sequence_name = os.path.basename(root_path)  # 例如: 013_fencing 或 002_basketball
    parent_name = os.path.basename(os.path.dirname(root_path))  # 例如: 03_fencing 或 04_basketball
    
    # 从序列名中提取运动类型
    sport_type = None
    if '_' in sequence_name:
        sport_type = sequence_name.split('_', 1)[1]  # 从 013_fencing 提取 fencing
    elif '_' in parent_name:
        sport_type = parent_name.split('_', 1)[1]  # 从 03_fencing 提取 fencing
    
    if not sport_type:
        print(f"警告: 无法从路径 {root_path} 中提取运动类型")
        return {}
    
    # 构建config文件路径的多种可能位置
    config_paths = []
    
    # 第一种：相对于当前数据目录的config路径
    base_data_path = root_path
    while base_data_path and os.path.basename(base_data_path) != 'EgoHumans':
        base_data_path = os.path.dirname(base_data_path)
    
    if base_data_path:
        config_paths.append(os.path.join(base_data_path, 'egohumans', 'configs', sport_type, f'{sequence_name}.yaml'))
    
    # 第二种：从当前脚本位置查找
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_paths.append(os.path.join(script_dir, 'egohumans', 'configs', sport_type, f'{sequence_name}.yaml'))
    
    # 第三种：相对路径
    config_paths.append(os.path.join(root_path, '..', '..', 'egohumans', 'configs', sport_type, f'{sequence_name}.yaml'))
    
    config = {}
    for config_path in config_paths:
        config_path = os.path.normpath(config_path)  # 规范化路径
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"成功加载配置文件: {config_path}")
                break
            except Exception as e:
                print(f"警告: 加载配置文件失败: {config_path}, 错误: {e}")
    
    if not config:
        print(f"警告: 未找到运动类型 '{sport_type}' 序列 '{sequence_name}' 的配置文件")
        print(f"尝试过的路径: {config_paths}")
    
    return config


def parse_colmap_files(folder):
    """解析COLMAP相机参数文件"""
    cameras_path = f"{folder}/cameras.txt"
    cameras = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(list(map(float, elems[4:])))
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

def read_text(file_path):
    """读取COLMAP images.txt文件"""
    view_lines = []
    pattern = re.compile(r'cam.*\.jpg')
    quat = []
    transl = []
    fnames = []
    cam_ids = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if pattern.search(line):
                view_lines.append(line.strip())
                quat.append(view_lines[-1].split(' ')[1:5])
                transl.append(view_lines[-1].split(' ')[5:8])
                fnames.append(view_lines[-1].split(' ')[-1])
                cam_ids.append(view_lines[-1].split(' ')[-2])
    quat = np.array(quat, dtype=np.float32)
    transl = np.array(transl, dtype=np.float32)
    return quat, transl, fnames, cam_ids

def parse_camera(folder):
    """解析相机外参"""
    quat, transl, fnames, cam_ids = read_text(os.path.join(folder, "images.txt"))
    quat = np.vstack([quat[:, 0], -quat[:, 3], quat[:, 2], quat[:, 1]]).T
    rotmat = R.from_quat(quat).as_matrix()
    transl[:, 0] *= -1
    transl = rotmat @ transl[..., None]
    P = np.eye(4, 4)[None].repeat(len(quat), axis=0)
    P[:, :3, :3] = rotmat
    P[:, :3, 3:4] = transl
    return P[:, :3, :3], P[:, :3, 3:4], fnames, cam_ids

def fisheye_to_K(params):
    """转换鱼眼参数为内参矩阵"""
    fx, fy, cx, cy = params[:4]
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    return K

def get_distortion_params(params):
    """提取畸变参数"""
    fx, fy, cx, cy = params[:4]
    k1, k2, k3, k4 = params[4:8] if len(params) > 4 else [0, 0, 0, 0]
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([k1, k2, k3, k4])
    
    return K, D

def undistort_points_fisheye(points_2d, K, D):
    """使用鱼眼模型对2D点进行去畸变处理"""
    if points_2d is None:
        return None
        
    points_2d = np.array(points_2d)
    if len(points_2d.shape) == 1:
        points_2d = points_2d.reshape(1, -1)
    
    if points_2d.shape[0] == 0:
        return points_2d
    
    try:
        # 提取x, y坐标 (假设前两列是坐标)
        xy_coords = points_2d[:, :2].astype(np.float32)
        
        # 重塑为OpenCV要求的格式 (N, 1, 2)
        points_cv = xy_coords.reshape(-1, 1, 2)
        
        # 使用OpenCV进行鱼眼去畸变
        undistorted_points_cv = cv2.fisheye.undistortPoints(
            points_cv, K, D, None, K
        )
        
        # 重塑回原格式
        undistorted_xy = undistorted_points_cv.reshape(-1, 2)
        
        # 保持原始数据的其他列（如置信度等）
        result = points_2d.copy().astype(np.float64)
        result[:, :2] = undistorted_xy
        
        return result
        
    except Exception as e:
        print(f"警告: 2D点去畸变失败: {e}")
        # 如果去畸变失败，返回原始数据
        return points_2d

def undistort_bbox_fisheye(bbox, K, D):
    """使用鱼眼模型对2D边界框进行去畸变处理"""
    if bbox is None:
        return None
    
    bbox = np.array(bbox)
    if len(bbox) < 4:
        return bbox
    
    try:
        # 假设边界框格式为 [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox[:4]
        
        # 创建边界框的四个角点
        corners = np.array([
            [x1, y1],
            [x2, y1], 
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        
        # 对角点进行去畸变
        undistorted_corners = undistort_points_fisheye(corners, K, D)
        
        # 计算去畸变后的边界框
        x_coords = undistorted_corners[:, 0]
        y_coords = undistorted_corners[:, 1]
        
        new_x1, new_x2 = np.min(x_coords), np.max(x_coords)
        new_y1, new_y2 = np.min(y_coords), np.max(y_coords)
        
        # 保持原始边界框的格式
        result = bbox.copy().astype(np.float64)
        result[:4] = [new_x1, new_y1, new_x2, new_y2]
        
        return result
        
    except Exception as e:
        print(f"警告: 边界框去畸变失败: {e}")
        # 如果去畸变失败，返回原始数据
        return bbox

def merge_temporal_data(root_path, output_path, scale_factor=2.0, invalid_arias=None, invalid_exos=None, manual_exo_cameras=None):
    """合并时序数据，包含去畸变和缩放处理"""
    print(f"处理数据: {root_path}")

    # 加载配置文件
    config = load_config(root_path)
    
    invalid_arias = config.get('INVALID_ARIAS', [])
    invalid_exos = config.get('INVALID_EXOS', [])
    calibration_config = config.get('CALIBRATION', {})
    manual_exo_cameras = calibration_config.get('MANUAL_EXO_CAMERAS', [])

    # 首先清理输出目录
    cleanup_output_directory(output_path)

    # 获取帧数（从有效的相机中获取）
    exo_dir = f"{root_path}/exo"
    valid_cam_dirs = [d for d in os.listdir(exo_dir) if d.startswith('cam') and os.path.isdir(os.path.join(exo_dir, d))]
    # valid_cam_dirs = [d for d in all_cam_dirs if d not in invalid_exos]
    
    # IPython.embed()
    
    first_cam = sorted(valid_cam_dirs)[0]
    images_dir = f"{root_path}/exo/{first_cam}/undistorted_images_scale2.0"
    num_frames = len(os.listdir(images_dir))
    print(f"总帧数: {num_frames}")
    
    # 解析相机参数
    intrinsics = parse_colmap_files(f"{root_path}/colmap/workplace")
    Rotations, Translations, fnames, cam_ids = parse_camera(f"{root_path}/colmap/workplace")
    
    # 读取变换矩阵
    colmap_transforms_file = f"{root_path}/colmap/workplace/colmap_from_aria_transforms.pkl"
    with open(colmap_transforms_file, 'rb') as f:
        colmap_transforms = pickle.load(f)
    primary_transform = colmap_transforms['aria01']
    colmap_reconstruction = pycolmap.Reconstruction(f"{root_path}/colmap/workplace") 

    # 解析外参
    all_extrinsics = {}
    for image_id, image in colmap_reconstruction.images.items():
        image_path = image.name
        image_camera_name = image_path.split('/')[0]
        if image_camera_name.startswith('cam') and image_camera_name not in invalid_exos:
            all_extrinsics[image_camera_name] = np.eye(4, 4)
            transform = image.cam_from_world()
            all_extrinsics[image_camera_name][:3, :3] = transform.rotation.matrix()
            all_extrinsics[image_camera_name][:3, 3] = transform.translation
    
    # 加载手动标注的相机参数并覆盖
    for manual_cam in manual_exo_cameras:
        manual_cam_file = f"{root_path}/colmap/workplace/{manual_cam}.npy"
        if os.path.exists(manual_cam_file):
            try:
                manual_extrinsics = np.load(manual_cam_file)
                if manual_extrinsics.shape == (4, 4):
                    # 手动标注的外参矩阵不需要缩放调整
                    # 外参表示的是相机在世界坐标系中的位置和朝向，与图像缩放无关
                    # 只有内参矩阵K需要在后续处理中进行缩放
                    all_extrinsics[manual_cam] = manual_extrinsics.copy()
                    print(f"加载手动标注相机参数: {manual_cam} (外参矩阵不需要缩放调整)")
                else:
                    print(f"警告: 手动标注相机参数文件格式错误: {manual_cam} (期望4x4矩阵)")
            except Exception as e:
                print(f"警告: 加载手动标注相机参数失败: {manual_cam}, 错误: {e}")
        else:
            print(f"警告: 未找到手动标注相机参数文件: {manual_cam_file}")

    # 解析内参和畸变参数
    intrinsics_all = {}
    distortion_params = {}
    for fid, fname in enumerate(fnames):
        cam_name = fname.split('/')[0]
        cam_id = cam_ids[fid]
        K = fisheye_to_K(intrinsics[int(cam_id)]["params"])
        K_dist, D_dist = get_distortion_params(intrinsics[int(cam_id)]["params"])
        intrinsics_all[cam_name] = K
        distortion_params[cam_name] = {'K': K_dist, 'D': D_dist}
    
    # 为手动标注的相机确保有内参数据（如果它们在COLMAP中不存在）
    for manual_cam in manual_exo_cameras:
        if manual_cam not in intrinsics_all:
            # 如果手动标注的相机没有内参，我们需要从COLMAP数据中找到匹配的相机ID
            # 通常手动标注的相机在cameras.txt中还是存在的，只是在images.txt中可能没有
            cam_id = None
            for camera_id, camera_data in intrinsics.items():
                # 假设手动标注的相机ID可以通过某种方式确定
                # 这里可能需要根据具体的数据格式来调整
                potential_cam_name = f"cam{camera_id:02d}"
                if potential_cam_name == manual_cam:
                    cam_id = camera_id
                    break
            
            if cam_id is not None:
                K = fisheye_to_K(intrinsics[cam_id]["params"])
                K_dist, D_dist = get_distortion_params(intrinsics[cam_id]["params"])
                intrinsics_all[manual_cam] = K
                distortion_params[manual_cam] = {'K': K_dist, 'D': D_dist}
                print(f"为手动标注相机 {manual_cam} 添加内参数据 (相机ID: {cam_id})")
            else:
                print(f"警告: 无法找到手动标注相机 {manual_cam} 的内参数据")

    # 合并相机数据
    print("合并相机数据...")
    for cam_name in all_extrinsics.keys():
        extrinsic = all_extrinsics[cam_name]
        extrinsic = np.dot(extrinsic, primary_transform)
        extrinsic = np.linalg.inv(extrinsic)
        
        # 手动标注相机参数已在前面处理，这里不需要重复处理
        
        if cam_name in manual_exo_cameras:
            print(f"手动标注相机 {cam_name} 已完成变换处理 (primary_transform + inverse)")


        # 应用缩放到内参
        K_original = intrinsics_all[cam_name].copy()
        K_scaled = K_original.copy()
        if scale_factor != 1.0:
            K_scaled[0, 0] /= scale_factor  # fx
            K_scaled[1, 1] /= scale_factor  # fy
            K_scaled[0, 2] /= scale_factor  # cx
            K_scaled[1, 2] /= scale_factor  # cy
        
        save_cam_data = {
            'K': K_scaled,
            'R': extrinsic[:3, :3],
            'T': extrinsic[:3, 3],
            'extrinsic': extrinsic,
            'scale_factor': scale_factor
        }
        
        np.savez(os.path.join(output_path, f"{cam_name}.npz"), **save_cam_data)
    
    # 合并SMPL和关键点数据
    print("合并SMPL和关键点数据...")
    
    temporal_data = {}
    all_aria_names = ['aria01', 'aria02', 'aria03', 'aria04']
    # 过滤无效的aria
    valid_aria_names = [aria for aria in all_aria_names if aria not in invalid_arias]
    
    if not valid_aria_names:
        raise ValueError(f"没有有效的aria，所有aria都被标记为无效: {invalid_arias}")
    
    print(f"有效aria: {valid_aria_names}")
    
    for aria_name in valid_aria_names:
        temporal_data[aria_name] = {
            'frame_indices': [],
            'smpl_params': {'betas': [], 'body_pose': [], 'global_orient': [], 'transl': []},
            'keypoints3d': [],
            'keypoints2d': {},
            'bboxes2d': {}
        }
        
        for cam_name in all_extrinsics.keys():
            temporal_data[aria_name]['keypoints2d'][cam_name] = []
            temporal_data[aria_name]['bboxes2d'][cam_name] = []
    
    # 逐帧读取数据
    for frame_idx in tqdm(range(1, num_frames + 1), desc="处理帧数据"):
        # 读取SMPL参数
        smpl_file = f"{root_path}/processed_data/smpl/{frame_idx:05d}.npy"
        smpl_params = {}
        if os.path.exists(smpl_file):
            smpl_data = np.load(smpl_file, allow_pickle=True)
            smpl_params = dict(smpl_data.item())
            
        # 读取3D关键点
        pose3d_file = f"{root_path}/processed_data/refine_poses3d/{frame_idx:05d}.npy"
        pose3d_data = {}
        if os.path.exists(pose3d_file):
            pose3d = np.load(pose3d_file, allow_pickle=True)
            if pose3d is not None:
                pose3d_data = pose3d.item() if hasattr(pose3d, 'item') else {}
        
        # 处理每个有效的aria
        for aria_name in valid_aria_names:
            temporal_data[aria_name]['frame_indices'].append(frame_idx)
            
            # SMPL参数
            if aria_name in smpl_params:
                temporal_data[aria_name]['smpl_params']['betas'].append(smpl_params[aria_name]['betas'])
                temporal_data[aria_name]['smpl_params']['body_pose'].append(smpl_params[aria_name]['body_pose'])
                temporal_data[aria_name]['smpl_params']['global_orient'].append(smpl_params[aria_name]['global_orient'])
                temporal_data[aria_name]['smpl_params']['transl'].append(smpl_params[aria_name]['transl'])
            else:
                temporal_data[aria_name]['smpl_params']['betas'].append(None)
                temporal_data[aria_name]['smpl_params']['body_pose'].append(None)
                temporal_data[aria_name]['smpl_params']['global_orient'].append(None)
                temporal_data[aria_name]['smpl_params']['transl'].append(None)
            
            # 3D关键点
            if aria_name in pose3d_data:
                temporal_data[aria_name]['keypoints3d'].append(pose3d_data[aria_name])
            else:
                temporal_data[aria_name]['keypoints3d'].append(None)
            
            # 2D数据
            for cam_name in all_extrinsics.keys():
                # 2D关键点
                pose2d_file = f"{root_path}/processed_data/poses2d/{cam_name}/rgb/{frame_idx:05d}.npy"
                keypoints_2d = None
                if os.path.exists(pose2d_file):
                    try:
                        pose2d = np.load(pose2d_file, allow_pickle=True)
                        if isinstance(pose2d, np.ndarray) and pose2d.size > 0:
                            for person_dict in pose2d:
                                if isinstance(person_dict, dict) and person_dict.get('human_name') == aria_name:
                                    keypoints_2d = person_dict.get('keypoints')
                                    break
                    except:
                        pass
                temporal_data[aria_name]['keypoints2d'][cam_name].append(keypoints_2d)
                
                # 2D边界框
                bbox_file = f"{root_path}/processed_data/bboxes/{cam_name}/rgb/{frame_idx:05d}.npy"
                bbox_2d = None
                if os.path.exists(bbox_file):
                    try:
                        bbox2d = np.load(bbox_file, allow_pickle=True)
                        if isinstance(bbox2d, np.ndarray) and bbox2d.size > 0:
                            for person_dict in bbox2d:
                                if isinstance(person_dict, dict) and person_dict.get('human_name') == aria_name:
                                    bbox_2d = person_dict.get('bbox')
                                    break
                    except:
                        pass
                temporal_data[aria_name]['bboxes2d'][cam_name].append(bbox_2d)

    # 保存数据
    print("保存数据...")
    for aria_name, person_data in temporal_data.items():
        # 转换SMPL参数
        smpl_params_arrays = {}
        for key, values in person_data['smpl_params'].items():
            valid_values = [v for v in values if v is not None]
            smpl_params_arrays[key] = np.array(valid_values) if valid_values else np.array([])
        
        # 先去畸变，再缩放2D数据
        keypoints2d_processed = {}
        for cam, poses in person_data['keypoints2d'].items():
            processed_poses = []
            
            # 获取该相机的畸变参数
            if cam in distortion_params:
                K_dist = distortion_params[cam]['K']
                D_dist = distortion_params[cam]['D']
            else:
                K_dist = None
                D_dist = None
            
            for pose in poses:
                if pose is not None and hasattr(pose, 'shape') and len(pose.shape) >= 2:
                    processed_pose = pose.copy().astype(np.float64)  # 确保是浮点类型
                    
                    # 步骤1: 先去畸变
                    if K_dist is not None and D_dist is not None:
                        processed_pose = undistort_points_fisheye(processed_pose, K_dist, D_dist)
                    
                    # 步骤2: 再缩放
                    if scale_factor != 1.0:
                        processed_pose[:, :2] /= scale_factor
                    
                    processed_poses.append(processed_pose)
                else:
                    processed_poses.append(pose)
            
            keypoints2d_processed[cam] = np.array(processed_poses, dtype=object)
        
        bboxes2d_processed = {}
        for cam, bboxes in person_data['bboxes2d'].items():
            processed_bboxes = []
            
            # 获取该相机的畸变参数
            if cam in distortion_params:
                K_dist = distortion_params[cam]['K']
                D_dist = distortion_params[cam]['D']
            else:
                K_dist = None
                D_dist = None
            
            for bbox in bboxes:
                if bbox is not None and hasattr(bbox, 'shape') and len(bbox.shape) >= 1 and len(bbox) >= 4:
                    processed_bbox = bbox.copy().astype(np.float64)  # 确保是浮点类型
                    
                    # 步骤1: 先去畸变
                    if K_dist is not None and D_dist is not None:
                        processed_bbox = undistort_bbox_fisheye(processed_bbox, K_dist, D_dist)
                    
                    # 步骤2: 再缩放
                    if scale_factor != 1.0:
                        processed_bbox[:4] /= scale_factor
                    
                    processed_bboxes.append(processed_bbox)
                else:
                    processed_bboxes.append(bbox)
            
            bboxes2d_processed[cam] = np.array(processed_bboxes, dtype=object)
        
        save_data = {
            'frame_indices': np.array(person_data['frame_indices']),
            'smpl_params': smpl_params_arrays,
            'keypoints3d': np.array(person_data['keypoints3d'], dtype=object),
            'keypoints2d': keypoints2d_processed,
            'bboxes2d': bboxes2d_processed
        }
        
        output_file = os.path.join(output_path, f"smpl_{aria_name}.npz")
        np.savez(output_file, **save_data)

    print(f"完成! 输出目录: {output_path}")
    print(f"数据处理完成: 2D关键点和边界框已去畸变并按缩放系数 {scale_factor} 调整")

def load_merged_data(data_path, person_name=None, camera_name=None):
    """加载合并后的数据"""
    result = {}
    
    if camera_name:
        cam_file = os.path.join(data_path, f"{camera_name}.npz")
        if os.path.exists(cam_file):
            result['camera'] = dict(np.load(cam_file, allow_pickle=True))
    else:
        result['cameras'] = {}
        for file in os.listdir(data_path):
            if file.startswith('cam') and file.endswith('.npz'):
                cam_name = file[:-4]
                result['cameras'][cam_name] = dict(np.load(os.path.join(data_path, file), allow_pickle=True))
    
    if person_name:
        person_file = os.path.join(data_path, f"smpl_{person_name}.npz")
        if os.path.exists(person_file):
            result['person'] = dict(np.load(person_file, allow_pickle=True))
    else:
        result['persons'] = {}
        for file in os.listdir(data_path):
            if file.startswith('smpl_') and file.endswith('.npz'):
                person_name_from_file = file[5:-4]
                result['persons'][person_name_from_file] = dict(np.load(os.path.join(data_path, file), allow_pickle=True))
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并EgoHumans时序数据')
    parser.add_argument('--root_path', type=str, default="/gemini/user/private/3D/data/EgoHumans/04_basketball/002_basketball", help='数据根目录路径')
    parser.add_argument('--output_path', type=str, default="/gemini/user/private/3D/data/EgoHumans/04_basketball/002_basketball/data_para", help='输出目录路径')
    parser.add_argument('--scale_factor', type=float, default=2.0, help='缩放系数')
    parser.add_argument('--invalid_arias', type=str, nargs='*', default=None, help='无效的aria列表（如果不指定，会从config文件读取）')
    parser.add_argument('--invalid_exos', type=str, nargs='*', default=None, help='无效的exo相机列表（如果不指定，会从config文件读取）')
    parser.add_argument('--manual_exo_cameras', type=str, nargs='*', default=None, help='手动标注的exo相机列表（如果不指定，会从config文件读取）')
    parser.add_argument('--test_load', action='store_true', help='测试加载数据')
    
    args = parser.parse_args()
    
    # 如果命令行参数提供了空列表，保持为空列表；如果没有提供，则传递None让函数从配置文件读取
    invalid_arias = args.invalid_arias if args.invalid_arias is not None else None
    invalid_exos = args.invalid_exos if args.invalid_exos is not None else None
    manual_exo_cameras = args.manual_exo_cameras if args.manual_exo_cameras is not None else None
    
    merge_temporal_data(
        args.root_path, 
        args.output_path, 
        args.scale_factor,
        invalid_arias,
        invalid_exos,
        manual_exo_cameras
    )
    
    if args.test_load:
        print("\n测试加载数据...")
        data = load_merged_data(args.output_path)
        
        if 'cameras' in data:
            print(f"相机数量: {len(data['cameras'])}")
        if 'persons' in data:
            print(f"人体数量: {len(data['persons'])}") 