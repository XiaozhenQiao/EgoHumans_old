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
import logging
from datetime import datetime
from typing import Iterable, Optional, Union
import copy
import math
import torch.nn as nn


from pytorch3d.transforms import (
        axis_angle_to_matrix,
        axis_angle_to_quaternion,
        euler_angles_to_matrix,
        matrix_to_euler_angles,
        matrix_to_quaternion,
        matrix_to_rotation_6d,
        quaternion_to_axis_angle,
        quaternion_to_matrix,
        rotation_6d_to_matrix,
        )
PYTORCH3D_AVAILABLE = True


class Compose:
    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript.

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self,
                 rotation: Union[torch.Tensor, np.ndarray],
                 convention: str = 'xyz',
                 **kwargs):
        convention = convention.lower()
        if not (set(convention) == set('xyz') and len(convention) == 3):
            raise ValueError(f'Invalid convention {convention}.')
        if isinstance(rotation, np.ndarray):
            data_type = 'numpy'
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = 'tensor'
        else:
            raise TypeError(
                'Type of rotation should be torch.Tensor or numpy.ndarray')
        for t in self.transforms:
            if 'convention' in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == 'numpy':
            rotation = rotation.detach().cpu().numpy()
        return rotation

def aa_to_rotmat(
    axis_angle: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert axis_angle to rotation matrixs.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f'Invalid input axis angles shape f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix])
    return t(axis_angle)

def rotmat_to_aa(
    matrix: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Convert rotation matrixs to axis angles.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {"x", "y", and "z"}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion, quaternion_to_axis_angle])
    return t(matrix)

def combine_RT(R, T):
    if R.ndim == 2:
        R = R[None]

    batch_size = R.shape[0]
    T = T.view(batch_size, 3, 1)
    RT = torch.zeros(batch_size, 4, 4).to(R.device)
    RT[:, 3, 3] = 1
    RT[:, :3, :3] = R
    RT[:, :3, 3:] = T
    return RT

def eliminate_external_matrix(
    R: Union[np.ndarray, torch.Tensor],
    T: Union[np.ndarray, torch.Tensor],
    body_model: nn.Module,
    global_orient: Optional[Union[np.ndarray, torch.Tensor]] = None,
    body_pose: Optional[Union[np.ndarray, torch.Tensor]] = None,
    transl: Optional[Union[np.ndarray, torch.Tensor]] = None,
    betas: Optional[Union[np.ndarray, torch.Tensor]] = None,
    gender: Optional[Union[np.ndarray, torch.Tensor]] = None,
    **kwargs,
):
    device = global_orient.device

    # 先获取世界坐标系下的关节位置（特别是根关节/pelvis）
    body_model_output = body_model(global_orient=global_orient,
                                   betas=betas,
                                   body_pose=body_pose,
                                   transl=transl,
                                   gender=gender)
    joints = body_model_output['joints']
    pelvis_world = joints[:, 0]  # 根关节在世界坐标系中的位置
    
    # 计算相机坐标系下的全局旋转
    global_orient_cam = rotmat_to_aa(
        torch.bmm(R.to(device), aa_to_rotmat(global_orient)))
    
    # 计算相机坐标系下的根关节位置
    RT = combine_RT(R, T).to(device)
    pelvis_homo = torch.cat([pelvis_world, torch.ones_like(pelvis_world)[:, :1]], 1)
    pelvis_homo = torch.bmm(RT, pelvis_homo[..., None])
    pelvis_cam = pelvis_homo[:, :3, 0] / pelvis_homo[:, 3:4, 0]
    
    # 使用相机坐标系下的旋转重新计算body model，获取新的根关节位置
    params_cam = dict()
    params_cam['betas'] = betas
    params_cam['global_orient'] = global_orient_cam
    params_cam['body_pose'] = body_pose
    params_cam['transl'] = torch.zeros_like(transl)  # 先设为0，后面计算正确的transl
    params_cam['gender'] = gender
    
    body_model_output_cam = body_model(**params_cam)
    pelvis_cam_canonical = body_model_output_cam['joints'][:, 0]  # 相机坐标系下标准姿态的根关节位置
    
    # 计算相机坐标系下的正确平移
    transl_cam = pelvis_cam - pelvis_cam_canonical
    
    return global_orient_cam, transl_cam

def setup_logging(log_file=None):
    """设置日志配置"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"merge_dataset_save_{timestamp}.log"
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger(__name__)


def cleanup_output_directory(output_path, logger=None):
    """清理输出目录中的所有文件"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if os.path.exists(output_path):
        logger.info(f"清理输出目录: {output_path}")
        try:
            # 删除目录中的所有文件，但保留目录本身
            for file_path in glob.glob(os.path.join(output_path, "*")):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"删除文件: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.info(f"删除目录: {file_path}")
            logger.info("输出目录清理完成")
        except Exception as e:
            logger.warning(f"清理输出目录时出错: {e}")
    else:
        logger.info(f"创建输出目录: {output_path}")
        os.makedirs(output_path, exist_ok=True)


def load_config(root_path, logger=None):
    """从数据集序列对应的config文件中加载配置参数"""
    if logger is None:
        logger = logging.getLogger(__name__)
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
        logger.warning(f"无法从路径 {root_path} 中提取运动类型")
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
                logger.info(f"成功加载配置文件: {config_path}")
                break
            except Exception as e:
                logger.warning(f"加载配置文件失败: {config_path}, 错误: {e}")
    
    if not config:
        logger.warning(f"未找到运动类型 '{sport_type}' 序列 '{sequence_name}' 的配置文件")
        logger.warning(f"尝试过的路径: {config_paths}")
    
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
        logging.getLogger(__name__).warning(f"2D点去畸变失败: {e}")
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
        logging.getLogger(__name__).warning(f"边界框去畸变失败: {e}")
        # 如果去畸变失败，返回原始数据
        return bbox

def project_3d_to_2d(points_3d, K, R, T):
    """将3D点投影到2D像素坐标
    
    Args:
        points_3d: (N, 3) 3D关键点坐标
        K: (3, 3) 相机内参矩阵
        R: (3, 3) 旋转矩阵
        T: (3,) 平移向量
    
    Returns:
        points_2d: (N, 3) 2D关键点坐标 [x, y, confidence]
    """
    if points_3d is None:
        return None
        
    points_3d = np.array(points_3d)
    if len(points_3d.shape) != 2 or points_3d.shape[1] < 3:
        return None
    
    # 提取3D坐标和置信度
    points_3d_coords = points_3d[:, :3]  # (N, 3)
    confidence = points_3d[:, 3] if points_3d.shape[1] > 3 else np.ones(points_3d.shape[0])
    
    # 转换到相机坐标系：X_cam = R * X_world + T
    points_cam = (R @ points_3d_coords.T + T.reshape(-1, 1)).T  # (N, 3)
    
    # 投影到像素坐标系
    # 处理深度为0的情况
    valid_depth = points_cam[:, 2] > 1e-6
    points_2d = np.zeros((points_3d.shape[0], 3))
    
    if np.any(valid_depth):
        # 投影到归一化平面
        points_normalized = points_cam[valid_depth, :2] / points_cam[valid_depth, 2:3]  # (N_valid, 2)
        
        # 应用内参矩阵
        points_pixel = (K[:2, :2] @ points_normalized.T + K[:2, 2:3]).T  # (N_valid, 2)
        
        # 设置有效投影点
        points_2d[valid_depth, :2] = points_pixel
        points_2d[valid_depth, 2] = confidence[valid_depth]  # 保持原始置信度
    
    return points_2d

def calculate_bbox_from_keypoints(keypoints_2d, padding=10):
    """从2D关键点计算边界框
    
    Args:
        keypoints_2d: (N, 3) 2D关键点坐标 [x, y, confidence]
        padding: 边界框填充像素数
    
    Returns:
        bbox: [x1, y1, x2, y2] 边界框坐标
    """
    if keypoints_2d is None:
        return None
        
    keypoints_2d = np.array(keypoints_2d)
    if len(keypoints_2d.shape) != 2 or keypoints_2d.shape[1] < 2:
        return None
    
    # 只考虑置信度大于0的关键点
    if keypoints_2d.shape[1] > 2:
        valid_kps = keypoints_2d[keypoints_2d[:, 2] > 0]
    else:
        valid_kps = keypoints_2d
    
    if len(valid_kps) == 0:
        return None
    
    # 计算边界框
    x_coords = valid_kps[:, 0]
    y_coords = valid_kps[:, 1]
    
    x1 = np.min(x_coords) - padding
    y1 = np.min(y_coords) - padding
    x2 = np.max(x_coords) + padding
    y2 = np.max(y_coords) + padding
    
    return np.array([x1, y1, x2, y2])

def merge_temporal_data(root_path, output_path, scale_factor=2.0, invalid_arias=None, invalid_exos=None, manual_exo_cameras=None, max_frames=None, smpl_model_path=None, logger=None):
    """合并时序数据，包含去畸变和缩放处理"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"处理数据: {root_path}")

    # 初始化SMPL模型用于相机坐标系变换
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    body_model = SMPL(model_path=smpl_model_path, gender="neutral", create_transl=False).to(device)

    # 加载配置文件
    config = load_config(root_path, logger)
    
    invalid_arias = config.get('INVALID_ARIAS', [])
    invalid_exos = config.get('INVALID_EXOS', [])
    calibration_config = config.get('CALIBRATION', {})
    manual_exo_cameras = calibration_config.get('MANUAL_EXO_CAMERAS', [])

    # 首先清理输出目录
    cleanup_output_directory(output_path, logger)

    # 获取帧数（从有效的相机中获取）
    exo_dir = f"{root_path}/exo"
    valid_cam_dirs = [d for d in os.listdir(exo_dir) if d.startswith('cam') and os.path.isdir(os.path.join(exo_dir, d))]
    # valid_cam_dirs = [d for d in all_cam_dirs if d not in invalid_exos]
    
    # IPython.embed()
    
    first_cam = sorted(valid_cam_dirs)[0]
    images_dir = f"{root_path}/exo/{first_cam}/undistorted_images_scale2.0"
    total_frames = len(os.listdir(images_dir))
    
    # 应用帧数限制
    if max_frames is not None and max_frames > 0:
        num_frames = min(total_frames, max_frames)
        logger.info(f"总帧数: {total_frames}, 限制处理帧数: {num_frames}")
    else:
        num_frames = total_frames
        logger.info(f"总帧数: {num_frames}")
    
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
                    logger.info(f"加载手动标注相机参数: {manual_cam} (外参矩阵不需要缩放调整)")
                else:
                    logger.warning(f"手动标注相机参数文件格式错误: {manual_cam} (期望4x4矩阵)")
            except Exception as e:
                logger.warning(f"加载手动标注相机参数失败: {manual_cam}, 错误: {e}")
        else:
            logger.warning(f"未找到手动标注相机参数文件: {manual_cam_file}")

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
                logger.info(f"为手动标注相机 {manual_cam} 添加内参数据 (相机ID: {cam_id})")
            else:
                logger.warning(f"无法找到手动标注相机 {manual_cam} 的内参数据")

    # 合并相机数据
    logger.info("合并相机数据...")
    processed_extrinsics = {}  # 存储处理后的外参，用于投影
    processed_intrinsics = {}  # 存储处理后的内参，用于投影
    
    for cam_name in all_extrinsics.keys():
        extrinsic = all_extrinsics[cam_name]
        extrinsic = np.dot(extrinsic, primary_transform)
        extrinsic = np.linalg.inv(extrinsic)
        
        # 手动标注相机参数已在前面处理，这里不需要重复处理
        
        if cam_name in manual_exo_cameras:
            logger.info(f"手动标注相机 {cam_name} 已完成变换处理 (primary_transform + inverse)")

        # 应用缩放到内参
        K_original = intrinsics_all[cam_name].copy()
        K_scaled = K_original.copy()
        if scale_factor != 1.0:
            K_scaled[0, 0] /= scale_factor  # fx
            K_scaled[1, 1] /= scale_factor  # fy
            K_scaled[0, 2] /= scale_factor  # cx
            K_scaled[1, 2] /= scale_factor  # cy
        
        # 存储处理后的参数用于投影
        processed_extrinsics[cam_name] = extrinsic
        processed_intrinsics[cam_name] = K_scaled
        
        save_cam_data = {
            'K': K_scaled,
            'R': extrinsic[:3, :3],
            'T': extrinsic[:3, 3],
            'extrinsic': extrinsic,
            'scale_factor': scale_factor
        }
        
        np.savez(os.path.join(output_path, f"{cam_name}.npz"), **save_cam_data)
    
    # 合并SMPL和关键点数据
    logger.info("合并SMPL和关键点数据...")
    
    temporal_data = {}
    all_aria_names = ['aria01', 'aria02', 'aria03', 'aria04']
    # 过滤无效的aria
    valid_aria_names = [aria for aria in all_aria_names if aria not in invalid_arias]
    
    if not valid_aria_names:
        raise ValueError(f"没有有效的aria，所有aria都被标记为无效: {invalid_arias}")
    
    logger.info(f"有效aria: {valid_aria_names}")
    
    for aria_name in valid_aria_names:
        temporal_data[aria_name] = {
            'frame_indices': [],
            'smpl_params': {'betas': [], 'body_pose': [], 'global_orient': [], 'transl': []},
            'smpl_params_cam': {},  # 相机坐标系下的参数
            'keypoints3d': [],
            'keypoints2d': {},
            'bboxes2d': {}
        }
        
        # 为每个相机初始化相机坐标系下的SMPL参数
        for cam_name in all_extrinsics.keys():
            temporal_data[aria_name]['smpl_params_cam'][cam_name] = {
                'global_orient_cam': [],
                'transl_cam': []
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
            current_smpl_params = None
            if aria_name in smpl_params:
                current_smpl_params = smpl_params[aria_name]
                temporal_data[aria_name]['smpl_params']['betas'].append(current_smpl_params['betas'])
                temporal_data[aria_name]['smpl_params']['body_pose'].append(current_smpl_params['body_pose'])
                temporal_data[aria_name]['smpl_params']['global_orient'].append(current_smpl_params['global_orient'])
                temporal_data[aria_name]['smpl_params']['transl'].append(current_smpl_params['transl'])
            else:
                temporal_data[aria_name]['smpl_params']['betas'].append(None)
                temporal_data[aria_name]['smpl_params']['body_pose'].append(None)
                temporal_data[aria_name]['smpl_params']['global_orient'].append(None)
                temporal_data[aria_name]['smpl_params']['transl'].append(None)
            
            # 计算相机坐标系下的SMPL参数
            for cam_name in processed_extrinsics.keys():
                global_orient_cam = None
                transl_cam = None
                
                if (body_model is not None and current_smpl_params is not None):
                    # 准备输入数据
                    betas = torch.tensor(current_smpl_params['betas']).float().to(device).unsqueeze(0)
                    body_pose = torch.tensor(current_smpl_params['body_pose']).float().to(device).unsqueeze(0)
                    global_orient = torch.tensor(current_smpl_params['global_orient']).float().to(device).unsqueeze(0)
                    transl = torch.tensor(current_smpl_params['transl']).float().to(device).unsqueeze(0)
                    
                    # 获取相机外参 (processed_extrinsics是C2W，需要转换为W2C)
                    extrinsic_c2w = processed_extrinsics[cam_name]
                    extrinsic_w2c = np.linalg.inv(extrinsic_c2w)
                    R_cam = torch.tensor(extrinsic_w2c[:3, :3]).float().to(device).unsqueeze(0)
                    T_cam = torch.tensor(extrinsic_w2c[:3, 3]).float().to(device).unsqueeze(0)
                        
                    # 计算相机坐标系下的参数
                    global_orient_cam_tensor, transl_cam_tensor = eliminate_external_matrix(
                        R=R_cam,
                        T=T_cam,
                        body_model=body_model,
                        global_orient=global_orient,
                        body_pose=body_pose,
                        transl=transl,
                        betas=betas
                    )
                        
                    # 转换回numpy
                    global_orient_cam = global_orient_cam_tensor.squeeze(0).detach().cpu().numpy()
                    transl_cam = transl_cam_tensor.squeeze(0).detach().cpu().numpy()

                temporal_data[aria_name]['smpl_params_cam'][cam_name]['global_orient_cam'].append(global_orient_cam)
                temporal_data[aria_name]['smpl_params_cam'][cam_name]['transl_cam'].append(transl_cam)
            
            # 3D关键点
            if aria_name in pose3d_data:
                temporal_data[aria_name]['keypoints3d'].append(pose3d_data[aria_name])
            else:
                temporal_data[aria_name]['keypoints3d'].append(None)
            
            # 2D数据 - 从3D关键点投影计算
            for cam_name in processed_extrinsics.keys():
                # 从3D关键点投影计算2D关键点
                keypoints_2d = None
                bbox_2d = None
                
                # 获取当前人体的3D关键点
                if aria_name in pose3d_data:
                    keypoints_3d = pose3d_data[aria_name]
                    
                    if keypoints_3d is not None:
                        # 获取处理后的相机参数
                        K = processed_intrinsics[cam_name]  # 处理后的内参矩阵
                        
                        # processed_extrinsics是C2W矩阵，需要转换为W2C进行投影
                        extrinsic_c2w = processed_extrinsics[cam_name]
                        extrinsic_w2c = np.linalg.inv(extrinsic_c2w)  # 转换为W2C
                        R = extrinsic_w2c[:3, :3]  # W2C旋转矩阵
                        T = extrinsic_w2c[:3, 3]   # W2C平移向量
                        
                        # 投影3D关键点到2D
                        keypoints_2d = project_3d_to_2d(keypoints_3d, K, R, T)
                        
                        # 从2D关键点计算边界框
                        if keypoints_2d is not None:
                            bbox_2d = calculate_bbox_from_keypoints(keypoints_2d, padding=10)
                
                temporal_data[aria_name]['keypoints2d'][cam_name].append(keypoints_2d)
                temporal_data[aria_name]['bboxes2d'][cam_name].append(bbox_2d)

    # 保存数据
    logger.info("保存数据...")
    for aria_name, person_data in temporal_data.items():
        num_frames = len(person_data['frame_indices'])
        logger.info(f"处理 {aria_name}，总帧数: {num_frames}")
        
        # 首先收集所有有效数据的形状信息
        shapes_info = {
            'smpl_params': {},
            'smpl_params_cam': {},
            'keypoints3d': None,
            'keypoints2d': {},
            'bboxes2d': {}
        }
        
        # 收集SMPL参数形状
        for key, values in person_data['smpl_params'].items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                shapes_info['smpl_params'][key] = valid_values[0].shape
        
        # 收集相机坐标系下的SMPL参数形状
        for cam_name in all_extrinsics.keys():
            shapes_info['smpl_params_cam'][cam_name] = {}
            cam_params = person_data['smpl_params_cam'][cam_name]
            for key, values in cam_params.items():
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    shapes_info['smpl_params_cam'][cam_name][key] = valid_values[0].shape
        
        # 收集3D关键点形状
        valid_kp3d = [kp for kp in person_data['keypoints3d'] if kp is not None]
        if valid_kp3d:
            shapes_info['keypoints3d'] = valid_kp3d[0].shape
        
        # 收集2D关键点和边界框形状
        for cam_name in all_extrinsics.keys():
            # 2D关键点形状 - 找到最大的关键点数量
            valid_kp2d = [kp for kp in person_data['keypoints2d'][cam_name] if kp is not None and hasattr(kp, 'shape')]
            if valid_kp2d:
                # 找到最大的关键点数量
                max_keypoints = max(kp.shape[0] for kp in valid_kp2d)
                # 假设所有关键点都有相同的特征维度（通常是3: x, y, confidence）
                feature_dim = valid_kp2d[0].shape[1] if len(valid_kp2d[0].shape) > 1 else 3
                shapes_info['keypoints2d'][cam_name] = (max_keypoints, feature_dim)
                logger.info(f"相机 {cam_name} 的2D关键点最大形状: {shapes_info['keypoints2d'][cam_name]}")
            
            # 2D边界框形状
            valid_bbox = [bbox for bbox in person_data['bboxes2d'][cam_name] if bbox is not None and hasattr(bbox, 'shape')]
            if valid_bbox:
                # 找到最大的边界框维度
                max_bbox_dims = max(bbox.shape for bbox in valid_bbox)
                shapes_info['bboxes2d'][cam_name] = max_bbox_dims
                logger.info(f"相机 {cam_name} 的2D边界框最大形状: {shapes_info['bboxes2d'][cam_name]}")
        
        # 统一化SMPL参数并创建标志位
        smpl_params_arrays = {}
        has_smpl_flags = {}
        
        for key, values in person_data['smpl_params'].items():
            if key in shapes_info['smpl_params']:
                shape = shapes_info['smpl_params'][key]
                unified_array = np.zeros((num_frames, *shape), dtype=np.float64)
                flags = np.zeros(num_frames, dtype=bool)
                
                for i, value in enumerate(values):
                    if value is not None:
                        unified_array[i] = value
                        flags[i] = True
                
                smpl_params_arrays[key] = unified_array
                has_smpl_flags[f'has_{key}'] = flags
            else:
                smpl_params_arrays[key] = np.array([])
                has_smpl_flags[f'has_{key}'] = np.zeros(num_frames, dtype=bool)
        
        # 统一化相机坐标系下的SMPL参数并创建标志位
        smpl_params_cam_arrays = {}
        has_smpl_cam_flags = {}
        
        for cam_name in all_extrinsics.keys():
            smpl_params_cam_arrays[cam_name] = {}
            cam_params = person_data['smpl_params_cam'][cam_name]
            cam_shapes = shapes_info['smpl_params_cam'][cam_name]
            
            for key, values in cam_params.items():
                if key in cam_shapes:
                    shape = cam_shapes[key]
                    unified_array = np.zeros((num_frames, *shape), dtype=np.float64)
                    flags = np.zeros(num_frames, dtype=bool)
                    
                    for i, value in enumerate(values):
                        if value is not None:
                            unified_array[i] = value
                            flags[i] = True
                    
                    smpl_params_cam_arrays[cam_name][key] = unified_array
                    has_smpl_cam_flags[f'has_{key}_{cam_name}'] = flags
                else:
                    smpl_params_cam_arrays[cam_name][key] = np.array([])
                    has_smpl_cam_flags[f'has_{key}_{cam_name}'] = np.zeros(num_frames, dtype=bool)
        
        # 统一化3D关键点并创建标志位
        if shapes_info['keypoints3d'] is not None:
            shape = shapes_info['keypoints3d']
            keypoints3d_array = np.zeros((num_frames, *shape), dtype=np.float64)
            has_kp3d = np.zeros(num_frames, dtype=bool)
            
            for i, kp in enumerate(person_data['keypoints3d']):
                if kp is not None:
                    keypoints3d_array[i] = kp
                    has_kp3d[i] = True
        else:
            keypoints3d_array = np.array([])
            has_kp3d = np.zeros(num_frames, dtype=bool)
        
        # 统一化2D关键点数据并创建标志位
        keypoints2d_processed = {}
        has_kp2d_flags = {}
        
        for cam in all_extrinsics.keys():
            poses = person_data['keypoints2d'][cam]
            
            # 统一化处理 - keypoints2d 已经是通过投影计算得到的，使用缩放后的内参
            if cam in shapes_info['keypoints2d']:
                shape = shapes_info['keypoints2d'][cam]
                unified_kp2d = np.zeros((num_frames, *shape), dtype=np.float64)
                flags = np.zeros(num_frames, dtype=bool)
                
                for i, pose in enumerate(poses):
                    if pose is not None and hasattr(pose, 'shape') and len(pose.shape) >= 2:
                        processed_pose = pose.copy().astype(np.float64)
                        
                        # keypoints2d已经通过缩放后的内参矩阵投影得到，不需要额外的去畸变和缩放处理
                        
                        # 处理形状不匹配问题：如果当前pose小于目标形状，用零填充
                        target_shape = shape
                        if processed_pose.shape[0] <= target_shape[0]:
                            # 创建一个目标大小的零数组
                            padded_pose = np.zeros(target_shape, dtype=np.float64)
                            # 复制实际数据到前面的位置
                            padded_pose[:processed_pose.shape[0], :processed_pose.shape[1]] = processed_pose
                            unified_kp2d[i] = padded_pose
                        else:
                            # 如果当前pose大于目标形状，截断到目标大小
                            unified_kp2d[i] = processed_pose[:target_shape[0], :target_shape[1]]
                        
                        flags[i] = True
                
                keypoints2d_processed[cam] = unified_kp2d
                has_kp2d_flags[f'has_kp2d_{cam}'] = flags
            else:
                keypoints2d_processed[cam] = np.array([])
                has_kp2d_flags[f'has_kp2d_{cam}'] = np.zeros(num_frames, dtype=bool)
        
        # 统一化2D边界框数据并创建标志位
        bboxes2d_processed = {}
        has_bbox_flags = {}
        
        for cam in all_extrinsics.keys():
            bboxes = person_data['bboxes2d'][cam]
            
            # 统一化处理 - bboxes2d 已经是通过投影的keypoints2d计算得到的
            if cam in shapes_info['bboxes2d']:
                shape = shapes_info['bboxes2d'][cam]
                unified_bbox = np.zeros((num_frames, *shape), dtype=np.float64)
                flags = np.zeros(num_frames, dtype=bool)
                
                for i, bbox in enumerate(bboxes):
                    if bbox is not None and hasattr(bbox, 'shape') and len(bbox.shape) >= 1 and len(bbox) >= 4:
                        processed_bbox = bbox.copy().astype(np.float64)
                        
                        # bboxes2d已经通过投影的keypoints2d计算得到，不需要额外的去畸变和缩放处理
                        
                        # 处理形状不匹配问题：如果当前bbox小于目标形状，用零填充
                        target_shape = shape
                        if len(processed_bbox.shape) == 1:
                            # 1D数组情况
                            if processed_bbox.shape[0] <= target_shape[0]:
                                padded_bbox = np.zeros(target_shape, dtype=np.float64)
                                padded_bbox[:processed_bbox.shape[0]] = processed_bbox
                                unified_bbox[i] = padded_bbox
                            else:
                                unified_bbox[i] = processed_bbox[:target_shape[0]]
                        else:
                            # 多维数组情况
                            if processed_bbox.shape[0] <= target_shape[0]:
                                padded_bbox = np.zeros(target_shape, dtype=np.float64)
                                padded_bbox[:processed_bbox.shape[0]] = processed_bbox
                                unified_bbox[i] = padded_bbox
                            else:
                                unified_bbox[i] = processed_bbox[:target_shape[0]]
                        
                        flags[i] = True
                
                bboxes2d_processed[cam] = unified_bbox
                has_bbox_flags[f'has_bbox_{cam}'] = flags
            else:
                bboxes2d_processed[cam] = np.array([])
                has_bbox_flags[f'has_bbox_{cam}'] = np.zeros(num_frames, dtype=bool)
        
        # 合并所有数据和标志位
        save_data = {
            'frame_indices': np.array(person_data['frame_indices']),
            'smpl_params': smpl_params_arrays,
            'smpl_params_cam': smpl_params_cam_arrays,
            'keypoints3d': keypoints3d_array,
            'keypoints2d': keypoints2d_processed,
            'bboxes2d': bboxes2d_processed,
        }
        
        output_file = os.path.join(output_path, f"smpl_{aria_name}.npz")
        np.savez(output_file, **save_data)
        
        # 打印统一化结果统计
        logger.info(f"  - SMPL参数统计:")
        for key in smpl_params_arrays.keys():
            valid_count = np.sum(has_smpl_flags[f'has_{key}'])
            logger.info(f"    {key}: {valid_count}/{num_frames} 帧有效")
        
        # 打印相机坐标系下的SMPL参数统计
        logger.info(f"  - 相机坐标系SMPL参数统计:")
        for cam_name in all_extrinsics.keys():
            for key in smpl_params_cam_arrays[cam_name].keys():
                flag_key = f'has_{key}_{cam_name}'
                if flag_key in has_smpl_cam_flags:
                    valid_count = np.sum(has_smpl_cam_flags[flag_key])
                    logger.info(f"    {cam_name}_{key}: {valid_count}/{num_frames} 帧有效")
        
        kp3d_valid = np.sum(has_kp3d)
        logger.info(f"  - 3D关键点: {kp3d_valid}/{num_frames} 帧有效")
        
        logger.info(f"  - 2D关键点统计:")
        for cam in all_extrinsics.keys():
            if f'has_kp2d_{cam}' in has_kp2d_flags:
                valid_count = np.sum(has_kp2d_flags[f'has_kp2d_{cam}'])
                logger.info(f"    {cam}: {valid_count}/{num_frames} 帧有效")
        
        logger.info(f"  - 2D边界框统计:")
        for cam in all_extrinsics.keys():
            if f'has_bbox_{cam}' in has_bbox_flags:
                valid_count = np.sum(has_bbox_flags[f'has_bbox_{cam}'])
                logger.info(f"    {cam}: {valid_count}/{num_frames} 帧有效")

    logger.info(f"完成! 输出目录: {output_path}")
    logger.info(f"数据处理完成:")
    logger.info(f"  - 所有参数已统一为相同shape的numpy数组，缺失数据填充为0")
    logger.info(f"  - 2D关键点和边界框已从3D关键点投影计算得到，使用缩放系数 {scale_factor} 的内参矩阵")
    if body_model is not None:
        logger.info(f"  - 已计算相机坐标系下的SMPL参数 (global_orient_cam, transl_cam)")
    else:
        logger.info(f"  - 未计算相机坐标系下的SMPL参数 (SMPL模型未加载)")
    logger.info(f"  - 添加了时序标志位数组以标识每帧数据的有效性")
    logger.info(f"  - 标志位命名规则: has_kp3d, has_betas/has_body_pose/has_global_orient/has_transl, has_global_orient_cam_{{cam}}/has_transl_cam_{{cam}}, has_kp2d_{{cam}}, has_bbox_{{cam}}")

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
    parser.add_argument('--root_path', type=str, default="/gemini/user/private/3D/data/EgoHumans/01_tagging/002_tagging", help='数据根目录路径')
    parser.add_argument('--output_path', type=str, default="/gemini/user/private/3D/data/EgoHumans/01_tagging/002_tagging/data_para", help='输出目录路径')
    parser.add_argument('--scale_factor', type=float, default=2.0, help='缩放系数')
    parser.add_argument('--invalid_arias', type=str, nargs='*', default=None, help='无效的aria列表（如果不指定，会从config文件读取）')
    parser.add_argument('--invalid_exos', type=str, nargs='*', default=None, help='无效的exo相机列表（如果不指定，会从config文件读取）')
    parser.add_argument('--manual_exo_cameras', type=str, nargs='*', default=None, help='手动标注的exo相机列表（如果不指定，会从config文件读取）')
    parser.add_argument('--max_frames', type=int, default=None, help='限制处理的最大帧数（如果不指定，处理所有帧）')
    parser.add_argument('--smpl_model_path', type=str, default="/gemini/user/private/3D/data/body_models/smpl", help='SMPL模型路径（如果不指定，将尝试自动查找）')
    parser.add_argument('--test_load', action='store_true', help='测试加载数据')
    parser.add_argument('--log_file', type=str, default=None, help='日志文件路径（如果不指定，将使用默认文件名）')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_file)
    
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
        manual_exo_cameras,
        args.max_frames,
        args.smpl_model_path,
        logger
    )
    
    if args.test_load:
        logger.info("测试加载数据...")
        data = load_merged_data(args.output_path)
        
        if 'cameras' in data:
            logger.info(f"相机数量: {len(data['cameras'])}")
        if 'persons' in data:
            logger.info(f"人体数量: {len(data['persons'])}") 