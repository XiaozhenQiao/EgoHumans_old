#!/usr/bin/env python3
"""
优化版的单帧处理脚本
使用合并后的时序数据，支持指定帧号和灵活的渲染选项
"""

import torch
import numpy as np
import os
import cv2
from smplx import SMPL    
import argparse
from t3drender.render.render_functions import render_rgb
from t3drender.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


def load_merged_data(data_para_path):
    """加载合并后的摄像机和人体数据"""
    camera_data = {}
    person_data = {}
    
    # 加载摄像机数据
    for file in os.listdir(data_para_path):
        if file.startswith('cam') and file.endswith('.npz'):
            cam_name = file[:-4]
            cam_file = os.path.join(data_para_path, file)
            camera_data[cam_name] = dict(np.load(cam_file, allow_pickle=True))
    
    # 加载人体数据
    for file in os.listdir(data_para_path):
        if file.startswith('smpl_') and file.endswith('.npz'):
            person_name = file[5:-4]
            person_file = os.path.join(data_para_path, file)
            person_data[person_name] = dict(np.load(person_file, allow_pickle=True))
    
    return camera_data, person_data


def get_frame_data(person_data_dict, frame_idx):
    """从时序数据中提取指定帧的数据"""
    frame_data = {}
    
    for person_name, person_data in person_data_dict.items():
        frame_indices = person_data['frame_indices']
        
        if frame_idx in frame_indices:
            time_idx = np.where(frame_indices == frame_idx)[0][0]
            
            # 提取SMPL参数
            smpl_params = person_data['smpl_params'].item()
            frame_smpl = {
                'betas': smpl_params['betas'][time_idx],
                'body_pose': smpl_params['body_pose'][time_idx],
                'global_orient': smpl_params['global_orient'][time_idx],
                'transl': smpl_params['transl'][time_idx]
            }
            
            # 提取3D关键点
            poses3d = person_data['poses3d'][time_idx] if len(person_data['poses3d']) > time_idx else None
            
            # 提取2D数据
            poses2d = {}
            bboxes2d = {}
            
            if 'poses2d' in person_data:
                poses2d_all = person_data['poses2d'].item()
                for cam_name in poses2d_all.keys():
                    poses2d[cam_name] = poses2d_all[cam_name][time_idx] if len(poses2d_all[cam_name]) > time_idx else None
            
            if 'bboxes2d' in person_data:
                bboxes2d_all = person_data['bboxes2d'].item()
                for cam_name in bboxes2d_all.keys():
                    bboxes2d[cam_name] = bboxes2d_all[cam_name][time_idx] if len(bboxes2d_all[cam_name]) > time_idx else None
            
            frame_data[person_name] = {
                'smpl_params': frame_smpl,
                'poses3d': poses3d,
                'poses2d': poses2d,
                'bboxes2d': bboxes2d
            }
    
    return frame_data


def vis_smpl(smpl_verts, body_model, device, cameras, batch_size=30, resolution=(512, 512), color=[1, 1, 1], verbose=False):
    """渲染单个SMPL模型"""
    smpl_meshes = Meshes(
        verts=torch.Tensor(smpl_verts).to(device), 
        faces=torch.Tensor(body_model.faces.astype(np.int64)).to(device)[None].repeat_interleave(len(smpl_verts), dim=0)
    )
    color_tensor = torch.ones_like(smpl_meshes.verts_padded())
    color_tensor[..., :3] = torch.Tensor(color).to(device)[None][None]
    smpl_meshes.textures = TexturesVertex(color_tensor)
    image_tensors = render_rgb(smpl_meshes, device=device, resolution=resolution, cameras=cameras, batch_size=batch_size, verbose=verbose)
    return (image_tensors.cpu().numpy() * 255).astype(np.uint8)


def vis_multiple_smpl(smpl_verts_list, body_model, device, cameras, batch_size=30, resolution=(512, 512), colors=None, verbose=False):
    """渲染多个SMPL模型"""
    if not smpl_verts_list:
        return None
    
    # 默认颜色
    if colors is None:
        default_colors = [
            [1.0, 0.7, 0.7],  # Light red
            [0.7, 1.0, 0.7],  # Light green  
            [0.7, 0.7, 1.0],  # Light blue
            [1.0, 1.0, 0.7],  # Light yellow
            [1.0, 0.7, 1.0],  # Light magenta
            [0.7, 1.0, 1.0],  # Light cyan
        ]
        colors = [default_colors[i % len(default_colors)] for i in range(len(smpl_verts_list))]
    
    # 合并所有顶点和面
    all_verts = []
    all_faces = []
    all_colors = []
    
    vertex_offset = 0
    for i, smpl_verts in enumerate(smpl_verts_list):
        all_verts.append(torch.Tensor(smpl_verts).to(device))
        
        faces = torch.Tensor(body_model.faces.astype(np.int64)).to(device) + vertex_offset
        all_faces.append(faces)
        
        person_color = torch.ones((smpl_verts.shape[1], 3)).to(device) * torch.Tensor(colors[i]).to(device)
        all_colors.append(person_color)
        
        vertex_offset += smpl_verts.shape[1]
    
    # 连接所有数据
    combined_verts = torch.cat(all_verts, dim=1)
    combined_faces = torch.cat(all_faces, dim=0)
    combined_colors = torch.cat(all_colors, dim=0)
    
    # 创建网格
    batch_size_actual = combined_verts.shape[0]
    smpl_meshes = Meshes(
        verts=combined_verts, 
        faces=combined_faces[None].repeat_interleave(batch_size_actual, dim=0)
    )
    
    color_tensor = combined_colors[None].repeat_interleave(batch_size_actual, dim=0)
    smpl_meshes.textures = TexturesVertex(color_tensor)
    
    image_tensors = render_rgb(smpl_meshes, device=device, resolution=resolution, cameras=cameras, batch_size=batch_size, verbose=verbose)
    return (image_tensors.cpu().numpy() * 255).astype(np.uint8)


def draw_keypoints(image, keypoints, skeleton=None, color=(0, 255, 0), thickness=2, radius=3):
    """绘制2D关键点"""
    if keypoints is None:
        return image
    
    img = image.copy()
    keypoints = np.array(keypoints)
    
    # 绘制骨架连接
    if skeleton is not None:
        for connection in skeleton:
            start_idx, end_idx = connection
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                keypoints[start_idx][0] > 0 and keypoints[start_idx][1] > 0 and
                keypoints[end_idx][0] > 0 and keypoints[end_idx][1] > 0):
                pt1 = tuple(map(int, keypoints[start_idx][:2]))
                pt2 = tuple(map(int, keypoints[end_idx][:2]))
                cv2.line(img, pt1, pt2, color, thickness)
    
    # 绘制关键点
    for kp in keypoints:
        if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
            center = tuple(map(int, kp[:2]))
            cv2.circle(img, center, radius, color, -1)
    
    return img


def process_frame(frame_idx, data_para_path, model_path="/gemini/user/private/3D/data/body_models/smpl", 
                  output_dir="./output", cameras=None, persons=None, options=None):
    """
    处理指定帧
    
    Args:
        frame_idx: 帧索引
        data_para_path: 合并数据目录
        model_path: SMPL模型路径
        output_dir: 输出目录
        cameras: 要处理的摄像机列表，None表示所有摄像机
        persons: 要处理的人物列表，None表示所有人物
        options: 渲染选项字典
    """
    if options is None:
        options = {
            'render_individual': True,
            'render_all': True,
            'save_smpl': True,
            'save_2d': False,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        }
    
    device = torch.device(options['device'])
    print(f"🎬 处理帧 {frame_idx} (设备: {device})")
    
    # 加载SMPL模型
    body_model = SMPL(
        model_path=model_path,
        gender="neutral",
        create_transl=False
    ).to(device)
    
    # 加载合并数据
    camera_data, person_data = load_merged_data(data_para_path)
    
    # 过滤摄像机和人物
    if cameras is not None:
        camera_data = {k: v for k, v in camera_data.items() if k in cameras}
    if persons is not None:
        person_data = {k: v for k, v in person_data.items() if k in persons}
    
    print(f"📷 摄像机: {list(camera_data.keys())}")
    print(f"🚶 人物: {list(person_data.keys())}")
    
    # 获取该帧的数据
    frame_data = get_frame_data(person_data, frame_idx)
    if not frame_data:
        print(f"❌ 帧 {frame_idx} 没有数据")
        return
    
    # 获取图像分辨率
    root_path = os.path.dirname(data_para_path)
    frame_path = f"{root_path}/exo/cam01/images/{frame_idx:05d}.jpg"
    if os.path.exists(frame_path):
        image = cv2.imread(frame_path)
        H, W = image.shape[:2]
    else:
        H, W = 512, 512
    
    print(f"📐 分辨率: {W}x{H}")
    
    # 生成SMPL vertices
    all_smpl_vertices = []
    person_names = []
    
    for person_name, person_frame_data in frame_data.items():
        smpl_params = person_frame_data['smpl_params']
        
        betas = torch.Tensor(smpl_params["betas"]).to(device)[None]
        body_pose = torch.Tensor(smpl_params["body_pose"]).to(device)[None]
        global_orient = torch.Tensor(smpl_params["global_orient"]).to(device)[None]
        transl = torch.Tensor(smpl_params["transl"]).to(device)[None]
        
        smpl_output = body_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
        all_smpl_vertices.append(smpl_output.vertices)
        person_names.append(person_name)
    
    print(f"✅ 生成了 {len(all_smpl_vertices)} 个SMPL模型")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个摄像机渲染
    for cam_name, cam_params in camera_data.items():
        print(f"🎥 渲染摄像机 {cam_name}...")
        
        K = cam_params['K']
        R = cam_params['R'] 
        T = cam_params['T']
        
        render_camera = PerspectiveCameras(
            R=R,
            T=T,
            K=K,
            in_ndc=False,
            resolution=(H, W),
            device=device,
            convention="opencv"
        )
        
        # 单独渲染每个人物
        if options['render_individual'] and options['save_smpl']:
            for person_name, smpl_vertices in zip(person_names, all_smpl_vertices):
                rendered_image = vis_smpl(smpl_vertices, body_model, device, render_camera, 
                                        batch_size=30, resolution=(H, W), verbose=False)
                output_file = f"{output_dir}/{frame_idx:05d}_{person_name}_{cam_name}.png"
                cv2.imwrite(output_file, rendered_image[0, ..., :3])
                print(f"   💾 {output_file}")
        
        # 渲染所有人物在一起
        if options['render_all'] and len(all_smpl_vertices) > 0 and options['save_smpl']:
            rendered_image = vis_multiple_smpl(all_smpl_vertices, body_model, device, render_camera, 
                                             batch_size=30, resolution=(H, W), verbose=False)
            person_names_str = "_".join(person_names)
            output_file = f"{output_dir}/{frame_idx:05d}_{person_names_str}_{cam_name}_all.png"
            cv2.imwrite(output_file, rendered_image[0, ..., :3])
            print(f"   💾 {output_file}")
        
        # 保存2D关键点可视化
        if options['save_2d'] and os.path.exists(frame_path):
            base_image = cv2.imread(frame_path)
            for person_name, person_frame_data in frame_data.items():
                poses2d = person_frame_data['poses2d']
                if cam_name in poses2d and poses2d[cam_name] is not None:
                    kp_image = draw_keypoints(base_image, poses2d[cam_name])
                    output_file = f"{output_dir}/{frame_idx:05d}_{person_name}_{cam_name}_2d.png"
                    cv2.imwrite(output_file, kp_image)
                    print(f"   💾 {output_file}")


def main():
    parser = argparse.ArgumentParser(description='优化版单帧处理脚本')
    parser.add_argument('--frame', type=int, required=True, help='要处理的帧号')
    parser.add_argument('--data_path', type=str, required=True, help='合并数据目录路径')
    parser.add_argument('--model_path', type=str, default="/gemini/user/private/3D/data/body_models/smpl", help='SMPL模型路径')
    parser.add_argument('--output', type=str, default="./output", help='输出目录')
    parser.add_argument('--cameras', nargs='+', help='要处理的摄像机列表')
    parser.add_argument('--persons', nargs='+', help='要处理的人物列表') 
    parser.add_argument('--no-individual', action='store_true', help='不渲染单个人物')
    parser.add_argument('--no-all', action='store_true', help='不渲染所有人物合并')
    parser.add_argument('--save-2d', action='store_true', help='保存2D关键点可视化')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备')
    
    args = parser.parse_args()
    
    options = {
        'render_individual': not args.no_individual,
        'render_all': not args.no_all,
        'save_smpl': True,
        'save_2d': args.save_2d,
        'device': args.device
    }
    
    process_frame(
        frame_idx=args.frame,
        data_para_path=args.data_path,
        model_path=args.model_path,
        output_dir=args.output,
        cameras=args.cameras,
        persons=args.persons,
        options=options
    )
    
    print(f"✅ 帧 {args.frame} 处理完成!")


if __name__ == "__main__":
    main() 