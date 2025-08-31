import torch
import numpy as np
import os
import cv2
from smplx import SMPL    
import argparse

from t3drender.render.render_functions import render_mp, MeshRenderer, SoftPhongShader, PointLights
from t3drender.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex


def load_merged_camera_data(data_para_path):
    """加载合并后的摄像机参数"""
    camera_data = {}
    
    for file in os.listdir(data_para_path):
        if file.startswith('cam') and file.endswith('.npz'):
            cam_name = file[:-4]
            cam_file = os.path.join(data_para_path, file)
            cam_params = dict(np.load(cam_file, allow_pickle=True))
            camera_data[cam_name] = cam_params
    
    return camera_data


def load_merged_person_data(data_para_path):
    """加载合并后的人体时序数据"""
    person_data = {}
    
    for file in os.listdir(data_para_path):
        if file.startswith('smpl_') and file.endswith('.npz'):
            person_name = file[5:-4]
            person_file = os.path.join(data_para_path, file)
            person_params = dict(np.load(person_file, allow_pickle=True))
            person_data[person_name] = person_params

    return person_data


def draw_kps(image, keypoints, skeleton, point_color=(0, 255, 0), line_color=(255, 0, 0), point_radius=2, line_thickness=2):
    """绘制2D关键点"""
    if keypoints is None:
        return image
        
    keypoints = keypoints.copy()[:, :2].astype(np.int32)
    img = image.copy()
    
    # 绘制骨架连线
    for connection in skeleton:
        start_idx, end_idx = connection
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = tuple(keypoints[start_idx]) 
            end_point = tuple(keypoints[end_idx])      
            if np.all(start_point) and np.all(end_point):
                cv2.line(img, start_point, end_point, line_color, line_thickness)

    # 绘制关键点
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(img, (x, y), point_radius, point_color, -1)

    return img


# COCO关键点编号对应表：
# 0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
# 5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
# 9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
# 13:left_knee  14:right_knee  15:left_ankle  16:right_ankle

# 简化的身体主要关键点骨架连接（只保留核心姿态，去掉头部细节）
BODY_SKELETON_SIMPLE = [
    # 上半身主干
    (5, 6),                   # 左肩(5)->右肩(6)
    (5, 11), (6, 12),         # 左肩(5)->左臀(11), 右肩(6)->右臀(12)
    (11, 12),                 # 左臀(11)->右臀(12)
    
    # 左臂
    (5, 7), (7, 9),           # 左肩(5)->左肘(7)->左手腕(9)
    
    # 右臂  
    (6, 8), (8, 10),          # 右肩(6)->右肘(8)->右手腕(10)
    
    # 左腿
    (11, 13), (13, 15),       # 左臀(11)->左膝(13)->左脚踝(15)
    
    # 右腿
    (12, 14), (14, 16),       # 右臀(12)->右膝(14)->右脚踝(16)
]


def draw_keypoints_on_image(image, keypoints_2d, person_name, downsample_factor=1, 
                          point_radius=2, line_thickness=2):
    """在图像上绘制2D关键点"""
    if keypoints_2d is None:
        return image
    
    if downsample_factor > 1:
        keypoints_2d_scaled = keypoints_2d.copy()
        keypoints_2d_scaled[:, :2] /= downsample_factor
    else:
        keypoints_2d_scaled = keypoints_2d
    
    # 为不同人物使用不同颜色
    person_colors = {
        'aria01': ((0, 255, 0), (0, 200, 0)),
        'aria02': ((255, 0, 0), (200, 0, 0)),
        'aria03': ((0, 0, 255), (0, 0, 200)),
        'aria04': ((255, 255, 0), (200, 200, 0))
    }
    
    point_color, line_color = person_colors.get(person_name, ((0, 255, 0), (255, 0, 0)))
    
    return draw_kps(image, keypoints_2d_scaled, BODY_SKELETON_SIMPLE, 
                   point_color=point_color, line_color=line_color,
                   point_radius=point_radius, line_thickness=line_thickness)


def vis_smpl(smpl_verts, body_model, device, cameras, resolution=(512, 512), light_location=[0, 0, 0]):
    """渲染SMPL模型"""
    faces = torch.Tensor(body_model.faces.astype(np.int64)).to(device)[None].repeat_interleave(len(smpl_verts), dim=0)
    smpl_meshes = Meshes(verts=torch.Tensor(smpl_verts).to(device), faces=faces)
    color_tensor = torch.ones_like(smpl_meshes.verts_padded())
    color_tensor[..., :3] = torch.tensor([1, 1, 1]).to(device)[None][None]
    smpl_meshes.textures = TexturesVertex(color_tensor)
    
    image_tensors = render_rgb_with_lights(smpl_meshes, device=device, resolution=resolution, cameras=cameras, light_location=light_location)
    return (image_tensors.cpu().numpy() * 255).astype(np.uint8)


def vis_multiple_smpl(smpl_verts_list, body_model, device, cameras, resolution=(512, 512), colors=None, light_location=[0, 0, 0]):
    """渲染多个SMPL模型"""
    if not smpl_verts_list:
        return None
    
    if colors is None:
        default_colors = [
            [1.0, 0.7, 0.7], [0.7, 1.0, 0.7], [0.7, 0.7, 1.0],
            [1.0, 1.0, 0.7], [1.0, 0.7, 1.0], [0.7, 1.0, 1.0],
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
    
    combined_verts = torch.cat(all_verts, dim=1)
    combined_faces = torch.cat(all_faces, dim=0)
    combined_colors = torch.cat(all_colors, dim=0)
    
    batch_size_actual = combined_verts.shape[0]
    smpl_meshes = Meshes(
        verts=combined_verts, 
        faces=combined_faces[None].repeat_interleave(batch_size_actual, dim=0)
    )
    
    color_tensor = combined_colors[None].repeat_interleave(batch_size_actual, dim=0)
    smpl_meshes.textures = TexturesVertex(color_tensor)
    
    image_tensors = render_rgb_with_lights(smpl_meshes, device=device, resolution=resolution, cameras=cameras, light_location=light_location)
    return (image_tensors.cpu().numpy() * 255).astype(np.uint8)


def composite_mesh_on_rgb(rgb_image, rendered_mesh, alpha=0.7):
    """将渲染的mesh叠加到RGB图像上"""
    if rgb_image.shape[:2] != rendered_mesh.shape[:2]:
        rendered_mesh = cv2.resize(rendered_mesh, (rgb_image.shape[1], rgb_image.shape[0]))
    
    if rendered_mesh.shape[2] == 4:
        alpha_channel = rendered_mesh[:, :, 3]
        rendered_mesh_rgb = rendered_mesh[:, :, :3]
        
        if alpha_channel.max() > 0:
            mesh_mask = (alpha_channel > 0.5).astype(np.float32)
        else:
            return rgb_image
    else:
        rendered_mesh_rgb = rendered_mesh
        white_background = np.all(rendered_mesh_rgb >= 250, axis=2)
        black_background = np.all(rendered_mesh_rgb <= 5, axis=2)
        mesh_mask = (~white_background & ~black_background).astype(np.float32)
        
        if mesh_mask.sum() == 0:
            color_std = np.std(rendered_mesh_rgb, axis=2)
            mesh_mask = (color_std > 10).astype(np.float32)
    
    if mesh_mask.sum() > 0:
        kernel = np.ones((3,3), np.uint8)
        mesh_mask = cv2.morphologyEx(mesh_mask, cv2.MORPH_OPEN, kernel)
        mesh_mask = cv2.morphologyEx(mesh_mask, cv2.MORPH_CLOSE, kernel)
        mesh_mask = cv2.GaussianBlur(mesh_mask, (3, 3), 0.5)
        mesh_mask = np.clip(mesh_mask, 0, 1)
    
    mesh_mask = mesh_mask[:, :, np.newaxis]
    
    composite = rgb_image.astype(np.float32) * (1 - mesh_mask * alpha) + \
                rendered_mesh_rgb.astype(np.float32) * (mesh_mask * alpha)
    
    return composite.astype(np.uint8)


def render_mesh_with_background(smpl_verts_list, body_model, device, cameras, rgb_image, 
                               colors=None, alpha=0.7, light_location=[0, 0, 0]):
    """渲染mesh并与背景RGB图像合成"""
    H, W = rgb_image.shape[:2]
    
    if len(smpl_verts_list) == 1:
        rendered_mesh = vis_smpl(smpl_verts_list[0], body_model, device, cameras, 
                               resolution=(H, W), light_location=light_location)
    else:
        rendered_mesh = vis_multiple_smpl(smpl_verts_list, body_model, device, cameras,
                                        resolution=(H, W), colors=colors, light_location=light_location)
    
    if rendered_mesh is None:
        return rgb_image
    
    return composite_mesh_on_rgb(rgb_image, rendered_mesh[0], alpha=alpha)


def get_frame_data(person_data_dict, frame_idx):
    """从合并的时序数据中获取指定帧的数据"""
    frame_data = {}
    
    for person_name, person_data in person_data_dict.items():
        frame_indices = person_data['frame_indices']
        
        if frame_idx in frame_indices:
            time_idx = np.where(frame_indices == frame_idx)[0][0]
            
            smpl_params = person_data['smpl_params'].item()
            
            # 检查SMPL参数是否为空（如果这个人员没有有效数据）
            if (smpl_params['betas'].shape[0] == 0 or 
                smpl_params['body_pose'].shape[0] == 0 or
                smpl_params['global_orient'].shape[0] == 0 or
                smpl_params['transl'].shape[0] == 0):
                print(f"跳过 {person_name}: SMPL参数为空")
                continue
            
            frame_smpl = {
                'betas': smpl_params['betas'][time_idx],
                'body_pose': smpl_params['body_pose'][time_idx],
                'global_orient': smpl_params['global_orient'][time_idx],
                'transl': smpl_params['transl'][time_idx]
            }
            
            poses3d = person_data['keypoints3d'][time_idx] if person_data['keypoints3d'][time_idx] is not None else None
            
            keypoints2d = {}
            keypoints2d_all = person_data['keypoints2d'].item()
            
            for cam_name in keypoints2d_all.keys():
                keypoints2d[cam_name] = keypoints2d_all[cam_name][time_idx] if keypoints2d_all[cam_name][time_idx] is not None else None

            frame_data[person_name] = {
                'smpl_params': frame_smpl,
                'poses3d': poses3d,
                'keypoints2d': keypoints2d
            }

    return frame_data


def render_rgb_with_lights(meshes, device, resolution=(512, 512), cameras=None, light_location=[0.0, 0.0, 0.0]):
    """带自定义光源的RGB渲染函数"""
    mesh_renderer = MeshRenderer(resolution=resolution, shader=SoftPhongShader())

    if cameras is None:
        K = torch.eye(3, 3)[None]
        h, w = resolution
        K[:, 0, 0] = 512
        K[:, 1, 1] = 512
        K[:, 0, 2] = w / 2
        K[:, 1, 2] = h / 2
        cameras = PerspectiveCameras(in_ndc=False, K=K, convention='opencv', resolution=resolution)
    
    lights = PointLights(location=[light_location])
    rendered_frames = render_mp(renderer=mesh_renderer, meshes=meshes, lights=lights, cameras=cameras, device=device)
    return rendered_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用合并时序数据进行SMPL渲染')
    parser.add_argument('--root', type=str, 
                       default="/gemini/user/private/3D/data/egohumans/01_tagging/004_tagging",
                       help='数据根目录')
    parser.add_argument('--downsample', type=int, default=2, help='降采样因子')
    parser.add_argument('--frame_step', type=int, default=1000, help='帧步长')
    parser.add_argument('--mesh_alpha', type=float, default=1.0, help='mesh透明度')
    parser.add_argument('--light_position', type=str, default='0,0,0', help='光源位置')
    parser.add_argument('--keypoint_radius', type=int, default=2, help='关键点半径')
    parser.add_argument('--skeleton_thickness', type=int, default=2, help='骨架线条粗细')
    
    args = parser.parse_args()
    
    model_path = "/gemini/user/private/3D/data/body_models/smpl"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = args.root
    data_para_path = f"{root}/data_para"
    
    downsample_factor = args.downsample
    frame_step = args.frame_step
    mesh_alpha = args.mesh_alpha
    light_position = [float(x) for x in args.light_position.split(',')]
    keypoint_radius = args.keypoint_radius
    skeleton_thickness = args.skeleton_thickness
    
    downsample_suffix = f"_ds{downsample_factor}" if downsample_factor > 1 else ""
    if frame_step > 1:
        downsample_suffix += f"_step{frame_step}"
    downsample_suffix += "_comp_kps"

    print(f"🤖 初始化SMPL模型: neutral ({model_path})")
    body_model = SMPL(model_path=model_path, gender="neutral", create_transl=False).to(device)
    
    print("📷 加载摄像机参数...")
    camera_data = load_merged_camera_data(data_para_path)
    
    print("🚶 加载人体时序数据...")
    person_data_dict = load_merged_person_data(data_para_path)
    
    if person_data_dict:
        first_person = list(person_data_dict.keys())[0]
        frame_indices = person_data_dict[first_person]['frame_indices']
        num_frames = len(frame_indices)
        print(f"📊 总帧数: {num_frames}")
    else:
        print("❌ 没有找到人体数据")
        exit(1)
    
    output_dir = f"{root}/rendered_output{downsample_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    selected_frames = frame_indices[::frame_step]
    print(f"📊 处理帧数: {len(selected_frames)}/{len(frame_indices)}")
    
    for i, frame_idx in enumerate(selected_frames):
        if i % 10 == 0:
            print(f"   处理帧 {frame_idx} ({i+1}/{len(selected_frames)})")
        
        frame_data = get_frame_data(person_data_dict, frame_idx)
        if not frame_data:
            continue
        
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
        
        for cam_name, cam_params in camera_data.items():
            K_orig = cam_params['K']
            R = cam_params['R']
            T = cam_params['T']
            
            frame_path = f"{root}/exo/{cam_name}/undistorted_images_scale2.0/{frame_idx:05d}.jpg"
            if not os.path.exists(frame_path):
                continue
                
            rgb_image = cv2.imread(frame_path)
            if rgb_image is None:
                continue
                
            H_orig, W_orig = rgb_image.shape[:2]
            H = H_orig // downsample_factor
            W = W_orig // downsample_factor
            H = H - (H % 2)
            W = W - (W % 2)
            
            if downsample_factor > 1:
                rgb_image = cv2.resize(rgb_image, (W, H))
            
            K = K_orig.copy()
            if downsample_factor > 1:
                K[..., 0, 0] /= downsample_factor
                K[..., 1, 1] /= downsample_factor
                K[..., 0, 2] /= downsample_factor
                K[..., 1, 2] /= downsample_factor
            
            render_camera = PerspectiveCameras(
                R=R, T=T, K=K, in_ndc=False, resolution=(H, W),
                device=device, convention="opencv"
            )
            
            # 单独渲染每个人物并合成
            for person_name, smpl_vertices in zip(person_names, all_smpl_vertices):
                composite_image = render_mesh_with_background(
                    [smpl_vertices], body_model, device, render_camera, rgb_image,
                    alpha=mesh_alpha, light_location=light_position
                )
                
                person_frame_data = frame_data[person_name]
                keypoints_2d = person_frame_data['keypoints2d'].get(cam_name)
                if keypoints_2d is not None:
                    composite_image = draw_keypoints_on_image(
                        composite_image, keypoints_2d, person_name, 
                        downsample_factor=downsample_factor, 
                        point_radius=keypoint_radius, line_thickness=skeleton_thickness
                    )
                
                output_file = f"{output_dir}/{frame_idx:05d}_{person_name}_{cam_name}_comp.png"
                cv2.imwrite(output_file, composite_image)
            
            # 渲染所有人物在一起并合成
            if len(all_smpl_vertices) > 0:
                composite_image = render_mesh_with_background(
                    all_smpl_vertices, body_model, device, render_camera, rgb_image,
                    alpha=mesh_alpha, light_location=light_position
                )
                
                for person_name in person_names:
                    person_frame_data = frame_data[person_name]
                    keypoints_2d = person_frame_data['keypoints2d'].get(cam_name)
                    if keypoints_2d is not None:
                        composite_image = draw_keypoints_on_image(
                            composite_image, keypoints_2d, person_name, 
                            downsample_factor=downsample_factor, 
                            point_radius=keypoint_radius, line_thickness=skeleton_thickness
                        )
                
                person_names_str = "_".join(person_names)
                output_file = f"{output_dir}/{frame_idx:05d}_{person_names_str}_{cam_name}_comp_all.png"
                cv2.imwrite(output_file, composite_image)
    
    print(f"\n✅ 渲染完成! 输出目录: {output_dir}") 