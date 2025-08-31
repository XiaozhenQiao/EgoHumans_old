import torch
import numpy as np
import os
import cv2
from smplx import SMPL    
import pickle
import re
import pycolmap
from scipy.spatial.transform import Rotation as R

from t3drender.render.render_functions import render_rgb
from t3drender.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex


import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec



def parse_colmap_files(folder):
    """
    Parse COLMAP output files: images.txt, points3D.txt, cameras.txt.

    Returns:
        cameras (dict): Dictionary of camera intrinsics.
        images (list): List of images with their poses and 2D-3D correspondences.
        points3D (dict): Dictionary of 3D points and their coordinates.
    """
    # Parse cameras.txt
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
                'params': params  # Typically [fx, fy, cx, cy]
            }
    return cameras

     
def read_text(file_path):
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
    quat, transl, fnames, cam_ids = read_text(os.path.join(folder, "images.txt"))
    quat = np.vstack([quat[:, 0], -quat[:, 3], quat[:, 2], quat[:, 1]]).T
    rotmat = R.from_quat(quat).as_matrix()
    transl[:, 0] *= -1
    transl = rotmat @ transl[..., None]
    P = np.eye(4, 4)[None].repeat(len(quat), axis=0)
    P[:, :3, :3] = rotmat
    P[:, :3, 3:4] = transl
    # P = P @ CAM_CONVENTION_CHANGE[None]
    Rotmat = P[:, :3, :3]
    T = P[:, :3, 3:4]
    return Rotmat, T, fnames, cam_ids

def draw_kps(image, keypoints, skeleton, keypoint_names, with_text=False, point_color=(0, 255, 0), line_color=(255, 0, 0), point_radius=5, line_thickness=2):
    keypoints = keypoints.copy()[:, :2].astype(np.int32)
    img = image.copy()
    for connection in skeleton:
        start_idx, end_idx = connection
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = tuple(keypoints[start_idx]) 
            end_point = tuple(keypoints[end_idx])      
            if np.all(start_point) and np.all(end_point):  
                cv2.line(img, start_point, end_point, line_color, line_thickness)

    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(img, (x, y), point_radius, point_color, -1) 
            if with_text:
                cv2.putText(img, keypoint_names[i], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1, cv2.LINE_AA) 

    return img

def vis_smpl(smpl_verts, body_model, device, cameras, batch_size=30, resolution=(512, 512), color=[1, 1, 1], verbose=False):
    smpl_meshes = Meshes(verts=torch.Tensor(smpl_verts).to(device), faces=torch.Tensor(body_model.faces).to(device)[None].repeat_interleave(len(smpl_verts), dim=0))
    color_tensor = torch.ones_like(smpl_meshes.verts_padded()) # n, 6890, 3
    color_tensor[..., :3] = torch.Tensor(color).to(device)[None][None] # 1, 1, 3
    smpl_meshes.textures = TexturesVertex(color_tensor)
    image_tensors = render_rgb(smpl_meshes, device=device, resolution=resolution, cameras=cameras, batch_size=batch_size, verbose=verbose)
    return (image_tensors.cpu().numpy() * 255).astype(np.uint8)

def vis_multiple_smpl(smpl_verts, body_model, device, cameras, batch_size=30, resolution=(512, 512), color=None, verbose=False):
    # smpl_vertices: num_frames x num_person x 6890 x 3
    # colors: num_person x 3
    pass


def fisheye_to_K(params):
    fx, fy, cx, cy = params[:4]
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    return K

if __name__ == "__main__":
    model_path = "/home/wwj/datasets/body_models/smpl"
    gender = "neutral"
    device = torch.device("cuda:0")
    root = "/home/wwj/datasets/egohumans/004_tagging/"
    
    body_model = SMPL(
            model_path=model_path,
            gender=gender,
            create_transl=False).to(device)
    num_frames = len(os.listdir(f"{root}/exo/cam01/images"))
    intrinsics = parse_colmap_files(f"{root}/colmap/workplace")
    Rotations, Translations, fnames, cam_ids = parse_camera(f"{root}/colmap/workplace")
    colmap_transforms_file = f"{root}/colmap/workplace/colmap_from_aria_transforms.pkl"
    with open(colmap_transforms_file, 'rb') as f:
        colmap_transforms = pickle.load(f)
    primary_transform = colmap_transforms['aria01']
    colmap_reconstruction = pycolmap.Reconstruction(f"{root}/colmap/workplace") 

    all_extrinsics = {}
    for image_id, image in colmap_reconstruction.images.items():
        image_path = image.name
        image_camera_name = image_path.split('/')[0]
        time_stamp = int((image_path.split('/')[1]).replace('.jpg', ''))

        if image_camera_name.startswith('cam0'):
            all_extrinsics[image_camera_name] = np.eye(4, 4)
            all_extrinsics[image_camera_name][:3] = image.cam_from_world.matrix() ## 3 x 4

    intrinsics_all = {}
    for fid, fname in enumerate(fnames):
        cam_name = fname.split('/')[0]
        cam_id = cam_ids[fid]
        K = fisheye_to_K(intrinsics[int(cam_id)]["params"])
        intrinsics_all[cam_name] = K

    for frame_idx in range(num_frames):
        frame_idx = frame_idx + 1
        frame_path = f"{root}/exo/cam01/images/{frame_idx:05d}.jpg"
        image = cv2.imread(frame_path)
        H, W = image.shape[:2]
        smpl_params = np.load(f"{root}/processed_data/smpl/{frame_idx:05d}.npy", allow_pickle=True)
        smpl_params = dict(smpl_params.item())
        pose3d = np.load(f"{root}/processed_data/refine_poses3d/{frame_idx:05d}.npy", allow_pickle=True)

        for person_name in smpl_params.keys():
            betas = torch.Tensor(smpl_params[person_name]["betas"]).to(device)[None]
            body_pose = torch.Tensor(smpl_params[person_name]["body_pose"]).to(device)[None]
            global_orient = torch.Tensor(smpl_params[person_name]["global_orient"]).to(device)[None]
            transl = torch.Tensor(smpl_params[person_name]["transl"]).to(device)[None]

        smpl_output = body_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
        smpl_vertices = smpl_output.vertices

        for cam_name in all_extrinsics.keys():
            poses2d = np.load(f"{root}/processed_data/poses2d/{cam_name}/rgb/{frame_idx:05d}.npy", allow_pickle=True)
            # Draw 2D keypoints: kp2d
            # project 3d keypoints and draw 2d keypoints: kp3d_proj
            extrinsic = all_extrinsics[cam_name]
            K = intrinsics_all[cam_name]

            extrinsic = np.dot(extrinsic, primary_transform)
            extrinsic = np.linalg.inv(extrinsic)
            Rot = extrinsic[:3, :3]
            Trans = extrinsic[:3, 3]

            render_camera = PerspectiveCameras(
                R=Rot,
                T=Trans,
                K=K,
                in_ndc=False,
                resolution=(H, W),
                device=device,
                convention="opencv"
            )
            rendered_image = vis_smpl(smpl_vertices, body_model, device, render_camera, batch_size=30, resolution=(H, W), verbose=False)
            cv2.imwrite(f"{frame_idx:05d}_{person_name}_{cam_name}.png", rendered_image[0, ..., :3])

    # final annotation:
    # store all the cam0x K, R, T
    # store all the kp3d, kp2d, smpl_params for each person
    # final files: cam0x.npz * n_cameras, smpl_0y.npz * n_persons(smpl_params, kp2d, kp3d, bbox2d)
