import os
import numpy as np
import open3d as o3d
import cv2
from PIL import Image


def get_raw_depths(path,poses):
    raw_depths = []
    pose = []
    
    for view_id in range(len(poses)):
        file_path = os.path.join(path, f'depth_{view_id:05d}.tiff')
        try:
            img = Image.open(file_path)
            tiff_array = np.array(img)
            raw_depths.append(tiff_array)
            pose.append(poses[view_id])
        except:
            continue
    return np.array(raw_depths),pose
# load pose
def load_poses(path):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        poses.append(c2w)
    return poses
def get_all_subdirectories(root_path):
    subdirectories = []
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            subdirectories.append(os.path.join(root, dir_name))
    return subdirectories
def tsdf_mesh(poses, raw_rgbs, raw_depths, K):

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=3 * 0.01,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for pose, raw_rgb, raw_depth in zip(poses, raw_rgbs, raw_depths):

        color = o3d.geometry.Image(raw_rgb)
        depth = o3d.geometry.Image(raw_depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )

        H = raw_rgb.shape[0]
        W = raw_rgb.shape[1]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)

    return volume.extract_triangle_mesh()

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    # import pdb; pdb.set_trace()

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose


if __name__ == '__main__':

    total_views = 100
    H, W = 384, 384

    cam_file = '/root/autodl-tmp/data/scan6/cameras.npz'
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(total_views)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(total_views)]

    scale_mat_0 = scale_mats[0]
    world_mat_0 = world_mats[0]
    P = world_mat_0 @ scale_mat_0
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(None, P)
    # scale and offset the intrinsics to match the image size
    scale = 384 / 680
    offset = (1200 - 680) * 0.5
    intrinsics[0, 2] -= offset
    intrinsics[:2, :] *= scale

    traj_poses = load_poses(cam_file.replace('cameras.npz', 'traj.txt'))
    pose_all = []
    for idx in range(len(traj_poses)):
        if idx % 20 == 0:
            pose_all.append(traj_poses[idx])

    poses = np.array(pose_all)
    raw_rgbs = np.ones((len(poses), H, W, 3))
    raw_rgbs = (raw_rgbs * 255).astype(np.uint8)

    K = intrinsics
    depth_root_path = '/root/autodl-tmp/output/scan6/train/ours_30000/vis'
    mesh_save_path='/root/autodl-tmp/mesh/sacn6_sence_1/'
    os.makedirs(mesh_save_path,exist_ok=True)
    raw_depths,_=get_raw_depths(depth_root_path,poses)
    refine_mesh = tsdf_mesh(poses, raw_rgbs, raw_depths, K)
    save_path = os.path.join(mesh_save_path,'scan6_2_relative_depth_0.05.ply')
    o3d.io.write_triangle_mesh(save_path, refine_mesh)
    print(f'Save mesh to {save_path}')

    # depth_root_path = '/root/autodl-tmp/output/scan6/objects_con_bg'
    # mesh_save_path='/root/autodl-tmp/mesh/sacn6_3/'
    # os.makedirs(mesh_save_path,exist_ok=True)
    # objects=get_all_subdirectories(depth_root_path)
    # for idx,obj in enumerate(objects):
    #     depths_path=os.path.join(depth_root_path,obj)
    #     raw_depths,pose=get_raw_depths(depths_path,poses)
    #     raw_rgbs = np.ones((len(pose), H, W, 3))
    #     raw_rgbs = (raw_rgbs * 255).astype(np.uint8)
    #     refine_mesh = tsdf_mesh(pose, raw_rgbs, raw_depths, K)
    #     save_path = os.path.join(mesh_save_path,'scan6_3_{}.ply'.format(idx))
    #     o3d.io.write_triangle_mesh(save_path, refine_mesh)
    #     print(f'Save mesh to {save_path}')
    

