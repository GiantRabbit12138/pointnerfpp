'''
For nvs dataset preprocessing.
'''
import os
import glob
import numpy as np 

import open3d as o3d
import ply
import imageio
import tqdm

import loadCalibration

# scenes = ["train_00", "train_01", "train_02", "train_03", "train_04"] # 5 scenes. 
# scenes = ["test_00", "test_01", "test_02", "test_03", "test_04"] # 5 scenes. 
# scenes = ["test_04"] # 5 scenes. 
scenes = [ "test_04", "test_00", "test_01", "test_02", "test_03"] # 5 scenes. 

dir_root = "/ubc/cs/home/w/weiweis/scratch/data/datasets/kitti360/raw"
ply_field_names = ['x', 'y', 'z', '']

dir_data_2d_nvs_drop = os.path.join(dir_root, "data_2d_nvs_drop50")
dir_data_pcd = os.path.join(dir_root, "data_3d_semantics", 'test')
dir_poses = os.path.join(dir_root, "data_poses")
dir_calibration = os.path.join(dir_root, "calibration")


dir_save_root = 'kitti360_pcd_images'

class Camera:
    def __init__(self):
        
        # load intrinsics
        self.load_intrinsics(self.intrinsic_file)

        # load poses
        poses = np.loadtxt(self.pose_file)
        frames = poses[:,0]
        poses = np.reshape(poses[:,1:],[-1,3,4])
        self.cam2world = {}
        self.frames = frames
        for frame, pose in zip(frames, poses): 
            pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
            # consider the rectification for perspective cameras
            if self.cam_id==0 or self.cam_id==1:
                self.cam2world[frame] = np.matmul(np.matmul(pose, self.camToPose),
                                                  np.linalg.inv(self.R_rect))
            # fisheye cameras
            elif self.cam_id==2 or self.cam_id==3:
                self.cam2world[frame] = np.matmul(pose, self.camToPose)
            else:
                raise RuntimeError('Unknown Camera ID!')


    def world2cam(self, points, R, T, inverse=False):
        assert (points.ndim==R.ndim)
        assert (T.ndim==R.ndim or T.ndim==(R.ndim-1)) 
        ndim=R.ndim
        if ndim==2:
            R = np.expand_dims(R, 0) 
            T = np.reshape(T, [1, -1, 3])
            points = np.expand_dims(points, 0)
        if not inverse:
            points = np.matmul(R, points.transpose(0,2,1)).transpose(0,2,1) + T
        else:
            points = np.matmul(R.transpose(0,2,1), (points - T).transpose(0,2,1))

        if ndim==2:
            points = points[0]

        return points

    def cam2image(self, points):
        raise NotImplementedError

    def load_intrinsics(self, intrinsic_file):
        raise NotImplementedError
    
    def project_vertices(self, vertices, frameId, inverse=True):

        # current camera pose
        curr_pose = self.cam2world[frameId]
        T = curr_pose[:3,  3]
        R = curr_pose[:3, :3]

        # convert points from world coordinate to local coordinate 
        points_local = self.world2cam(vertices, R, T, inverse)

        # perspective projection
        u,v,depth = self.cam2image(points_local)

        return (u,v), depth 

    def __call__(self, obj3d, frameId):

        vertices = obj3d.vertices

        uv, depth = self.project_vertices(vertices, frameId)

        obj3d.vertices_proj = uv
        obj3d.vertices_depth = depth 
        obj3d.generateMeshes()


class CameraPerspective(Camera):

    def __init__(self, root_dir, seq='2013_05_28_drive_0009_sync', cam_id=0):
        # perspective camera ids: {0,1}, fisheye camera ids: {2,3}
        assert (cam_id==0 or cam_id==1)

        pose_dir = os.path.join(root_dir, 'data_poses', seq)
        calib_dir = os.path.join(root_dir, 'calibration')
        self.pose_file = os.path.join(pose_dir, "poses.txt")
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibration.loadCalibrationCameraToPose(fileCameraToPose)['image_%02d' % cam_id]
        self.cam_id = cam_id
        super(CameraPerspective, self).__init__()

    def load_intrinsics(self, intrinsic_file):
        ''' load perspective intrinsics '''
    
        intrinsic_loaded = False
        width = -1
        height = -1
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_%02d:' % self.cam_id:
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3,4])
                intrinsic_loaded = True
            elif line[0] == 'R_rect_%02d:' % self.cam_id:
                R_rect = np.eye(4) 
                R_rect[:3,:3] = np.array([float(x) for x in line[1:]]).reshape(3,3)
            elif line[0] == "S_rect_%02d:" % self.cam_id:
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert(intrinsic_loaded==True)
        assert(width>0 and height>0)
    
        self.K = K
        self.width, self.height = width, height
        self.R_rect = R_rect

    def cam2image(self, points):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(self.K[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth


def load_acc_pcd(sequence, img_frame_start_idx, img_frame_end_idx):
    dir_sequence = os.path.join(dir_data_pcd, sequence, 'static')
    fn_pcd_list =glob.glob(os.path.join(dir_sequence, "*.ply"))

    def contain_img_frame(fn_pcd):
        pcd_basename = os.path.basename(fn_pcd)
        pcd_frame_start_idx, pcd_frame_end_idx = pcd_basename.split('.')[0].split('_') 
        return (int(pcd_frame_end_idx) >= int(img_frame_end_idx) and int(pcd_frame_start_idx) <= int(img_frame_end_idx)) \
            or (int(pcd_frame_start_idx) <= int(img_frame_start_idx) and int(pcd_frame_end_idx) >= int(img_frame_start_idx))
    fn_pcd_list = [fn_pcd for fn_pcd in fn_pcd_list if contain_img_frame(fn_pcd)] 

    # 
    print(f"Reading {len(fn_pcd_list)} point clouds.")
    pcd_list = [ply.read_ply(fn_pcd) for fn_pcd in fn_pcd_list] 
    xyz = np.concatenate(
            [
                np.stack([pcd['x'], pcd['y'], pcd['z']], 1).astype(np.float32) \
                    for pcd in pcd_list
            ],
            0
        )
    color = np.concatenate(
            [
                np.stack([pcd['red'], pcd['green'], pcd['blue']], 1).astype(np.uint8) \
                    for pcd in pcd_list
            ],
            0
        )
    # print(f'Before {len(xyz)}, {len(color)} points')
    # ply.write_ply('temp.ply', [xyz, color], ['x', 'y', 'z', 'red', 'green', 'blue'])
    pcd = (xyz, color)
    return pcd 
    

def prune_pcd_visibility(pcd, frameidx_list, frame00_list, camera_list):
    xyz, color = pcd
    mask_all = np.zeros(len(xyz)) > 0
    assert len(frameidx_list) == len(frame00_list)
    for frameidx, frame00 in tqdm.tqdm(zip(frameidx_list, frame00_list)):
        for camera in camera_list:
            (u, v), depth = camera.project_vertices(xyz, frameidx)
            cam_id = camera.cam_id
            assert cam_id in [0, 1]

            if cam_id == 1:    
                fn_frame = frame00.replace("image_00", 'image_01')
            else:
                fn_frame = frame00

            visibility_mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)
            # visualize points within 30 meters
            visibility_mask = np.logical_and(np.logical_and(visibility_mask, depth>0), depth<80)
            mask_all = np.logical_or(mask_all, visibility_mask)
            # prepare depth map for visualization
            depthMap = np.zeros((camera.height, camera.width))
            depthImage = np.zeros((camera.height, camera.width, 3))
            mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)
            # visualize points within 30 meters
            mask = np.logical_and(np.logical_and(mask, depth>0), depth<30)
            depthMap[v[mask],u[mask]] = depth[mask]

            # visualize 
            import matplotlib.pyplot as plt
            layout = (5,1)  
            fig, axs = plt.subplots(*layout, figsize=(18,12))
            cm = plt.get_cmap('jet')
            # load RGB image for visualization
            colorImage = imageio.imread(fn_frame) / 255.0
            depthImage = cm(depthMap/depthMap.max())[...,:3]
            colorImage[depthMap>0] = depthImage[depthMap>0]

            ptsColorImage = np.ones((camera.height, camera.width, 3)) * 255.0
            ptsColorImage[v[mask],u[mask]] = color[mask] / 255.0 
            ptsColorImage_overlaid = imageio.imread(fn_frame) / 255.0
            ptsColorImage_overlaid[depthMap>0] = ptsColorImage[depthMap>0]
            input_colorImage = imageio.imread(fn_frame) / 255.0
            
            axs[0].imshow(depthMap, cmap='jet')
            axs[0].title.set_text('Projected Depth')
            axs[0].axis('off')
            axs[1].imshow(colorImage)
            axs[1].title.set_text('Projected Depth Overlaid on Image')
            axs[1].axis('off')
            axs[2].imshow(ptsColorImage)
            axs[2].title.set_text('Projected color images')
            axs[2].axis('off')

            axs[3].imshow(ptsColorImage_overlaid)
            axs[3].title.set_text('Projected color images overlaid with input')
            axs[3].axis('off')

            axs[4].imshow(input_colorImage)
            axs[4].title.set_text('Input images')
            axs[4].axis('off')

            # plt.show()
            plt.savefig(f'sample_{frameidx}_{cam_id}.png')
            # import pdb; pdb.set_trace()
    print(f"{len(xyz)} points! ")
    xyz, color = xyz[mask_all], color[mask_all]
    print(f"afeter {len(xyz)} points! ")
    return (xyz, color) 


def vis_pcd_2d(pcd, frameidx_list, frame00_list, camera_list, scene):
    xyz, color = pcd
    mask_all = np.zeros(len(xyz)) > 0
    assert len(frameidx_list) == len(frame00_list)
    for frameidx, frame00 in tqdm.tqdm(zip(frameidx_list, frame00_list)):
        for camera in camera_list:
            (u, v), depth = camera.project_vertices(xyz, frameidx)
            cam_id = camera.cam_id
            assert cam_id in [0, 1]

            if cam_id == 1:    
                fn_frame = frame00.replace("image_00", 'image_01')
            else:
                fn_frame = frame00
            # if frameidx != 8116:
            #     continue
            # scale = 3
            # camera.width, camera.height = scale * camera.width, scale *camera.height
            visibility_mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)
            # visualize points within 30 meters
            visibility_mask = np.logical_and(np.logical_and(visibility_mask, depth>0), depth<80)
            mask_all = np.logical_or(mask_all, visibility_mask)
            # prepare depth map for visualization
            depthMap = np.zeros((camera.height, camera.width))
            depthImage = np.zeros((camera.height, camera.width, 3))
            mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)
            # visualize points within 30 meters
            mask = np.logical_and(np.logical_and(mask, depth>0), depth<1000)
            depthMap[v[mask],u[mask]] = depth[mask]

            # visualize 
            import matplotlib.pyplot as plt
            layout = (1,1)  
            fig, axs = plt.subplots(*layout)
            cm = plt.get_cmap('jet')
            # load RGB image for visualization
            # colorImage = imageio.imread(fn_frame) / 255.0
            # depthImage = cm(depthMap/depthMap.max())[...,:3]
            # colorImage[depthMap>0] = depthImage[depthMap>0]
            scale = 1
            ptsColorImage = np.ones((camera.height*scale, camera.width*scale, 3)) * 255.0
            ptsColorImage[v[mask]*scale,u[mask]*scale] = color[mask] / 255.0 
            # ptsColorImage_overlaid = imageio.imread(fn_frame) / 255.0
            # ptsColorImage_overlaid[depthMap>0] = ptsColorImage[depthMap>0]
            # input_colorImage = imageio.imread(fn_frame) / 255.0
            
            # axs[0].imshow(depthMap, cmap='jet')
            # axs[0].title.set_text('Projected Depth')
            # axs[0].axis('off')
            # axs[1].imshow(colorImage)
            # axs[1].title.set_text('Projected Depth Overlaid on Image')
            # axs[1].axis('off')
            axs.imshow(ptsColorImage)
            # axs.title.set_text('Projected color images')
            axs.axis('off')

            # axs[3].imshow(ptsColorImage_overlaid)
            # axs[3].title.set_text('Projected color images overlaid with input')
            # axs[3].axis('off')

            # axs[4].imshow(input_colorImage)
            # axs[4].title.set_text('Input images')
            # axs[4].axis('off')

            # plt.show()
            plt.savefig(f'logs/logs_pcd_vis/kitti360/rerun_{scene}_sample_{frameidx}_{cam_id}_{scale}.png', pad_inches=0, bbox_inches='tight', dpi=300)
            plt.close()
    print(f"{len(xyz)} points! ")
    xyz, color = xyz[mask_all], color[mask_all]
    print(f"afeter {len(xyz)} points! ")
    return (xyz, color) 
def convert_scene(scene): 

    # read all frames
    fn_frame_list = os.path.join(dir_data_2d_nvs_drop, f'{scene}.txt')
    with open(fn_frame_list) as f:
        frame_00_list = f.read().splitlines()
    sequence_name = frame_00_list[0].split('/')[0]

    # full path 
    frame_00_list = [os.path.join(dir_data_2d_nvs_drop, scene, frame_00) for frame_00 in frame_00_list]
    # frame_01_list = [frame.replace('image_00', 'image_01') for frame in frame_00_list]
    frameidx_list = [int(os.path.basename(frame).split('.')[0]) for frame in frame_00_list] 
    # retrieve pcd
    frame_idx_sorted = sorted(frameidx_list)
    frame_start_idx, frame_end_idx = frame_idx_sorted[0], frame_idx_sorted[-1] 

    pcd = load_acc_pcd(sequence_name, frame_start_idx, frame_end_idx) # in world coordinate system. 
    camera_00 = CameraPerspective(dir_root, sequence_name, 0)
    camera_01 = CameraPerspective(dir_root, sequence_name, 1)  
    camera_list = [camera_00, camera_01] 

    pcd = vis_pcd_2d(pcd, frameidx_list, frame_00_list, camera_list, scene) 

    # save pcd
    # dir_pcd = os.path.join(dir_save_root, '')
    # if not os.path.exists(dir_pcd):
    #     os.makedirs(dir_pcd)
    # fn_pcd = os.path.join(dir_pcd, f'{scene}.ply')
    # xyz, color = pcd
    # ply.write_ply(fn_pcd, [xyz, color], ['x', 'y', 'z', 'red', 'green', 'blue']) 


def main():
    for scene in scenes:
        convert_scene(scene)
    
if __name__ == "__main__":
    main()
