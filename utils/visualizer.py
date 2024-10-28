import numpy as np
import os

from PIL import Image
import shutil
from collections import OrderedDict
import time
import datetime
import torch
import imageio
from utils.util import to8b
from models.mvs.mvs_utils import *

def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)

def save_image(img_array, filepath):
    assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                         and img_array.shape[2] in [3, 4])

    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_array).save(filepath)

def save_points(xyz, dir, total_steps):
    if xyz.ndim < 3:
        xyz = xyz[None, ...]
    os.makedirs(dir, exist_ok=True)
    for i in range(xyz.shape[0]):
        if isinstance(total_steps,str):
            filename = 'step-{}-{}.txt'.format(total_steps, i)
        else:
            filename = 'step-{:04d}-{}.txt'.format(total_steps, i)
        filepath = os.path.join(dir, filename)
        np.savetxt(filepath, xyz[i, ...].reshape(-1, xyz.shape[-1]), delimiter=";")


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def cam2image(points, K):
    ndim = points.ndim
    if ndim == 2:
        points = np.expand_dims(points, 0)
    points_proj = np.matmul(K[:3,:3].reshape([1,3,3]), points)
    depth = points_proj[:,2,:]
    depth[depth==0] = -1e-6
    u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.int)
    v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.int)

    if ndim==2:
        u = u[0]; v=v[0]; depth=depth[0]
    return u, v, depth

def world2cam(points, R, T, inverse=True):
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

def project_vertices(vertices, curr_pose, K):
    # current camera pose
    T = curr_pose[:3,  3]
    R = curr_pose[:3, :3]

    # convert points from world coordinate to local coordinate 
    points_local = world2cam(vertices, R, T)

    # perspective projection
    u,v,depth = cam2image(points_local, K)
    return (u,v), depth 
    

def vis_pcd_2d(pcd, t, k, height, width, name, edge=0): 
    xyz, color = pcd[:, :3], pcd[:, 3:] 
    mask_all = np.zeros(len(xyz)) > 0
    (u, v), depth = project_vertices(xyz, t, k)

    visibility_mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<width), v>=0), v<height)
    # visualize points within 30 meters
    visibility_mask = np.logical_and(np.logical_and(visibility_mask, depth>0), depth<80)
    mask_all = np.logical_or(mask_all, visibility_mask)
    # prepare depth map for visualization
    mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<width), v>=0), v<height)
    # visualize points within 30 meters
    mask = np.logical_and(np.logical_and(mask, depth>0), depth<200)

    # visualize 
    layout = (1,1)  
    fig, axs = plt.subplots(*layout)
    cm = plt.get_cmap('jet')
    # load RGB image for visualization
    # colorImage = imageio.imread(fn_frame) / 255.0
    # depthImage = cm(depthMap/depthMap.max())[...,:3]
    # colorImage[depthMap>0] = depthImage[depthMap>0]
    scale = 1
    ptsColorImage = np.ones((height*scale, width*scale, 3)) * 255.0
    ptsColorImage[v[mask]*scale,u[mask]*scale] = color[mask] / 255.0 
    np.clip(ptsColorImage, 0.0, 1.0, out=ptsColorImage)
    # ptsColorImage_overlaid = imageio.imread(fn_frame) / 255.0
    # ptsColorImage_overlaid[depthMap>0] = ptsColorImage[depthMap>0]
    # input_colorImage = imageio.imread(fn_frame) / 255.0
            
    # axs[0].imshow(depthMap, cmap='jet')
    # axs[0].title.set_text('Projected Depth')
    # axs[0].axis('off')
    # axs[1].imshow(colorImage)
    # axs[1].title.set_text('Projected Depth Overlaid on Image')
    # axs[1].axis('off')
    if edge > 0:
        ptsColorImage = ptsColorImage[edge:-edge, edge:-edge]
    os.makedirs(os.path.dirname(name), exist_ok=True)
    imageio.imwrite(name, (ptsColorImage*255.0).astype(np.uint8))


def depth2color(depth, near_plane, far_plane):
    """ depth: hw
    """
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = (np.clip(depth, 0, 1) * 255).astype(np.int32)
    colormap = "turbo"
    colors = np.array(matplotlib.colormaps[colormap].colors)
    depth_color = colors[depth] 
    return depth_color
    
    

class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.image_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        self.point_dir = os.path.join(opt.checkpoints_dir, opt.name, 'points')
        self.vid_dir = os.path.join(opt.checkpoints_dir, opt.name, 'vids')
        os.makedirs(self.vid_dir, exist_ok=True)

        if opt.show_tensorboard > 0:
            from tensorboardX import SummaryWriter
            self.tb_writer = SummaryWriter(
                os.path.join(
                    opt.checkpoints_dir, opt.name,
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def save_image(self, img_array, filepath):
        assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                             and img_array.shape[2] in [3, 4])

        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        Image.fromarray(img_array).save(filepath)

    def read_image(self, filepath, dtype=None):
        image = np.asarray(Image.open(filepath))
        if dtype is not None and dtype==np.float32:
            image = (image / 255).astype(dtype)
        return image

    def display_current_results(self, visuals, total_steps, opt=None):
        for name, img in visuals.items():
            if opt is not None and name in opt.visual_items:
                img = np.array(img)
                filename = 'step-{:04d}-{}.png'.format(total_steps, name)
                filepath = os.path.join(self.image_dir, filename)

                if 'scannet' in opt.dataset_name and opt.split == "render":
                    edge = 10
                    img = img[edge:-edge, edge:-edge]
                save_image(img, filepath)
            if name == 'coarse_depth':
                img = np.array(img)
                near = img.min()
                far = img.max()
                img = depth2color(img, 1.0, 80.0)
                filename = 'step-{:04d}-{}.png'.format(total_steps, name)
                filepath = os.path.join(self.image_dir, filename) 
                if 'scannet' in opt.dataset_name and opt.split == "render":
                    edge = 10
                    img = img[edge:-edge, edge:-edge]
                save_image(img, filepath)
            


    def save_pts_2d(self, data, total_steps, opt=None):
        intrinsic = data["intrinsic"].squeeze(0).cpu().numpy()
        pts = data["pts_for_vis"]
        c2w = data["c2w"].squeeze(0).cpu().numpy()
        height = data["h"].item()
        width = data["w"].item() 
        filename = 'step-{:04d}-{}.png'.format(total_steps, 'pcdproj')
        filename = os.path.join(self.image_dir, filename) 
        edge = 10 if 'scannet' in opt.dataset_name and opt.split=="render" else 0 
        vis_pcd_2d(pts, c2w, intrinsic, height, width, filename, edge)

        

    def check_if_rendered(self, visuals_list, total_steps, opt=None):
        if_rendered = True 
        for name in visuals_list:
            if opt is not None and name in opt.visual_items:
                filename = 'step-{:04d}-{}.png'.format(total_steps, name)
                filename = os.path.join(self.image_dir, filename) 
                if not os.path.exists(filename):
                    if_rendered = False
        return if_rendered

    def display_video(self, visual_lst, total_steps):
        for name in visual_lst[0].keys():
            stacked_imgs = [to8b(visuals[name]) for visuals in visual_lst]
            filename = 'video_{:04d}_{}.mov'.format(total_steps, name)
            imageio.mimwrite(os.path.join(self.vid_dir, filename), stacked_imgs, fps=5, quality=8)
            filename = 'video_{:04d}_{}.gif'.format(total_steps, name)
            imageio.mimwrite(os.path.join(self.vid_dir, filename), stacked_imgs, fps=5, format='GIF')

    def gen_video(self, name, steps, total_step):
        img_lst = []
        for i in steps:
            img_filepath = os.path.join(self.image_dir, 'step-{:04d}-{}.png'.format(i, name))
            img_arry = self.read_image(img_filepath, dtype=np.float32)
            img_lst.append(img_arry)
        stacked_imgs = [to8b(img_arry) for img_arry in img_lst]
        filename = 'video_{:04d}_{}.mov'.format(total_step, name)
        imageio.mimwrite(os.path.join(self.vid_dir, filename), stacked_imgs, fps=20, quality=10)
        filename = 'video_{:04d}_{}.gif'.format(total_step, name)
        imageio.mimwrite(os.path.join(self.vid_dir, filename), stacked_imgs, fps=5, format='GIF')

    def save_neural_points(self, total_steps, xyz, features, data, save_ref=0):
        if features is None:
            if torch.is_tensor(xyz):
                # xyz = xyz.detach().cpu().numpy()
                xyz = xyz.detach().cpu().numpy()
            save_points(xyz, self.point_dir, total_steps)
        elif features.shape[-1] == 9:
            pnt_lst = []
            for i in range(0,3):
                points = torch.cat([xyz, features[0, ..., i*3:i*3+3] * 255], dim=-1)
                if torch.is_tensor(points):
                    # xyz = xyz.detach().cpu().numpy()
                    points = points.detach().cpu().numpy()
                pnt_lst.append(points)
            save_points(np.stack(pnt_lst,axis=0), self.point_dir, total_steps)
        else:
            points = torch.cat([xyz, features[0, ..., :3] * 255], dim=-1)
            if torch.is_tensor(points):
                # xyz = xyz.detach().cpu().numpy()
                points = points.detach().cpu().numpy()
            save_points(points, self.point_dir, total_steps)

        if save_ref and "images" in data:
            self.save_ref_views(data, total_steps)


    def save_ref_views(self, data, total_steps, subdir=None):
            dir = self.point_dir if subdir is None else os.path.join(self.point_dir, subdir)
            for i in range(data['images'].shape[1]):
                img = data['images'][0,i].permute(1,2,0).cpu().numpy()
                filename = 'step-{}-{}-ref{}.png'.format(total_steps, 0, i)
                filepath = os.path.join(dir, filename)
                save_image(img, filepath)

            if data['images'].shape[1] > 3:
                img = data['images'][0,3].permute(1, 2, 0).cpu().numpy()
                filename = 'step-{}-{}-trgt.png'.format(total_steps, 0)
                filepath = os.path.join(dir, filename)
                save_image(img, filepath)

    def reset(self):
        self.start_time = time.time()
        self.acc_iterations = 0
        self.acc_losses = OrderedDict()

    def accumulate_losses(self, losses):
        self.acc_iterations += 1
        for k, v in losses.items():
            if k not in self.acc_losses:
                self.acc_losses[k] = 0
            self.acc_losses[k] += v
            if k.endswith('raycolor'):
                psnrkey = k + "_psnr"
                if psnrkey not in self.acc_losses:
                    self.acc_losses[psnrkey] = 0
                self.acc_losses[psnrkey] += mse2psnr(v)


    def get_psnr(self, key):
        return self.acc_losses[key + "_psnr"] / self.acc_iterations

    def print_losses(self, total_steps):
        m = 'End of iteration {} \t Number of batches {} \t Time taken: {:.2f}s\n'.format(
            total_steps, self.acc_iterations, (time.time() - self.start_time))
        m += '[Average Loss] '
        for k, v in self.acc_losses.items():
            m += '{}: {:.10f}   '.format(k, v / self.acc_iterations)
        filepath = os.path.join(self.log_dir, 'log.txt')
        with open(filepath, 'a') as f:
            f.write(m + '\n')
        print(m)


    def print_details(self, str):
        filepath = os.path.join(self.log_dir, 'log.txt')
        with open(filepath, 'a') as f:
            f.write(str + '\n')
        print(str)

    def plot_current_losses_with_tb(self, step, losses):
        if not self.opt.show_tensorboard > 0:
            return

        for key in losses.keys():
            curr_loss = losses[key]
            self.tb_writer.add_scalar(key, float(curr_loss), step)
