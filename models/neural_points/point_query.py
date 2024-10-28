import torch
import torch.nn
import torch.nn.functional as F
import os
import numpy as np
from numpy import dot
from math import sqrt
import matplotlib.pyplot as plt
import pickle
import time
from models.rendering.diff_ray_marching import near_far_linear_ray_generation, near_far_disparity_linear_ray_generation
parent_dir = os.path.dirname(os.path.abspath(__file__))


from torch.utils.cpp_extension import load as load_cuda

query_worldcoords_cuda = load_cuda(
    name='query_worldcoords_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/query_worldcoords.cpp', 'cuda/query_worldcoords.cu']],
    verbose=True)

def np_to_gpuarray(device, *args):
    result = []
    for x in args:
        if isinstance(x, np.ndarray):
            result.append(torch.from_numpy(x).to(device))
        else:
            print("trans",x)
    return result


class lighting_fast_querier():

    def __init__(self, device, opt):

        print("querier device", device, device.index)
        self.device="cuda"
        self.gpu = device.index
        self.opt = opt
        self.inverse = self.opt.inverse
        self.count=0
        self.radius_limit_np = np.asarray(self.opt.radius_limit_scale * max(self.opt.vsize[0], self.opt.vsize[1])).astype(np.float32)
        self.vscale_np = np.array(self.opt.vscale, dtype=np.int32)
        self.scaled_vsize_np = (self.opt.vsize * self.vscale_np).astype(np.float32)
        self.scaled_vsize_tensor = torch.as_tensor(self.scaled_vsize_np, device=device)
        self.kernel_size = np.asarray(self.opt.kernel_size, dtype=np.int32)
        self.kernel_size_tensor = torch.as_tensor(self.kernel_size, device=device)
        self.query_size = np.asarray(self.opt.query_size, dtype=np.int32)
        self.query_size_tensor = torch.as_tensor(self.query_size, device=device)

    def clean_up(self):
        pass

    # def get_hyperparameters(self, vsize_np, point_xyz_w_tensor, ranges=None):
    #     '''
    #     :param l:
    #     :param h:
    #     :param w:
    #     :param zdim:
    #     :param ydim:
    #     :param xdim:
    #     :return:
    #     '''
    #     min_xyz, max_xyz = torch.min(point_xyz_w_tensor, dim=-2)[0][0], torch.max(point_xyz_w_tensor, dim=-2)[0][0]
    #     ranges_min = torch.as_tensor(ranges[:3], dtype=torch.float32, device=min_xyz.device)
    #     ranges_max = torch.as_tensor(ranges[3:], dtype=torch.float32, device=min_xyz.device)
    #     if ranges is not None:
    #         # print("min_xyz", min_xyz.shape)
    #         # print("max_xyz", max_xyz.shape)
    #         # print("ranges", ranges)
    #         min_xyz, max_xyz = torch.max(torch.stack([min_xyz, ranges_min], dim=0), dim=0)[0], torch.min(torch.stack([max_xyz, ranges_max], dim=0), dim=0)[0]
    #     min_xyz = min_xyz - torch.as_tensor(self.scaled_vsize_np * self.opt.kernel_size / 2, device=min_xyz.device, dtype=torch.float32)
    #     max_xyz = max_xyz + torch.as_tensor(self.scaled_vsize_np * self.opt.kernel_size / 2, device=min_xyz.device, dtype=torch.float32)

    #     ranges_tensor = torch.cat([min_xyz, max_xyz], dim=-1)
    #     vdim_np = (max_xyz - min_xyz).cpu().numpy() / vsize_np
    #     scaled_vdim_np = np.ceil(vdim_np / self.vscale_np).astype(np.int32)

    #     return ranges_tensor, vsize_np, scaled_vdim_np 

    def get_hyperparameters(self, vsize_np, point_xyz_w_tensor, ranges=None):
        '''
        :param l:
        :param h:
        :param w:
        :param zdim:
        :param ydim:
        :param xdim:
        :return:
        '''
        min_xyz, max_xyz = torch.min(point_xyz_w_tensor, dim=-2)[0][0], torch.max(point_xyz_w_tensor, dim=-2)[0][0]
        vscale_np = np.array(self.opt.vscale, dtype=np.int32)
        scaled_vsize_np = (vsize_np * vscale_np).astype(np.float32)
        if ranges is not None:
            # print("min_xyz", min_xyz.shape)
            # print("max_xyz", max_xyz.shape)
            # print("ranges", ranges)
            min_xyz, max_xyz = torch.max(torch.stack([min_xyz, torch.as_tensor(ranges[:3], dtype=torch.float32, device=min_xyz.device)], dim=0), dim=0)[0], torch.min(torch.stack([max_xyz, torch.as_tensor(ranges[3:], dtype=torch.float32,  device=min_xyz.device)], dim=0), dim=0)[0]
        min_xyz = min_xyz - torch.as_tensor(scaled_vsize_np * self.opt.kernel_size / 2, device=min_xyz.device, dtype=torch.float32)
        max_xyz = max_xyz + torch.as_tensor(scaled_vsize_np * self.opt.kernel_size / 2, device=min_xyz.device, dtype=torch.float32)

        ranges_np = torch.cat([min_xyz, max_xyz], dim=-1).cpu().numpy().astype(np.float32)
        # print("ranges_np",ranges_np)
        vdim_np = (max_xyz - min_xyz).cpu().numpy() / vsize_np

        scaled_vdim_np = np.ceil(vdim_np / vscale_np).astype(np.int32)
        ranges_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu = np_to_gpuarray(
            point_xyz_w_tensor.device,
            ranges_np, scaled_vsize_np, scaled_vdim_np, vscale_np, np.asarray(self.opt.kernel_size, dtype=np.int32),
            np.asarray(self.opt.query_size, dtype=np.int32),
            )
        radius_limit_np, depth_limit_np = self.opt.radius_limit_scale * max(vsize_np[0], vsize_np[1]), self.opt.depth_limit_scale * vsize_np[2]
        return np.asarray(radius_limit_np).astype(np.float32), np.asarray(depth_limit_np).astype(np.float32), ranges_np, vsize_np, vdim_np, scaled_vsize_np, scaled_vdim_np, vscale_np, ranges_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, kernel_size_gpu, query_size_gpu


    def query_points(self, pixel_idx_tensor, point_xyz_pers_tensor, point_xyz_w_tensor, actual_numpoints_tensor, h, w, intrinsic, near_depth, far_depth, ray_dirs_tensor, cam_pos_tensor, cam_rot_tensor, raypos_tensor):
        if self.opt.nf_opt == "none": 
            near_depth, far_depth = np.asarray(near_depth).item() , np.asarray(far_depth).item()
        # ranges_tensor, vsize_np, scaled_vdim_np = self.get_hyperparameters(self.opt.vsize, point_xyz_w_tensor, ranges=self.opt.ranges)
        radius_limit_np, depth_limit_np, ranges_np, vsize_np, vdim_np, scaled_vsize_np, scaled_vdim_np, \
            vscale_np, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, \
                kernel_size_gpu, query_size_gpu = self.get_hyperparameters(self.opt.vsize, point_xyz_w_tensor, ranges=self.opt.ranges) 
        
        # print("self.opt.ranges", self.opt.ranges, range_gpu, ray_dirs_tensor)

        if raypos_tensor is None:
            # reuse when querying from multi-level feature.
            if self.opt.inverse > 0:
                raypos_tensor, _, _, _ = near_far_disparity_linear_ray_generation(cam_pos_tensor, ray_dirs_tensor, self.opt.z_depth_dim, near=near_depth, far=far_depth, jitter=0.3 if self.opt.is_train > 0 else 0.)
            else:
                raypos_tensor, _, _, _ = near_far_linear_ray_generation(cam_pos_tensor, ray_dirs_tensor, self.opt.z_depth_dim, near=near_depth, far=far_depth, jitter=0.3 if self.opt.is_train > 0 else 0.)

        D = raypos_tensor.shape[2]
        R = pixel_idx_tensor.reshape(point_xyz_w_tensor.shape[0], -1, 2).shape[1]

        sample_pidx_tensor, sample_loc_w_tensor, sample_loc_mask_tensor, ray_mask_tensor, raypos_mask_tensor= \
            query_worldcoords_cuda.woord_query_grid_point_index(pixel_idx_tensor, raypos_tensor, point_xyz_w_tensor, actual_numpoints_tensor, kernel_size_gpu,
                                                                query_size_gpu, self.opt.SR, self.opt.K, R, D,
                                                                scaled_vdim_gpu,
                                                                self.opt.max_o, self.opt.P, radius_limit_np, range_gpu,
                                                                scaled_vsize_gpu,
                                                                self.opt.gpu_maxthr, self.opt.NN)
        # sample_pidx_tensor, sample_loc_w_tensor, sample_loc_mask_tensor, ray_mask_tensor, raypos_mask_tensor = self.query_grid_point_index(
        #     h, w, pixel_idx_tensor, raypos_tensor, point_xyz_w_tensor, actual_numpoints_tensor, kernel_size_gpu, query_size_gpu, self.opt.SR, self.opt.K, ranges_np, scaled_vsize_np, scaled_vdim_np, vscale_np, self.opt.max_o, self.opt.P, radius_limit_np, depth_limit_np, range_gpu, scaled_vsize_gpu, scaled_vdim_gpu, vscale_gpu, ray_dirs_tensor, cam_pos_tensor, kMaxThreadsPerBlock=self.opt.gpu_maxthr)

        sample_ray_dirs_tensor = torch.masked_select(ray_dirs_tensor, ray_mask_tensor[..., None]>0).reshape(ray_dirs_tensor.shape[0],-1,3)[...,None,:].expand(-1, -1, self.opt.SR, -1).contiguous()
        # print("sample_ray_dirs_tensor", sample_ray_dirs_tensor.shape)
        return sample_pidx_tensor, self.w2pers(sample_loc_w_tensor, cam_rot_tensor, cam_pos_tensor), \
               sample_loc_w_tensor, sample_loc_mask_tensor, sample_ray_dirs_tensor, ray_mask_tensor, raypos_mask_tensor, vsize_np, range_gpu.cpu().numpy(), raypos_tensor


    def w2pers(self, point_xyz_w, camrotc2w, campos):
        #     point_xyz_pers    B X M X 3
        xyz_w_shift = point_xyz_w - campos[:, None, :]
        xyz_c = torch.sum(xyz_w_shift[..., None,:] * torch.transpose(camrotc2w, 1, 2)[:, None, None,...], dim=-1)
        z_pers = xyz_c[..., 2]
        x_pers = xyz_c[..., 0] / xyz_c[..., 2]
        y_pers = xyz_c[..., 1] / xyz_c[..., 2]
        return torch.stack([x_pers, y_pers, z_pers], dim=-1)