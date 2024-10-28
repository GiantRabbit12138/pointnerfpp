import numpy as np 
import torch
import torch.nn as nn
import torch_scatter

try:
    import MinkowskiEngine as ME
except:
    print("ME import error")
from ..backbone.voxelization_utils import sparse_quantize
from ..backbone import transform_utils 
from ..backbone import mink_utils 

def set_parameters(module, tensor, var_name, grad=True):     
    para = torch.nn.Parameter(tensor.clone().detach(), requires_grad=grad)
    module.register_parameter(var_name, para)   

def set_buffer(module, tensor, var_name): 
    module.register_buffer(var_name, tensor)

def voxelize(coords, voxel_size=0.02):
    assert coords.shape[1] == 3 and coords.shape[0]

    voxelization_matrix = np.eye(4)
    scale = 1 / voxel_size
    np.fill_diagonal(voxelization_matrix[:3, :3], scale)
    # Apply transformations
    rigid_transformation = voxelization_matrix  

    homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
    coords_aug = homo_coords @ rigid_transformation.T[:, :3]

    # Align all coordinates to the origin.
    min_coords = coords_aug.min(0)
    M_t = np.eye(4)
    M_t[:3, -1] = -min_coords
    rigid_transformation = M_t @ rigid_transformation
    coords_aug = coords_aug - min_coords
    coords_aug = np.floor(coords_aug)
    inds, inds_reconstruct = sparse_quantize(coords_aug, return_index=True)
    coords_aug= coords_aug[inds]
    return coords_aug, rigid_transformation, inds, inds_reconstruct

def pts2sparsetensor(coords, feat, voxel_size):   
    device = feat.device
    coords_vox, transform_world2grid, inds, inds_inverse = voxelize(
        coords.detach().cpu().numpy(), voxel_size)  
    
    inds = torch.from_numpy(inds).long().to(device) 
    inds_inverse = torch.from_numpy(inds_inverse).long().to(device) 

    coords_vox = torch.from_numpy(coords_vox).int().to(device) 
    # feat_vox = torch.from_numpy(feat_vox).float().cuda()
    transform_world2grid = torch.from_numpy(transform_world2grid).float().cuda()
    coords = ME.utils.batched_coordinates(
        [coords_vox], dtype=coords_vox.dtype, 
        device=device) 
    feat_vox = feat[inds]
    sinput = ME.SparseTensor(
        features=feat_vox, coordinates=coords)
    return sinput, inds, inds_inverse, transform_world2grid

def grid_sampling(coords, voxel_size):   
    device = coords.device
    _, _, _, inds_inverse = voxelize(
        coords.detach().cpu().numpy(), voxel_size)
    inds_inverse = torch.from_numpy(inds_inverse).long().to(device) 
    coords_sampled = torch_scatter.scatter(coords, inds_inverse, dim=0, reduce="mean")
    return coords_sampled 


class Backbone(nn.Module):
    def __init__(self, device, opt):
        super().__init__()  
        self.opt = opt
        self.device = device
        self.SR_init = opt.SR
        self.K_init = opt.K
        self.voxel_size = self.opt.backbone_voxel_size 

        if self.opt.backbone_opt.startswith('mink_unet'):
            from ..backbone.mink_unet import mink_unet 
            from ..backbone import sparse_pointnet 
            backbone_pts_dim = self.opt.point_features_dim 
            out_dim = self.opt.point_features_dim  

            if self.opt.point_init_emb_opt == 'learned_allinput':
                if self.opt.point_conf_mode == "1":
                    out_dim += 1
                    backbone_pts_dim += 1
                if self.opt.point_dir_mode == "1":
                    out_dim += 3
                    backbone_pts_dim += 3
                if self.opt.point_color_mode == "1":
                    out_dim += 3 
                    backbone_pts_dim += 3
            elif self.opt.point_init_emb_opt == "fixed_ones":
                backbone_pts_dim = 3
            
            
            if self.opt.points2vox_opt == "pointnet_ae": 
                # backbone_pts_dim = self.opt.point_features_dim 
                self.pointnet_encoder = sparse_pointnet.SparsePointnetEncoder(
                    dim=3+backbone_pts_dim, num_blocks=1 if self.opt.num_point_emb_level == -1 else 3, c_dim=backbone_pts_dim, hidden_dim=32)

            backbone_arch_opt = self.opt.backbone_opt.split('-')
            arch_opt = 'MinkUNet18A' if len(backbone_arch_opt) == 1 else backbone_arch_opt[-1]
            self.pts_backbone = mink_unet(
                in_channels=backbone_pts_dim, 
                D=3, arch=arch_opt)
            
            self.voxel_size = self.opt.backbone_voxel_size 
            if self.opt.num_point_emb_level == -1:
                self.channel_list = self.pts_backbone.FPN_PLANES 
                self.opt.num_point_emb_level = len(self.channel_list)
            else:
                self.channel_list = self.pts_backbone.FPN_PLANES[-self.opt.num_point_emb_level:] 

            out_dim_hier = out_dim
            if self.opt.point_conf_mode == "1":
                out_dim_hier += 1
            if self.opt.point_dir_mode == "1":
                out_dim_hier += 3
            if self.opt.point_color_mode == "1":
                out_dim_hier += 3 
            for idx_layer, in_channel in enumerate(self.channel_list): 
                setattr(self, f"linear_{idx_layer}", nn.Linear(in_channel, out_dim_hier))
            
            if self.opt.points2vox_opt == "pointnet_ae": 
                self.pointnet_decoder = sparse_pointnet.SparsePointnetDecoder(
                    dim=self.channel_list[-1]+3, num_blocks= 1 if self.opt.num_point_emb_level == -1 else 2, c_dim=out_dim, hidden_dim=32) 
            else:
                self.linear_last = nn.Linear(self.pts_backbone.FPN_PLANES[-1], out_dim)

        elif self.opt.backbone_opt.startswith('pointnet2'):
            from ..backbone.pointnet2 import get_model 
            self.pts_backbone = get_model(self.opt.point_features_dim)
            self.channel_list = []

            for idx_layer, in_channel in enumerate(self.channel_list):
                setattr(self, f"linear_{idx_layer}", nn.Linear(in_channel, self.opt.point_features_dim))

        elif self.opt.backbone_opt == 'none':
            pass 
        elif self.opt.backbone_opt == 'identity':
            pass 
        else:
            raise NotImplementedError 

    def mink_forward(self, pts_xyz, pts_feat, points_dir=None, points_conf=None, points_color=None, **kwargs): 
        # points 2 voxels.
        pts_feat_in = pts_feat
        pts_feat = pts_feat.squeeze(0)
        sinput, inds, inds_inverse, pts_transform = pts2sparsetensor(pts_xyz, pts_feat, self.voxel_size)
        if self.opt.points2vox_opt == "pointnet_ae": 
            coords_vox = sinput.decomposed_coordinates[0] 
            coords_w= transform_utils.grid2world(coords_vox.float(), pts_transform)
            pts_rel_xyz = pts_xyz - coords_w[inds_inverse]
            pointnet_feat_in = torch.cat([pts_rel_xyz, pts_feat], dim=-1)
            feat_vox = self.pointnet_encoder(
                pointnet_feat_in, inds_inverse, pooling=True)  
            coords = ME.utils.batched_coordinates(
                [coords_vox], dtype=coords_vox.dtype, 
                device=coords_vox.device) 
            sinput = ME.SparseTensor(
                features=feat_vox, coordinates=coords)

        soutput, fpn_list = self.pts_backbone(sinput, return_fpn=True)
        
        # level 0
        aux = {}
        source_coords_w_list = []
        source_feat_list = []
        vsize_list = [] 
        SR_list = [] 
        K_list = [] 
        points_dir_list = []
        points_conf_list = []
        points_color_list = []

        feat_dim = self.opt.point_features_dim
        if self.opt.num_point_emb_level >= 0:
            fpn_list = fpn_list[-self.opt.num_point_emb_level:] 
        elif self.opt.num_point_emb_level == -1:
            fpn_list = fpn_list
        else:
            raise NotImplementedError

        if self.opt.num_point_emb_level > 0:
            # soutput = soutput[inds_inverse]
            source_coords_w_list += [
                transform_utils.grid2world_batched(
                sparse_tensor.decomposed_coordinates, pts_transform[None]) \
                    for sparse_tensor in fpn_list 
            ]
            # tensor_stride_list = [
            #     sparse_tensor.tensor_stride[0]
            #         for sparse_tensor in fpn_list 
            # ] 
            # aux["tensor_stride_list"] = tensor_stride_list
            vsize_list += [
                self.voxel_size * sparse_tensor.tensor_stride[0]
                    for sparse_tensor in fpn_list 
            ] 
            SR_list += [
                min(int(vsize_ / self.opt.vsize[0] * self.SR_init * 0.5), 48)
                    for vsize_ in vsize_list 
            ] 
            K_list += [
                max(int(self.opt.vsize[0] / vsize_* self.K_init), 2)
                    for vsize_ in vsize_list 
            ]
            # print(SR_list)
            for idx_layer, sparse_tensor in enumerate(fpn_list):
                feat = getattr(self, f"linear_{idx_layer}", None)(sparse_tensor.features[None])
                source_feat_list += [feat[:, :, :feat_dim]] 

                if self.opt.point_conf_mode == "1":
                    if self.opt.mid_conf_mode == "constant_ones":
                        points_conf_list += [torch.sigmoid(feat[:, :, feat_dim:feat_dim+1])] 
                    else:
                        points_conf_list += [torch.ones_like(feat[:, :, feat_dim:feat_dim+1])] 
                if self.opt.point_dir_mode == "1":
                    points_dir_list += [feat[:, :, feat_dim+1:feat_dim+4]] 
                if self.opt.point_color_mode == "1":
                    points_color_list += [feat[:, :, feat_dim+4:]] 

        if self.opt.points2vox_opt == "pointnet_ae": 
            pts_feat = self.pointnet_decoder(
                pts_rel_xyz, soutput.F, inds_inverse)[None]  
        else:
            pts_feat = self.linear_last(soutput.F[inds_inverse])[None]

        emb = pts_feat[..., :self.opt.point_features_dim]       
        if self.opt.pts_feat_out_opt == "residual_input":
            emb = emb + pts_feat_in[..., :self.opt.point_features_dim] 

        if self.opt.point_init_emb_opt == 'learned_allinput':
            if self.opt.point_conf_mode == "1":
                if self.opt.pts_feat_out_opt == "residual_input":
                    points_conf_list += [pts_feat[:, :, feat_dim:feat_dim+1] + points_conf] 
                else:
                    points_conf_list += [torch.sigmoid(pts_feat[:, :, feat_dim:feat_dim+1])] # with sigmoid, it's more stable.
            if self.opt.point_dir_mode == "1":
                if self.opt.pts_feat_out_opt == "residual_input":
                    points_dir_list += [pts_feat[:, :, feat_dim:feat_dim+1] + points_dir] 
                else: 
                    points_dir_list += [pts_feat[:, :, feat_dim+1:feat_dim+4]]
            if self.opt.point_color_mode == "1":
                if self.opt.pts_feat_out_opt == "residual_input":
                    points_color_list += [pts_feat[:, :, feat_dim:feat_dim+1] + points_color] 
                else: 
                    points_color_list += [pts_feat[:, :, feat_dim+4:]] 
        else:
            if self.opt.point_conf_mode == "1":
                points_conf_list += [points_conf]
            if self.opt.point_dir_mode == "1":
                points_dir_list += [points_dir]
            if self.opt.point_color_mode == "1":
                points_color_list += [points_color]

        source_coords_w_list += [pts_xyz]
        source_feat_list += [emb]
        vsize_list += [self.opt.vsize[0]]

        SR_list += [self.opt.SR]
        K_list += [self.opt.K] 
        aux["vsize_list"] = vsize_list 
        aux["SR_list"] = SR_list 
        aux["K_list"] = K_list 
        if self.opt.point_conf_mode == "1":
            aux['points_conf'] = points_conf_list 
        if self.opt.point_dir_mode == "1":
            aux['points_dir'] = points_dir_list 
        if self.opt.point_color_mode == "1":
            aux['points_color'] = points_color_list 

        return source_coords_w_list,  source_feat_list, aux  
    
    def pointnet2_forward(self, pts_xyz, pts_feat, **kwargs): 
        aux = {}
        pts_xyz_list, pts_feat_list, tensor_stride_list = self.pts_backbone(
            torch.cat([pts_xyz[None], pts_feat], dim=-1).transpose(2, 1))
        aux["tensor_stride_list"] = tensor_stride_list
        return pts_xyz_list,  pts_feat_list, aux  
    
    def build_pyramid(self, pts_xyz, pts_feat, points_dir=None, points_conf=None, points_color=None, **kwargs):
        # Same as network.
        pts_feat = pts_feat.squeeze(0)
        self.voxel_size_list = []
        self.plane_size_list = [] 
        self.offset_opt = None if len(self.opt.pyramid_opt.split("-")) == 1 else self.opt.pyramid_opt.split("-")[-1]
        stride = self.opt.backbone_stride
        if self.opt.pyramid_opt == "smallminknet_fpn":
            cm, mapkey_list, stride_list, transform_world2grid = \
                mink_utils.build_fpn(
                    pts_xyz.detach().cpu().numpy(), self.voxel_size, self.opt.num_point_emb_level)
            for idx_layer, (stride, mapkey) in enumerate(zip(stride_list, mapkey_list)):
                add_pts_dim = (self.opt.num_point_emb_level - idx_layer) * self.opt.point_features_dim
                coords_vox = cm.get_coordinates(mapkey)[:, 1:]
                coords_w = transform_utils.grid2world(
                    coords_vox.float(), transform_world2grid)
                self.voxel_size_list += [self.voxel_size * stride]
                num_points = len(coords_vox) 
                set_parameters(
                    self, coords_w, 
                    f"coords_{idx_layer}", grad=False)
                set_parameters(
                    self, (torch.rand(num_points, pts_feat.shape[-1] + add_pts_dim).to(pts_feat) - 0.5) * 0.01, 
                    f"feat_{idx_layer}") 

                if self.opt.point_conf_mode == "1":
                    if self.opt.mid_conf_mode == "constant_ones":
                        set_parameters(
                            self, torch.ones(num_points, points_conf.shape[-1]).to(pts_feat), 
                            f"conf_{idx_layer}", grad=False) # Not sure if we want to learn this.
                    else:
                        set_parameters(
                            self, torch.ones(num_points, points_conf.shape[-1]).to(pts_feat), 
                            f"conf_{idx_layer}") # Not sure if we want to learn this.

                if self.opt.point_dir_mode == "1":
                    set_parameters(
                        self, torch.rand(num_points, points_dir.shape[-1]).to(pts_feat) * 2 - 1, 
                        f"dir_{idx_layer}") 

                if self.opt.point_color_mode == "1":
                    set_parameters(
                        self, torch.rand(num_points, points_color.shape[-1]).to(pts_feat) * 2 - 1, 
                        f"color_{idx_layer}") 
        elif self.opt.pyramid_opt.startswith("grid_sampling"): 

            for idx_layer in range(self.opt.num_point_emb_level):
                if self.opt.agg_opt in ["planes", 'pts_planes']: 
                    ori_point_features_dim = self.opt.point_features_dim
                    ori_plane_size = self.opt.plane_size 
                    current_level = (self.opt.num_point_emb_level - idx_layer - 1)  
                    current_level = current_level - self.opt.plane_start_idx
                    cur_stride = stride**current_level
                    # int(self.voxel_size / self.opt.vsize[0])
                    # cur_stride = current_level * int(self.voxel_size / self.opt.vsize[0])
                    # import pdb; pdb.set_trace()
                    cur_plane_size = int(ori_plane_size * cur_stride)

                    cur_plane_size = 1 if current_level < 0 else cur_plane_size 
                    self.plane_size_list += [cur_plane_size]
                    cur_point_features_dim = (cur_plane_size)**2 * 3 * ori_point_features_dim
                    add_pts_dim = cur_point_features_dim - self.opt.point_features_dim 
                    add_pts_dim = 0 if current_level < 0 else add_pts_dim 
                else:
                    add_pts_dim = (self.opt.num_point_emb_level - idx_layer) * self.opt.point_features_dim
                
                voxel_size = self.voxel_size * stride **(self.opt.num_point_emb_level - idx_layer - 1) 
                coords_w = grid_sampling(pts_xyz, voxel_size)
                self.voxel_size_list += [voxel_size]
                num_points = len(coords_w) 
                set_parameters(
                    self, coords_w, 
                    f"coords_{idx_layer}", grad=False)
                if self.offset_opt == "offset":
                    set_parameters(
                        self, torch.zeros_like(coords_w), 
                        f"coords_offset_{idx_layer}", grad=True)
                elif self.offset_opt is None:
                    pass
                else:
                    raise NotImplementedError
                    
                set_parameters(
                    self, (torch.rand(num_points, self.opt.point_features_dim + add_pts_dim).to(pts_feat) - 0.5) * 0.01, 
                    f"feat_{idx_layer}") 

                if self.opt.point_conf_mode == "1":
                    if self.opt.mid_conf_mode == "constant_ones":
                        set_parameters(
                            self, torch.ones(num_points, points_conf.shape[-1]).to(pts_feat), 
                            f"conf_{idx_layer}", False) # Not sure if we want to learn this.
                    else:
                        set_parameters(
                            self, torch.ones(num_points, points_conf.shape[-1]).to(pts_feat), 
                            f"conf_{idx_layer}") # Not sure if we want to learn this.

                if self.opt.point_dir_mode == "1":
                    set_parameters(
                        self, torch.rand(num_points, points_dir.shape[-1]).to(pts_feat) * 2 - 1, 
                        f"dir_{idx_layer}") 

                if self.opt.point_color_mode == "1":
                    set_parameters(
                        self, torch.rand(num_points, points_color.shape[-1]).to(pts_feat) * 2 - 1, 
                        f"color_{idx_layer}") 
        else:
            pass 

        print('')
        for idx_layer in range(self.opt.num_point_emb_level):
            coords = getattr(self, f'coords_{idx_layer}') 
            feat = getattr(self, f'feat_{idx_layer}') 
            print(f'{idx_layer} coords shape: {coords.shape}')
            print(f'{idx_layer} feat size : {feat.shape[0], feat.shape[1]}')
            if len(self.plane_size_list)> 0:
                print(f'plane size: {self.plane_size_list[idx_layer]}')
            print(f'voxel_size: {self.voxel_size_list[idx_layer]}')
        print(f'init feature size: {pts_xyz.shape[0] * 32}')

    def restore_pyramid(self, saved_features): 
        pts_xyz = nn.Parameter(saved_features["neural_points.xyz"])     
        suffix = "neural_points.backbone."

        self.voxel_size_list = []
        self.plane_size_list = [] 
        self.offset_opt = None if len(self.opt.pyramid_opt.split("-")) == 1 else self.opt.pyramid_opt.split("-")[-1]
        stride = self.opt.backbone_stride

        for idx_layer in range(self.opt.num_point_emb_level):
            if self.opt.agg_opt in ["planes", 'pts_planes']: 
                ori_point_features_dim = self.opt.point_features_dim
                ori_plane_size = self.opt.plane_size 
                current_level = (self.opt.num_point_emb_level - idx_layer - 1)  
                current_level = current_level - self.opt.plane_start_idx
                cur_stride = stride**current_level
                # int(self.voxel_size / self.opt.vsize[0])
                # cur_stride = current_level * int(self.voxel_size / self.opt.vsize[0])
                # import pdb; pdb.set_trace()
                cur_plane_size = int(ori_plane_size * cur_stride)

                cur_plane_size = 1 if current_level < 0 else cur_plane_size 
                self.plane_size_list += [cur_plane_size]
                cur_point_features_dim = (cur_plane_size)**2 * 3 * ori_point_features_dim
                add_pts_dim = cur_point_features_dim - self.opt.point_features_dim 
                add_pts_dim = 0 if current_level < 0 else add_pts_dim 
            else:
                add_pts_dim = (self.opt.num_point_emb_level - idx_layer) * self.opt.point_features_dim
            voxel_size = self.voxel_size * stride **(self.opt.num_point_emb_level - idx_layer - 1) 
            # coords_w_ori = grid_sampling(pts_xyz, voxel_size)
            self.voxel_size_list += [voxel_size]
            
            coords_w = saved_features[suffix+f"coords_{idx_layer}"]
            set_parameters(
                self, coords_w, 
                f"coords_{idx_layer}", grad=False)
            if self.offset_opt == "offset":
                set_parameters(
                    self, saved_features[suffix+f"coords_offset_{idx_layer}"], 
                    f"coords_offset_{idx_layer}", grad=True)
            elif self.offset_opt is None:
                pass
            else:
                raise NotImplementedError
            feat = saved_features[suffix+f'feat_{idx_layer}'] 
            set_parameters(
                self, feat, f"feat_{idx_layer}") 
            if self.opt.point_conf_mode == "1":
                if self.opt.mid_conf_mode == "constant_ones":
                    
                    set_parameters(
                        self, saved_features[suffix+f"conf_{idx_layer}"], 
                        f"conf_{idx_layer}", False) # Not sure if we want to learn this.
                else:
                    set_parameters(
                        self, saved_features[suffix+f"conf_{idx_layer}"], 
                        f"conf_{idx_layer}") # Not sure if we want to learn this.

            if self.opt.point_dir_mode == "1":
                set_parameters(
                    self, saved_features[suffix+f"dir_{idx_layer}"],
                    f"dir_{idx_layer}") 

            if self.opt.point_color_mode == "1":
                set_parameters(
                    self, saved_features[suffix + f'color_{idx_layer}'],
                    f"color_{idx_layer}") 
        else:
            pass 

        print('')
        for idx_layer in range(self.opt.num_point_emb_level):
            coords = getattr(self, f'coords_{idx_layer}') 
            feat = getattr(self, f'feat_{idx_layer}') 
            print(f'{idx_layer} coords shape: {coords.shape}')
            print(f'{idx_layer} feat size : {feat.shape[0], feat.shape[1]}')
            if len(self.plane_size_list)> 0:
                print(f'plane size: {self.plane_size_list[idx_layer]}')
            print(f'voxel_size: {self.voxel_size_list[idx_layer]}')
        print(f'init feature size: {pts_xyz.shape[0] * 32}')
        
    
    def identity_forward(self, pts_xyz, pts_feat, points_dir=None, points_conf=None, points_color=None, **kwargs): 
        # level 0
        aux = {}
        source_coords_w_list = []
        source_feat_list = []
        vsize_list = [] 
        planesize_list = []
        SR_list = [] 
        K_list = [] 
        points_dir_list = []
        points_conf_list = []
        points_color_list = []

        for idx_layer in range(self.opt.num_point_emb_level):
            coords_w = getattr(self, f'coords_{idx_layer}')
            if self.offset_opt == "offset":
                offset = torch.tanh(getattr(self, f'coords_{idx_layer}')) * 0.5 * self.voxel_size_list[idx_layer]
                coords_w = coords_w + offset 
            
            source_coords_w_list += [
                coords_w
            ]
            source_feat_list += [getattr(self, f'feat_{idx_layer}')[None]]
            if self.opt.point_conf_mode == "1":
                # TODO:
                points_conf_list += [getattr(self, f'conf_{idx_layer}')[None]] 
            if self.opt.point_dir_mode == "1":
                points_dir_list += [getattr(self, f'dir_{idx_layer}')[None]] 
            if self.opt.point_color_mode == "1":
                points_color_list += [getattr(self, f'color_{idx_layer}')[None]]
            vsize_list += [
                self.voxel_size_list[idx_layer]
            ]
            if self.opt.agg_opt in ["planes", "pts_planes"]:
                planesize_list += [
                    self.plane_size_list[idx_layer]
                ] 
        if self.opt.dataset_name in ['kitti360']:
            SR_list += [
                min(int(vsize_ / self.opt.vsize[0] * self.SR_init * 1.5), int(self.opt.z_depth_dim))
                    for vsize_ in vsize_list 
            ] 
            K_list += [
                max(int(self.opt.vsize[0] / vsize_* self.K_init), self.opt.min_k)
                    for vsize_ in vsize_list 
            ]
        else:
            SR_list += [
                min(int(vsize_ / self.opt.vsize[0] * self.SR_init * 0.5), self.opt.z_depth_dim // 2)
                    for vsize_ in vsize_list 
            ] 
            K_list += [
                max(int(self.opt.vsize[0] / vsize_* self.K_init), 2)
                    for vsize_ in vsize_list 
            ]
        # original resolution 
        source_coords_w_list += [pts_xyz]
        source_feat_list += [pts_feat]
        if self.opt.point_conf_mode == "1":
            points_conf_list += [points_conf]
        if self.opt.point_dir_mode == "1":
            points_dir_list += [points_dir]
        if self.opt.point_color_mode == "1":
            points_color_list += [points_color]
        vsize_list += [self.opt.vsize[0]]
        SR_list += [self.opt.SR]
        K_list += [self.opt.K] 
        planesize_list += [self.opt.plane_size]

        aux["vsize_list"] = vsize_list 
        aux["SR_list"] = SR_list 
        aux["K_list"] = K_list 
        if self.opt.agg_opt in ["planes", 'pts_planes']:
            aux["plane_size_list"] = planesize_list
        if self.opt.point_conf_mode == "1":
            aux['points_conf'] = points_conf_list 
        if self.opt.point_dir_mode == "1":
            aux['points_dir'] = points_dir_list 
        if self.opt.point_color_mode == "1":
            aux['points_color'] = points_color_list 
        return source_coords_w_list,  source_feat_list, aux  
         
    def forward(self, pts_xyz, pts_feat, **kwargs):
        if self.opt.backbone_opt.startswith('mink_unet'):
            return self.mink_forward(pts_xyz, pts_feat, **kwargs)
        elif self.opt.backbone_opt.startswith('pointnet2'):
            return self.pointnet2_forward(pts_xyz, pts_feat, **kwargs)
        elif self.opt.backbone_opt.startswith('identity'):
            return self.identity_forward(pts_xyz, pts_feat, **kwargs)
        else:
            raise NotImplementedError