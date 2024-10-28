try:
    import MinkowskiEngine as ME
except:
    print("import error")

import torch
from ..backbone.voxelization_utils import sparse_quantize
import numpy as np 

def voxelize(coords, feats, voxel_size=0.02):
    assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]

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
    coords_aug, feats = coords_aug[inds], feats[inds]
    return coords_aug, feats, rigid_transformation
    
def build_fpn(pcd, voxel_size, nlevels=5):   
    coords, feat = pcd[:, :3], pcd[:, 3:]
    coords_vox, _, transform_world2grid= voxelize(coords, feat, voxel_size)
    coords_vox = torch.from_numpy(coords_vox).int().cuda() 
    transform_world2grid = torch.from_numpy(transform_world2grid).float().cuda()
    coords = ME.utils.batched_coordinates(
        [coords_vox], dtype=coords_vox.dtype, 
        device=coords_vox.device)
    dummy_sparse = ME.SparseTensor(
        features=coords.float(), coordinates=coords)
    mapkey_stride1 = dummy_sparse.coordinate_map_key
    cm = dummy_sparse.coordinate_manager 
    mapkey_list = [mapkey_stride1]
    stride_list = [1]
    stride_to_list = [ 2**i for i in range(1, nlevels)]
    for stride in stride_to_list: 
        mapkey_list+= [cm.stride(mapkey_stride1, stride=stride)]
        stride_list += [stride]
    # from low to high resolution
    mapkey_list = mapkey_list[::-1]
    stride_list = stride_list[::-1]
    return cm, mapkey_list, stride_list, transform_world2grid 

def reinitalize_cm(sparse_feat_in):    
    cm = sparse_feat_in.coordinate_manager
    mapkey = sparse_feat_in.coordinate_map_key
    feature = sparse_feat_in.features
    stride = sparse_feat_in.tensor_stride
    sparse_feat_out = ME.SparseTensor(
        features=feature, 
        coordinates=cm.get_coordinates(mapkey),
        tensor_stride=stride,
        )
    return sparse_feat_out