import torch
import torch.nn.functional as F

def world2grid(pts_w, transform_voxel, batch_idx=None):
    '''
    '''
    pts_w = F.pad(pts_w, pad=(0, 1), mode='constant', value=1.0) # nx4
    pts_grid = pts_w @ transform_voxel.transpose(-2, -1) 
    return pts_grid[..., :3]    

def grid2world(pts_grid, transform_voxel):
    '''
    '''
    pts_grid = F.pad(pts_grid, pad=(0, 1), mode='constant', value=1.0) # nx4
    transform_voxel_inverse = torch.inverse(transform_voxel)
    pts_world = pts_grid @ transform_voxel_inverse.transpose(-2, -1) 
    return pts_world[..., :3]

def grid2world_batched(pts_grid_batched, transform_voxel):
    '''
    '''
    return torch.cat(
        [grid2world(pts_grid.float(), transform_voxel[i]) \
         for i, pts_grid in enumerate(pts_grid_batched)], 0)

def get_world_coords_from_sparse_tensor(sparse_tensor, pts_transform):
    w_coords = grid2world_batched(
        sparse_tensor.decomposed_coordinates, pts_transform)
    return w_coords  