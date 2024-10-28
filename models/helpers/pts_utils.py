import os

try:
    import open3d as o3d
except:
    print('import open3d faiedl')
    
import torch


def export_pointcloud(name, points, colors=None, normals=None):

    save_dir = os.path.dirname(name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    
    if len(points.shape) > 2:
        points = points[0]
        if normals is not None:
            normals = normals[0]
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        if normals is not None:
            normals = normals.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(name, pcd)