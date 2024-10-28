import os
import numpy as np
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
    

def vis_pcd_2d(pcd, t, k, height, width, name): 
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
    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    plt.savefig(name, pad_inches=0, bbox_inches='tight')
    plt.close()