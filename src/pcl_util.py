import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

bones_h36m = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13],]
cpm_2_h36m = [14, 8, 9, 10, 11, 12, 13, 14, 1, 0, 0, 5, 6, 7, 2, 3, 4]

def gaussianSmooth(x, dim=0):
    assert dim==0
    gaussian_kernel = torch.FloatTensor([[[0.061, 0.242, 0.383, 0.242, 0.061]]])
    num_points = len(x)
    num_smoothin_iterations = 2
    size_padded = list(x.shape)
    size_padded[dim] += num_smoothin_iterations*(2+2) # pad by two on both sides, as kernael of window 5 is used
    x_padded = torch.zeros(size_padded)
    x_padded[num_smoothin_iterations*2:-2*num_smoothin_iterations] = x
    x_padded[:num_smoothin_iterations*2] = x[0]
    x_padded[-num_smoothin_iterations*2:] = x[-1]

    results = []
    for i in range(len(x[dim])): # loop over remaingin dimensions
        # smooth position
        row = x_padded[:,i].unsqueeze(0).unsqueeze(0)
        for j in range(num_smoothin_iterations):
            row = F.conv1d(row, gaussian_kernel)
        results.append(row.squeeze())
    return torch.stack(results,dim=1)

def affine_2_pos_scale(affine_torch):
    bbox_pos_img = affine_torch[:, :, 2]
    bbox_size_img = torch.stack([affine_torch[:, 0, 0], affine_torch[:, 1, 1]], dim=-1)
    return bbox_pos_img, bbox_size_img

def affine_torch_2_pos_scale_unit(affine_torch):
    bbox_pos_img  = torch.stack([(affine_torch[:, 0, 2]+1)/2,
                                (-affine_torch[:, 1, 2]+1)/2], dim=-1)
    bbox_size_img = torch.stack([affine_torch[:, 0, 0], affine_torch[:, 1, 1]], dim=-1)
    return bbox_pos_img, bbox_size_img

def K_px2K_torch(K_px, img_w_h):
    K_torch = K_px.clone()
    K_torch[:, 0, 0] = K_px[:, 0, 0] * 2 / img_w_h[0]  # spread out from 0..w to -1..1
    K_torch[:, 1, 1] = K_px[:, 1, 1] * 2 / img_w_h[1]  # spread out from 0..h to -1..1
    K_torch[:, 0, 2] = K_px[:, 0, 2] * 2 / img_w_h[0] - 1  # move image origin bottom left corner to to 1/2 image width
    K_torch[:, 1, 2] = K_px[:, 1, 2] * 2 / img_w_h[1] - 1  # move image origin to 1/2 image width

    K_torch[:, 1, 1] *= -1  # point y coordinates downwards (image coordinates start in top-left corner in pytorch)
    K_torch[:, 1, 2] *= -1  # point y coordinates downwards (image coordinates start in top-left corner in pytorch)
    return K_torch

def K_new_resolution_px(K_px, img_w_h_orig, img_w_h_small):
    img_w_h_small = img_w_h_small.unsqueeze(0).repeat(img_w_h_orig.shape[0], 1)
    K_px2 = K_px.clone()
    K_px2[:, 0, :] *= (img_w_h_small[:,0]/img_w_h_orig[:,0]).unsqueeze(-1)  # spread out from 0..w to -1..1
    K_px2[:, 1, :] *= (img_w_h_small[:,1]/img_w_h_orig[:,1]).unsqueeze(-1)  # spread out from 0..h to -1..1
    return K_px2    

def K_torch2K_px(K_torch, img_w_h):
    K_px = K_torch.clone()
    K_px[:, 0, 0] = K_torch[:, 0, 0] / 2 * img_w_h[0]  # spread out from 0..w to -1..1
    K_px[:, 1, 1] = K_torch[:, 1, 1] / 2 * img_w_h[1]  # spread out from 0..h to -1..1
    K_px[:, 0, 2] = (1-K_torch[:, 0, 2]) / 2 * img_w_h[0] # move image origin bottom left corner to to 1/2 image width
    K_px[:, 1, 2] = (1-K_torch[:, 1, 2]) / 2 * img_w_h[1] # move image origin to 1/2 image width
    return K_px

def grid_img_2_gird_pytorch(grid):
    return grid

def plot_3Dpose(ax, pose_3d, bones=bones_h36m, linewidth=5, alpha=0.95, colormap='gist_rainbow', autoAxisRange=True, flip_yz=True, change_view=True):
    "Used for live application (real-time)"
    pose_3d = np.reshape(pose_3d.numpy().transpose(), (3, -1))
    pose_3d[1,:] *= -1

    if flip_yz:
        X, Y, Z = np.squeeze(np.array(pose_3d[0, :])), np.squeeze(np.array(pose_3d[2, :])), np.squeeze(
            np.array(pose_3d[1, :]))
    else:
        X, Y, Z = np.squeeze(np.array(pose_3d[0, :])), np.squeeze(np.array(pose_3d[1, :])), np.squeeze(
            np.array(pose_3d[2, :]))
    XYZ = np.vstack([X, Y, Z])

    if change_view:
        ax.view_init(elev=0, azim=-90)
    cmap = plt.get_cmap(colormap)

    maximum = len(bones) 

    for i, bone in enumerate(bones):
        colorIndex = cmap.N - cmap.N * i/float(maximum) # cmap.N - to start from back (nicer color)
        color = cmap(int(colorIndex))
        depth = max(XYZ[1, bone])
        zorder = -depth # otherwise bones with be ordered in the order of drawing or something even more weird...
        ax.plot(XYZ[0, bone], XYZ[1, bone], XYZ[2, bone], color=color, linewidth=linewidth, zorder=zorder,
                              alpha=alpha, solid_capstyle='round')

    # maintain aspect ratio
    if autoAxisRange:
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_axis_off()