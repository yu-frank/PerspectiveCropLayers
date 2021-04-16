import sys
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import torchvision
from mpl_toolkits.mplot3d import Axes3D
import math
import torch
from PIL import Image


import pcl
import pcl_util

import dataset_h36m
from margipose import data_specs

def roll_axis(img):
    img = np.rollaxis(img, -1, 0)
    img = np.rollaxis(img, -1, 0)
    return img

"""General Purpose Helper Fucntions"""
def generate_unique_run_name(name, model_save_path, run_save_path):
    run_string = "-run="
    run_count = 0
    not_unique = True
    new_run_name = name + run_string
    while not_unique:
        temp_new_run_name = new_run_name + str(run_count)
        temp_model_save_path = os.path.join(model_save_path, temp_new_run_name)
        temp_run_save_path = os.path.join(run_save_path, temp_new_run_name)
        if os.path.exists(temp_model_save_path) or os.path.exists(temp_run_save_path):
            run_count += 1
        else:
            new_run_name = temp_new_run_name
            not_unique = False
    return new_run_name

"""Helper Functions for Generating Bounding Box Labels"""
def intermediate_2_pixel_locations_2dPose(pose2d, width, height):
    pose2d_new = pose2d.clone()
    pose2d_new[:,0] = (pose2d_new[:,0]+0)/1 * width
    pose2d_new[:,1] = (pose2d_new[:,1]+0)/1 * height
    return pose2d_new

def pytorch_2_pixel_locations(pose2d, width, height):
    pose2d_new = pose2d.clone()
    pose2d_new[:,0] = (pose2d_new[:,0]+1)/2 * width
    pose2d_new[:,1] = (pose2d_new[:,1]+1)/2 * height
    return pose2d_new

def pixel_2_pytorch_locations(px_location, image_height, image_width):
    """
    Converts the pixel locations in an image to the pytorch coordinate system of [-1, 1]
    pixel_locations given in [width, height]
    """
    pixel_locations = px_location.clone()
    pixel_locations[:,0] = pixel_locations[:,0] / (image_width/2) - 1
    pixel_locations[:,1] = pixel_locations[:,1] / (image_height/2) - 1
    return pixel_locations
        

def world_2_camera_coordinates(P_px, world_coordinates):
    """
    Converts the world coordinates in blender to the pixel locations in the image given a
    perspective matrix
    """
    camera_coord = []
    if torch.is_tensor(world_coordinates):
        for coordinate in world_coordinates:
            hm_coord_2d = torch.matmul(P_px, coordinate)
            coord_2d = torch.FloatTensor([hm_coord_2d[0]/hm_coord_2d[2], hm_coord_2d[1]/hm_coord_2d[2]])
            camera_coord.append(coord_2d)
        return torch.stack(camera_coord)
    else:
        for coordinate in world_coordinates:
            hm_coord_2d = P_px.dot(coordinate)
            coord_2d = np.array([hm_coord_2d[0]/hm_coord_2d[2], hm_coord_2d[1]/hm_coord_2d[2]])
            camera_coord.append(coord_2d)
        return np.array(camera_coord)

def generate_gt_location(P_px, centers, width, height):
    center_px = world_2_camera_coordinates(P_px, centers)
    return center_px[0] # don't want the extra dimension  

def generate_2d_pose(P_px, all_joints, width, height):
    all_joints_px = world_2_camera_coordinates(P_px, all_joints)
    all_joints_px_py = pixel_2_pytorch_locations(all_joints_px, height, width)
    return all_joints_px_py # don't want the extra dimension 

def generate_gt_scales_cube(K, side_length, centers, width, height):
    gt_scales = []
    for center in centers:
        scale_cam = np.array((side_length*math.sqrt(3),side_length*math.sqrt(3), center[2]))
        scale_image_plane = scale_cam/scale_cam[2] # projection
        scale_image_px = K @ scale_image_plane # intrinsics matrix K (without translations)
        scale_image_normalized = scale_image_px[:2] / np.array([width, height])
        gt_scales.append(scale_image_normalized.float())
    return gt_scales[0] # because we don't want the extra dimension

def generate_gt_scales(K, side_length, centers, width, height):
    gt_scales = []
    for center in centers:
        scale_cam = np.array((side_length,side_length, center[2]))
        scale_image_plane = scale_cam/scale_cam[2] # projection
        scale_image_px = K @ scale_image_plane # intrinsics matrix K (without translations)
        gt_scales.append(scale_image_px[:2].float())
    return gt_scales[0] # because we don't want the extra dimension

def generate_gt_scales_from2d(pose_2d):
    max_y = torch.max(pose_2d[:,1])
    min_y = torch.min(pose_2d[:,1])
    max_x = torch.max(pose_2d[:,0])
    min_x = torch.min(pose_2d[:,0])
    scale_y = max_y - min_y
    scale_x = max_x - min_x
    return torch.tensor([scale_x, scale_y])
    
def relative_to_absolute(relative_location, center_location):
    center_location = np.repeat(center_location[:, np.newaxis, :], 8, axis=1)
    return relative_location + center_location

def generate_relative_coordinates(labels):
    labels = np.array(labels)
    cube_centers = np.mean(labels, axis=1)
    cube_centers_calc = np.repeat(cube_centers[:, np.newaxis, :], 8, axis=1)
    normalized_labels = labels - cube_centers_calc
    return normalized_labels, cube_centers

def load_cube_dataset_labels(root_dir):
    absolute_labels = []
    with (open(os.path.join(root_dir, "vertices.pkl"), "rb")) as f:
        absolute_labels = np.array(pickle.load(f))

    copy = np.copy(absolute_labels)
    absolute_labels[:,:,0] = copy[:,:,1] * -1

    absolute_labels[:,:,1] = copy[:,:,2]

    absolute_labels[:,:,2] = copy[:,:,0]

    absolute_labels = np.float32(absolute_labels)
    labels, centers = generate_relative_coordinates(absolute_labels)
    return labels, centers, absolute_labels

"""Training Helper Functions"""

def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    mean = torch.FloatTensor([0.0, 0.0, 0.0])
    std = torch.FloatTensor([0.0, 0.0, 0.0])
    for i, data in enumerate(dataloader):
        img = data['img_big']
        img = img.squeeze(0)
        mean[0] += torch.mean(img[0,:,:])
        mean[1] += torch.mean(img[1,:,:])
        mean[2] += torch.mean(img[2,:,:])
        std[0] += torch.std(img[0,:,:])
        std[1] += torch.std(img[1,:,:])
        std[2] += torch.std(img[2,:,:])

    mean = mean / len(dataloader)
    std = std / len(dataloader)
    return mean, std

def prep_Ks_H36(Ks_px, image_resolution_px):
    """NEW"""
    combined = []
    for i, resolution in enumerate(image_resolution_px):
        """H36m has different resolutions for different cameras so we have to computer K_px for each camera seperately"""
        combined.append(pcl_util.K_px2K_torch(Ks_px, resolution)[i])
    return torch.stack(combined)

def prep_Ks(Ks_px, batch_size, image_resolution_px):
    Ks_px = Ks_px.unsqueeze(0).expand([batch_size,3,3])
    Ks = pcl_util.K_px2K_torch(Ks_px, image_resolution_px)
    return Ks

def normalize_human_joints(joints, mean, std):
    #Inputs should all be 32x3
    normalized_joints = joints.clone()
    normalized_joints[1:,:] = torch.div(joints[1:,:] - mean[1:,:], std[1:,:])
    return normalized_joints

def batch_normalize_canon_pcl_human_joints(joints, mean, std):
    #Input Joints should all be bx17x3
    #mean and std are 17x3
    #joint index 14 is the pelvis which is has mean and std of 0
    batch_size = joints.shape[0]
    mean = mean.unsqueeze(0).repeat(batch_size, 1, 1)
    std = std.unsqueeze(0).repeat(batch_size, 1, 1)
    normalized_joints = joints.clone()
    normalized_joints[:,:14,:] = torch.div(joints[:,:14,:] - mean[:,:14,:], std[:,:14,:])
    normalized_joints[:,14,:] = joints[:,14,:] - mean[:,14,:]
    normalized_joints[:,15:,:] = torch.div(joints[:,15:,:] - mean[:,15:,:], std[:,15:,:])
    return normalized_joints

def batch_normalize_HRNet_human_joints(joints, mean, std):
    #Input Joints should all be bx17x3
    #mean and std are 17x3
    #joint index 14 is the pelvis which is has mean and std of 0
    batch_size = joints.shape[0]
    mean = mean.unsqueeze(0).repeat(batch_size, 1, 1)
    std = std.unsqueeze(0).repeat(batch_size, 1, 1)
    normalized_joints = joints.clone()
    normalized_joints[:,:6,:] = torch.div(joints[:,:6,:] - mean[:,:6,:], std[:,:6,:])
    normalized_joints[:,6,:] = joints[:,6,:] - mean[:,6,:]
    normalized_joints[:,7:,:] = torch.div(joints[:,7:,:] - mean[:,7:,:], std[:,7:,:])
    return normalized_joints

def batch_normalize_canon_human_joints(joints, mean, std):
    #Input Joints should all be bx17x3
    #mean and std are 17x3
    #joint index 14 is the pelvis which is has mean and std of 0
    batch_size = joints.shape[0]
    mean = mean.unsqueeze(0).repeat(batch_size, 1, 1)
    std = std.unsqueeze(0).repeat(batch_size, 1, 1)
    normalized_joints = joints.clone()
    normalized_joints[:,:14,:] = torch.div(joints[:,:14,:] - mean[:,:14,:], std[:,:14,:])
    normalized_joints[:,15:,:] = torch.div(joints[:,15:,:] - mean[:,15:,:], std[:,15:,:])
    return normalized_joints

def batch_normalize_human_joints(joints, mean, std):
    #Input Joints should all be bx32x3
    #mean and std are 32x3
    batch_size = joints.shape[0]
    mean = mean.unsqueeze(0).repeat(batch_size, 1, 1)
    std = std.unsqueeze(0).repeat(batch_size, 1, 1)
    normalized_joints = joints.clone()
    normalized_joints[:,1:,:] = torch.div(joints[:,1:,:] - mean[:,1:,:], std[:,1:,:])
    return normalized_joints

def denorm_human_joints(normalized_joints, mean, std):
    return torch.mul(normalized_joints, std) + mean 

def denormalize_batch(tensor, mean, std):
    for i, t in enumerate(tensor):
        tensor[i,:,:,:] = data_specs.denormalize_pixels(t, mean, std)
    return tensor

def position2D_torch_to_position2D_px(positions_torch, image_resolution_px):
    return (positions_torch+1)*image_resolution_px/2

def scale_torch_to_scale_px(scale_torch, image_resolution_px):
    return scale_torch*image_resolution_px
