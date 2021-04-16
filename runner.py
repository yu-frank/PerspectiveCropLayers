from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pickle
import os

from torch.utils.tensorboard import SummaryWriter

from model import Resnet_H36m, LinearModel, weight_init
from utils import prep_Ks_H36, denormalize_batch

import dataset_h36m
import dataset_3dhp

from margipose import data_specs
from margipose.data_specs import ImageSpecs
from margipose.data.skeleton import CanonicalSkeletonDesc

import margipose
import utils
from runner_utils import pcl_postprocess, calculate_batch_mpjpe, calculate_loss

import pcl
import pcl_util

import time

def runner_2d3d(epoch, data_loader, model, optimizer, loss_fn, device, use_dataset, 
                use_pcl, slant_compensation, denormalize_during_training, training):
    if training:
        model.train()
    else:
        model.eval()
        validation_loss = 0.0

    for i, data in enumerate(data_loader):
        Ks_px_orig = data['camera_original']

        orig_img_shape = data['original_img_shape']

        label = data['normalized_skel_mm'].to(device)
        label_no_norm = data['non_normalized_3d']

        pelvis_location = data['pelvis_location_mm']

        P_px = data['perspective_matrix'] 

        location_px = data['crop_location'].float()
        scale_px = data['crop_scale'].float()

        label_2d_px = data['pose2d_original'] # no hip location removal, 32x2

        square_scale = torch.tensor([torch.max(scale_px.squeeze(0)), torch.max(scale_px.squeeze(0))])
        square_scale_py = square_scale / data['original_img_shape'].squeeze(0)

        scale_py = square_scale_py.unsqueeze(0)
        location_py = utils.pixel_2_pytorch_locations(location_px.cpu(), orig_img_shape[:,0], orig_img_shape[:,1]).to(device)

        if use_dataset == "H36m":
            hips = label_2d_px[:,0,:].unsqueeze(1).repeat(1,label_2d_px.shape[1],1)
        else:
            hips = hips = label_2d_px[:,14,:].unsqueeze(1).repeat(1,label_2d_px.shape[1],1)

        label_2d_no_hip = label_2d_px - hips

        if use_dataset == 'H36m':
            canon_label_2d = dataset_h36m.h36m_to_canonical_skeleton(label_2d_no_hip.cpu()).to(device) # 
            canon_label_3d = dataset_h36m.h36m_to_canonical_skeleton(label.cpu()).to(device) # also normalized by mean and std
            label_no_norm = dataset_h36m.h36m_to_canonical_skeleton(label_no_norm)
        else:
            canon_label_2d = label_2d_no_hip.to(device) # 
            canon_label_3d = data['normalized_skel_mm'].to(device) # also normalized by mean and std

        num_joints = canon_label_2d.shape[1]
        bs = canon_label_2d.shape[0] 

        if use_pcl:
            model_input = data['preprocess-model_input'].to(device)
            canon_virt_2d = data['preprocess-canon_virt_2d'].to(device)
            R_virt2orig = data['preprocess-R_virt2orig']
        else:
            model_input = canon_label_2d.detach().clone()
            model_input = model_input / scale_py.unsqueeze(1).to(device) 

            if use_dataset == 'H36m':
                if slant_compensation:
                    mean = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36m_2d_STN_Mean_2dScale).to(device)
                    std = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36m_2d_STN_Std_2dScale).to(device)
                    model_input = utils.batch_normalize_canon_human_joints(model_input, mean, std)
                else:
                    mean = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36m_2d_Mean).to(device)
                    std = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36m_2d_Std).to(device)
                    model_input = utils.batch_normalize_canon_human_joints(model_input, mean, std)
            
            else:
                if slant_compensation:
                    mean = dataset_3dhp.mpi_2d_stn_slant_mean.to(device)
                    std = dataset_3dhp.mpi_2d_stn_slant_std.to(device)
                    model_input = utils.batch_normalize_canon_human_joints(model_input, mean, std)
                else:
                    mean = dataset_3dhp.mpi_2d_stn_3dscale_mean.to(device)
                    std = dataset_3dhp.mpi_2d_stn_3dscale_std.to(device)
                    model_input = utils.batch_normalize_canon_human_joints(model_input, mean, std)        

            model_input = model_input.view(bs, -1)
        
        if training:
            optimizer.zero_grad()
        output = model(model_input.to(device))
        
        if use_pcl:
            postprocess = pcl_postprocess(bs, num_joints, output, R_virt2orig, device, use_dataset)
            if denormalize_during_training:
                output = postprocess['output_no_norm']
            else:
                output = postprocess['output']
            normalized_output = postprocess['output']
            pre_transform = postprocess['pre_transform']

        else: 
            output = output.view(canon_label_3d.shape[0], -1, 3)
            if denormalize_during_training:
                if use_dataset == 'H36m':
                    canonical_mean = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36mMean).to(device)
                    canonical_std = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36mStd).to(device)
                else:
                    canonical_mean = dataset_3dhp.mpi_3d_Mean.to(device)
                    canonical_std = dataset_3dhp.mpi_3d_Std.to(device)
                output = utils.denorm_human_joints(output, canonical_mean, canonical_std)
        
        if training:
            if denormalize_during_training:
                loss = loss_fn(output, label_no_norm.to(device))
            else:
                loss = loss_fn(output, canon_label_3d)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
            optimizer.step()

            if i % 1000 == 0:
                print('Epoch: {} | Iteration: {} | Loss: {}'.format(epoch, i, loss.cpu().item()))
        
        else:
            if not denormalize_during_training:
                if use_dataset == 'H36m':
                    canonical_mean = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36mMean).to(device)
                    canonical_std = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36mStd).to(device)
                else:
                    canonical_mean = dataset_3dhp.mpi_3d_Mean.to(device)
                    canonical_std = dataset_3dhp.mpi_3d_Std.to(device)
                output = utils.denorm_human_joints(output, canonical_mean, canonical_std)

            loss = calculate_batch_mpjpe(output.detach(), label_no_norm.to(device))
            validation_loss += loss.item()

    if training:
        return model, optimizer 
    else:
        return validation_loss / len(data_loader)


def runner_3dFromImage(epoch, data_loader, model, optimizer, loss_fn, device, use_dataset, 
                use_pcl, use_canonical, training):

    if training:
        model.train()
    else:
        model.eval()
        validation_loss = 0.0

    for i, data in enumerate(data_loader):
        input_small = data['input'].to(device)
        input_big = data['input_big'].to(device)
        img_big = data['input_big_img']

        Ks_px_orig = data['camera_original'].to(device)
        orig_img_shape = data['original_img_shape']

        label = data['normalized_skel_mm']

        if use_dataset == 'H36m':
            if use_canonical:
                label_no_norm = dataset_h36m.h36m_to_canonical_skeleton(data['non_normalized_3d']).to(device)
            else:
                label_no_norm = data['non_normalized_3d'].to(device)
        else:
            label_no_norm = data['non_normalized_3d'].to(device)

        pelvis_location = data['pelvis_location_mm']

        location_px = data['crop_location'].float().to(device) 
        scale_px = data['crop_scale'].float().to(device)

        """CONVERT BACK TO PYTORCH COORDINATES"""
        location_py = utils.pixel_2_pytorch_locations(location_px.cpu(), orig_img_shape[:,0], orig_img_shape[:,1]).to(device)
        if use_pcl:
            scale_py = scale_px.cpu() / orig_img_shape
            scale_py = scale_py.to(device)
        else:
            scale_py = data['stn_square_scale_py'].to(device)

        img_w_h_shape = torch.tensor([input_big.shape[3], input_big.shape[2]]).to(device)

        if use_dataset == 'H36m':
            if use_canonical:
                label_2d = dataset_h36m.h36m_to_canonical_skeleton(data['pose2d_original'])
            else:
                label_2d = data['pose2d_original']
        else:
            label_2d = data['pose2d_original']

        P_px = data['perspective_matrix'] 
        
        if training:
            optimizer.zero_grad()

        if use_pcl:
            Ks = pcl_util.K_new_resolution_px(Ks_px_orig, orig_img_shape.to(device), img_w_h_shape).to(device)

            if use_dataset == 'H36m':
                output_dict = model(input_big, input_small, Ks, position_gt=location_py, scale_gt=scale_py, rectangular_images=False)
            else:
                # because some of the validation set images are rectangular instead of square
                output_dict = model(input_big, input_small, Ks, position_gt=location_py, scale_gt=scale_py, rectangular_images=True)
                
            output, pre_transform, pcl_out, grid_sparse, theta, output_no_norm = output_dict["output"], output_dict["pre_transform"], output_dict["pcl_out"], \
                output_dict["grid_sparse"], output_dict["theta"], output_dict["output_no_norm"]
        else:
            output_dict = model(input_big, input_small, position_gt=location_py, scale_gt=scale_py)
            output, stn_out, affine, theta = output_dict["output"], output_dict["stn_out"], output_dict["affine"], output_dict["theta"]

            if use_dataset == 'H36m':
                if use_canonical:
                    output_no_norm = utils.denorm_human_joints(output, dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36mMean).to(device),\
                            dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36mStd).to(device))
                else:
                    output_no_norm = utils.denorm_human_joints(output, dataset_h36m.H36mMean.to(device), dataset_h36m.H36mStd.to(device))
            else:
                output_no_norm = utils.denorm_human_joints(output, dataset_3dhp.mpi_3d_Mean.to(device), dataset_3dhp.mpi_3d_Std.to(device))

        if training:
            loss, loc_loss, scale_loss, regression_loss = calculate_loss(loss_fn, theta, output_no_norm, label_no_norm, \
                device, scale_py, location_py)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('Epoch: {} | Iteration: {} | Loss: {}'.format(epoch, i, loss.cpu().item()))
            
            return model, optimizer
        
        else:
            loss = calculate_batch_mpjpe(output.detach(), label_no_norm.to(device))
            validation_loss += loss.item()

            return validation_loss / len(data_loader)