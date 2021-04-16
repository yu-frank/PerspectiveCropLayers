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

from dataset_h36m import H36MDataset
from dataset_3dhp import MpiInf3dDataset

from model import Resnet_H36m, LinearModel, weight_init

import time

from runner import runner_2d3d, runner_3dFromImage

import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    # for logging / monitoring
    parser.add_argument('--use_pcl', action='store_true',
                        help='use pcl')
    parser.add_argument('--model_file', type=str,
                        help='where to store model', default='model')        
    parser.add_argument('--use_dataset', type=str,
                        help='which dataset to use', default='H36m')
    parser.add_argument('--exp_type', type=str,
                        help='type of experiment to run', default='2d3d')
    parser.add_argument('--dataset_root', type=str,
                        help='path to the dataset')
    parser.add_argument('--use_2d_scale', action='store_true',
                        help='use 2D scale for poses')        
    parser.add_argument('--use_slant_compensation', action='store_true',
                        help='use slant compensation (property for pcl) (should be on for images and if using 2d scale)')       
    parser.add_argument('--use_canonical', action='store_true',
                        help='use the canonical 3d pose for H36m (should be on)')
    parser.add_argument('--use_resnet50', action='store_true',
                        help='use the pretrained model for ResNet50 or ResNet18') 
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_type = args.exp_type
    dataset_root = args.dataset_root
    use_dataset = args.use_dataset
    use_slant_compensation = args.use_slant_compensation
    use_pcl = args.use_pcl
    use_canonical = args.use_canonical
    use_resnet50 = args.use_resnet50
    use_2d_scale = args.use_2d_scale

    if exp_type == '2d3d':
        # 2D to 3D keypoint lifting using GT 2D pose
        if use_dataset == 'H36m':
            validation_dataset = H36MDataset(dataset_root, subset='test', without_image=True, use_pcl=use_pcl, \
                calculate_scale_from_2d=use_2d_scale, use_slant_compensation=use_slant_compensation)
        else:
            # val_path = os.path.join(dataset_root, 'val')
            val_path = dataset_root
            validation_dataset = MpiInf3dDataset(val_path,  without_image=True, use_pcl=use_pcl, \
                calculate_scale_from_2d=use_2d_scale, use_slant_compensation=use_slant_compensation)
    
        model = model = LinearModel()
        model.to(device)
        model.apply(weight_init)

    else:
        if use_dataset == 'H36m':
            validation_dataset = H36MDataset(dataset_root, subset='test', without_image=False, use_pcl=use_pcl, \
                calculate_scale_from_2d=use_2d_scale, use_slant_compensation=use_slant_compensation)
            
            if use_canonical:
                num_joints = 17
            else:
                num_joints = 32

        else:
            validation_dataset = MpiInf3dDataset(val_path, without_image=False, use_pcl=use_pcl, \
                calculate_scale_from_2d=use_2d_scale, use_slant_compensation=use_slant_compensation)
            
            num_joints = 17
        
        model = Resnet_H36m(device, num_joints=num_joints, use_pcl=use_pcl, use_resnet50=use_resnet50,\
                            use_pretrain=False, dataset=use_dataset).to(device)

    checkpoint = torch.load(args.model_file, map_location='cuda:0')
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint['epoch']
    model.to(device)
    model.eval()

    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)
    print('DATASET: ', use_dataset)
    print('USE PCL: ', use_pcl)
    print('USE CANONICAL POSE: ', use_canonical)
    print('USING 2D SCALE: ', use_2d_scale)
    print('USING SLANT COMPENSATION: ', use_slant_compensation)
    
    if exp_type == "2d3d":        
        time.sleep(2) # solution found so training doesn't randomly crash
        validation_loss = runner_2d3d(epoch, validation_loader, model, None, None, device,\
                                        use_dataset, use_pcl, use_slant_compensation,\
                                        denormalize_during_training=True, training=False) 

    # Experiments for 3D Pose from Image Regression
    else:
        time.sleep(2) # solution found so training doesn't randomly crash
        validation_loss = runner_3dFromImage(epoch, validation_loader, model, None, None, device, use_dataset, use_pcl,\
                                                use_canonical, training=False)
    
    print("MPJPE on Validation Dataset after Epoch {} = {}".format(epoch, validation_loss))