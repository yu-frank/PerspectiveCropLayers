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
    parser.add_argument('--total_epochs', type=int, default=50,
                        help='total number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate during training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for training/evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for the dataloaders')
    parser.add_argument('--exp_type', type=str,
                        help='type of experiment to run', default='2d3d')
    parser.add_argument('--use_dataset', type=str,
                        help='which dataset to use', default='H36m')
    parser.add_argument('--use_canonical', action='store_true',
                        help='use the canonical 3d pose for H36m (should be on)')

    parser.add_argument('--run_name', type=str,
                        help='experiment name', default='test')
    parser.add_argument('--model_path', type=str,
                        help='where to store model', default='model')
    parser.add_argument('--dataset_root', type=str,
                        help='path to the dataset')
    parser.add_argument('--denormalize_during_training', action='store_true',
                        help='denormalize 3d pose during training') 
    parser.add_argument('--use_slant_compensation', action='store_true',
                        help='use slant compensation (property for pcl) (should be on)')    
    parser.add_argument('--use_2d_scale', action='store_true',
                        help='use 2D scale for poses (property for pcl) (should be on)')      

    parser.add_argument('--use_mpi_aug', action='store_true',
                        help='use background augmentation for MPI-INF-3DHP')
    parser.add_argument('--use_resnet50', action='store_true',
                        help='use ResNet50 for backbone network (3d from img) else use ResNet18') 
    parser.add_argument('--use_pretrain', action='store_true',
                        help='use the pretrained model for ResNet50 or ResNet18 (should use pretrain)')   
    parser.add_argument('--seed', type=int, default=1,
                        help='seed to use')                     
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    use_pcl = args.use_pcl
    total_epochs = args.total_epochs 
    learning_rate = args.lr
    batch_size = args.batch_size
    dataset_root = args.dataset_root
    num_workers = args.num_workers
    denormalize_during_training = args.denormalize_during_training
    use_slant_compensation = args.use_slant_compensation
    use_2d_scale = args.use_2d_scale
    model_path = args.model_path
    run_name = args.run_name

    exp_type = args.exp_type

    use_dataset = args.use_dataset
    use_canonical = args.use_canonical

    use_mpi_aug = args.use_mpi_aug
    use_resnet50 = args.use_resnet50
    use_pretrain = args.use_pretrain

    torch.manual_seed(args.seed)

    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_folder = os.path.join(model_path, run_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Predict 3d pose from 2d pose
    if exp_type == '2d3d':
        # 2D to 3D keypoint lifting using GT 2D pose
        if use_dataset == 'H36m':
            train_dataset = H36MDataset(dataset_root, subset='trainval', without_image=True, use_pcl=use_pcl, \
                calculate_scale_from_2d=use_2d_scale, use_slant_compensation=use_slant_compensation)
            validation_dataset = H36MDataset(dataset_root, subset='test', without_image=True, use_pcl=use_pcl, \
                calculate_scale_from_2d=use_2d_scale, use_slant_compensation=use_slant_compensation)
        else:
            train_path = os.path.join(dataset_root, 'train')
            val_path = os.path.join(dataset_root, 'val')
            train_dataset = MpiInf3dDataset(train_path, without_image=True, use_pcl=use_pcl, \
            calculate_scale_from_2d=use_2d_scale, use_slant_compensation=use_slant_compensation)
            validation_dataset = MpiInf3dDataset(val_path,  without_image=True, use_pcl=use_pcl, \
                calculate_scale_from_2d=use_2d_scale, use_slant_compensation=use_slant_compensation)
    
        model = model = LinearModel()
        model.to(device)
        model.apply(weight_init)

    else:
        if use_dataset == 'H36m':
            train_dataset = H36MDataset(dataset_root, subset='trainval', without_image=False, use_pcl=use_pcl,  \
                calculate_scale_from_2d=use_2d_scale, use_slant_compensation=use_slant_compensation)
            validation_dataset = H36MDataset(dataset_root, subset='test', without_image=False, use_pcl=use_pcl, \
                calculate_scale_from_2d=use_2d_scale, use_slant_compensation=use_slant_compensation)
            
            if use_canonical:
                num_joints = 17
            else:
                num_joints = 32

        else:
            train_dataset = MpiInf3dDataset(train_path, without_image=False, use_pcl=use_pcl, \
                calculate_scale_from_2d=use_2d_scale, use_slant_compensation=True, use_aug=use_mpi_aug)
            validation_dataset = MpiInf3dDataset(val_path, without_image=False, use_pcl=use_pcl, \
                calculate_scale_from_2d=use_2d_scale, use_slant_compensation=True, use_aug=use_mpi_aug)
            
            num_joints = 17
        
        model = Resnet_H36m(device, num_joints=num_joints, use_pcl=use_pcl, use_resnet50=use_resnet50,\
                            use_pretrain=use_pretrain, dataset=use_dataset).to(device)

        

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    lowest_validation_loss = 1e7

    loss_fn = nn.MSELoss()

    print('Starting Training')
    for epoch in range(total_epochs):
        time.sleep(2) # solution found so training doesn't randomly crash

        # Experiments for 2D to 3D Keypoint Lifting
        if exp_type == "2d3d":
            model, optimizer = runner_2d3d(epoch, train_loader, model, optimizer, loss_fn, device,\
                                            use_dataset, use_pcl, use_slant_compensation,\
                                            denormalize_during_training, training=True)
            
            time.sleep(2) # solution found so training doesn't randomly crash
            validation_loss = runner_2d3d(epoch, validation_loader, model, optimizer, None, device,\
                                            use_dataset, use_pcl, use_slant_compensation,\
                                            denormalize_during_training, training=False) 

        # Experiments for 3D Pose from Image Regression
        else:
            model, optimizer = runner_3dFromImage(epoch, train_loader, model, optimizer, loss_fn, device, use_dataset, use_pcl,\
                                                  use_canonical, training=True)

            time.sleep(2) # solution found so training doesn't randomly crash
            validation_loss = runner_3dFromImage(epoch, validation_loader, model, optimizer, None, device, use_dataset, use_pcl,\
                                                  use_canonical, training=False)
        
        print("MPJPE on Validation Dataset after Epoch {} = {}".format(epoch, validation_loss))

        """SAVE MODEL AND OPTIMIZER"""
        training_file = os.path.join(model_folder, "latest_validation.tar")
        torch.save({
                    'epoch': epoch,
                    'dataset': use_dataset,
                    'exp_type': exp_type,
                    'batch_size': batch_size,
                    'validation_loss': validation_loss,
                    'lowest_validation_loss':lowest_validation_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
        }, training_file)

        if validation_loss < lowest_validation_loss:
            lowest_validation_loss = validation_loss
            validation_model = os.path.join(model_folder, "lowest_validation_model.tar")
            torch.save({
                        'epoch': epoch,
                        'dataset': use_dataset,
                        'exp_type': exp_type,
                        'batch_size': batch_size,
                        'validation_loss': lowest_validation_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
            }, validation_model)

    print('Finished Training')
            

            