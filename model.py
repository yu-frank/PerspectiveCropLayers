import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
import sys

import pcl

import pcl_util
import utils
import dataset_h36m
import dataset_3dhp

class Resnet_H36m(nn.Module):
    def __init__(self, device, num_joints=32, use_pcl=False, use_resnet50=True, use_pretrain=True, focal_at_image_plane=True, 
    slant_compensation=True, dataset="H36m", use_resnet34=False):
        super(Resnet_H36m, self).__init__()
        self.num_joints = num_joints
        self.focal_at_image_plane = focal_at_image_plane
        self.slant_compensation = slant_compensation

        if self.num_joints == 17:
            self.canonical_model = True
        else:
            self.canonical_model = False
        
        if dataset == 'H36m':
            self.dataset = 'H36m'
        else:
            self.dataset = '3DHP'

        self.localization = models.resnet18(pretrained=use_pretrain)
        self.localization.fc = nn.Linear(512, 4)
        
        self.localization.fc.weight.data.zero_()
        self.localization.fc.bias.data.copy_(torch.tensor([0, 0, 0, 0], dtype=torch.float)) # initialize @0 because applying sigmoid

        if use_resnet50:
            self.convNet = models.resnet50(pretrained=use_pretrain)
            self.convNet.fc = nn.Linear(2048, self.num_joints * 3)
        else:
            self.convNet = models.resnet18(pretrained=use_pretrain)
            self.convNet.fc = nn.Linear(512, self.num_joints * 3)

        if use_resnet34:
            self.convNet = models.resnet34(pretrained=use_pretrain)
            self.convNet.fc = nn.Linear(512, self.num_joints * 3)

        self.device = device
        self.use_pcl = use_pcl

    # Spatial transformer network forward function
    def stn(self, x, x_small, position_gt, scale_gt):
        if position_gt is None or scale_gt is None:
            theta = self.localization(x_small)

            theta[:, 0:2] = F.sigmoid(theta[:, 0:2]) 

        else:
            positions = position_gt
            scales = scale_gt
            theta = torch.zeros((x_small.shape[0], 4), dtype=torch.float32).to(self.device)
            
            theta[:, 2:] = positions
            theta[:, 0:2] = scales

        affine = torch.zeros([x.shape[0], 2, 3], dtype=torch.float32).to(self.device)
        #set affine matrix
        affine[:, 0, 0] = theta[:, 0] 
        affine[:, 1, 1] = theta[:, 1]
        affine[:, 0, 2] = theta[:, 2] #x loc
        affine[:, 1, 2] = theta[:, 3] #y loc
        grid = F.affine_grid(affine, x_small.size()) #crop size (could be same size as xsmall)
        
        x = F.grid_sample(x, grid)
        return x, affine, theta
    
    def pcl_layer(self, x, x_small, Ks, position_gt, scale_gt, rectangular_images=False):
        if position_gt is None or scale_gt is None:
            theta = self.localization(x_small) 
            
            theta[:, 0:2] = F.sigmoid(theta[:, 0:2]) 
            
            positions = theta[:, 2:] 
            scales = theta[:, 0:2] 
        else:
            positions = position_gt
            scales = scale_gt
            theta = torch.zeros((x_small.shape[0], 4), dtype=torch.float32).to(self.device)
            
            theta[:, 2:] = positions
            theta[:, 0:2] = scales

        img_w_h_shape = torch.tensor([x.shape[3], x.shape[2]]).to(self.device)
        scales_px = scales * img_w_h_shape
        positions_px = utils.pytorch_2_pixel_locations(positions, img_w_h_shape[0], img_w_h_shape[1])
   
        P_virt2orig, R_virt2orig, K_virt = pcl.pcl_transforms(positions_px, scales_px, Ks,\
             focal_at_image_plane=self.focal_at_image_plane, slant_compensation=self.slant_compensation, rectangular_images=rectangular_images)
        grid_perspective = pcl.perspective_grid(P_virt2orig, torch.tensor([x.shape[3], x.shape[2]]).to(self.device),\
             torch.tensor([x_small.shape[3], x_small.shape[2]]).to(self.device), transform_to_pytorch=True)
        grid_sparse = pcl.perspective_grid(P_virt2orig, torch.tensor([x.shape[3], x.shape[2]]).to(self.device),\
             torch.tensor([3, 3]).to(self.device), transform_to_pytorch=True)
        x = F.grid_sample(x, grid_perspective)
        return x, grid_sparse, theta, R_virt2orig

    def forward(self, x, x_small, Ks=None, position_gt=None, scale_gt=None, rectangular_images=False):
        if not self.use_pcl:
            stn_out, affine, theta = self.stn(x, x_small, position_gt, scale_gt) 
            convOut = self.convNet(stn_out)

            convOut = convOut.view(x.shape[0], self.num_joints, -1)

            return {"output":convOut, "stn_out":stn_out, "affine":affine, "theta":theta}
        else:
            pcl_out, grid_sparse, theta, R_virt2orig = self.pcl_layer(x, x_small, Ks, position_gt, scale_gt, rectangular_images) 

            conv_out = self.convNet(pcl_out)

            R_virt2orig = R_virt2orig.unsqueeze(1).repeat(1, self.num_joints, 1, 1) #Repeats along 2nd dimension num_joint times (1 for each joint)
            new_pre_transform = conv_out.view(x.shape[0], self.num_joints, -1)

            if self.canonical_model:
                if self.dataset == "H36m":
                    canonical_mean = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36mMean).to(self.device)
                    canonical_std = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36mStd).to(self.device)
                else:
                    canonical_mean = dataset_3dhp.mpi_3d_Mean.to(self.device)
                    canonical_std = dataset_3dhp.mpi_3d_Std.to(self.device)
                new_pre_transform = utils.denorm_human_joints(new_pre_transform, canonical_mean, canonical_std).unsqueeze(3)
            else:
                new_pre_transform = utils.denorm_human_joints(new_pre_transform, dataset_h36m.H36mMean.to(self.device), dataset_h36m.H36mStd.to(self.device)).unsqueeze(3)
            
            new_pre_transform = new_pre_transform.view(x.shape[0] * self.num_joints, 3, 1)
            R_virt2orig = R_virt2orig.view(x.shape[0] * self.num_joints, 3, 3)
            new_output = torch.bmm(R_virt2orig, new_pre_transform)
            new_output = new_output.squeeze(-1).view(x.shape[0], self.num_joints, -1)

            if self.canonical_model:
                if self.dataset == "H36m":
                    canonical_mean = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36mMean).to(self.device)
                    canonical_std = dataset_h36m.h36m_to_canonical_skeleton(dataset_h36m.H36mStd).to(self.device)
                else:
                    canonical_mean = dataset_3dhp.mpi_3d_Mean.to(self.device)
                    canonical_std = dataset_3dhp.mpi_3d_Std.to(self.device)
                normalized_new_output = utils.batch_normalize_canon_human_joints(new_output, mean=canonical_mean, std=canonical_std)
                new_pre_transform = new_pre_transform.squeeze(-1).view(x.shape[0], self.num_joints, -1)
                new_pre_transform = utils.batch_normalize_canon_human_joints(new_pre_transform, mean=canonical_mean, std=canonical_std)
            
            else:
                normalized_new_output = utils.batch_normalize_human_joints(new_output, mean=dataset_h36m.H36mMean.to(self.device), std=dataset_h36m.H36mStd.to(self.device))
                new_pre_transform = new_pre_transform.squeeze(-1).view(x.shape[0], self.num_joints, -1)
                new_pre_transform = utils.batch_normalize_human_joints(new_pre_transform, mean=dataset_h36m.H36mMean.to(self.device), std=dataset_h36m.H36mStd.to(self.device))

            return {"output": normalized_new_output, "pre_transform":new_pre_transform, "pcl_out":pcl_out, "grid_sparse":grid_sparse, "theta":theta, \
                "output_no_norm":new_output}


""" MODEL FOR 2D POSE TO 3D POSE LIFTING"""
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class LinearModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5,
                 input_joints=17,
                 output_joints=17):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        """Changed to Match H36m 17 joint canonical skeleton"""
        # 2d joints
        self.input_size =  input_joints * 2
        # 3d joints
        self.output_size = output_joints * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y