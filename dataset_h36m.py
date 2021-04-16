"""
Data loader for the Human 3.6M dataset.
Dataset home page: http://vision.imar.ro/human3.6m/
"""

from glob import iglob
from os import path
from torchvision import transforms

import h5py
import numpy as np
import torch
import torch.nn.functional
from PIL import Image
from pose3d_utils.camera import CameraIntrinsics
from pose3d_utils.coords import ensure_homogeneous, homogeneous_to_cartesian

from margipose.data import PoseDataset, collate
from margipose.data.skeleton import CanonicalSkeletonDesc, SkeletonDesc
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs
from margipose.eval import prepare_for_3d_evaluation, gather_3d_metrics

from torchvision import transforms
import utils
import pcl

import json
import constants

# Load in Constants (Mean and Stds)
H36mMean = constants.H36mMean
H36mStd = constants.H36mStd
H36m_2d_Mean = constants.H36m_2d_Mean
H36m_2d_Std = constants.H36m_2d_Std
H36m_2d_PCL_Mean = constants.H36m_2d_PCL_Mean
H36m_2d_PCL_Std = constants.H36m_2d_PCL_Std
H36m_2d_PCL_Mean_2dScale = constants.H36m_2d_PCL_Mean_2dScale
H36m_2d_PCL_Std_2dScale = constants.H36m_2d_PCL_Std_2dScale
H36m_2d_STN_Mean_2dScale = constants.H36m_2d_STN_Mean_2dScale
H36m_2d_STN_Std_2dScale = constants.H36m_2d_STN_Std_2dScale
HRNet_pcl_mean = constants.HRNet_pcl_mean
HRNet_pcl_std = constants.HRNet_pcl_std
HRNet_stn_mean = constants.HRNet_stn_mean
HRNet_stn_std = constants.HRNet_stn_std

H36MSkeletonDesc = SkeletonDesc(
    joint_names=[
        # 0-3
        'pelvis', 'right_hip', 'right_knee', 'right_ankle',
        # 4-7
        'right_toes', 'right_site1', 'left_hip', 'left_knee',
        # 8-11
        'left_ankle', 'left_toes', 'left_site1', 'spine1',
        # 12-15
        'spine', 'neck', 'head', 'head_top',
        # 16-19
        'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',
        # 20-23
        'left_thumb', 'left_site2', 'left_wrist2', 'left_site3',
        # 24-27
        'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
        # 28-31
        'right_thumb', 'right_site2', 'right_wrist2', 'right_site3'
    ],
    joint_tree=[
        0, 0, 1, 2,
        3, 4, 0, 6,
        7, 8, 9, 0,
        11, 12, 13, 14,
        12, 16, 17, 18,
        19, 20, 19, 22,
        12, 24, 25, 26,
        27, 28, 27, 30,
    ],
    hflip_indices=[
        0, 6, 7, 8,
        9, 10, 1, 2,
        3, 4, 5, 11,
        12, 13, 14, 15,
        24, 25, 26, 27,
        28, 29, 30, 31,
        16, 17, 18, 19,
        20, 21, 22, 23,
    ]
)

H36M_Actions = {
    1:  'Miscellaneous',
    2:  'Directions',
    3:  'Discussion',
    4:  'Eating',
    5:  'Greeting',
    6:  'Phoning',
    7:  'Posing',
    8:  'Purchases',
    9:  'Sitting',
    10: 'SittingDown',
    11: 'Smoking',
    12: 'TakingPhoto',
    13: 'Waiting',
    14: 'Walking',
    15: 'WalkingDog',
    16: 'WalkingTogether',
}

def pcl_preprocess(batch_size, num_joints, canon_label_2d_with_hip, orig_img_shape, Ks_px_orig, location, scale, \
    normalize=True, use_slant_compensation=False, HRNet_input=False):       
        canon_virt_2d, R_virt2orig, P_virt2orig = pcl.pcl_transforms_2d(canon_label_2d_with_hip, location, scale, Ks_px_orig,\
                                                                        focal_at_image_plane=True, slant_compensation=use_slant_compensation)
        model_input = canon_virt_2d.clone()

        if normalize:
            if HRNet_input:
                model_input = utils.batch_normalize_HRNet_human_joints(model_input, HRNet_pcl_mean, HRNet_pcl_std)
            else:
                if use_slant_compensation:
                    model_input = utils.batch_normalize_canon_pcl_human_joints(model_input, H36m_2d_PCL_Mean_2dScale, H36m_2d_PCL_Std_2dScale)
                else:
                    model_input = utils.batch_normalize_canon_pcl_human_joints(model_input, H36m_2d_PCL_Mean, H36m_2d_PCL_Std)

        model_input = model_input.view(batch_size, -1)

        return {'model_input':model_input, 'canon_virt_2d':canon_virt_2d, 'R_virt2orig':R_virt2orig, 'P_virt2orig':P_virt2orig}

def h36m_to_canonical_skeleton(skel):
    assert skel.size(-2) == H36MSkeletonDesc.n_joints

    canonical_joints = [
        H36MSkeletonDesc.joint_names.index(s)
        for s in CanonicalSkeletonDesc.joint_names
    ]
    size = list(skel.size())
    size[-2] = len(canonical_joints)
    canonical_joints_tensor = torch.LongTensor(canonical_joints).unsqueeze(-1).expand(size)
    return skel.gather(-2, canonical_joints_tensor)


class H36MDataset(PoseDataset):
    '''Create a Dataset object for loading Human 3.6M pose data (protocol 2).
    Args:
        data_dir (str): path to the data directory
        data_specs (DataSpecs):
        subset (str): subset of the data to load ("train", "val", "trainval", or "test")
        use_aug (bool): set to `True` to enable random data augmentation
        max_length (int):
        universal (bool): set to `True` to use universal skeleton scale
    '''

    def __init__(self, data_dir, human_height=2000, data_specs=None, subset='train', use_aug=False, max_length=None,
                 universal=False, focal_diff=0, without_image=False, use_pcl=True, calculate_scale_from_2d=False, use_slant_compensation=False,\
                 use_predicted_2D=False, predicted_input_dir=None, img_small_size=128, img_big_size=256):
        if data_specs is None:
            data_specs = DataSpecs(
                ImageSpecs(img_small_size, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
                JointsSpecs(H36MSkeletonDesc, n_dims=2),
            )

        super().__init__(data_specs)

        if not path.isdir(data_dir):
            raise NotADirectoryError(data_dir)
        
        """NEW"""
        self.human_height = human_height
        self.focal_diff = focal_diff
        self.use_pcl = use_pcl
        self.calculate_scale_from_2d = calculate_scale_from_2d
        self.use_slant_compensation = use_slant_compensation
        self.img_big_size = img_big_size
        """"""

        self.subset = subset
        self.use_aug = use_aug
        self.data_dir = data_dir

        annot_files = sorted(iglob(path.join(data_dir, 'S*', '*', 'annot.h5')))

        keys = ['pose/2d', 'pose/3d', 'pose/3d-univ', 'camera', 'frame',
                'subject', 'action', 'subaction']
        datasets = {}
        self.camera_intrinsics = []

        intrinsics_ds = 'intrinsics-univ' if universal else 'intrinsics'
        self.predicted_input_dir = predicted_input_dir
        self.use_predicted_2D = use_predicted_2D
        if self.use_predicted_2D:
            json_file_dict = {}
            if self.predicted_input_dir is None:
                predicted_pose_files = sorted(iglob(path.join(data_dir, 'hrnet_predictions', 'S*', '*', '*.json')))
            else:
                predicted_pose_files = sorted(iglob(path.join(predicted_input_dir, 'hrnet_predictions', 'S*', '*', '*.json')))
            for json_file in predicted_pose_files:
                with open(json_file) as f:
                    data = json.load(f)
                    for img in data:
                        frame_name = img['image_id']
                        directory_name = path.join(json_file, frame_name)
                        keypoints = img['keypoints']
                        json_file_dict[directory_name] = np.array(img['keypoints'], dtype=np.float32)
            self.json_dict = json_file_dict


        for annot_file in annot_files:
            with h5py.File(annot_file) as annot:
                for k in keys:
                    if k in datasets:
                        datasets[k].append(annot[k].value)
                    else:
                        datasets[k] = [annot[k].value]
                cams = {}
                for camera_id in annot[intrinsics_ds].keys():
                    alpha_x, x_0, alpha_y, y_0 = list(annot[intrinsics_ds][camera_id])
                    cams[int(camera_id)] = CameraIntrinsics.from_ccd_params(alpha_x, alpha_y, x_0, y_0)
                for camera_id in annot['camera']:
                    self.camera_intrinsics.append(cams[camera_id])
        datasets = {k: np.concatenate(v) for k, v in datasets.items()}

        self.frame_ids = datasets['frame']
        self.subject_ids = datasets['subject']
        self.action_ids = datasets['action']
        self.subaction_ids = datasets['subaction']
        self.camera_ids = datasets['camera']
        self.joint_3d = datasets['pose/3d-univ'] if universal else datasets['pose/3d']
        self.joint_2d = datasets['pose/2d']

        # Protocol #2
        train_subjects = {1, 5, 6, 7, 8}
        test_subjects = {9, 11}

        train_ids = []
        test_ids = []

        for index, subject_id in enumerate(self.subject_ids):
            if subject_id in train_subjects:
                train_ids.append(index)
            if subject_id in test_subjects:
                test_ids.append(index)

        if subset == 'trainval':
            self.example_ids = np.array(train_ids, np.uint32)
        elif subset == 'test':
            self.example_ids = np.array(test_ids, np.uint32)
        else:
            raise Exception('Only trainval and test subsets are supported')

        if max_length is not None:
            self.example_ids = self.example_ids[:max_length]

        self.without_image = without_image
        self.multicrop = False

    def to_canonical_skeleton(self, skel):
        if self.skeleton_desc.canonical:
            return skel
        return h36m_to_canonical_skeleton(skel)

    def get_orig_skeleton(self, index):
        id = self.example_ids[index]
        original_skel = ensure_homogeneous(torch.from_numpy(self.joint_3d[id]), d=3)
        if self.skeleton_desc.canonical:
            if original_skel.size(-2) == H36MSkeletonDesc.n_joints:
                original_skel = h36m_to_canonical_skeleton(original_skel)
            else:
                raise Exception('unexpected number of joints: ' + original_skel.size(-2))
        return original_skel

    def _load_image(self, id):
        if self.without_image:
            return None
        image_file = path.join(
            self.data_dir,
            'S{:d}'.format(self.subject_ids[id]),
            '{}-{:d}'.format(H36M_Actions[self.action_ids[id]], self.subaction_ids[id]),
            'imageSequence',
            str(self.camera_ids[id]),
            'img_{:06d}.jpg'.format(self.frame_ids[id])
        )
        return Image.open(image_file)

    """NEW!!!"""
    def _load_predicted_pose2d(self, id):
        pose_file_path = path.join(
            self.predicted_input_dir,
            'hrnet_predictions',
            'S{:d}'.format(self.subject_ids[id]),
            '{}-{:d}'.format(H36M_Actions[self.action_ids[id]], self.subaction_ids[id]),

            'hrnet_detections_{}.json'.format(str(self.camera_ids[id])),
            'img_{:06d}.jpg'.format(self.frame_ids[id])
        )
        
        return self.json_dict[pose_file_path]

    def _evaluate_3d(self, index, original_skel, norm_pred, camera_intrinsics, transform_opts):
        assert self.skeleton_desc.canonical, 'can only evaluate canonical skeletons'
        expected, actual = prepare_for_3d_evaluation(original_skel, norm_pred, self,
                                                     camera_intrinsics, transform_opts,
                                                     known_depth=True)
        return gather_3d_metrics(expected, actual)

    def __len__(self):
        return len(self.example_ids)

    def _build_sample(self, index, orig_camera, orig_image, orig_skel, transform_opts, transform_opts_big, extrinsics, human_height, focal_diff):

        ctx = self.create_transformer_context(transform_opts)
        _, img, _ = ctx.transform(image= orig_image)

        big_ctx = self.create_transformer_context(transform_opts_big)
        _, img_big, _ = big_ctx.transform(image=orig_image)

        """Could comment this out"""

        temp_dict = dict(transform_opts)
        temp_dict.pop('in_camera')

        sample = {
            'index': index,  # Index in the dataset
            'valid_depth': 1,

            'original_skel': orig_skel.float(),

            'camera_original': orig_camera.matrix[:,:-1].float(),
            'original_img_shape': torch.FloatTensor(orig_image.size),

        }

        img_transform = transforms.Compose([transforms.ToTensor()])

        if img:
            sample['input'] = self.input_to_tensor(img)

        if img_big:
            sample['input_big'] = self.input_to_tensor(img_big)
            sample['input_big_img'] = img_transform(img_big)

        # Generate the GT location and Scale of Crop
        pelvis_joint = sample['original_skel'][0,:-1].unsqueeze(0) #because of legacy code in utils that take a list of centers
        all_joints = sample['original_skel'][:,:-1]
        sample['world_coord_skel_mm'] = all_joints
        relative_joints = all_joints - pelvis_joint

        sample['non_normalized_3d'] = relative_joints

        #Normalize the Joints!
        normalized_joints = utils.normalize_human_joints(relative_joints, H36mMean, H36mStd)

        sample['normalized_skel_mm'] = normalized_joints
        sample['pelvis_location_mm'] = pelvis_joint

        Ks_px = sample['camera_original']

        K = Ks_px.clone()
        K[0,2] = 0.
        K[1,2] = 0.
        P_px = Ks_px.clone()

        pose_2d = utils.world_2_camera_coordinates(P_px, all_joints.float())
        sample['pose2d_original'] = pose_2d
        sample['perspective_matrix'] = P_px

        if focal_diff != 0:
            Ks_px[0,0] *= focal_diff
            Ks_px[1,1] *= focal_diff
            sample['camera_original'] = Ks_px

        """generate_gt_scales_from2d"""
        if self.calculate_scale_from_2d:
            scale = utils.generate_gt_scales_from2d(pose_2d)
            square_scale = torch.tensor([torch.max(scale), torch.max(scale)])
        else:
            scale = utils.generate_gt_scales(K, human_height, pelvis_joint, sample['original_img_shape'][0], sample['original_img_shape'][1]) # 2000 is the height in mm
            square_scale = scale.clone()

        square_scale_py = square_scale / sample['original_img_shape']
        sample['stn_square_scale_py'] = square_scale_py

        location = utils.generate_gt_location(P_px, pelvis_joint, sample['original_img_shape'][0], sample['original_img_shape'][1])

        sample['crop_scale'] = torch.FloatTensor(scale)
        sample['crop_location'] = torch.FloatTensor(location)

        return sample

    def _build_sample_without_image(self, id, index, orig_camera, orig_skel, human_height, focal_diff):

        sample = {
            'index': index,  # Index in the dataset
            'valid_depth': 1,

            'original_skel': orig_skel.float(),

            'camera_original': orig_camera.matrix[:,:-1].float(),
            'original_img_shape': torch.FloatTensor([1000, 1000]),
        }

        # Generate the GT location and Scale of Crop
        pelvis_joint = sample['original_skel'][0,:-1].unsqueeze(0) #because of legacy code in utils that take a list of centers
        all_joints = sample['original_skel'][:,:-1]
        sample['world_coord_skel_mm'] = all_joints
        relative_joints = all_joints - pelvis_joint

        sample['non_normalized_3d'] = relative_joints

        #Normalize the Joints!
        normalized_joints = utils.normalize_human_joints(relative_joints, H36mMean, H36mStd)

        sample['normalized_skel_mm'] = normalized_joints
        sample['pelvis_location_mm'] = pelvis_joint

        Ks_px = sample['camera_original']
        
        K = Ks_px.clone()
        K[0,2] = 0.
        K[1,2] = 0.
        P_px = Ks_px.clone()

        pose_2d = utils.world_2_camera_coordinates(P_px, all_joints.float())
        sample['pose2d_original'] = pose_2d
        sample['perspective_matrix'] = P_px

        if focal_diff != 0:
            Ks_px[0,0] *= focal_diff
            Ks_px[1,1] *= focal_diff
            sample['camera_original'] = Ks_px

        """generate_gt_scales_from2d"""
        if self.calculate_scale_from_2d:
            scale = utils.generate_gt_scales_from2d(pose_2d)
            square_scale = torch.tensor([torch.max(scale), torch.max(scale)])
        else:
            scale = utils.generate_gt_scales(K, human_height, pelvis_joint, sample['original_img_shape'][0], sample['original_img_shape'][1]) # 2000 is the height in mm
            square_scale = scale.clone()

        square_scale_py = square_scale / sample['original_img_shape']
        sample['stn_square_scale_py'] = square_scale_py

        location = utils.generate_gt_location(P_px, pelvis_joint, sample['original_img_shape'][0], sample['original_img_shape'][1])

        sample['crop_scale'] = torch.FloatTensor(scale)
        sample['crop_location'] = torch.FloatTensor(location)

        if self.use_predicted_2D:
            json_keypoints = torch.FloatTensor(self._load_predicted_pose2d(id))
            new_keypoints = json_keypoints.view(16,3)
            keypoints_removed_score = new_keypoints[:,:-1]

            input_pose_py = utils.pixel_2_pytorch_locations(keypoints_removed_score, 64, 64)
            input_pose_px = utils.pytorch_2_pixel_locations(input_pose_py, sample['original_img_shape'][0], sample['original_img_shape'][1])

            input_pose_scale_match_px = input_pose_px * sample['stn_square_scale_py'].unsqueeze(0)
            hip = input_pose_scale_match_px[6,:].unsqueeze(0)
            input_pose_scale_match_px = input_pose_scale_match_px - hip
            predicted_pose2d_input = input_pose_scale_match_px + sample['crop_location']
            sample['predicted_2d_input'] = predicted_pose2d_input # 16x2

            scale = utils.generate_gt_scales_from2d(predicted_pose2d_input)
            square_scale = torch.tensor([torch.max(scale), torch.max(scale)])
            
            square_scale_py = square_scale / sample['original_img_shape']
            sample['stn_square_scale_py'] = square_scale_py

            location = predicted_pose2d_input[6,:]

            sample['crop_scale'] = torch.FloatTensor(scale)
            sample['crop_location'] = torch.FloatTensor(location)

            if self.use_pcl:
                canon_label_2d_with_hip = sample['predicted_2d_input'].unsqueeze(0) 
                preprocess = pcl_preprocess(1, canon_label_2d_with_hip.shape[1], canon_label_2d_with_hip, sample['original_img_shape'].unsqueeze(0), \
                    sample['camera_original'].unsqueeze(0), location.unsqueeze(0), scale.unsqueeze(0),\
                        normalize=True, use_slant_compensation=True, HRNet_input=True)

                sample['preprocess-model_input'] = preprocess['model_input'].squeeze(0)
                sample['preprocess-canon_virt_2d'] = preprocess['canon_virt_2d'].squeeze(0)
                sample['preprocess-R_virt2orig'] = preprocess['R_virt2orig'].squeeze(0)

        else:
            if self.use_pcl:
                canon_label_2d_with_hip = h36m_to_canonical_skeleton(pose_2d).unsqueeze(0) 
                preprocess = pcl_preprocess(1, canon_label_2d_with_hip.shape[1], canon_label_2d_with_hip, sample['original_img_shape'].unsqueeze(0), \
                    sample['camera_original'].unsqueeze(0), location.unsqueeze(0), scale.unsqueeze(0),\
                        normalize=True, use_slant_compensation=self.use_slant_compensation)
            
                sample['preprocess-model_input'] = preprocess['model_input'].squeeze(0)
                sample['preprocess-canon_virt_2d'] = preprocess['canon_virt_2d'].squeeze(0)
                sample['preprocess-R_virt2orig'] = preprocess['R_virt2orig'].squeeze(0)
        return sample

    def __getitem__(self, index):
        id = self.example_ids[index]

        if not self.without_image: 
            orig_image = self._load_image(id)
            if orig_image:
                img_w, img_h = orig_image.size
            else:
                img_w = img_h = 1000
            img_short_side = min(img_h, img_w)

            extrinsics = torch.eye(4).double()
            orig_camera = self.camera_intrinsics[id]

            orig_skel = self.get_orig_skeleton(index)

            # Bounding box details
            joints2d = homogeneous_to_cartesian(
                orig_camera.project(ensure_homogeneous(orig_skel, d=3)))
            min_x = joints2d[:, 0].min().item()
            max_x = joints2d[:, 0].max().item()
            min_y = joints2d[:, 1].min().item()
            max_y = joints2d[:, 1].max().item()
            bb_cx = (min_x + max_x) / 2
            bb_cy = (min_y + max_y) / 2
            bb_size = 1.5 * max(max_x - min_x, max_y - min_y)

            out_width = self.data_specs.input_specs.width
            out_height = self.data_specs.input_specs.height

            if self.multicrop:
                samples = []
                for aug_hflip in [False, True]:
                    for offset in [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)]:
                        aug_x = offset[0] * 8
                        aug_y = offset[1] * 8

                        transform_opts = {
                            'in_camera': orig_camera,
                            'in_width': img_w,
                            'in_height': img_h,
                            'centre_x': bb_cx + aug_x,
                            'centre_y': bb_cy + aug_y,
                            'rotation': 0,
                            'scale': bb_size / img_short_side,
                            'hflip_indices': self.skeleton_desc.hflip_indices,
                            'hflip': aug_hflip,
                            'out_width': out_width,
                            'out_height': out_height,
                            'brightness': 1,
                            'contrast': 1,
                            'saturation': 1,
                            'hue': 0,
                        }

                        samples.append(self._build_sample(index, orig_camera, orig_image, orig_skel,
                                                        transform_opts, extrinsics, self.human_height, self.focal_diff))

                return collate(samples)
            else:
                aug_hflip = False
                aug_brightness = aug_contrast = aug_saturation = 1.0
                aug_hue = 0.0
                aug_x = aug_y = 0.0
                aug_scale = 1.0
                aug_rot = 0

                if self.use_aug:
                    aug_hflip = np.random.uniform() < 0.5
                    if np.random.uniform() < 0.3:
                        aug_brightness = np.random.uniform(0.8, 1.2)
                    if np.random.uniform() < 0.3:
                        aug_contrast = np.random.uniform(0.8, 1.2)
                    if np.random.uniform() < 0.3:
                        aug_saturation = np.random.uniform(0.8, 1.2)
                    if np.random.uniform() < 0.3:
                        aug_hue = np.random.uniform(-0.1, 0.1)
                    aug_x = np.random.uniform(-16, 16)
                    aug_y = np.random.uniform(-16, 16)
                    aug_scale = np.random.uniform(0.9, 1.1)
                    if np.random.uniform() < 0.4:
                        aug_rot = np.clip(np.random.normal(0, 30), -30, 30)

                transform_opts = {
                    'in_camera': orig_camera,
                    'in_width': img_w,
                    'in_height': img_h,
                    'centre_x': bb_cx + aug_x,
                    'centre_y': bb_cy + aug_y,
                    'rotation': aug_rot,
                    'scale': bb_size * aug_scale / img_short_side,
                    'hflip_indices': self.skeleton_desc.hflip_indices,
                    'hflip': aug_hflip,
                    'out_width': out_width,
                    'out_height': out_height,
                    'brightness': aug_brightness,
                    'contrast': aug_contrast,
                    'saturation': aug_saturation,
                    'hue': aug_hue,
                }

                transform_opts_big = {
                    'in_camera': orig_camera,
                    'in_width': img_w,
                    'in_height': img_h,
                    'centre_x': bb_cx + aug_x,
                    'centre_y': bb_cy + aug_y,
                    'rotation': aug_rot,
                    'scale': bb_size * aug_scale / img_short_side,
                    'hflip_indices': self.skeleton_desc.hflip_indices,
                    'hflip': aug_hflip,
                    'out_width': self.img_big_size,
                    'out_height': self.img_big_size,
                    'brightness': aug_brightness,
                    'contrast': aug_contrast,
                    'saturation': aug_saturation,
                    'hue': aug_hue,
                }

                return self._build_sample(index, orig_camera, orig_image, orig_skel, transform_opts, transform_opts_big,
                                        extrinsics, self.human_height, self.focal_diff)
        
        else: #self.without_image == True
            orig_camera = self.camera_intrinsics[id]

            orig_skel = self.get_orig_skeleton(index)
            return self._build_sample_without_image(id, index, orig_camera, orig_skel, self.human_height, self.focal_diff)
