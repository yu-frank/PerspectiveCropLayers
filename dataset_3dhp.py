import re
from glob import iglob
from os import path
from torchvision import transforms

import h5py
import numpy as np
import torch
from PIL import Image, ImageOps
from pose3d_utils.coords import homogeneous_to_cartesian, ensure_homogeneous
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

from margipose.data import PoseDataset, collate
from margipose.data.mpi_inf_3dhp.common import Annotations, parse_camera_calibration, Constants, \
    MpiInf3dhpSkeletonDesc
from margipose.data.skeleton import CanonicalSkeletonDesc, VNect_Common_Skeleton
from margipose.data_specs import DataSpecs, ImageSpecs, JointsSpecs
from margipose.eval import prepare_for_3d_evaluation, gather_3d_metrics

import utils
import pcl
import pcl_util
import constants

# Load in Constants (Mean and Stds)
mpi_3d_Mean = constants.mpi_3d_Mean
mpi_3d_Std = constants.mpi_3d_Std
mpi_2d_pcl_slant_mean = constants.mpi_2d_pcl_slant_mean
mpi_2d_pcl_slant_std = constants.mpi_2d_pcl_slant_std
mpi_2d_pcl_3dscale_mean = constants.mpi_2d_pcl_3dscale_mean
mpi_2d_pcl_3dscale_std = constants.mpi_2d_pcl_3dscale_std
mpi_2d_stn_slant_mean = constants.mpi_2d_stn_slant_mean
mpi_2d_stn_slant_std = constants.mpi_2d_stn_slant_std
mpi_2d_stn_3dscale_mean = constants.mpi_2d_stn_3dscale_mean
mpi_2d_stn_3dscale_std = constants.mpi_2d_stn_3dscale_std


def pcl_preprocess(batch_size, num_joints, canon_label_2d_with_hip, orig_img_shape, Ks_px_orig, location, scale, \
    normalize=True, use_slant_compensation=False):

    canon_virt_2d, R_virt2orig, P_virt2orig = pcl.pcl_transforms_2d(canon_label_2d_with_hip, location, scale, Ks_px_orig,\
                                                                        focal_at_image_plane=True, slant_compensation=use_slant_compensation)
    model_input = canon_virt_2d.clone()

    if normalize:
        if use_slant_compensation:
            model_input = utils.batch_normalize_canon_pcl_human_joints(model_input, mpi_2d_pcl_slant_mean, mpi_2d_pcl_slant_std)
        else:
            model_input = utils.batch_normalize_canon_pcl_human_joints(model_input, mpi_2d_pcl_3dscale_mean, mpi_2d_pcl_3dscale_std)

    model_input = model_input.view(batch_size, -1)

    return {'model_input':model_input, 'canon_virt_2d':canon_virt_2d, 'R_virt2orig':R_virt2orig, 'P_virt2orig':P_virt2orig}

class FrameRef:
    def __init__(self, subject_id, sequence_id, camera_id, frame_index, activity_id=None):
        self.subject_id = subject_id
        self.sequence_id = sequence_id
        self.camera_id = camera_id
        self.frame_index = frame_index
        self.activity_id = activity_id

    @property
    def image_file(self):
        return 'S{}/Seq{}/imageSequence/video_{}/img_{:06d}.jpg'.format(
            self.subject_id, self.sequence_id, self.camera_id, self.frame_index + 1
        )

    @property
    def bg_mask_file(self):
        return 'S{}/Seq{}/foreground_mask/video_{}/img_{:06d}.png'.format(
            self.subject_id, self.sequence_id, self.camera_id, self.frame_index + 1
        )

    @property
    def ub_mask_file(self):
        return 'S{}/Seq{}/up_body_mask/video_{}/img_{:06d}.png'.format(
            self.subject_id, self.sequence_id, self.camera_id, self.frame_index + 1
        )

    @property
    def lb_mask_file(self):
        return 'S{}/Seq{}/low_body_mask/video_{}/img_{:06d}.png'.format(
            self.subject_id, self.sequence_id, self.camera_id, self.frame_index + 1
        )

    @property
    def annot_file(self):
        return 'S{}/Seq{}/annot.mat'.format(self.subject_id, self.sequence_id)

    @property
    def camera_file(self):
        return 'S{}/Seq{}/camera.calibration'.format(self.subject_id, self.sequence_id)

    @property
    def metadata_file(self):
        return 'S{}/Seq{}/metadata.h5'.format(self.subject_id, self.sequence_id)

    @property
    def bg_augmentable(self):
        seq_path = 'S{}/Seq{}'.format(self.subject_id, self.sequence_id)
        return Constants['seq_info'][seq_path]['bg_augmentable'] == 1

    @property
    def ub_augmentable(self):
        seq_path = 'S{}/Seq{}'.format(self.subject_id, self.sequence_id)
        return Constants['seq_info'][seq_path]['ub_augmentable'] == 1

    @property
    def lb_augmentable(self):
        seq_path = 'S{}/Seq{}'.format(self.subject_id, self.sequence_id)
        return Constants['seq_info'][seq_path]['lb_augmentable'] == 1

    def to_dict(self):
        return {
            'subject_id': self.subject_id,
            'sequence_id': self.sequence_id,
            'camera_id': self.camera_id,
            'frame_index': self.frame_index,
            'activity_id': self.activity_id,
        }


def random_texture():
    files = list(iglob('resources/textures/*.png'))
    file = files[np.random.randint(0, len(files))]
    texture = Image.open(file).convert('L')
    texture = ImageOps.colorize(
        texture,
        'black',
        (np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256))
    )
    return texture


def augment_clothing(img, mask, texture):
    a = np.array(img)
    grey = a.mean(axis=-1)
    blackness = (255 - grey).clip(min=0) / 255

    texture = np.array(texture, dtype=np.float)
    texture -= blackness[..., np.newaxis] * texture
    texture = Image.fromarray(texture.round().astype(np.uint8))

    return Image.composite(texture, img, mask)


def random_background():
    files = list(iglob('resources/backgrounds/*.jpg'))
    file = files[np.random.randint(0, len(files))]
    bg = Image.open(file)
    bg = RandomHorizontalFlip()(RandomCrop(768)(bg))
    return bg


def augment_background(img, mask, bg):
    return Image.composite(img, bg, mask)


class MpiInf3dDataset(PoseDataset):
    preserve_root_joint_at_univ_scale = False

    def __init__(self, data_dir, data_specs=None, use_aug=False, disable_mask_aug=False, \
        without_image=True, human_height=2000, focal_diff=0, use_pcl=True, calculate_scale_from_2d=True, use_slant_compensation=True):
        if data_specs is None:
            data_specs = DataSpecs(
                ImageSpecs(128, mean=ImageSpecs.IMAGENET_MEAN, stddev=ImageSpecs.IMAGENET_STDDEV),
                JointsSpecs(MpiInf3dhpSkeletonDesc, n_dims=3),
            )

        super().__init__(data_specs)

        """NEW"""
        self.human_height = human_height
        self.focal_diff = focal_diff
        self.use_pcl = use_pcl
        self.calculate_scale_from_2d = calculate_scale_from_2d
        self.use_slant_compensation = use_slant_compensation

        if not path.isdir(data_dir):
            raise NotADirectoryError(data_dir)

        metadata_files = sorted(iglob(path.join(data_dir, 'S*', 'Seq*', 'metadata.h5')))
        frame_refs = []
        univ_scale_factors = {}

        for metadata_file in metadata_files:
            # match = re.match(r'.*S(\d+)/Seq(\d+)/metadata.h5', metadata_file)
            match = re.match(r'.*S(\d+)\\Seq(\d+)\\metadata.h5', metadata_file)
            subject_id = int(match.group(1))
            sequence_id = int(match.group(2))

            activity_ids = None
            mat_annot_file = path.join(path.dirname(metadata_file), 'annot_data.mat')
            if path.isfile(mat_annot_file):
                with h5py.File(mat_annot_file, 'r') as f:
                    activity_ids = f['activity_annotation'][:].flatten().astype(int)

            with h5py.File(metadata_file, 'r') as f:
                keys = f['interesting_frames'].keys()
                for key in keys:
                    camera_id = int(re.match(r'camera(\d)', key).group(1))
                    for frame_index in f['interesting_frames'][key]:
                        activity_id = None
                        if activity_ids is not None:
                            activity_id = activity_ids[frame_index]
                        frame_refs.append(FrameRef(subject_id, sequence_id, camera_id, frame_index, activity_id))
                univ_scale_factors[(subject_id, sequence_id)] = f['scale'][0]

        self.data_dir = data_dir
        self.use_aug = use_aug
        self.disable_mask_aug = disable_mask_aug
        self.frame_refs = frame_refs
        self.univ_scale_factors = univ_scale_factors
        self.without_image = without_image
        self.multicrop = False

    @staticmethod
    def _mpi_inf_3dhp_to_canonical_skeleton(skel):
        assert skel.size(-2) == MpiInf3dhpSkeletonDesc.n_joints

        canonical_joints = [
            MpiInf3dhpSkeletonDesc.joint_names.index(s)
            for s in CanonicalSkeletonDesc.joint_names
        ]
        size = list(skel.size())
        size[-2] = len(canonical_joints)
        canonical_joints_tensor = torch.LongTensor(canonical_joints).unsqueeze(-1).expand(size)
        return skel.gather(-2, canonical_joints_tensor)

    def to_canonical_skeleton(self, skel):
        if self.skeleton_desc.canonical:
            return skel

        return self._mpi_inf_3dhp_to_canonical_skeleton(skel)

    def _get_skeleton_3d(self, index):
        frame_ref = self.frame_refs[index]
        metadata_file = path.join(self.data_dir, frame_ref.metadata_file)
        with h5py.File(metadata_file, 'r') as f:
            # Load the pose joint locations
            original_skel = torch.from_numpy(
                f['joints3d'][frame_ref.camera_id, frame_ref.frame_index]
            )

        if original_skel.shape[-2] == MpiInf3dhpSkeletonDesc.n_joints:
            # The training/validation skeletons have 28 joints.
            skel_desc = MpiInf3dhpSkeletonDesc
        elif original_skel.shape[-2] == CanonicalSkeletonDesc.n_joints:
            # The test set skeletons have the 17 canonical joints only.
            skel_desc = CanonicalSkeletonDesc
        else:
            raise Exception('unexpected number of joints: ' + original_skel.shape[-2])

        if self.skeleton_desc.canonical:
            if skel_desc == MpiInf3dhpSkeletonDesc:
                original_skel = self._mpi_inf_3dhp_to_canonical_skeleton(original_skel)
            elif skel_desc == CanonicalSkeletonDesc:
                # No conversion necessary.
                pass
            else:
                raise Exception()
            skel_desc = CanonicalSkeletonDesc

        return original_skel, skel_desc

    def _to_univ_scale(self, skel_3d, skel_desc, univ_scale_factor):
        univ_skel_3d = skel_3d.clone()

        # Scale the skeleton to match the universal skeleton size
        if self.preserve_root_joint_at_univ_scale:
            # Scale the skeleton about the root joint position. This should give the same
            # joint position coordinates as the "univ_annot3" annotations.
            root = skel_3d[..., skel_desc.root_joint_id:skel_desc.root_joint_id+1, :]
            univ_skel_3d -= root
            univ_skel_3d /= univ_scale_factor
            univ_skel_3d += root
        else:
            # Scale the skeleton about the camera position. Useful for breaking depth/scale
            # ambiguity.
            univ_skel_3d /= univ_scale_factor

        return univ_skel_3d

    def _evaluate_3d(self, index, original_skel, norm_pred, camera_intrinsics, transform_opts):
        assert self.skeleton_desc.canonical, 'can only evaluate canonical skeletons'
        expected, actual = prepare_for_3d_evaluation(original_skel, norm_pred, self,
                                                     camera_intrinsics, transform_opts,
                                                     known_depth=False)
        included_joints = [
            CanonicalSkeletonDesc.joint_names.index(joint_name)
            for joint_name in VNect_Common_Skeleton
        ]
        return gather_3d_metrics(expected, actual, included_joints)

    def __len__(self):
        return len(self.frame_refs)

    def _build_sample(self, index, orig_camera, orig_image, orig_skel, transform_opts, transform_opts_big):
        frame_ref = self.frame_refs[index]
        # out_width = self.data_specs.input_specs.width
        # out_height = self.data_specs.input_specs.height
        if orig_skel.shape[0] != 17:
            canonical_original_skel = self._mpi_inf_3dhp_to_canonical_skeleton(ensure_homogeneous(orig_skel, d=3)).float()
        else:
            canonical_original_skel = ensure_homogeneous(orig_skel, d=3).float()

        ctx = self.create_transformer_context(transform_opts)
        _, img, _ = ctx.transform(image=orig_image)

        big_ctx = self.create_transformer_context(transform_opts_big)
        _, img_big, _ = big_ctx.transform(image=orig_image)


        sample = {
            'index': index,  # Index in the dataset

            'original_skel': canonical_original_skel, 

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
        """14 is the location of the hip in canonical skeleton!"""
        pelvis_joint = sample['original_skel'][14,:-1].unsqueeze(0) #because of legacy code in utils that take a list of centers
        all_joints = sample['original_skel'][:,:-1]
        sample['world_coord_skel_mm'] = all_joints
        relative_joints = all_joints - pelvis_joint

        sample['non_normalized_3d'] = relative_joints

        #Normalize the Joints!
        normalized_joints = utils.batch_normalize_canon_human_joints(relative_joints.unsqueeze(0), mpi_3d_Mean, mpi_3d_Std).squeeze(0)

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

        if self.focal_diff != 0:
            Ks_px[0,0] *= self.focal_diff
            Ks_px[1,1] *= self.focal_diff
            sample['camera_original'] = Ks_px

        """generate_gt_scales_from2d"""
        if self.calculate_scale_from_2d:
            scale = utils.generate_gt_scales_from2d(pose_2d)
            square_scale = torch.tensor([torch.max(scale), torch.max(scale)])
        else:
            scale = utils.generate_gt_scales(K, self.human_height, pelvis_joint, sample['original_img_shape'][0], sample['original_img_shape'][1]) # 2000 is the height in mm
            square_scale = scale.clone()

        square_scale_py = square_scale / sample['original_img_shape']
        sample['stn_square_scale_py'] = square_scale_py

        location_2d3d = utils.generate_gt_location(P_px, pelvis_joint, sample['original_img_shape'][0], sample['original_img_shape'][1])
        sample['crop_location_2d3d'] = location_2d3d

        # Location that is centered in the middle of the 2D pose (NOTE: not the same as the location calculation in 2D->3D)
        location = torch.FloatTensor([(torch.max(pose_2d[:,0]) + torch.min(pose_2d[:,0]))/2, (torch.max(pose_2d[:,1]) + torch.min(pose_2d[:,1]))/2])

        sample['crop_scale'] = torch.FloatTensor(scale)
        sample['crop_location'] = torch.FloatTensor(location)

        return sample
    

    def _build_sample_without_image(self, index, orig_skel, orig_camera, img_wh):
        frame_ref = self.frame_refs[index]
        if orig_skel.shape[0] != 17:
            canonical_original_skel = self._mpi_inf_3dhp_to_canonical_skeleton(ensure_homogeneous(orig_skel, d=3)).float()
        else:
            canonical_original_skel = ensure_homogeneous(orig_skel, d=3).float()
        Ks_px_video_cam = orig_camera.matrix[:,:-1].float().unsqueeze(0) #originally was 2048 x 2048, need to resize to 768 x 768
        img_w_h_orig = torch.FloatTensor([2048, 2048]).unsqueeze(0)
        img_w_h_small = torch.FloatTensor([img_wh[0], img_wh[1]])
        Ks_px_image_cam = pcl_util.K_new_resolution_px(Ks_px_video_cam, img_w_h_orig, img_w_h_small).squeeze(0)
        sample = {
            'index': index,  # Index in the dataset

            'original_skel': canonical_original_skel, 

            # Transformed data
            'camera_original': Ks_px_image_cam,
            'original_img_shape': torch.FloatTensor(img_wh)

        }

        # Generate the GT location and Scale of Crop
        """HIP IS Position 14 in """
        pelvis_joint = sample['original_skel'][14,:-1].unsqueeze(0) #because of legacy code in utils that take a list of centers
        all_joints = sample['original_skel'][:,:-1]
        sample['world_coord_skel_mm'] = all_joints
        relative_joints = all_joints - pelvis_joint

        sample['non_normalized_3d'] = relative_joints

        #Normalize the Joints!
        normalized_joints = utils.batch_normalize_canon_human_joints(relative_joints.unsqueeze(0), mpi_3d_Mean, mpi_3d_Std).squeeze(0)

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

        if self.focal_diff != 0:
            Ks_px[0,0] *= self.focal_diff
            Ks_px[1,1] *= self.focal_diff
            sample['camera_original'] = Ks_px

        """generate_gt_scales_from2d"""
        if self.calculate_scale_from_2d:
            scale = utils.generate_gt_scales_from2d(pose_2d)
            square_scale = torch.tensor([torch.max(scale), torch.max(scale)])
        else:
            scale = utils.generate_gt_scales(K, self.human_height, pelvis_joint, sample['original_img_shape'][0], sample['original_img_shape'][1]) # 2000 is the height in mm
            square_scale = scale.clone()

        square_scale_py = square_scale / sample['original_img_shape']
        sample['stn_square_scale_py'] = square_scale_py

        location = utils.generate_gt_location(P_px, pelvis_joint, sample['original_img_shape'][0], sample['original_img_shape'][1])

        sample['crop_scale'] = torch.FloatTensor(scale)
        sample['crop_location'] = torch.FloatTensor(location)

        if self.use_pcl:
            canon_label_2d_with_hip = pose_2d.unsqueeze(0)
            preprocess = pcl_preprocess(1, canon_label_2d_with_hip.shape[1], canon_label_2d_with_hip, sample['original_img_shape'].unsqueeze(0), \
                sample['camera_original'].unsqueeze(0), location.unsqueeze(0), scale.unsqueeze(0),\
                     normalize=True, use_slant_compensation=self.use_slant_compensation)
        
            sample['preprocess-model_input'] = preprocess['model_input'].squeeze(0)
            sample['preprocess-canon_virt_2d'] = preprocess['canon_virt_2d'].squeeze(0)
            sample['preprocess-R_virt2orig'] = preprocess['R_virt2orig'].squeeze(0)

        return sample

    def __getitem__(self, index):
        if not self.without_image:
            frame_ref = self.frame_refs[index]

            skel_3d, skel_desc = self._get_skeleton_3d(index)
            univ_scale_factor = self.univ_scale_factors[(frame_ref.subject_id, frame_ref.sequence_id)]
            orig_skel = self._to_univ_scale(skel_3d, skel_desc, univ_scale_factor)

            if self.without_image:
                orig_image = None
                img_w = img_h = 768
            else:
                orig_image = Image.open(path.join(self.data_dir, frame_ref.image_file))
                img_w, img_h = orig_image.size

            with open(path.join(self.data_dir, frame_ref.camera_file), 'r') as f:
                cam_cal = parse_camera_calibration(f)[frame_ref.camera_id]

            # Correct the camera to account for the fact that video frames were
            # stored at a lower resolution.
            orig_camera = cam_cal['intrinsics'].clone()
            old_w = cam_cal['image_width']
            old_h = cam_cal['image_height']
            orig_camera.scale_image(img_w / old_w, img_h / old_h)

            extrinsics = cam_cal['extrinsics']

            # Bounding box details
            skel_2d = orig_camera.project_cartesian(skel_3d)
            min_x = skel_2d[:, 0].min().item()
            max_x = skel_2d[:, 0].max().item()
            min_y = skel_2d[:, 1].min().item()
            max_y = skel_2d[:, 1].max().item()
            bb_cx = (min_x + max_x) / 2
            bb_cy = (min_y + max_y) / 2
            bb_size = 1.5 * max(max_x - min_x, max_y - min_y)

            img_short_side = min(img_h, img_w)
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
                                                        transform_opts, extrinsics))

                return collate(samples)
            else:
                aug_bg = aug_ub = aug_lb = False
                aug_hflip = False
                aug_brightness = aug_contrast = aug_saturation = 1.0
                aug_hue = 0.0
                aug_x = aug_y = 0.0
                aug_scale = 1.0
                aug_rot = 0

                if self.use_aug:
                    if not self.disable_mask_aug:
                        aug_bg = frame_ref.bg_augmentable and np.random.uniform() < 0.6
                        aug_ub = frame_ref.ub_augmentable and np.random.uniform() < 0.2
                        aug_lb = frame_ref.lb_augmentable and np.random.uniform() < 0.5
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

                if orig_image:
                    if aug_bg:
                        orig_image = augment_background(
                            orig_image,
                            Image.open(path.join(self.data_dir, frame_ref.bg_mask_file)),
                            random_background()
                        )
                    if aug_ub:
                        orig_image = augment_clothing(
                            orig_image,
                            Image.open(path.join(self.data_dir, frame_ref.ub_mask_file)),
                            random_texture()
                        )
                    if aug_lb:
                        orig_image = augment_clothing(
                            orig_image,
                            Image.open(path.join(self.data_dir, frame_ref.lb_mask_file)),
                            random_texture()
                        )

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
                    'out_width': 256,
                    'out_height': 256,
                    'brightness': aug_brightness,
                    'contrast': aug_contrast,
                    'saturation': aug_saturation,
                    'hue': aug_hue,
                }

                return self._build_sample(index, orig_camera, orig_image, orig_skel, transform_opts, transform_opts_big)

        else:
            # dataloader when no image is required
            frame_ref = self.frame_refs[index]
            skel_3d, skel_desc = self._get_skeleton_3d(index)
            univ_scale_factor = self.univ_scale_factors[(frame_ref.subject_id, frame_ref.sequence_id)]
            img_w = img_h = 768

            with open(path.join(self.data_dir, frame_ref.camera_file), 'r') as f:
                cam_cal = parse_camera_calibration(f)[frame_ref.camera_id]

            # Correct the camera to account for the fact that video frames were
            # stored at a lower resolution.
            orig_camera = cam_cal['intrinsics'].clone()
            orig_skel = self._to_univ_scale(skel_3d, skel_desc, univ_scale_factor)
            return self._build_sample_without_image(index, orig_skel, orig_camera, [img_w, img_h])