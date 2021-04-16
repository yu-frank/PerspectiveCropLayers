import torch

def perspective_grid(P_virt2orig, image_pixel_size, crop_pixel_size_wh, transform_to_pytorch=False):
    batch_size = P_virt2orig.shape[0]

    # create a grid of linearly increasing indices (one for each pixel, going from 0..1)
    device = P_virt2orig.device
    xs = torch.linspace(0, 1, crop_pixel_size_wh[0]).to(device)
    ys = torch.linspace(0, 1, crop_pixel_size_wh[1]).to(device)

    #rs, cs = torch.meshgrid([xs, ys])  # for pytorch >0.4 instead of following two lines
    cs = ys.view(1, -1).repeat(xs.size(0), 1)
    rs = xs.view(-1, 1).repeat(1, ys.size(0))
    zs = torch.ones(rs.shape).to(device)  # init homogeneous coordinate to 1
    pv = torch.stack([rs, cs, zs])

    # same input grid for all batch elements, expand along batch dimension
    grid = pv.unsqueeze(0).expand([batch_size, 3, crop_pixel_size_wh[0], crop_pixel_size_wh[1]])

    # linearize the 2D grid to a single dimension, to apply transformation
    bpv_lin = grid.view([batch_size, 3, -1])

    # do the projection
    bpv_lin_orig = torch.bmm(P_virt2orig, bpv_lin)
    eps = 0.00000001
    bpv_lin_orig_p = bpv_lin_orig[:, :2, :] / (eps + bpv_lin_orig[:, 2:3, :]) # projection, divide homogeneous coord

    # go back from linear to two–dimensional outline of points
    bpv_orig = bpv_lin_orig_p.view(batch_size, 2, crop_pixel_size_wh[0], crop_pixel_size_wh[1])

    # the sampling function assumes the position information on the last dimension
    bpv_orig = bpv_orig.permute([0, 3, 2, 1])

    # the transformed points will be in pixel coordinates ranging from 0 up to the image width/height (unmapped from the original intrinsics matrix)
    # but the pytorch grid_sample function assumes it in -1,..,1; the direction is already correct (assuming negative y axis, which is also assumed by bytorch)
    if transform_to_pytorch:
        bpv_orig /= image_pixel_size.view([1,1,1,2]) # map to 0..1
        bpv_orig *= 2 # to 0...2
        bpv_orig -= 1 # to -1...1

    return bpv_orig

def pcl_transforms(bbox_pos_img, bbox_size_img, K, focal_at_image_plane=True, slant_compensation=True, rectangular_images=False, internal_call=False):
    K_inv = torch.inverse(K)
    # get target position from image coordinates (normalized pixels)
    p_position = bmm_homo(K_inv, bbox_pos_img)

    # get rotation from orig to new coordinate frame
    R_virt2orig = virtualCameraRotationFromPosition(p_position)

    # determine target frame
    K_virt = bK_virt(p_position, K, bbox_size_img, focal_at_image_plane, slant_compensation, maintain_aspect_ratio=True,\
                    rectangular_images=rectangular_images)

    K_virt_inv = torch.inverse(K_virt)
    # projective transformation orig to virtual camera
    P_virt2orig = torch.bmm(K, torch.bmm(R_virt2orig, K_virt_inv))

    
    if not internal_call:
        return P_virt2orig, R_virt2orig, K_virt
    else:
        R_orig2virt = torch.inverse(R_virt2orig)
        P_orig2virt = torch.inverse(P_virt2orig)
        return P_virt2orig, R_virt2orig, K_virt, R_orig2virt, P_orig2virt

def pcl_transforms_2d(pose2d, bbox_pos_img, bbox_size_img, K, focal_at_image_plane=True, slant_compensation=True, rectangular_images=False):
    # create canonical labels
    batch_size = pose2d.shape[0]
    num_joints = pose2d.shape[1]
    ones = torch.ones([batch_size, num_joints, 1])

    P_virt2orig, R_virt2orig, K_virt, R_orig2virt, P_orig2virt = pcl_transforms(bbox_pos_img, bbox_size_img, K,\
                                                                                focal_at_image_plane, slant_compensation, rectangular_images,\
                                                                                internal_call=True)

    # Vector manipulation to use torch.bmm and transform original 2D pose to virtual camera coordinates
    P_orig2virt = P_orig2virt.unsqueeze(1).repeat(1, num_joints, 1, 1)
    P_orig2virt = P_orig2virt.view(batch_size*num_joints, 3, 3)

    canonical_2d_pose = torch.cat((pose2d, ones), dim=-1).unsqueeze(-1)
    canonical_2d_pose = canonical_2d_pose.view(batch_size*num_joints, 3, 1)
    PCL_canonical_2d_pose = torch.bmm(P_orig2virt, canonical_2d_pose)
    PCL_canonical_2d_pose = PCL_canonical_2d_pose.squeeze(-1).view(batch_size, num_joints, -1)

    # Convert from homogeneous coordinate by dividing x and y by z
    virt_2d_pose = torch.div(PCL_canonical_2d_pose[:,:,:-1], PCL_canonical_2d_pose[:,:,-1].unsqueeze(-1))
    return virt_2d_pose, R_virt2orig, P_virt2orig

def virtPose2CameraPose(virt_pose, R_virt2orig, batch_size, num_joints):
    # for input 3d pose
    R_virt2orig = R_virt2orig.unsqueeze(1).repeat(1, num_joints, 1, 1)

    virt_pose = virt_pose.view(batch_size * num_joints, 3, 1)
    R_virt2orig = R_virt2orig.view(batch_size * num_joints, 3, 3)

    # Matrix Multiplication
    camera_pose = torch.bmm(R_virt2orig, virt_pose)
    camera_pose = camera_pose.squeeze(-1).view(batch_size, num_joints, -1)
    
    return camera_pose

def bK_virt(p_position, K, bbox_size_img, focal_at_image_plane, slant_compensation, maintain_aspect_ratio=True, rectangular_images=False):
    batch_size = bbox_size_img.shape[0]
    p_length = torch.norm(p_position, dim=1, keepdim=True)
    focal_length_factor = 1
    if focal_at_image_plane:
        focal_length_factor *= p_length
    if slant_compensation:
        sx = 1.0 / torch.sqrt(p_position[:,0]**2+p_position[:,2]**2)  # this is cos(phi)
        sy = torch.sqrt(p_position[:,0]**2+1) / torch.sqrt(p_position[:,0]**2+p_position[:,1]**2 + 1)  # this is cos(theta)
        bbox_size_img = bbox_size_img * torch.stack([sx,sy], dim=1)

    if not rectangular_images:
        if maintain_aspect_ratio:
            max_width,_ = torch.max(bbox_size_img, dim=-1, keepdims=True)
            bbox_size_img = torch.cat([max_width, max_width],dim=-1)
        f_orig = torch.stack([K[:,0,0], K[:,1,1]], dim=1)
        f_compensated = focal_length_factor * f_orig / bbox_size_img # dividing by the target bbox_size_img will make the coordinates normalized to 0..1, as needed for the perspective grid sample function; an alternative would be to make the grid_sample operate on pixel coordinates
        K_virt        = torch.zeros([batch_size,3,3], dtype=torch.float).to(f_compensated.device)
        K_virt[:,2,2] = 1
        # Note, in unit image coordinates ranging from 0..1
        K_virt[:, 0, 0] = f_compensated[:, 0]
        K_virt[:, 1, 1] = f_compensated[:, 1]
        K_virt[:,:2, 2] = 0.5
        return K_virt
    else:
        f_orig = torch.stack([K[:,0,0], K[:,1,1]], dim=1)
        f_re_scaled = f_orig / bbox_size_img
        if maintain_aspect_ratio:
            min_factor,_ = torch.min(f_re_scaled, dim=-1, keepdims=True)
            f_re_scaled = torch.cat([min_factor, min_factor],dim=-1)
        f_compensated = focal_length_factor * f_re_scaled 
        K_virt        = torch.zeros([batch_size,3,3], dtype=torch.float).to(f_compensated.device)
        K_virt[:,2,2] = 1
        K_virt[:, 0, 0] = f_compensated[:, 0]
        K_virt[:, 1, 1] = f_compensated[:, 1]
        K_virt[:,:2, 2] = 0.5
        return K_virt

def virtualCameraRotationFromPosition(position):
    x, y, z = position[:, 0], (position[:, 1]), position[:, 2]
    n1x = torch.sqrt(1 + x ** 2)
    d1x = 1 / n1x
    d1xy = 1 / torch.sqrt(1 + x ** 2 + y ** 2)
    d1xy1x = 1 / torch.sqrt((1 + x ** 2 + y ** 2) * (1 + x ** 2))
    R_virt2orig = torch.stack([d1x, -x * y * d1xy1x, x * d1xy,
                               0*x,      n1x * d1xy, y * d1xy,
                          -x * d1x,     -y * d1xy1x, 1 * d1xy], dim=1).reshape([-1, 3, 3])
    return R_virt2orig

def bmm_homo(K_inv, bbox_center_img):
    batch_size = bbox_center_img.shape[0]
    ones = torch.ones([batch_size, 1], dtype=torch.float).to(bbox_center_img.device)
    bbox_center_px_homo = torch.cat([bbox_center_img, ones],dim=1).reshape([batch_size,3,1])
    cam_pos = torch.bmm(K_inv, bbox_center_px_homo).view(batch_size,-1)
    return cam_pos
