import torch
from math import sqrt
from scipy import optimize

from pose3d_utils.camera import CameraIntrinsics


def make_projection_matrix(z_ref, intrinsics, height, width):
    """Build a matrix that projects from camera space into clip space.

    Args:
        z_ref (float): The reference depth (will become z=0).
        intrinsics (CameraIntrinsics): The camera object specifying focal length and optical centre.
        height (float): The image height.
        width (float): The image width.

    Returns:
        torch.Tensor: The projection matrix.
    """

    # Set the z-size (depth) of the viewing frustum to be equal to the
    # size of the portion of the XY plane at z_ref which projects
    # onto the image.
    size = z_ref * max(width / intrinsics.alpha_x, height / intrinsics.alpha_y)

    # Set near and far planes such that:
    # a) z_ref will correspond to z=0 after normalisation
    #    $z_ref = 2fn/(f+n)$
    # b) The distance from z=-1 to z=1 (normalised) will correspond
    #    to `size` in camera space
    #    $f - n = size$
    far = 0.5 * (sqrt(z_ref ** 2 + size ** 2) + z_ref - size)
    near = 0.5 * (sqrt(z_ref ** 2 + size ** 2) + z_ref + size)

    # Construct the perspective projection matrix.
    # More details: http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
    m_proj = intrinsics.matrix.new([
        [intrinsics.alpha_x / intrinsics.x_0, 0, 0, 0],
        [0, intrinsics.alpha_y / intrinsics.y_0, 0, 0],
        [0, 0, -(far + near) / (far - near), 2 * far * near / (far - near)],
        [0, 0, 1, 0],
    ])

    return m_proj


def camera_space_to_ndc(Xc, P):
    """Transform point(s) from camera space to normalised device coordinates.

    Args:
        Xc (torch.Tensor): homogeneous point(s) in camera space.
        P (torch.Tensor): projection matrix.

    Returns:
        Xn (torch.Tensor): homogeneous point(s) in normalised device coordinates.
    """
    # Camera space -> homogeneous clip space
    Xh = torch.matmul(Xc, P.t())
    # Homogeneous clip space -> normalised device coordinates
    w = Xh[..., 3:4]
    Xn = Xh / w
    return Xn


def ndc_to_camera_space(Xn, P):
    """Transform point(s) from normalised device coordinates to camera space.

    Args:
        Xn (torch.Tensor): homogeneous point(s) in normalised device coordinates.
        P (torch.Tensor): projection matrix.

    Returns:
        Xc (torch.Tensor): homogeneous point(s) in camera space.
    """
    # Normalised device coordinates -> homogeneous clip space
    z = Xn[..., 2:3]
    w = P[2, 3] / (z - P[2, 2])
    Xh = Xn * w
    # Homogeneous clip space -> camera space
    Xc = torch.matmul(Xh, P.inverse().t())
    return Xc


class SkeletonNormaliser:
    def normalise_skeleton(self, denorm_skel, z_ref, intrinsics, height, width):
        """Normalise the skeleton, removing scale and z position.

        Joints within the frame should have coordinate values between -1 and 1.

        Args:
            denormalised_skel (torch.DoubleTensor): The denormalised skeleton.
            z_ref (float): The depth of the plane which will become z=0.
            intrinsics (CameraIntrinsics): The camera which projects 3D points onto the 2D image.
            height (float): The image height.
            width (float): The image width.

        Returns:
            torch.DoubleTensor: The normalised skeleton.
        """
        m_proj = make_projection_matrix(z_ref, intrinsics, height, width).type_as(denorm_skel)
        return camera_space_to_ndc(denorm_skel, m_proj)

    def denormalise_skeleton(self, norm_skel, z_ref, intrinsics, height, width):
        """Denormalise the skeleton, adding scale and z position.

        Args:
            normalised_skel (torch.DoubleTensor): The normalised skeleton.
            z_ref (float): Depth of the root joint.
            intrinsics (CameraIntrinsics): The camera which projects 3D points onto the 2D image.
            height (float): The image height.
            width (float): The image width.

        Returns:
            torch.DoubleTensor: The denormalised skeleton.
        """
        m_proj = make_projection_matrix(z_ref, intrinsics, height, width).type_as(norm_skel)
        return ndc_to_camera_space(norm_skel, m_proj)

    def infer_depth(self, norm_skel, eval_scale, intrinsics, height, width, z_upper=20000):
        """Infer the depth of the root joint.

        Args:
            norm_skel (torch.DoubleTensor): The normalised skeleton.
            eval_scale (function): A function which evaluates the scale of a denormalised skeleton.
            intrinsics (CameraIntrinsics): The camera which projects 3D points onto the 2D image.
            height (float): The image height.
            width (float): The image width.
            z_upper (float): Upper bound for depth.

        Returns:
            float: `z_ref`, the depth of the root joint.
        """
        def f(z_ref):
            z_ref = float(z_ref)
            skel = self.denormalise_skeleton(norm_skel, z_ref, intrinsics, height, width)
            k = eval_scale(skel)
            return (k - 1.0) ** 2
        z_lower = max(intrinsics.alpha_x, intrinsics.alpha_y)
        z_ref = float(optimize.fminbound(f, z_lower, z_upper, maxfun=200, disp=0))
        return z_ref
