class SkeletonDesc:
    def __init__(self, joint_names, joint_tree, hflip_indices):
        """Description of a particular skeleton representation.

        Args:
            joint_names (list of str): Names of the joints.
            joint_tree (list of int): References to the parent of each joint.
            hflip_indices (list of int): References to the horizontal mirror of each joint.
        """
        self.joint_names = joint_names
        self.joint_tree = joint_tree
        self.hflip_indices = hflip_indices

    @property
    def n_joints(self):
        """The number of joints in the skeleton."""
        return len(self.joint_names)

    @property
    def root_joint_id(self):
        """The ID (index) of the root joint."""
        return self.joint_names.index('pelvis')

    def get_joint_metadata(self, joint_id):
        name = self.joint_names[joint_id]
        if name.startswith('left_'):
            group = 'left'
        elif name.startswith('right_'):
            group = 'right'
        else:
            group = 'centre'
        return dict(parent=self.joint_tree[joint_id], group=group)


MPI3D_SKELETON_DESC = SkeletonDesc(
    joint_names=[
        # 0-3
        'spine3', 'spine4', 'spine2', 'spine',
        # 4-7
        'pelvis', 'neck', 'head', 'head_top',
        # 8-11
        'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',
        # 12-15
        'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow',
        # 16-19
        'right_wrist', 'right_hand', 'left_hip', 'left_knee',
        # 20-23
        'left_ankle', 'left_foot', 'left_toe', 'right_hip',
        # 24-27
        'right_knee', 'right_ankle', 'right_foot', 'right_toe'
    ],
    joint_tree=[
        2, 0, 3, 4,
        4, 1, 5, 6,
        5, 8, 9, 10,
        11, 5, 13, 14,
        15, 16, 4, 18,
        19, 20, 21, 4,
        23, 24, 25, 26
    ],
    hflip_indices=[
        0, 1, 2, 3,
        4, 5, 6, 7,
        13, 14, 15, 16,
        17, 8, 9, 10,
        11, 12, 23, 24,
        25, 26, 27, 18,
        19, 20, 21, 22
    ]
)


CANONICAL_SKELETON_DESC = SkeletonDesc(
    joint_names=[
        # 0-4
        'head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
        # 5-9
        'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
        # 10-14
        'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'pelvis',
        # 15-16
        'spine', 'head',
    ],
    joint_tree=[
        1, 15, 1, 2, 3,
        1, 5, 6, 14, 8,
        9, 14, 11, 12, 14,
        14, 1
    ],
    hflip_indices=[
        0, 1, 5, 6, 7,
        2, 3, 4, 11, 12,
        13, 8, 9, 10, 14,
        15, 16
    ]
)
