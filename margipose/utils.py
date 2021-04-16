from mpl_toolkits.mplot3d import Axes3D
# import plotly.graph_objs as go

from PIL import ImageDraw
import torch
import numpy as np
import random
from time import perf_counter
from contextlib import contextmanager

import matplotlib.pyplot as plt

"""NEW"""
H36mMean = torch.FloatTensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 2.7933e+00, -7.5145e+00, -1.2986e+00],
         [-1.9678e+00,  3.5349e+02,  8.1133e+01],
         [-7.4258e+00,  7.2981e+02,  1.6748e+02],
         [-8.6249e+00,  7.8462e+02,  1.8294e+02],
         [-8.7289e+00,  7.8115e+02,  1.8342e+02],
         [-2.7933e+00,  7.5144e+00,  1.2986e+00],
         [-6.7001e+00,  3.6386e+02,  8.1523e+01],
         [-1.1538e+01,  7.4941e+02,  1.6670e+02],
         [-1.2512e+01,  7.9438e+02,  1.7917e+02],
         [-1.2651e+01,  7.8386e+02,  1.7793e+02],
         [ 1.0626e-03, -8.7544e-02, -1.9741e-02],
         [ 3.2130e+00, -2.1617e+02, -4.9807e+01],
         [ 6.0882e+00, -4.4400e+02, -1.0059e+02],
         [ 6.6411e+00, -5.0350e+02, -1.1218e+02],
         [ 7.8687e+00, -5.9624e+02, -1.3426e+02],
         [ 6.0882e+00, -4.4400e+02, -1.0059e+02],
         [ 2.6216e+00, -3.9252e+02, -8.9596e+01],
         [-2.9501e+00, -1.9117e+02, -4.4392e+01],
         [-3.7014e+00, -1.1458e+02, -2.2266e+01],
         [-3.7014e+00, -1.1458e+02, -2.2266e+01],
         [-2.4662e+00, -1.4720e+02, -2.9274e+01],
         [-4.8120e+00, -9.6083e+01, -1.6530e+01],
         [-4.8120e+00, -9.6083e+01, -1.6530e+01],
         [ 6.0882e+00, -4.4400e+02, -1.0059e+02],
         [ 8.2758e+00, -3.9039e+02, -8.8588e+01],
         [ 7.9857e+00, -1.9748e+02, -4.4392e+01],
         [ 5.4043e+00, -1.5638e+02, -3.2351e+01],
         [ 5.4043e+00, -1.5638e+02, -3.2351e+01],
         [ 5.7490e+00, -1.9011e+02, -3.9752e+01],
         [ 5.3365e+00, -1.4585e+02, -2.9284e+01],
         [ 5.3365e+00, -1.4585e+02, -2.9284e+01]])

H36mStd = torch.FloatTensor([[0.0000e+00, 0.0000e+00, 0.0000e+00],
         [1.0793e+02, 2.2247e+01, 7.6768e+01],
         [1.4502e+02, 1.5901e+02, 1.7766e+02],
         [1.8004e+02, 2.0789e+02, 2.1721e+02],
         [2.0026e+02, 2.2317e+02, 2.5215e+02],
         [2.2086e+02, 2.3238e+02, 2.8068e+02],
         [1.0793e+02, 2.2247e+01, 7.6767e+01],
         [1.4518e+02, 1.7028e+02, 1.8135e+02],
         [1.6732e+02, 2.1575e+02, 2.1751e+02],
         [1.9338e+02, 2.3060e+02, 2.5808e+02],
         [2.1522e+02, 2.3875e+02, 2.8799e+02],
         [1.7809e-02, 3.3567e-02, 2.3308e-02],
         [4.8436e+01, 4.3687e+01, 5.9935e+01],
         [8.7637e+01, 8.6227e+01, 1.1160e+02],
         [1.1114e+02, 1.1087e+02, 1.4665e+02],
         [1.1552e+02, 1.1319e+02, 1.4564e+02],
         [8.7637e+01, 8.6227e+01, 1.1160e+02],
         [1.3746e+02, 8.4760e+01, 1.2866e+02],
         [2.3449e+02, 1.1672e+02, 1.9532e+02],
         [2.4567e+02, 2.0682e+02, 2.3384e+02],
         [2.4567e+02, 2.0682e+02, 2.3384e+02],
         [2.3050e+02, 2.1808e+02, 2.2910e+02],
         [2.8250e+02, 2.5210e+02, 2.7550e+02],
         [2.8250e+02, 2.5210e+02, 2.7550e+02],
         [8.7637e+01, 8.6227e+01, 1.1160e+02],
         [1.3283e+02, 8.6854e+01, 1.2973e+02],
         [2.2889e+02, 1.3102e+02, 2.0094e+02],
         [2.4406e+02, 2.3335e+02, 2.4977e+02],
         [2.4406e+02, 2.3335e+02, 2.4977e+02],
         [2.3442e+02, 2.3222e+02, 2.4601e+02],
         [2.8775e+02, 3.0466e+02, 3.0345e+02],
         [2.8775e+02, 3.0466e+02, 3.0345e+02]])


def seed_all(seed):
    """Seed all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_algorithms(deterministic=False):
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _make_joint_metadata_fn(skel_desc):
    def joint_metadata_fn(joint_id):
        group = 'centre'
        if skel_desc.joint_names[joint_id].startswith('left_'):
            group = 'left'
        if skel_desc.joint_names[joint_id].startswith('right_'):
            group = 'right'
        return {
            'parent': skel_desc.joint_tree[joint_id],
            'group': group
        }
    return joint_metadata_fn


# def plotly_skeleton_figure(skel3d, skel_desc):
#     meta_fn = _make_joint_metadata_fn(skel_desc)

#     cxs = []
#     cys = []
#     czs = []

#     lxs = []
#     lys = []
#     lzs = []

#     rxs = []
#     rys = []
#     rzs = []

#     xt = list(skel3d[:, 0])
#     zt = list(-skel3d[:, 1])
#     yt = list(skel3d[:, 2])

#     for j, p in enumerate(skel_desc.joint_tree):
#         metadata = meta_fn(j)
#         if metadata['group'] == 'left':
#             xs, ys, zs = lxs, lys, lzs
#         elif metadata['group'] == 'right':
#             xs, ys, zs = rxs, rys, rzs
#         else:
#             xs, ys, zs = cxs, cys, czs

#         xs += [xt[j], xt[p], None]
#         ys += [yt[j], yt[p], None]
#         zs += [zt[j], zt[p], None]

#     points = go.Scatter3d(
#         x=list(skel3d[:, 0]),
#         z=list(-skel3d[:, 1]),
#         y=list(skel3d[:, 2]),
#         text=skel_desc.joint_names,
#         mode='markers',
#         marker=dict(color='grey', size=3, opacity=0.8),
#     )

#     centre_lines = go.Scatter3d(
#         x=cxs,
#         y=cys,
#         z=czs,
#         mode='lines',
#         line=dict(color='magenta', width=1),
#         hoverinfo='none',
#     )

#     left_lines = go.Scatter3d(
#         x=lxs,
#         y=lys,
#         z=lzs,
#         mode='lines',
#         line=dict(color='blue', width=1),
#         hoverinfo='none',
#     )

#     right_lines = go.Scatter3d(
#         x=rxs,
#         y=rys,
#         z=rzs,
#         mode='lines',
#         line=dict(color='red', width=1),
#         hoverinfo='none',
#     )

#     layout = go.Layout(
#         margin=go.Margin(l=20, r=20, b=20, t=20, pad=0),
#         hovermode='closest',
#         scene=go.Scene(
#             aspectmode='data',
#             yaxis=go.YAxis(title='z'),
#             zaxis=go.ZAxis(title='y'),
#         ),
#         showlegend=False,
#     )
#     fig = go.Figure(data=[points, centre_lines, left_lines, right_lines], layout=layout)

#     return fig

def plot_skeleton_on_axes3d(skel, skel_desc, invert=True, alpha=1.0):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    # NOTE: y and z axes are swapped
    xs = skel.narrow(-1, 0, 1).numpy()
    ys = skel.narrow(-1, 2, 1).numpy()
    zs = skel.narrow(-1, 1, 1).numpy()

    # Correct aspect ratio (https://stackoverflow.com/a/21765085)
    max_range = np.array([
        xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
    ]).max() / 2.0
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # ax.set_aspect('equal')

    if invert:
        ax.invert_zaxis()

    # Set starting view
    ax.view_init(elev=20, azim=-100)

    get_joint_metadata = _make_joint_metadata_fn(skel_desc)
    for joint_id, joint in enumerate(skel):
        meta = get_joint_metadata(joint_id)
        color = 'magenta'
        if meta['group'] == 'left':
            color = 'blue'
        if meta['group'] == 'right':
            color = 'red'
        parent = skel[meta['parent']]
        offset = parent - joint
        ax.quiver(
            [joint[0]], [joint[2]], [joint[1]],
            [offset[0]], [offset[2]], [offset[1]],
            color=color,
            alpha=alpha,
        )

    ax.scatter(xs, ys, zs, color='grey', alpha=alpha)

    return fig

"""NEW"""
def denorm_human_joints(normalized_joints, mean, std):
    return torch.mul(normalized_joints, std) + mean 

"""NEW"""
def plot_two_skeleton_on_axes3d(skel, skel2, skel_desc, legend_one, legend_two, invert=True, alpha=1.0, use_canonical=False):
    skel = skel[0,:,:]
    skel2 = skel2[0,:,:]
    
    # Denormalize the skeleton for printing!
    if not use_canonical:
        skel = denorm_human_joints(skel, H36mMean, H36mStd)
        skel2 = denorm_human_joints(skel2, H36mMean, H36mStd)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    # NOTE: y and z axes are swapped
    xs = skel.narrow(-1, 0, 1).numpy()
    ys = skel.narrow(-1, 2, 1).numpy()
    zs = skel.narrow(-1, 1, 1).numpy()

    # Correct aspect ratio (https://stackoverflow.com/a/21765085)
    max_range = np.array([
        xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
    ]).max() / 2.0
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # ax.set_aspect('equal')

    if invert:
        ax.invert_zaxis()

    # Set starting view
    ax.view_init(elev=20, azim=-100)

    get_joint_metadata = _make_joint_metadata_fn(skel_desc)
    for joint_id, joint in enumerate(skel):
        meta = get_joint_metadata(joint_id)
        color = 'magenta'
        if meta['group'] == 'left':
            color = 'blue'
        if meta['group'] == 'right':
            color = 'red'
        parent = skel[meta['parent']]
        offset = parent - joint
        ax.quiver(
            [joint[0].item()], [joint[2].item()], [joint[1].item()],
            [offset[0].item()], [offset[2].item()], [offset[1].item()],
            color=color,
            alpha=alpha,
        )

    ax.scatter(xs, ys, zs, color='grey', alpha=alpha, label=legend_one)

    xs = skel2.narrow(-1, 0, 1).numpy()
    ys = skel2.narrow(-1, 2, 1).numpy()
    zs = skel2.narrow(-1, 1, 1).numpy()

    # Correct aspect ratio (https://stackoverflow.com/a/21765085)
    max_range = np.array([
        xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
    ]).max() / 2.0
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # ax.set_aspect('equal')

    get_joint_metadata = _make_joint_metadata_fn(skel_desc)

    for joint_id, joint in enumerate(skel2):
        meta = get_joint_metadata(joint_id)
        color = 'black'
        if meta['group'] == 'left':
            color = 'black'
        if meta['group'] == 'right':
            color = 'black'
        parent = skel2[meta['parent']]
        offset = parent - joint
        ax.quiver(
            [joint[0].item()], [joint[2].item()], [joint[1].item()],
            [offset[0].item()], [offset[2].item()], [offset[1].item()],
            color=color,
            alpha=alpha,
        )
    
    ax.scatter(xs, ys, zs, color='grey', alpha=alpha, label=legend_two)

    return fig

"""NEW"""
def plot_two_can_skeleton_on_axes3d(skel, skel2, skel_desc, legend_one, legend_two, invert=True, alpha=1.0):
    skel = skel[0,:,:]
    skel2 = skel2[0,:,:]
    
    # Input must already be denormalized!

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    # NOTE: y and z axes are swapped
    xs = skel.narrow(-1, 0, 1).numpy()
    ys = skel.narrow(-1, 2, 1).numpy()
    zs = skel.narrow(-1, 1, 1).numpy()

    # Correct aspect ratio (https://stackoverflow.com/a/21765085)
    max_range = np.array([
        xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
    ]).max() / 2.0
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # ax.set_aspect('equal')

    if invert:
        ax.invert_zaxis()

    # Set starting view
    ax.view_init(elev=20, azim=-100)

    get_joint_metadata = _make_joint_metadata_fn(skel_desc)
    for joint_id, joint in enumerate(skel):
        meta = get_joint_metadata(joint_id)
        color = 'magenta'
        if meta['group'] == 'left':
            color = 'blue'
        if meta['group'] == 'right':
            color = 'red'
        parent = skel[meta['parent']]
        offset = parent - joint
        ax.quiver(
            [joint[0].item()], [joint[2].item()], [joint[1].item()],
            [offset[0].item()], [offset[2].item()], [offset[1].item()],
            color=color,
            alpha=alpha,
        )

    ax.scatter(xs, ys, zs, color='grey', alpha=alpha, label=legend_one)

    xs = skel2.narrow(-1, 0, 1).numpy()
    ys = skel2.narrow(-1, 2, 1).numpy()
    zs = skel2.narrow(-1, 1, 1).numpy()

    # Correct aspect ratio (https://stackoverflow.com/a/21765085)
    max_range = np.array([
        xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
    ]).max() / 2.0
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # ax.set_aspect('equal')

    get_joint_metadata = _make_joint_metadata_fn(skel_desc)

    for joint_id, joint in enumerate(skel2):
        meta = get_joint_metadata(joint_id)
        color = 'black'
        if meta['group'] == 'left':
            color = 'black'
        if meta['group'] == 'right':
            color = 'black'
        parent = skel2[meta['parent']]
        offset = parent - joint
        ax.quiver(
            [joint[0].item()], [joint[2].item()], [joint[1].item()],
            [offset[0].item()], [offset[2].item()], [offset[1].item()],
            color=color,
            alpha=alpha,
        )
    
    ax.scatter(xs, ys, zs, color='grey', alpha=alpha, label=legend_two)

    return fig

def plot_skeleton_on_axes(skel, skel_desc, ax, alpha=1.0):
    get_joint_metadata = _make_joint_metadata_fn(skel_desc)
    for joint_id, joint in enumerate(skel):
        meta = get_joint_metadata(joint_id)
        color = 'magenta'
        if meta['group'] == 'left':
            color = 'blue'
        if meta['group'] == 'right':
            color = 'red'
        parent = skel[meta['parent']]
        offset = parent - joint
        if offset.norm(2) >= 1:
            ax.arrow(
                joint[0], joint[1],
                offset[0], offset[1],
                color=color,
                alpha=alpha,
                head_width=2,
                length_includes_head=True,
            )

    xs = skel.narrow(-1, 0, 1).numpy()
    ys = skel.narrow(-1, 1, 1).numpy()
    ax.scatter(xs, ys, color='grey', alpha=alpha)

def plot_skeleton_on_axes_changed(skel, skel_desc, ax, use_color, alpha=1.0):
    get_joint_metadata = _make_joint_metadata_fn(skel_desc)
    for joint_id, joint in enumerate(skel):
        meta = get_joint_metadata(joint_id)
        color = use_color
        if meta['group'] == 'left':
            color = use_color
        if meta['group'] == 'right':
            color = use_color
        parent = skel[meta['parent']]
        offset = parent - joint
        if offset.norm(2) >= 1:
            ax.arrow(
                joint[0], joint[1],
                offset[0], offset[1],
                color=color,
                alpha=alpha,
                head_width=2,
                length_includes_head=True,
            )

    xs = skel.narrow(-1, 0, 1).numpy()
    ys = skel.narrow(-1, 1, 1).numpy()
    ax.scatter(xs, ys, color='grey', alpha=alpha)


def draw_skeleton_2d(img, skel2d, skel_desc, make_copy=True, mask=None, width=1):
    """COMMENTED"""
    # assert skel2d.size(-1) == 2, 'coordinates must be 2D'
    """NEW"""
    
    if make_copy:
        copy_img = img.copy()
        draw = ImageDraw.Draw(copy_img)
        get_joint_metadata = _make_joint_metadata_fn(skel_desc)
        for joint_id in range(skel_desc.n_joints):
            meta = get_joint_metadata(joint_id)
            color = (255, 0, 255)
            if meta['group'] == 'left':
                color = (0, 0, 255)
            if meta['group'] == 'right':
                color = (255, 0, 0)
            if mask is not None:
                if mask[joint_id] == 0 or mask[meta['parent']] == 0:
                    color = (128, 128, 128)
            draw.line(
                [*skel2d[joint_id], *skel2d[meta['parent']]],
                color, width=width
            )
        """ ADDED """
        return copy_img
    else:
        draw = ImageDraw.Draw(img)
        get_joint_metadata = _make_joint_metadata_fn(skel_desc)
        for joint_id in range(skel_desc.n_joints):
            meta = get_joint_metadata(joint_id)
            color = (255, 255, 255)
            if meta['group'] == 'left':
                color = (255, 255, 255)
            if meta['group'] == 'right':
                color = (255, 255, 255)
            if mask is not None:
                if mask[joint_id] == 0 or mask[meta['parent']] == 0:
                    color = (255, 255, 255)
            draw.line(
                [*skel2d[joint_id], *skel2d[meta['parent']]],
                color, width=width
            )
        """ ADDED """
        return img


@contextmanager
def timer(meter, n=1):
    start_time = perf_counter()
    yield
    time_elapsed = perf_counter() - start_time
    meter.add(time_elapsed / n)


def generator_timer(iterable, meter):
    iterator = iter(iterable)
    while True:
        try:
            with timer(meter):
                vals = next(iterator)
            yield vals
        except StopIteration:
            return


