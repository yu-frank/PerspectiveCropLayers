import numpy as np


def identity():
    return np.eye(4, dtype=np.float64)


def affine(A=None, t=None):
    aff = identity()
    if A is not None:
        aff[0:3, 0:3] = np.array(A, dtype=aff.dtype)
    if t is not None:
        aff[0:3, 3] = np.array(t, dtype=aff.dtype)
    return aff


def flip_x():
    """Flip horizontally."""
    return affine(A=[[-1, 0, 0],
                     [ 0, 1, 0],
                     [ 0, 0, 1]])


def rotate(axis, theta):
    assert axis.shape == (3,), 'rotation axis must be a 3D vector'
    axis = axis / np.linalg.norm(axis, 2)
    ux, uy, uz = list(axis)
    cos = np.cos(theta)
    sin = np.sin(theta)
    return affine(
        A=[[cos + ux*ux*(1 - cos),  ux*uy*(1-cos) - uz*sin, ux*uz*(1-cos) + uy*sin],
           [uy*ux*(1-cos) + uz*sin, cos + uy*uy*(1-cos),    uy*uz*(1-cos) - ux*sin],
           [uz*ux*(1-cos) - uy*sin, uz*uy*(1-cos) + ux*sin, cos + uz*uz*(1-cos)]]
    )
