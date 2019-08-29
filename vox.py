# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.io
import mayavi.mlab


def stl_stack(bone):
    """stack .png images into voxel array"""

    file_list = sorted(glob.glob(str(bone)+'/*.png'))
    # reads images into a list, then stacks list
    result = []
    for file in file_list:
        img = plt.imread(file)
        result.append(img)
    result = np.stack(result, axis=2)

    return result


def mat2array(file):
    """loads a .mat file and returns a 3d array"""
    mat_obj = scipy.io.loadmat(file)
    arrary = mat_obj['v']
    return arrary


def voxel_xyz(arr):
    """voxel arry to xyz coordiates"""
    x, y, z = np.where(arr == 1)
    x -= arr.shape[1]
    y -= arr.shape[0]
    x *= -1
    y *= -1
    xyz = np.array([x, y, z]).T
    return xyz


def bone_pca(bone):
    """PCA on bone array"""
    xyz = voxel_xyz(bone)
    # covaraince of xyz points
    cov = np.cov(xyz.T)
    # eiganvalues and vectors
    (eig_val, eig_vec) = np.linalg.eig(cov)

    mean_x, mean_y, mean_z = (
        [np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])]
    )

    return mean_x, mean_y, mean_z, eig_val, eig_vec


def vox_plot(*args, user_colours=None, plot_PCA=True, plot_inv=True):
    """plots voxel array; can take n bones and plot PCA vectors"""

    # Sorting out colours
    colour_dict = {'purple': (0.6, 0, 0.5),
                   'yellow': (0.9, 0.9, 0),
                   'pastel_blue': (0.7, 1, 1),
                   'dark_blue': (0, 0.3, 0.7),
                   'orange': (0.8, 0.3, 0)}

    plot_colours = []

    for col in user_colours:
        x = colour_dict.get(col)
        plot_colours.append(x)

    for n, bone in enumerate(args):

        mayavi.mlab.points3d(voxel_xyz(bone)[:, 0],
                             voxel_xyz(bone)[:, 1],
                             voxel_xyz(bone)[:, 2],
                             mode="cube",
                             color=plot_colours[n],
                             scale_factor=1)

        # PCA on bones
        mean_x, mean_y, mean_z, eig_val, eig_vec = bone_pca(bone)
        x, y, z = [mean_x, mean_y, mean_z]

        # plot eignevectors
        u0, v0, w0 = eig_vec[:, 0].T * 100
        u0_inv, v0_inv, w0_inv = u0 * -1, v0 * -1, w0 * -1

        u1, v1, w1 = eig_vec[:, 1].T * 100
        u1_inv, v1_inv, w1_inv = u1 * -1, v1 * -1, w1 * -1

        u2, v2, w2 = eig_vec[:, 2].T * 100
        u2_inv, v2_inv, w2_inv = u2 * -1, v2 * -1, w2 * -1

        print(f"{n}th bone PCA vectors: \n {eig_vec} \n ")

        if plot_PCA is True:
            mayavi.mlab.quiver3d(x, y, z, u0, v0, w0,
                                 line_width=6,
                                 scale_factor=0.7,
                                 color=(1, 0, 0))
            mayavi.mlab.quiver3d(x, y, z, u1, v1, w1,
                                 line_width=6,
                                 scale_factor=0.5,
                                 color=(0, 1, 0))
            mayavi.mlab.quiver3d(x, y, z, u2, v2, w2,
                                 line_width=6,
                                 scale_factor=0.3,
                                 color=(0, 0, 1))

        # ploting the inverse of eigen vectors
        if plot_inv is True:
            mayavi.mlab.quiver3d(x, y, z, u0_inv, v0_inv, w0_inv,
                                 line_width=6,
                                 scale_factor=0.7,
                                 color=(1, 0, 0))
            mayavi.mlab.quiver3d(x, y, z, u1_inv, v1_inv, w1_inv,
                                 line_width=6,
                                 scale_factor=0.5,
                                 color=(0, 1, 0))
            mayavi.mlab.quiver3d(x, y, z, u2_inv, v2_inv, w2_inv,
                                 line_width=6,
                                 scale_factor=0.3,
                                 color=(0, 0, 1))

    return mayavi.mlab.show()
