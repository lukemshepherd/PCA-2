import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import scipy.io

def mat2array (file):
    """loads a .mat file and returns a 3d array"""
    mat_obj = scipy.io.loadmat(file)
    arrary = mat_obj['v']
    return arrary


def bone_pca(bone, plot = True, stack = False):
    """PCA on 3D objects from either png voxel slices from stl-voxel
     or directly from a user loaded array (vaules 1 or 0)

     plot:      True: if you want plot to be automaticaly shown,
                False: fig is returned

     stack:     True: bone is file begigin for .png slices,
                False: bone is array of voxels

     bone:      (see stack)
     """

    if stack:
        #creates list of all file names
        file_list = sorted(glob.glob(str(bone)+'/*.png'))
        # reads images into a list, then stacks list
        result = []
        for file in file_list:
            img = plt.imread(file)
            result.append(img)
        result = np.stack(result, axis =2)

    else:
        result=bone

    #for number, enumiate in len(bone).enumiate :
        #def pca(result):

            #enishalise point cloud
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(np.argwhere(result==1.0))

    xyz = np.array(point_cloud.points)

            #mean/covaraance
    (mean, cov) = o3d.geometry.compute_point_cloud_mean_and_covariance(point_cloud)
    mean_x = mean[0]
    mean_y = mean[1]
    mean_z = mean[2]
            #eignvaules
    (eig_val_cov, eig_vec_cov) = np.linalg.eig(cov)

            #return out

    #downsalmple xyz for ploting
    point_downsample = o3d.uniform_down_sample(point_cloud,20)
    xyz_downsample = np.array(point_downsample.points)

    #mean scale data
    x_points = xyz_downsample[:,0]- np.mean(xyz[:,0])
    y_points = xyz_downsample[:,1]- np.mean(xyz[:,1])
    z_points = xyz_downsample[:,2]- np.mean(xyz[:,2])

    #vector orgigin points
    x,y,z = [0,0,0]

    #plot eignevectors
    u0,v0,w0 = eig_vec_cov[:,0].T * abs(np.amax(x_points))
    u1,v1,w1 = eig_vec_cov[:,1].T * abs(np.amax(y_points))
    u2,v2,w2 = eig_vec_cov[:,2].T * abs(np.amax(z_points))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(x,y,z,u0,v0,w0, color ='r')
    ax.quiver(x,y,z,u1,v1,w1, color ='b')
    ax.quiver(x,y,z,u2,v2,w2, color ='g')

    #plot mean scaled, downsampled point cloud
    ax.scatter(x_points,y_points,z_points, \
    color ='gray', \
    alpha= 0.5, \
    edgecolors= 'black')

    if plot:
        return plt.show(fig)

    else:
        return fig
