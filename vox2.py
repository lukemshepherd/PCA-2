import numpy as np
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.pyplot import cm
import mayavi.mlab

def mat2array (file):
    """loads a .mat file and returns a 3d array"""
    mat_obj = scipy.io.loadmat(file)
    arrary = mat_obj['v']
    return arrary

def voxel_xyz (arr):
    """voxel arry to xyz coordiates"""
    x,y,z = np.where(arr == 1)
    x -= arr.shape[1]
    y -= arr.shape[0]
    x *= -1
    y *= -1
    xyz = np.array([x,y,z]).T
    return xyz

def bone_pca_dev(bone):
    """PCA on bone array"""
    xyz = voxyz(bone)
    #covaraince of xyz points
    cov = np.cov(xyz)
    #eiganvalues and vectors
    (eig_val, eig_vec) = np.linalg.eig(cov)

    mean_x = xyz[:,0].mean()
    mean_y = xyz[:,1].mean()
    mean_z = xyz[:,2].mean()

    return mean_x, mean_y, mean_z, eig_val, eig_vec


def vox_plot_surf(xlim,ylim,zlim,*args):
    """plots voxel array as surface mesh can take n bones"""
    # sets colour map
    color=iter(cm.viridis(np.linspace(0,1,len(args))))

    #enshalise figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlim(0,xlim)
    ax.set_ylim(0,ylim)
    ax.set_zlim(0,zlim)

    for bone in (args):
        #plots surface mesh
        verts, faces, normals, values = measure.marching_cubes_lewiner(bone, 0)
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor(next(color))
        ax.add_collection3d(mesh)

    return plt.show(fig)


def vox_plot_dev(xlim,ylim,zlim,*args):
    """plots voxel array as surface mesh can take n bones and plot PCA"""
    # sets colour map
    color=iter(cm.viridis(np.linspace(0,1,len(args))))

    #enshalise figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #ax.set_aspect("equal")
    ax.set_xlim(0,xlim)
    ax.set_ylim(0,ylim)
    ax.set_zlim(0,zlim)

    for bone in (args):
        #plots surface mesh
        verts, faces, normals, values = measure.marching_cubes_lewiner(bone, 0)
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor(next(color))
        ax.add_collection3d(mesh)

        #PCA on bones
        mean_x, mean_y, mean_z, eig_val, eig_vec = bone_pca(args)
        x,y,z = [mean_x,mean_y,mean_z]

        #plot eignevectors
        u0,v0,w0 = eig_vec[:,0].T * abs(np.amax(x_points))
        u0_inv,v0_inv,w0_inv = u0 -1 ,v0 -1, w0-1

        u1,v1,w1 = eig_vec[:,1].T * abs(np.amax(x_points))
        u1_inv,v1_inv,w1_inv = u1 -1 ,v1 -1, w1-1

        u2,v2,w2 = eig_vec[:,2].T * abs(np.amax(x_points))
        u2_inv,v2_inv,w2_inv = u2 -1 ,v2 -1, w2-1

        ax.quiver(x,y,z,u0,v0,w0, color ='r')
        ax.quiver(x,y,z,u1,v1,w1, color ='b')
        ax.quiver(x,y,z,u2,v2,w2, color ='g')

    return plt.show(fig)

def vox_plot(*args):
    """plots voxel array; can take n bones and plot PCA vectors"""

    for bone in (args):
        #random colour tuple
        rand_colour = tuple(np.random.uniform(0.0,1.0,3))
        
        mayavi.mlab.points3d(voxel_xyz(bone)[:,0],
                             voxel_xyz(bone)[:,1],
                             voxel_xyz(bone)[:,2],
                     mode="cube",
                     color= rand_colour,
                     scale_factor=1)

        #PCA on bones
        mean_x, mean_y, mean_z, eig_val, eig_vec = bone_pca(bone)
        x,y,z = [mean_x,mean_y,mean_z]

        #plot eignevectors
        u0,v0,w0 = eig_vec[:,0].T * 100
        #u0_inv,v0_inv,w0_inv = u0 -1 ,v0 -1, w0-1

        u1,v1,w1 = eig_vec[:,1].T * 100
        #u1_inv,v1_inv,w1_inv = u1 -1 ,v1 -1, w1-1

        u2,v2,w2 = eig_vec[:,2].T * 100
        #u2_inv,v2_inv,w2_inv = u2 -1 ,v2 -1, w2-1

        mayavi.mlab.quiver3d(x,y,z,u0,v0,w0, 
                             line_width =6,
                             scale_factor=1,
                             color= rand_colour)
        
        mayavi.mlab.quiver3d(x,y,z,u1,v1,w1, 
                             line_width =6,
                             scale_factor=1,
                             color=rand_colour)
        
        mayavi.mlab.quiver3d(x,y,z,u2,v2,w2, 
                             line_width =6,
                             scale_factor=1, 
                             color=rand_colour)
        
    return mayavi.mlab.show()
