#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.io
import os
import pandas as pd
import math
from pathlib import Path
from mayavi import mlab
from pyquaternion import Quaternion


class matlab_loader:
    """ Loads .mat files from a direcotory 
            
            root_dir (str): path to the directory that containes files
            mat_file (str): name of file 
     
         returns:
             self.array (np.array)
             self.mat_file (Path object)
    """
    def __init__(self, root_dir, mat_file):  
        
        root_dir = Path(root_dir)
        mat_file = Path(mat_file)
        
        mat_obj = scipy.io.loadmat(root_dir/mat_file)
        obj= mat_obj.keys()
        obj = list(obj)
        array = mat_obj[obj[-1]]
        
        self.array = array
        self.mat_file = root_dir/mat_file
        
    def __repr__(self):
        return f'{self.mat_file}'


def xyz(arr, filter_level):
    """Convert 3D voxel arry to xyz coordiates.
    
            arr (np.array): 3D voxel array  
            
            filter_level (int/float): (inherited from `bone` class)sets the threshold level for 
            what is considered a voxel. Everything below filter level is
            rounded to 0, everything above rouned to 1 (ie voxel)
            
            returns: 
                np.array (n x 3)
    """
    
    # Everything above filter level is converted to 1
    arr = np.where(arr < filter_level, 0, 1)
    
    x, y, z = np.where(arr == 1)
    
    # converts the xyz so z is is *up* 
    x -= arr.shape[1]
    y -= arr.shape[0]
    x *= -1
    y *= -1
    xyz = np.array([x, y, z]).T
    return xyz



def pca(xyz):
    """PCA on a xyz points array
    
            xyz(np.array): n x 3 array of xyz coordinates
            
            returns: mean_x, mean_y, mean_z, eig_val, eig_vec
    """
    
    #covaraince of xyz points
    cov = np.cov(xyz.T)
    
    #eiganvalues and vectors
    (eig_val, eig_vec) = np.linalg.eig(cov)

    mean_x, mean_y, mean_z = [np.mean(xyz[:,0]),np.mean(xyz[:,1]),np.mean(xyz[:,2])]

    #NB eiganvaules not used the calculations
    return mean_x, mean_y, mean_z, eig_val, eig_vec


class foot_bone(bone):
    def __init__(self, name = 'UN-NAMED', **kwargs):
        
        """
        The higest level class.
        
            name: sets the bone name ie('tibia')
        """
        # Allows for n subclasses 
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.name = name
        
    def __repr__(self):
        return self.name



class frame_rec(foot_bone):
    def __init__(self, frame_rec_name, f1 = None ,f2 = None):
        
        """
        Sets the frame and diffent reconstructions

            f1: sets f1 images
            f2: sets f1 images
            frame_rec_name: name of image reconstructions eg('3 angles')
        """
        self.f1 = bone(f1)
        self.f2 = bone(f2)
        
        # tell what the name would be 
        self.n_name = frame_rec_name
        
    def __repr__(self):
        return f'{self.n_name} ' 


class bone:
        def __init__(self, array, filter_level=0.001):
            
            """
            Performs clacuations on the voxel array objects
            
                array (np.array): binaryvoxel object)  
            
                filter_level (int/float): sets the threshold level for 
            what is considered a voxel. Everything below filter level is
            rounded to 0, everything above rouned to 1 (ie voxel)
            """
          
            if array is None:
                pass
            
            else:
                self.array = array
                self.filter_level = filter_level
                self.xyz = xyz(array, filter_level)

                mean_x, mean_y, mean_z, eig_val, eig_vec = pca(self.xyz) 

                self.PCs = eig_vec
                self.PC1 = eig_vec[:,0]
                self.PC2 = eig_vec[:,1]
                self.PC3 = eig_vec[:,2]   

                self.mean = (mean_x, mean_y, mean_z)


def voxel_points(bone, color):
    """
    plots voxel array 
    """
    mlab.points3d(bone[:,0],
                  bone[:,1],
                  bone[:,2], 
                  mode="cube", color=color)


def voxel_PCAs(bone,pc,x=0,y=0,z=0):
    """
    plots each PC of bone object
    """
    pc_color = [(1,0,0),(0,1,0),(0,0,1)]
    
    for n in range(1,4):
      # adds scalling to the PCAs  
#     if n == 1:
#             x = .7
#         if n == 2:
#             x = .5
#         if n==3:
#             x = .3
            
        mlab.quiver3d(x,y,z, 
                      getattr(bone,f'{pc}{n}')[0],
                      getattr(bone,f'{pc}{n}')[1], 
                      getattr(bone,f'{pc}{n}')[2], 
                      line_width =6, scale_factor= 100, color= pc_color[n-1])
        

def bone_plot(*args, user_colours = None, plot_PCA = True, plot_inv = False):
    """
    plots voxel array that has an xyz atribute;
    can take n bones and plot PCA vectors
    PC1 Red 
    PC2 Blue
    PC3 Green

    
    plot_PCA: plots the PCAs as vectors on the bone
    plot_inv: plots the inverse of each PCA vector (PCAs go in both directiosn)
    
    """

    # Sorting out colours
    colour_dict = {'yellow':(0.9,0.9,0),
                   'pastel_blue':(0.7,1,1),
                   'purple':(0.6,0,0.5),
                   'orange':(0.8,0.3,0),
                   'dark_blue':(0,0.3,0.7),}
    
    if user_colours is None:
        user_colours = colour_dict
    
    plot_colours = []
    
    for col in user_colours:
        x = colour_dict.get(col)
        plot_colours.append(x)
        
    
    for n, bone in enumerate(args):

        mlab.points3d(bone.xyz[:,0], 
                      bone.xyz[:,1], 
                      bone.xyz[:,2],
                     mode="cube",
                     color= plot_colours[n],
                     scale_factor=1)
        
        x,y,z = bone.mean

        #plot princible vectors
        u0,v0,w0 = bone.PC1 * 100
        u0_inv,v0_inv,w0_inv = bone.PC1 * 100 * -1

        u1,v1,w1 = bone.PC2 * 100
        u1_inv,v1_inv,w1_inv = bone.PC2 * 100 * -1

        u2,v2,w2 = bone.PC3 * 100
        u2_inv,v2_inv,w2_inv = bone.PC3 * 100 * -1

        #print(f"{n}th bone PCA vectors: \n {bone.vec} \n ")
        
        
        if plot_PCA is True:
            mlab.quiver3d(x,y,z,u0,v0,w0,
                                 line_width =6,
                                 scale_factor=0.7,
                                 color= (1,0,0))
            mlab.quiver3d(x,y,z,u1,v1,w1,
                                 line_width =6,
                                 scale_factor= 0.5,
                                 color= (0,1,0))
            mlab.quiver3d(x,y,z,u2,v2,w2, 
                                 line_width =6,
                                 scale_factor=0.3,
                                 color=(0,0,1))


        #ploting the inverse of eigen vectors
        if plot_inv is True:
            mlab.quiver3d(x,y,z,u0_inv,v0_inv,w0_inv,
                                 line_width =6,
                                 scale_factor=0.7,
                                 color= (1,0,0))
            mlab.quiver3d(x,y,z,u1_inv,v1_inv,w1_inv,
                                 line_width =6,
                                 scale_factor=0.5,
                                 color= (0,1,0))
            mlab.quiver3d(x,y,z,u2_inv,v2_inv,w2_inv,
                                 line_width =6,
                                 scale_factor=0.3,
                                 color=(0,0,1))

    return mlab.show()


def bone_plot2(*args, user_colours = None, plot_PCA = True, plot_inv = False):
    """plots voxel array that has an xyz atribute;
    can take n bones and plot PCA vectors"""

    # Sorting out colours
    colour_dict = {'yellow':(0.9,0.9,0),
                   'pastel_blue':(0.7,1,1),
                   'purple':(0.6,0,0.5),
                   'orange':(0.8,0.3,0),
                   'dark_blue':(0,0.3,0.7)}
    
    if user_colours is None:
        user_colours = colour_dict
    
    plot_colours = []
    
    for col in user_colours:
        x = colour_dict.get(col)
        plot_colours.append(x)
    
    for n, bone in enumerate(args):

        voxel_points(bone.xyz, plot_colours[n])
        
        x,y,z = bone.mean
        
        if plot_PCA is True:
            voxel_PCAs(bone,'PC',x,y,z)

        #ploting the inverse of PCs
        if plot_inv is True:
            voxel_PCAs(-bone,'PC',x,y,z)

            
    return mlab.show()



def mag(v):
    """Finds magnitude of vector
        v (np.array): vector
    """
    return math.sqrt(np.dot(v,v))


def angle(v1, v2):
    """Finds angel between 2 vectors"""
    try:
        x = math.acos(np.dot(v1, v2) / (mag(v1) * mag(v2)))
    
    except:
        x = 0
        print ('angles are the same')
        
    return x



def df_angles(bone_phantom,bone_target, name ='UN-NAMED BONE'):
    """,
    Compares the PCA angles between to bones.
    
    Input:  bone_phantom = bone.phantom 
            bone_target = bone.frame_rec
               
    Returns: pandas dataframe
    """
    
    df = pd.DataFrame()
    # loops over each PCA
    for n in range(1,4):
        
        # Sets the collom names
         df.loc[f'{name} {bone_phantom} f1: PC{n}',
               f'{name} {bone_target} f2: PC{n}'] = angle(
             # gets the PC vaule from the object
            getattr(bone_phantom.f1,f'PC{n}'),
            getattr(bone_target.f2,f'PC{n}'))
            
    return df


def voxel_center(bone):
    #moves f1 onto f2
    tfm =  np.asarray(bone.f1.mean) - np.asarray(bone.f2.mean)
    
    #changing bone matrix coords f1
    bone.f1.tfm_xyz = bone.f1.xyz + tfm
    
    #sets mean to origin
    for n, i in enumerate(bone.f1.mean):
         bone.f1.tfm_xyz[:,n] -= bone.f1.mean[n] 
    
    #changing bone matrix coords f2
    bone.f2.tfm_xyz = bone.f2.xyz.astype(np.float64)
    
    #sets mean to origin
    for n, i in enumerate(bone.f2.mean):
         bone.f2.tfm_xyz[:,n] -= bone.f1.mean[n] 
            
    return bone


def roation(bone):
    
    # init tfm_PCn
    for n in range(1,4):
        setattr(bone.f1,f'tfm_PC{n}',getattr(bone.f1,f'PC{n}'))
        
    # for each pc rotate    
    for n in range(1,4):
    
        f1_PCn = getattr(bone.f1,f'tfm_PC{n}')
        f2_PCn = getattr(bone.f2,f'PC{n}')
            
        # angle between PCs
        ang =  angle(f1_PCn, f2_PCn)
        
        #cross product between PCs
        cx,cy,cz = np.cross(f1_PCn, f2_PCn)

        r = Quaternion(axis = [cx,cy,cz], angle = ang)
        
       # applys rotaition rowise down the xyz list
        bone.f1.tfm_xyz = np.apply_along_axis(r.rotate, 1, bone.f1.tfm_xyz)
        print(f'{n} {r}')
        
        #rotate PCs
        for n in range(1,4):
            setattr(bone.f1,
                    f'tfm_PC{n}',
                    r.rotate(getattr(bone.f1,f'tfm_PC{n}')))
                                            
    return bone


def rotation_plot(bone):
    
   #plot rotated f1
    voxel_points(bone.f1.tfm_xyz,color=(0,0.7,0))
    
    # plots orginal f2 
    voxel_points(bone.f2.tfm_xyz,color=(0.7,0,0))

    #f2 PCA 
    voxel_PCAs(bone.f2,pc='PC')
    
    #w/ rotion
    voxel_PCAs(bone.f1,pc='tfm_PC') 

    mlab.show()
