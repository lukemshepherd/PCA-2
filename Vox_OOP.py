# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Vox'))
	print(os.getcwd())
except:
	pass
# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Libraries" data-toc-modified-id="Libraries-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Libraries</a></span><ul class="toc-item"><li><span><a href="#OS" data-toc-modified-id="OS-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>OS</a></span></li><li><span><a href="#Python" data-toc-modified-id="Python-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Python</a></span></li><li><span><a href="#mayavi" data-toc-modified-id="mayavi-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>mayavi</a></span><ul class="toc-item"><li><span><a href="#mayavi-install" data-toc-modified-id="mayavi-install-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>mayavi install</a></span></li></ul></li><li><span><a href="#pyquaternion" data-toc-modified-id="pyquaternion-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>pyquaternion</a></span></li><li><span><a href="#Optional:-inline-3d-plotting" data-toc-modified-id="Optional:-inline-3d-plotting-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Optional: inline 3d plotting</a></span></li><li><span><a href="#matlab_loader" data-toc-modified-id="matlab_loader-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>matlab_loader</a></span></li><li><span><a href="#PCA" data-toc-modified-id="PCA-1.7"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>PCA</a></span></li></ul></li><li><span><a href="#Classes:" data-toc-modified-id="Classes:-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Classes:</a></span><ul class="toc-item"><li><span><a href="#Hierarchy." data-toc-modified-id="Hierarchy.-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Hierarchy.</a></span></li><li><span><a href="#What-are-f1-and-f2" data-toc-modified-id="What-are-f1-and-f2-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>What are f1 and f2</a></span><ul class="toc-item"><li><span><a href="#Foot-bone" data-toc-modified-id="Foot-bone-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Foot bone</a></span></li><li><span><a href="#Indivual-bone-classes" data-toc-modified-id="Indivual-bone-classes-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Indivual bone classes</a></span></li></ul></li><li><span><a href="#Bone" data-toc-modified-id="Bone-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Bone</a></span></li></ul></li><li><span><a href="#Plotting:" data-toc-modified-id="Plotting:-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Plotting:</a></span><ul class="toc-item"><li><span><a href="#bone_plot" data-toc-modified-id="bone_plot-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>bone_plot</a></span></li><li><span><a href="#Angels" data-toc-modified-id="Angels-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Angels</a></span></li><li><span><a href="#Table-of-Angles" data-toc-modified-id="Table-of-Angles-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Table of Angles</a></span></li></ul></li><li><span><a href="#Rotation" data-toc-modified-id="Rotation-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Rotation</a></span><ul class="toc-item"><li><span><a href="#Step-1:-Center-the-2-means" data-toc-modified-id="Step-1:-Center-the-2-means-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Step 1: Center the 2 means</a></span></li></ul></li><li><span><a href="#Quaternions" data-toc-modified-id="Quaternions-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Quaternions</a></span><ul class="toc-item"><li><span><a href="#Rotation-around-PCs" data-toc-modified-id="Rotation-around-PCs-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Rotation around PCs</a></span></li><li><span><a href="#What-is-the-difference-with-this-and-the-other-bone_plot?" data-toc-modified-id="What-is-the-difference-with-this-and-the-other-bone_plot?-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>What is the difference with this and the other <code>bone_plot</code>?</a></span></li></ul></li><li><span><a href="#How-to-use:" data-toc-modified-id="How-to-use:-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>How to use:</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#1.-Set-the-root-directory-for-the-matlab-file-loader" data-toc-modified-id="1.-Set-the-root-directory-for-the-matlab-file-loader-6.0.1"><span class="toc-item-num">6.0.1&nbsp;&nbsp;</span>1. Set the root directory for the matlab file loader</a></span></li><li><span><a href="#2.-Load-the-data-that-you-want-to-use" data-toc-modified-id="2.-Load-the-data-that-you-want-to-use-6.0.2"><span class="toc-item-num">6.0.2&nbsp;&nbsp;</span>2. Load the data that you want to use</a></span></li><li><span><a href="#3.-Construct-the-bone-classes" data-toc-modified-id="3.-Construct-the-bone-classes-6.0.3"><span class="toc-item-num">6.0.3&nbsp;&nbsp;</span>3. Construct the bone classes</a></span></li><li><span><a href="#4.-Change-xyz-coordinates-so-f1-is-in-f2-position" data-toc-modified-id="4.-Change-xyz-coordinates-so-f1-is-in-f2-position-6.0.4"><span class="toc-item-num">6.0.4&nbsp;&nbsp;</span>4. Change xyz coordinates so <code>f1</code> is in <code>f2</code> position</a></span></li><li><span><a href="#5.-Rotate-bone-using-PCs" data-toc-modified-id="5.-Rotate-bone-using-PCs-6.0.5"><span class="toc-item-num">6.0.5&nbsp;&nbsp;</span>5. Rotate bone using PCs</a></span></li><li><span><a href="#6.-Plotting-the-rotation" data-toc-modified-id="6.-Plotting-the-rotation-6.0.6"><span class="toc-item-num">6.0.6&nbsp;&nbsp;</span>6. Plotting the rotation</a></span></li></ul></li></ul></li></ul></div>

# %%
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from mayavi import mlab
#from pyquaternion import Quaternion


# %%
from IPython.core.debugger import set_trace


# %%
import quaternion as quat

# %% [markdown]
# # Libraries
# See environment.yml 
# 
# Activate desired environment
# 
#     conda activate myenv
#   
#     conda env update -f environment.yml 
# 
# ## OS
# Has been written (and runs) on both Windows 10 and MacOS
# 
# ## Python
# This was written on python 3.7 (although 3.6 *should* work- although not tested) python 2 versions won't work due to the use of f strings
# 
# ## mayavi
# This is the 3d plotting library used for rendering the plots. mayvai will launch a qt window to display the plot- meaning that you will need an X serve session for the plots to load. If you want to plot things inline you will need to use jupyter notebooks, not jupyter lab.
# 
# ### mayavi install
# https://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-conda-forge
# 
#     conda install vtk
#     conda install qyqt5
#     
#     conda install mayavi
#     
# ## pyquaternion
# http://kieranwynn.github.io/pyquaternion/
# 
#     pip install pyquaternion
#     
#     
# ## Optional: inline 3d plotting
# http://docs.enthought.com/mayavi/mayavi/tips.html#using-mayavi-in-jupyter-notebooks

# %%
# Only works with notebooks not lab
# mlab.init_notebook('x3d', 500, 500)


# %%
# test inline rendering
# s = mlab.test_plot3d()
# s

# %% [markdown]
# ## matlab_loader
# %% [markdown]
# This makes a data loader class for matlab files

# %%
class matlab_loader:
    """ Loads .mat files from a directory 
            
            root_dir (str): path to the directory that contains files
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

# %% [markdown]
# Why do you have have to remove noise

# %%
def xyz(arr, filter_level):
    """Convert 3D voxel array to xyz coordinates.
    
            arr (np.array): 3D voxel array  
            
            filter_level (int/float): (inherited from `bone` class)sets the threshold level for 
            what is considered a voxel. Everything below filter level is
            rounded to 0, everything above rounded to 1 (ie voxel)
            
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

# %% [markdown]
# ## PCA

# %%
def pca(xyz):
    """PCA on a xyz points array
    
            xyz(np.array): n x 3 array of xyz coordinates
            
            returns: mean_x, mean_y, mean_z, eig_val, eig_vec
    """
    
    #covariance of xyz points
    cov = np.cov(xyz.T)
    
    #eigenvalues and vectors
    (eig_val, eig_vec) = np.linalg.eig(cov)

    mean_x, mean_y, mean_z = [np.mean(xyz[:,0]),np.mean(xyz[:,1]),np.mean(xyz[:,2])]

    #NB eigenvalues not used the calculations
    return mean_x, mean_y, mean_z, eig_val, eig_vec

# %% [markdown]
# # Classes:
# %% [markdown]
# ## Hierarchy.
# 
# 1. `foot_bone` : highest level bone eg 'tibia' 
#    
# 2. `frame_rec` : sets the frame position of the bone, f1 or f2 and how many 
# 
# 3. `bone` : It calls the PCA on xyz of the bone and stores the output
# 
# ## What are f1 and f2 
# 
# These are the 2 different position of the bone. F1 is the starting position of the bone using the phantom image.
# 
# F2 is the the bone in it's second frame of motion. The image is created by using a multi angle reconstruction.
# %% [markdown]
# ## Bone

# %%
class bone:
        def __init__(self, array, filter_level=0.001):
            
            """
            Performs calculations on the voxel array objects
            
                array (np.array): binary voxel object)  
            
                filter_level (int/float): sets the threshold level for 
            what is considered a voxel. Everything below filter level is
            rounded to 0, everything above rounded to 1 (ie voxel)
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

# %% [markdown]
# ### Foot bone

# %%
class foot_bone(bone):
    def __init__(self, name = 'UN-NAMED', **kwargs):
        
        """
        The highest level class.
        
            name: sets the bone name ie('tibia')
        """
        # Allows for n subclasses 
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.name = name
        
    def __repr__(self):
        return self.name

# %% [markdown]
# ### Indivual bone classes

# %%
class frame_rec(foot_bone):
    def __init__(self, frame_rec_name, f1 = None ,f2 = None):
        
        """
        Sets the frame and different reconstructions

            f1: sets f1 images
            f2: sets f1 images
            frame_rec_name: name of image reconstructions eg('3 angles')
        """
        self.f1 = bone(f1)
        self.f2 = bone(f2)
        
        # set what the name will be 
        self.n_name = frame_rec_name
        
    def __repr__(self):
        return f'{self.n_name} ' 

# %% [markdown]
# # Plotting:
# %% [markdown]
# Creates plots that show both the PCs and the voxelized bones

# %%
#Creates some repeated blocks for functions

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
        

# %% [markdown]
# ## bone_plot
# %% [markdown]
# This plots an arbitrary number of voxelized bones.
# 
# eg: `bone_plot(bone.bone_name.f1)`
# 
# `plot_PCA`: plots the PCAs as vectors on the bone
# 
# `plot_inv`: plots the inverse of each PCA so the axes go in both directions
# 
# You can use your own colours by passing a colour dictionary
# 
# `my_colours = {'red':(1,0,0),'green':(0,1,0),'blue':(0,0,1)`
#          
# The first bone will be plotted with the first colour in the dictionary, the second with the second and so on.
# 

# %%
# NEED TO REFACTOR
def bone_plot(*args, user_colours = None, plot_PCA = True, plot_inv = False):
    """
    plots voxel array that has an xyz attribute;
    can take n bones and plot PCA vectors
    PC1 Red 
    PC2 Blue
    PC3 Green

    
    plot_PCA: plots the PCAs as vectors on the bone
    plot_inv: plots the inverse of each PCA vector (PCAs go in both directions)
    
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

        #plot PCAs
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


        #plotting the inverse of PCAs
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


# %%
# cleaner code
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

        #plotting the inverse of PCAs
        if plot_inv is True:
            voxel_PCAs(-bone,'PC',x,y,z)

            
    return mlab.show()


# %%
# Option user colours
# user_colours=['yellow','purple']

# %% [markdown]
# ## Angels

# %%
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


# %%
# # write a file loader 

# file list = walk through dir
#     for file in file list:
#          file[:-4] = matlab_loader(file)
#          setattr(bone, file[:-7])

# %% [markdown]
# ## Table of Angles

# %%
def df_angles(bone_phantom,bone_target, name ='UN-NAMED BONE'):
    """
    Compares the PCA angles between to bones.
    
    Input:  bone_phantom = bone.phantom 
            bone_target = bone.frame_rec
               
    Returns: pandas dataframe
    """
    
    df = pd.DataFrame()
    # loops over each PCA
    for n in range(1,4):
        
        # Sets the column names
         df.loc[f'{name} {bone_phantom} f1: PC{n}',
               f'{name} {bone_target} f2: PC{n}'] = angle(
        # gets the PC value from the object
            getattr(bone_phantom.f1,f'PC{n}'),
            getattr(bone_target.f2,f'PC{n}'))
            
    return df

# %% [markdown]
# # Rotation
# %% [markdown]
# ## Step 1: Center the 2 means
# %% [markdown]
# Moves the f1 bone to the f2 position; 
# 
# Creates new attribute `bone.f1.tfm_xyz` and `bone.f2.tfm_xyz`

# %%
def voxel_center(bone):
    #displacment between f1 onto f2
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

# %% [markdown]
# # Quaternions
# %% [markdown]
# ## Rotation around PCs
# %% [markdown]
# The bones are rotated with quaternions.
# 
# The angle between the two PC1 vectors is taken. The object is  then rotated (by a quaternion) around the cross product between the PC1 vectors.
# 
# The new angles between the next PCs are calculates and the process is repeated for the other PCs

# %%
def rotation(bone):
    
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
        
       # applies rotation row wise down the xyz list
        bone.f1.tfm_xyz = np.apply_along_axis(r.rotate, 1, bone.f1.tfm_xyz)
        print(f'{n} {r}')
        
        #rotate PCs
        for n in range(1,4):
            setattr(bone.f1, f'tfm_PC{n}', r.rotate(getattr(bone.f1,f'tfm_PC{n}')))
                                            
    return bone




#%%

def quaternion_rot(v, c_axis,theta):

    rot_axis = np.array([0.] + c_axis)
    axis_angle = (theta*0.5) * rot_axis/np.linalg.norm(rot_axis)

    vec = quat.quaternion(*v)
    # quaternion from exp of axis angle
    qlog = quat.quaternion(*axis_angle)
    q = np.exp(qlog)

    #double cover quaternion rotation
    v_prime = q * vec * np.conjugate(q)

    return v_prime.imag


# %%
def rotation2(bone):
    
    # init tfm_PCn for the class 
    for n in range(1,4):
        setattr(bone.f1,f'tfm_PC{n}',getattr(bone.f1,f'PC{n}'))
        
    # for each pairwise    
    for n in range(1,4):
    
        f1_PC = getattr(bone.f1,f'tfm_PC{n}')
        f2_PC = getattr(bone.f2,f'PC{n}')
            
        # angle between PCs
        theta =  angle(f1_PC, f2_PC)
        
        #cross product between PCs
        axis = np.cross(f1_PC, f2_PC) 

        bone.f1.tfm_xyz = np.apply_along_axis(
            func1d=quaternion_rot, 
            axis=1, 
            arr= bone.f1.tfm_xyz, 
            c_axis=axis, 
            theta=theta)        

            #rotate PCs
        for n in range(1,4):

            tfm_PC = getattr(bone.f1,f'tfm_PC{n}')
            
            tfm_PC = quaternion_rot(v=tfm_PC,c_axis=axis,theta=theta)

            setattr(bone.f1, f'tfm_PC{n}',tfm_PC)
                    
# %% [markdown]
# ## What is the difference with this and the other `bone_plot`?
# 
# Internally nothing- both uses the same functions.
# 
# `rotation_plot` provides a quick validation that the rotation has been done right and only takes one argument.
# 
# `bone_plot` allows you to plot all the bones/colours you want 
# 

# %%
def rotation_plot(bone):
    
   #plot rotated f1
    voxel_points(bone.f1.tfm_xyz,color=(0,0.7,0))
    
    # plots original f2 
    voxel_points(bone.f2.tfm_xyz,color=(0.7,0,0))

    #f2 PCA 
    voxel_PCAs(bone.f2,pc='PC')
    
    #w/ rotation
    voxel_PCAs(bone.f1,pc='tfm_PC') 

    mlab.show()

# %% [markdown]
# # How to use:
# %% [markdown]
# ### 1. Set the root directory for the matlab file loader

# %%
root_dir = Path('C://Users/luke/OneDrive - University College London/Marta/data')

# %% [markdown]
# ### 2. Load the data that you want to use

# %%
# load data phantoms
tibia_phant_f2 = matlab_loader(root_dir, mat_file = 'phantom/phantom_tibia_f2.mat' )
tibia_phant_f1 = matlab_loader(root_dir, mat_file = 'phantom/phantom_tibia_f1.mat')

# load multiple angles
tibia_3_f2 = matlab_loader(root_dir, mat_file = 'fista_recons/3 angles/tibia_f2.mat')
tibia_6_f2 = matlab_loader(root_dir, mat_file = 'fista_recons/6 angles/tibia_f2.mat')

# %% [markdown]
# ### 3. Construct the bone classes

# %%
# set the foot bone
tibia = foot_bone(name = 'tibia')

# load f1/f2 phantom data
tibia.phantom = frame_rec(f1=tibia_phant_f1.array,
                      f2=tibia_phant_f2.array,
                      frame_rec_name= 'phantom')

# load f2 3 angel data
tibia.ang3 = frame_rec(f2 = tibia_3_f2.array,
                   frame_rec_name= 'angle 3')

# %% [markdown]
# ### 4. Change xyz coordinates so `f1` is in `f2` position

# %%
voxel_center(tibia.phantom)

# %% [markdown]
# ### 5. Rotate bone using PCs

# %%
rotation2(tibia.phantom)


# %% [markdown]
# ### 6. Plotting the rotation

# %%
rotation_plot(tibia.phantom)

# %% [markdown]
# ## Table of Angels

# %%
#df_angles(tibia.phantom,tibia.phantom, name= tibia)


# %%

bone = tibia.phantom

getattr(bone.f1,f'PC1')# %%

