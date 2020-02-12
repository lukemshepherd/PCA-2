import math
import numpy as np
import pandas as pd
import scipy.io

from mayavi import mlab
import quaternion as quat
from sklearn.decomposition import PCA


#%% Classes 
class bone:

    filter_level=0.001

    def __init__(self, array):
               
        """ Performs calculations on the voxel array objects    
            array 
            
            (np.array): binary voxel object)   

            filter_level (int/float): sets the threshold level for 
            what is considered a voxel. 
            
            Everything below filter level is
            rounded to 0, everything above rounded to 1 (ie voxel)
        """ 
        self.array = array
        self.get_xyz()


    def get_xyz(self,array):
        """ Convert 3D voxel array to xyz coordinates.
    
            array (np.array): 3D voxel array  
                
            filter_level (int/float): (inherited from `bone` class) sets the threshold level for 
            what is considered a voxel. Everything below filter level is
            rounded to 0, everything above rounded to 1 (ie voxel)
                
            returns: 
                np.array( [n x 3]) 
        """
        # Everything above filter level is converted to 1
        array = np.where(array < self.filter_level, 0, 1)
        
        # records coordiates where there is a 1
        x, y, z = np.where(arr == 1)

        self.xyz = np.array([x, y, z]).T


    def get_pca(self):
        """ PCA on the xyz points array
    
            xyz(np.array): n x 3 array of xyz coordinates
            
            returns:    self.pc1
                        self.pc2
                        self.pc3             
        """
         
        pca = PCA(n_components=3, svd_solver='full')
        pca.fit_transform(self.xyz)

        self.pc1 = pca.components_[0]
        self.pc2 = pca.components_[1]
        self.pc3 = pca.components_[2]    



    def get_mean(self):
        """ The mean of the xyz atriube
    
            returns:
                tupple (mean_of_x, mean_of_y ,mean_of_z)
        """

        #mean_x, mean_y, mean_z
        return (np.mean(self.xyz[:,0]), np.mean(self.xyz[:,1]), np.mean(self.xyz[:,2]))
        

    def center_to_origin(self):
        """ sets the mean of the bone to 0,0,0
        """    

        # set transformation (tfm) value
        self.tfm = self.get_mean()

        self.xyz = self.xyz - self.tfm


    def reset_position(self):
        """ resets the position of the bone to its orginal one
        """
        
        self.xyz = self.xyz + self.tfm

    # Alternative constructor:
    # Import directly from matlab path

    @classmethod 
    def from_matlab_path(cls, path):
        """ Imports matlab file drectly
        
            path: path object/string 

            retruns np.array (n x n x n )
        """
        
        matlab_object = scipy.io.loadmat(path)
        obj = matlab_object.keys()
        obj = list(obj)
        array = matlab_object[obj[-1]]
        
        return array

def mag(v):
    """ Finds magnitude of vector

        v (np.array): vector
    """
    return math.sqrt(np.dot(v,v))

def angle(v1, v2):
    """ Finds angel between 2 vectors
    
    returns: ang , v1
    """
    try:
        ang = math.acos(np.dot(v1, v2) / (mag(v1) * mag(v2)))
   
        if ang >= math.pi/2:
            v1 = -v1
            ang = math.acos(np.dot(v1, v2) / (mag(v1) * mag(v2)))
            print(f'{ang} PC inverted')
        
        else:
            print(f'{ang} no invert')


    except:
        #ang = 0
        print (f'ERROR: angles are the same v1= {v1}, v2= {v2}')
        ang = 0
       
    return ang , v1

def quaternion_rotation(v, c_axis,theta):

    rotation_axis = np.array([0.] + c_axis)
    axis_angle = (theta*0.5) * rotation_axis/np.linalg.norm(rotation_axis)

    vec = quat.quaternion(*v)

    # quaternion from exp of axis angle
    qlog = quat.quaternion(*axis_angle)
    q = np.exp(qlog)

    # double cover quaternion rotation
    v_prime = q * vec * np.conjugate(q)

    return v_prime.imag


def voxel_rotate(bone_f1, bone_f2):

    #center bones too 0,0,0,
    bone_f1.center_to_origin()
    bone_f2.center_to_origin()

    # PCA on bones
    bone_f1.get_pca()
    bone_f2.get_pca()
    
    # for 1 to 3 principle conponents of the object
    for n in range(1,4):

        # takes cross product axis
        cross_product_axis = np.cross(
            getattr(bone_f1,f'PC{n}'),
            getattr(bone_f2,f'PC{n}'))         

        # finds angle between PCs for f1 vs f2
        theta, vector = angle(
            getattr(bone_f1,f'PC{n}'),
            getattr(bone_f2,f'PC{n}'))

        # sets any new values needed
        setattr(bone_f1,f'PC{n}',vector)

        #rotates each PC
        for n in range(1,4):
            transformed_PC = quaternion_rotation(
                v = getattr(bone_f1,f'PC{n}'),
                c_axis=cross_product_axis,
                theta=theta)

            #sets new PCA
            setattr(bone_f1, f'PC{n}',transformed_PC)

        #rotates xyz array
        rotated_xyz = np.apply_along_axis(
                    quaternion_rotation, 1, getattr(bone_f1,'xyz'),
                    c_axis=cross_product_axis, 
                    theta=theta)  

        setattr(bone_f1,'xyz',rotated_xyz)


def df_angles(bone_f1,bone_f2, name ='UN-NAMED BONE'):
    """
    Compares the PCA angles between to bones.
    
    Input:  bone_f1 = bone in 1st position
            bone_f2 = bone in 2nd position
               
    Returns: pandas dataframe
    """
    
    df = pd.DataFrame()

    # loops over each PCA
    for n in range(1,4):
        theta, _ = angle(getattr(bone_f1,f'PC{n}'),getattr(bone_f2.f2,f'PC{n}'))

        # Sets the column names
        df.loc[f'{name} {bone_f1} f1: PC{n}',f'{name} {bone_f2} f2: PC{n}'] = theta
            
    return df

def bone_plot(*args, user_colours = None, plot_PCA = True, plot_inv = False):
    """ Plots voxel array that has an xyz attribute;
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
        
        x,y,z = bone.get_mean()

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
