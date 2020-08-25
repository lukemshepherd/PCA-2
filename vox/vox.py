import math
import numpy as np
import pandas as pd
import scipy.io
from scipy.ndimage import zoom
from pathlib import Path

# import numba
from stl import mesh
from mayavi import mlab
import quaternion as quat
from sklearn.decomposition import PCA

# Bone class:
class bone:

    filter_level = 0.001
    default_color = (0.7, 1, 1)

    def __init__(self, data, dtype):
        """
        Performs calculations on the voxel array objects    
        array (np.array): binary voxel object)      
        filter_level (int/float): sets the threshold level for 
        what is considered a voxel. Everything below filter level is
        rounded to 0, everything above rounded to 1 (ie voxel)
        """
        
        self.dtype = dtype
        self.data = data
        
        self.get_xyz()

                
        
    def get_xyz(self):
        """Convert 3D voxel array or STL to xyz coordinates.

        filter_level (int/float): (inherited from `bone` class) sets the threshold level for 
        what is considered a voxel. Everything below filter level is
        rounded to 0, everything above rounded to 1 (ie voxel)

        returns: 
            np.array( [n x 3] )"""

        
        if self.dtype == 'voxel':
        
            # Everything above filter level is converted to 1
            filtered_array = np.where(self.data < self.filter_level, 0, 1)

            # records coordiates where there is a 1
            x, y, z = np.where(filtered_array == 1)

            self.xyz = np.array([x, y, z]).T
            
            
        elif self.dtype == 'stl':
            self.xyz = np.concatenate((self.data.v0,
                                         self.data.v1,
                                         self.data.v2), axis=0)

                
    def get_pca(self):
        """PCA on the xyz points array

            xyz(np.array): n x 3 array of xyz coordinates

            returns:    self.pc1
                        self.pc2
                        self.pc3"""

        pca = PCA(svd_solver='full')
        pca.fit(self.xyz)
         
        self.pca_list = pca.components_
        self.pc1 = pca.components_[0]
        self.pc2 = pca.components_[1]
        self.pc3 = pca.components_[2]

   
    
    @property
    def mean(self):
        """The mean of the xyz atriube 
            returns:
            tupple (mean_of_x, mean_of_y ,mean_of_z)"""
    
        return (np.mean(self.xyz[:, 0]), np.mean(self.xyz[:, 1]), np.mean(self.xyz[:, 2]))


    def center_to_origin(self):
        """ sets the mean of the bone to 0,0,0"""

        # set transformation (tfm) value
        self.tfm = self.mean

        self.xyz = self.xyz - self.mean

    def reset_position(self):
        """ resets the position of the bone to its orginal one"""
        self.xyz = self.xyz + self.tfm


    def plot(self, user_color=None, mesh=False, PCA_inv=False, PCA=True):
        """ Plot voxels with optional PCA, and colours
        
            user_color (tupple): RGB color of the bone where 1 is maxium
                                    eg: red = (1,0,0)
                                    
            PCA (boolean): plots the PCAs of the voxel
            
            PCA_inv (boolean): plots the inverse of each PCA so the axes go in both directions
        """
                
        if hasattr(self, 'pc1') is False:
            self.get_pca()
        
    
        if user_color is None:
            user_color = self.default_color
            
            
            
        if mesh is False:
            #plots points
            mlab.points3d(self.xyz[:, 0],
                          self.xyz[:, 1],
                          self.xyz[:, 2],
                          mode = "cube",
                          color= user_color,
                          scale_factor = 1)  
            
        else:
             mlab.mesh(self.data.x, self.data.y, self.data.z)
        

        def quiver_pca(n,i):
            mlab.quiver3d(*self.mean, *(getattr(self,f'pc{n}')*i),
                                  line_width=6,
                                  scale_factor=100/n,
                                  color=c)

        for n in range(1,4):
            #sets color
            c = [0,0,0]
            c[n-1] = 1
            c = tuple(c)
            
            # plots pca arrows
            if PCA is True:
                quiver_pca(n,1)
                
            #plots the pca *-1      
            if PCA_inv is True:
                quiver_pca(n,-1)
            
        
    def scale(self, n):
        """ up-scales the bone """
        self.data = zoom(self.data, (n, n, n))
        
        #update xyz
        self.get_xyz()
            
            
    def xyz_to_array(self,array_dim=(256,256,256)):
        vx_array = np.zeros(array_dim,dtype=bool)
    
        for i in self.xyz:
            if np.allclose(i, np.around(i), rtol= 0.5, equal_nan=True):
                vx_array[tuple(np.around(i).astype(int))] = True
                
        x = np.count_nonzero(vx_array)/self.xyz.shape[0]
        
        print(f'{x*100}% reconstructed')
        
        return vx_array


    @classmethod
    def from_matlab_path(cls, matlab_file):
        """Imports matlab file drectly

           path: path object/string 

           retruns np.array (n x n x n )"""

        matlab_file = Path(matlab_file)

        matlab_object = scipy.io.loadmat(matlab_file)
        obj = matlab_object.keys()
        obj = list(obj)
        data = matlab_object[obj[-1]]

        return cls(data, dtype="voxel")

    @classmethod
    def from_stl_path(cls, stl_file):
        """Imports stl file drectly

       path: path object/string 

       retruns np.array (n x n x n )"""

        stl_file = Path(stl_file)

        data = mesh.Mesh.from_file(stl_file)

        return cls(data, dtype="stl")
        
# Functions:
def mag(v):
    """ Finds magnitude of vector

        v (np.array): vector"""
    return math.sqrt(np.dot(v, v))


def angle(v1, v2):
    """ Finds angel between 2 vectors

    returns: ang , v1"""

    try:

        ang = math.atan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

        if ang > math.pi / 2:
            v1 = -v1
            ang = math.atan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

            print(f"{ang} PC inverted")

        else:
            print(f"{ang} no invert")

    except:
        # vang = 0
        print(f"ERROR: vectors v1= {v1}, v2= {v2}")
        ang = "ERROR"

    return ang, v1


def quaternion_rotation_from_angle(v, c_axis, theta):

    rotation_axis = np.array([0.] + c_axis)
    axis_angle = (theta*0.5) * rotation_axis/np.linalg.norm(rotation_axis)

    vec = quat.quaternion(*v)

    # quaternion from exp of axis angle
    qlog = quat.quaternion(*axis_angle)
    q = np.exp(qlog)

    # double cover quaternion rotation
    v_prime = q * vec * np.conjugate(q)

    return v_prime.imag , q


def quaternion_rotation_from_quaternion(v, q):
    # double cover quaternion rotation
    vec = quat.quaternion(*v)
    
    v_prime = q * vec * np.conjugate(q)
    
    return v_prime.imag


def rotate(bone_f1, bone_f2, interpolate = False, scale_factor= 2):

    if interpolate is True: 
        print(f'scalling bone by {scale_factor}')
        bone_f1.scale(scale_factor)
    
    # center bones too 0,0,0,
    bone_f1.center_to_origin()
    bone_f2.center_to_origin()

    # PCA on bones
    bone_f1.get_pca()
    bone_f2.get_pca()

    # for 1 to 3 principle conponents of the object
    for n in range(1, 4):

        # takes cross product axis
        cross_product_axis = np.cross(
            getattr(bone_f1, f'pc{n}'),
            getattr(bone_f2, f'pc{n}'))

        # finds angle between PCs for f1 vs f2
        theta, vector = angle(
            getattr(bone_f1, f'pc{n}'),
            getattr(bone_f2, f'pc{n}'))

        # sets any new values needed
        setattr(bone_f1, f'pc{n}', vector)

        # rotates each PC
        for n in range(1, 4):
            transformed_pc, q = quaternion_rotation_from_angle(
                v=getattr(bone_f1, f'pc{n}'),
                c_axis=cross_product_axis,
                theta=theta)
            
            if hasattr(bone_f1,'q_prod') == True:
                bone_f1.q_prod = bone_f1.q_prod * q
                
            else:
                setattr(bone_f1,'q_prod', q)
                
            # sets new PCA
            setattr(bone_f1, f'pc{n}', transformed_pc)


    # rotates xyz array with the quaterion product
    rotated_xyz = np.apply_along_axis(
        quaternion_rotation_from_quaternion,1, 
        getattr(bone_f1, 'xyz'),
        bone_f1.q_prod)
        
    setattr(bone_f1, 'xyz', rotated_xyz)

    bone_f1.reset_position()
    bone_f2.reset_position()

    # reduce bone to orginal size 
    if interpolate is True: 
        print(f'scalling bone by {1/scale_factor}')
        bone_f1.scale(1/scale_factor)   
        

    if bone_f1.dtype is 'stl':

        #update internal data
        bone_f1.data.v0 , bone_f1.data.v1, bone_f1.data.v2 = np.array_split(bone_f1.xyz, 3)
        bone_f1.data.update_normals()

def df_angles(bone_f1, bone_f2, name="UN-NAMED BONE"):
    """
    Compares the PCA angles between to bones.

    Input:  bone_f1 = bone in 1st position
            bone_f2 = bone in 2nd position

    Returns: pandas dataframe
    """

    df = pd.DataFrame()

    # Check for PCAs
    if hasattr(bone_f1, "pc1") is False:
        bone_f1.get_pca()

    if hasattr(bone_f2, "pc1") is False:
        bone_f2.get_pca()

    # loops over each PCA
    for n in range(1, 4):
        theta, _ = angle(getattr(bone_f1, f"pc{n}"), getattr(bone_f2, f"pc{n}"))

        # Sets the column names
        df.loc[f"{name} f1: pc{n}", f"{name} f2: pc{n}"] = theta

    return df