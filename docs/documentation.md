# Documentation

## Bone class methods

### `.get_xyz(self)`
Convert 3D voxel array to xyz coordinates.

`array` (np.array): 3D voxel array  

`filter_level` (int/float): (inherited from `bone` class) sets the threshold level for what is considered a voxel. Everything below filter level is rounded to 0, everything above rounded to 1 (ie voxel)

    returns: 
        np.array([n x 3])


### `.get_pca(self)`
Performs PCA on the xyz points array

`xyz(np.array)`: n x 3 array of xyz coordinates

    return (self.pc1
            self.pc2
            self.pc3)

### `.get_mean(self)`
Gets the mean xyz coordinates for xyz array 

    return (
            np.mean(self.xyz[:, 0]),
            np.mean(self.xyz[:, 1]),
            np.mean(self.xyz[:, 2]),

### `.center_to_origin(self)`
Sets the mean of the bone to (0,0,0)

### `.reset_position(self)`
Returns bone to original position in space

### `.plot(self, user_color=None, PCA_inv=None, PCA=True)`
Plot voxels with optional PCA, and colours
        
`user_color` (tuple): RGB color of the bone where 1 is maxium eg: red = (1,0,0)

*[www.colortools.net/color_mixer.html](https://www.colortools.net/color_mixer.html) is a good tool to work out what the best colour to use is.*
                        
`PCA` (boolean): plots the PCAs of the voxel

`PCA_inv` (boolean): plots the inverse of each PCA so the axes go in both directions


### `.scale(self, n)`

`n` = scale factor

scales the voxel data array by n

### `xyz_to_array(self. array_dim=(256,256,256))`
Creates a numpy voxel array the size of `array_dim`from `.xyz` attribute.

`array_dim` list/tupple (1x3)

    return vx_array

## Functions

### `mag(v)`

`v` = vector (np.array 1x3)

Finds magnitude of vector

### `angle (v1, v2)`

`v1, v2` = vector (np.array 1x3)

Finds the angel between two vectors

    return ang, v1

### `quaternion_rotation_from_angle(v, c_axis, theta)`

`v` = vector (np.array 1x3)

`c_axis` = cross product between two principle conponets 

`theta` = angle of rotation (radians) 

### `rotate(bone_f1, bone_f2, interpolate=False, scale_factor=2)`

Aligns and rotates bone_f1 to bone_f2

`bone_f1, bone_f2` = bone class object

`interpolate` = (boolean) if `True` bone_f1 is upscaled, rotated and downscaled to increase point density.

`scale_factor` = set how much the bone will be upscaled to increase the density.