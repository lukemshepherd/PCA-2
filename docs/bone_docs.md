# Bone class methods:

## `.get_xyz()`
Convert 3D voxel array to xyz coordinates.

    array (np.array): 3D voxel array  

`filter_level` (int/float): (inherited from `bone` class) sets the threshold level for what is considered a voxel. Everything below filter level is rounded to 0, everything above rounded to 1 (ie voxel)

    returns: 
        np.array( [n x 3] )


## `.get_pca()`
Performs PCA on the xyz points array

    xyz(np.array): n x 3 array of xyz coordinates

    returns:    self.pc1
                self.pc2
                self.pc3

## `.get_mean()`
Gets the mean xyz coordinates for xyz array 

## `.center_to_origin()`
Sets the mean of the bone to (0,0,0)

## `.reset_position()`
Returns bone to original position in space

## `.plot()`
Plot voxels with optional PCA, and colours
        
`user_color` (tuple): RGB color of the bone where 1 is maxium eg: red = (1,0,0)
                        
`PCA` (boolean): plots the PCAs of the voxel

`PCA_inv` (boolean): plots the inverse of each PCA so the axes go in both directions


## `.dense()`
up-scales the bone 
