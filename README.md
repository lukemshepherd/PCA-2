# Vox
*voxel ploting and translation*

## Dependencies 
*See environment.yml* 

You can create a copy my conda `sci` enviroment with these comands:

    conda env create -f environment.yml

    conda activate sci

## OS
Has been written (and runs) on both Windows 10 and MacOS

## Python
This was written on python 3.6. Python 2 versions won't work due to the use of *f strings*

## mayavi
This is the 3d plotting library used for rendering the plots. mayvai will launch a qt window to display the plot- meaning that you will need an X serve session for the plots to load. If you want to plot things inline you will need to use jupyter notebooks, not jupyter lab.

### mayavi install
https://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-conda-forge
    
## numpy-quaternion 
https://github.com/moble/quaternion
https://quaternion.readthedocs.io/en/latest/

    conda install -c conda-forge quaternion
    
 or
 
    pip install numpy-quaternion
    
    
## Optional: inline 3d plotting
http://docs.enthought.com/mayavi/mayavi/tips.html#using-mayavi-in-jupyter-notebooks


# How to use

## 1. Set the root directory for the matlab file loader
    root_dir = Path('C://Users/some/file/path/data')

## 2. Load the data that you want to use
    tibia_f2 = bone.from_matlab_path(root_dir, matlab_file='phantom/phantom_tibia_f2.mat')

    tibia_f1 = bone.from_matlab_path(root_dir, matlab_file='phantom/phantom_tibia_f1.mat')

### *Set custom filter level (optional)*
    bone.filter_level = 0.1 

## Rotate the Bone
    voxel_rotate(tibia_f1, tibia_f2)

## Plotting the Rotation
    bone_plot(tibia_f1, tibia_f2)

## Table of Angles
    df_angles(tibia_f1, tibia_f2, name='tibia')


# Contact
If there are an issues or questions please be in touch:

luke.shepherd.17@ucl.ac.uk
