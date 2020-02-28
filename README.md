# Vox
*voxel plotting and translation*

![bone_image](/images/bone.png)

## Dependencies 
*See environment.yml* 

You can create a copy of my conda `vox` environment with these commands:

    conda env create -f environment.yml

    conda activate vox

## OS
Has been written (and runs) on both Windows 10 and MacOS

## Python
This was written on python 3.6. Python 2 versions won't work due to the use of *f strings*

## mayavi
This is the 3d plotting library used for rendering the plots. Mayvai will launch a qt window to display the plot so you can't use this if you are using something remote like docker- you could set up an X serve session and with SSH but I would recommend just running it locally. There is a jupyter notebooks Extention want to plot things inline in your notebook - however, this is not the most stable or recommended way of plotting.

### Optional: inline 3d plotting
[inline plotting docs](http://docs.enthought.com/mayavi/mayavi/tips.html#using-mayavi-in-jupyter-notebooks)
  
  
### mayavi install

[mayavi install docs](https://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-conda-forge)
 
Mayavi plots images by calling the VTK libray and displying it a qt window- this means it is very very fast, however can be a bit of pain to install. Part of this is caused by its abity to work with difent qt packages, which makes it very flexable but does also mean it can get a bit confused!


## numpy-quaternion 

Numpy doesn't nativly suport quaternions as a data type- this package always you to pass quaterions properly and makes muliplication an returning the imaginary conponent a lot easier.

[numpy-quaternion github](https://github.com/moble/quaternion)

[numpy-quaternion docs](https://quaternion.readthedocs.io/en/latest/)

    conda install -c conda-forge quaternion
    
 or
 
    pip install numpy-quaternion
    
# How to use

## 1. Set the root directory for the matlab file loader
    root_dir = Path('C://Users/some/file/path/data')

### *Set custom filter level (optional)*
    bone.filter_level = 0.1

## 2. Load the data that you want to use
    tibia_f2 = bone.from_matlab_path(root_dir, matlab_file='phantom/phantom_tibia_f2.mat')

    tibia_f1 = bone.from_matlab_path(root_dir, matlab_file='phantom/phantom_tibia_f1.mat')


## 3. Rotate the Bone
    voxel_rotate(tibia_f1, tibia_f2)

## 4. Plotting the Rotation
    bone_plot(tibia_f1, tibia_f2)

![rotated_image](/images/rotated.png)


## 5. Table of Angles
    df_angles(tibia_f1, tibia_f2, name='tibia')


## Questions
If there are any issues or questions please do ask
