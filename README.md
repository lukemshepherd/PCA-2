# How to use

## 1. Set the root directory for the matlab file loader
`root_dir = Path('C://Users/luke/OneDrive - University College London/Marta/data')`

## 2. Load the data that you want to use
`tibia_f2 = bone.from_matlab_path(root_dir, matlab_file='phantom/phantom_tibia_f2.mat')`

`tibia_f1 = bone.from_matlab_path(root_dir, matlab_file='phantom/phantom_tibia_f1.mat')`

### *Set custom filter level (optional)*
`bone.filter_level = 0.1`

## Rotate the Bone
`voxel_rotate(tibia_f1, tibia_f2)`

## Plotting the Rotation
`bone_plot(tibia_f1, tibia_f2)`

## Table of Angles
`df_angles(tibia_f1, tibia_f2, name='tibia')`
    
