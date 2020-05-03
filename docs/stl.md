# STL files

STL objects are treated pretty much the same as normal voxel objects. They can be rotated/plotted etc. Also they can be used with voxel ones- i.e you can rotate an stl by a voxel.

## Loading data:

    tibia_stl = bone.from_stl_path(stl_file='phantom/phantom_tibia_f2.stl')

## How the data is stored

The full stl data is stored in `tibia_stl.data` 

The stl files are loaded from [numpy-stl](https://numpy-stl.readthedocs.io/en/latest/) module 


## Saving bones back into 

    tibia_stl.data.save('file.stl')