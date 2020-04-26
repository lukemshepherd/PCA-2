# How to install

## Clone conda environment

If you use conda environments this is probably the quickest way to install dependencies is to update/ create an environment from my `vox_enviroment.yml` file. This will install all of the required packages.

$ conda env create -f environment.yml

$ conda activate vox

## Installing dependencies

### Python
This was written on python 3.6. Python 2 versions won't work due to the use of *f strings*

### mayavi

[mayavi install docs](https://docs.enthought.com/mayavi/mayavi/installation.html)
Mayavi plots images by calling the VTK library and displaying it a qt window- this means it is very very fast, however can be a bit of pain to install. Part of this is caused by its ability to work with different qt packages, which makes it very flexible but does also mean it can get a bit confused!

You can use conda to install it but using pip seems to be easier and will sort out the VTK install for you.

$ pip install mayavi

$ pip install PyQt5

### numpy-quaternion 

Numpy doesn't natively support quaternions as a data type- this package always you to pass quaternions properly and makes multiplication and returning the imaginary component a lot easier.

[numpy-quaternion github](https://github.com/moble/quaternion)

[numpy-quaternion docs](https://quaternion.readthedocs.io/en/latest/)

$ conda install -c conda-forge quaternion
or
$ pip install numpy-quaternion
