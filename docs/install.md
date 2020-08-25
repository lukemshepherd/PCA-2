# How to install

### Clone git repo

    $ git pull https://github.com/lukemshepherd/PCA-2.git

    $ cd PCA-2 


### Create conda environment 

    $ conda env create -f environment.yml

    $ conda activate PCA-2

### Create vox package    

    $ pip install -e .


## Installing packages individually 

### Python
Python 3.6 and higher

### mayavi

[mayavi install docs](https://docs.enthought.com/mayavi/mayavi/installation.html)
Mayavi plots images by calling the VTK library and displaying it a qt window- this means it is very very fast, however can be a bit of pain to install. Part of this is caused by its ability to work with different qt packages, which makes it very flexible but does also mean it can get a bit confused!

You can use conda to install it but using pip seems to be easier and will sort out the VTK install for you.

    $ pip install mayavi

    $ pip install PyQt5

### numpy-quaternion 

Numpy doesn't natively support quaternions as a data type- this package allows you to pass quaternions properly and makes multiplication and returning the imaginary component a lot easier.

[numpy-quaternion github](https://github.com/moble/quaternion)

[numpy-quaternion docs](https://quaternion.readthedocs.io/en/latest/)

    $ conda install -c conda-forge quaternion

or

    $ pip install numpy-quaternion

### numpy-stl
Numpy-stl adds support for loading and breaking down stl data.

[numpy-stl github](https://github.com/WoLpH/numpy-stl)

[numpy-stl documentation](https://numpy-stl.readthedocs.io/en/latest/)

    $ pip install numpy-stl
