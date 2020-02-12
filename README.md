# Dependencies 
See environment.yml 

Activate desired environment

    conda activate myenv
  
    conda env update -f environment.yml 

## OS
Has been written (and runs) on both Windows 10 and MacOS

## Python
This was written on python 3.6. Python 2 versions won't work due to the use of f strings

## mayavi
This is the 3d plotting library used for rendering the plots. mayvai will launch a qt window to display the plot- meaning that you will need an X serve session for the plots to load. If you want to plot things inline you will need to use jupyter notebooks, not jupyter lab.

### mayavi install
https://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-conda-forge

    conda install vtk
    conda install pyqt5
    
    conda install mayavi
    
## pyquaternion
https://github.com/moble/quaternion
https://quaternion.readthedocs.io/en/latest/

    conda install -c conda-forge quaternion
    
 or
 
    pip install numpy-quaternion
    
    
## Optional: inline 3d plotting
http://docs.enthought.com/mayavi/mayavi/tips.html#using-mayavi-in-jupyter-notebooks
