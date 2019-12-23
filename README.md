# Vox
# Libaries
See environment.yml 

Activate desired enviroment

    conda activate myenv
  
    conda env update -f environment.yml 

## OS
Has been written (and runs) on both Windows 10 and MacOS

## Python
This was written on python 3.6.7, python 2 versions won't work due to the use of f strings

## mayavi
This is the 3d plotting libray used for rendering the plots. Mayvai will launch a qt window to display the plot- meaning that you will need an X serve session for the plots to load. If you want to plot things inline you will need to use jupyter notebooks, not jupyter lab.

### mayavi install
https://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-conda-forge

    conda install vtk
    conda install qyqt5
    
    conda install mayavi
    
## pyquaternion
http://kieranwynn.github.io/pyquaternion/

    pip install pyquaternion
    
## Optional: inline 3d plotting
http://docs.enthought.com/mayavi/mayavi/tips.html#using-mayavi-in-jupyter-notebooks


# How to use:

### 1. Set the root directory for the matlab file loader

### 2. Load the data that you want to use

### 3. Contstruct the bone classes

### 4. Change xyz coordiates so `f1` is in `f2` position

### 5. Rotate bone using PCs
### 6. Plotting the rotation
