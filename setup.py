from setuptools import setup

setup(
    name="vox",
    version="0.1.0",
    description="Package for rotating, aligning and plotting voxel and stl (bone) objects",
    url="http://github.com/lukemshepherd/vox",
    author="Luke M Shepherd",
    author_email="l.m.shepherd@outlook.com",
    license="MIT",
    packages=["vox"],
    install_requires=[
        "numpy",
        "pandas",
        "numpy-stl",
        "numpy-quaternion",
        "scipy",
        "mayavi",
        "scikit-learn",
    ],
    zip_safe=False,
)
