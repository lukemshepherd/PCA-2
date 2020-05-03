from setuptools import setup

setup(
    name="vox",
    version="0.1",
    description="rotating and alineing voxel and stl objects",
    url="http://github.com/lukemshepherd/vox",
    author="Luke M Shepherd",
    author_email="l.m.shepherd@outlook.com",
    license="MIT",
    packages=["vox"],
    install_requires=["numpy", "math" "pandas", "scipy.io", "stl,", "quaternion"],
    zip_safe=False,
)
