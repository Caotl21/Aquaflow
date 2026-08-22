from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['hofa_mpc_ros'],
    package_dir={'': 'src'},
)

setup(**d)
