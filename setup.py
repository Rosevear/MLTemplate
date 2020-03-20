#Setup.py is used to package the source code as a python module. 
# See https://stackoverflow.com/questions/1471994/what-is-setup-py and https://docs.python.org/3/installing/index.html#installing-index

from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Research into VEE exception automation',
    author='Harris Utilities Smartworks',
    license='',
)
