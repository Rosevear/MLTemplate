
#Setup.py is used for packaging the source code of the project into a python module. See https://stackoverflow.com/questions/1471994/what-is-setup-py and https://docs.python.org/3/installing/index.html#installing-index
from setuptools import find_packages, setup

setup(
    name='VEEPrototype',
    packages=find_packages("src"),
    package_dir={"": "src"},
    version='0.1.0',
    description='Research prototype for VEE exception automation',
    author='Harris Utilities Smartworks',
    license='',
)
