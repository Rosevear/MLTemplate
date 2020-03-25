VEE Automation Exception Prototype

This repo is a research prototype for the VEE Automation Exception Project by Harris Utilities Smartworks.
See https://confluence.harrissmartworks.com/display/RD/Project+Notes%3A+VEE+Exception+Automation for more detail.

########## Dependency installation ########

It is possible to install dependencies via the use of an environment file (in .yml format) which captures the relevant conda and pip dependencies.

conda env create -f conda.yml

#########Setting up an ML development environment with Anaconda#########

NOTE: These instructions are for Windows 10

Anaconda is both a package and a virtual environment manager. The former aspect allows for the quick and easy installation/updating/management of hundreds of modules within the python ecosystem.

Virtual environments serve as a way to isolate dependencies required for different projects, rather than installing everything globally and risking conflicts due to differing requirements between projects.

1. Download the Anaconda distribution (ideally installing for all users of the system) from https://www.anaconda.com/distribution/

2. Open the anaconda prompt (which should be installed by the anaconda distribution) and type as below to find where conda is being called from within your shell
Add the revealed path for the .exe file and Scripts folder to your PATH environment variable (if you want to be able to call conda from the terminal)

    where conda

Result for my case: C:\ProgramData\Anaconda3\Scripts\conda.exe

So, for example, I would add C:\ProgramData\Anaconda3\Scripts\ and  C:\ProgramData\Anaconda3\Scripts\conda.exe to my windows PATH variable. See here for how to to so on Windows 10: https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/

3.  In order to use conda with powershell specifically one must initialize the shell, and then restart it

    conda init powershell

Other shells likely need to perform something similar, and attempting to use conda from a given shell should result in a prompt to initialize for your respective shell

4. Create a new virtual environment with a python version of 3.6 (keras supports python versions Python 2.7-3.6)

conda create --name <you_env_name> python=3.6

5. To activate the environment

    conda activate <your_env_name>

6. To install tensorflow: Tensorflow is an open source machine learning and numerical computing platform https://www.tensorflow.org/

    conda install tensorflow

7. Verify that the installation was successful: you should see the version displayed without any errors (warnings about unoptimized binaries for your compute architecture are OK. There are different versions of tensorflow that allow for CPU specific optimizations to train models faster, but this typically requires building tensorflow from source. See here for more info).

python -c "import tensorflow as tf; print(tf.__version__)"

Anaconda Cloud also has a gpu-specific installation of tensorflow that will download and manage the necessary GPU API's (e.g. CUDA) as well. 
See the following for more information on the relationship between CPU and GPU packages within the Conda ecosystem: https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/ and https://www.anaconda.com/tensorflow-in-anaconda/

NOTE: GPE enabled tensorflow requires that the relevant CUDA and CUDAnn libraries are also installed, along with an NVIDIA driver for your GPU. The Conda package manager will handle the former software requirements but not the latter.

Finally, intended GPU itself must be CUDA enabled. See here for a list of CUDA enabled Nvidia products: https://developer.nvidia.com/cuda-gpus

See here https://www.tensorflow.org/install/gpu for installation instructions and requirements for dpu enabled tensorflow

8. To install keras: Keras is a high level Deep Learning API that supports multiple ML frameworks as a backend (like Tensorflow): https://keras.io/.

Since tensorflow 2.0 tensorflow comes shipped with it's own implementation of the Keras API, with tensorflow specific enhancements. See here for more detail: https://blog.tensorflow.org/2018/12/standardizing-on-keras-guidance.html

To install the Keras github project (independent of Tensorflow), perform the following.

    conda install keras

9. Verify keras installation

    python -c "import keras; print(keras.__version__)"

10. Install sci-kit learn: scikit-learn is a traditional machine learning library, that also features a number of helpful functions for running ML experiments more generally: https://scikit-learn.org/stable/

    conda install scikit-learn

11. Validate sklearn installed

    python -c "import sklearn; print(sklearn.__version__)"

12. Install pandas: the de facto standard data analysis library in the python ecosystem: https://pandas.pydata.org/
    conda install pandas

13. Install mlflow: a leading open source experiment management framework. This is not yet part of the official Anaconda distribution. However, the associated Anaconda Cloud package repository contains a number of additional 'channels' which conda can search and pull additional packages from. One of the most well known is conda-forge. Perform the steps below to download and install mlflow from the forge, or see here about configuring conda to search the forge automatically: https://conda-forge.org/docs/user/introduction.html
    conda install -c conda-forge mlflow

    NOTE: There is currently an open issue with getting the tracker UI to work using the conda-forge version of mlflow as of Friday, March 20th, 2020. See here: https://github.com/mlflow/mlflow/issues/1951

    Workaround is to use the pip package instead

    pip install mlflow  –upgrade-strategy only-if-needed

14. Install DVC: an open source data version control system
    conda install -c conda-forge dvc

######Setting up VSCode########

TODO:

######Setting up Jupyter Lab#######

The Anaconda distribution comes with the Anaconda Navigator: a program that allows you to navigate a number of tools for the python ecosystem.
One of the useful programs included in the distribution is JupyterLab: a platform for interactive computing that offers a number of useful features integrated into one place. See https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html

It comes shipped with the IPython (https://ipython.org/) interactive shell as its kernel (the computational engine that actually runs the code), but supports a number of additional kernels for other programming languages as well.

In order to connect JupyterLab to your conda environments so that they will be visible within JupyterLab. See here: https://github.com/Anaconda-Platform/nb_conda_kernels or here if having trouble: https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook

Open up your shell AS an ADMINISTRATOR and run the following command below to install the nb_conda_kernels and make the provided environment visible to JupyterLab
    conda install -n <environment_name>  nb_conda_kernels

Any additional environments to the one specified in the above command must have their own kernel installed in order to be accessible to JupyterLab For instance, to access a Python environment, it must have the ipykernel package; e.g.

    conda install -n <other_environment> ipykernel

There appears to be an issue with conda version 4.7 currently (as of Friday, March 20th, 2020). See https://github.com/conda/conda/issues/8836
If encountering this error try running conda init bash and restarting the shell as per https://github.com/conda/conda/issues/8836#issuecomment-514026318

######Project Configuration#######
This project uses MLFlow as an experiment management framework (see https://mlflow.org/docs/latest/index.html).

The MLProject.txt file specifies both the execution environment (docker, conda, system), and the entry points for running different portions of the project. This particular project uses a conda environment, which details all of the dependencies and channels used to download those dependencies in the conda.yml file. To run experiments via specific entry points see: https://mlflow.org/docs/latest/projects.html#running-projects




Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>