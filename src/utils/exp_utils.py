import random
import os
import config
import numpy as np
import pandas as pd
import logging
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, TimeSeriesSplit, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
from utils.data_utils import SlidingWindowTimeSeriesSplit
import tensorflow as tf

def freeze_random_generators(random_seed):
    """
    Configures all of the random processed that depend on following packages,
    so that they start from the provided random_seed variable.

    This is done to ensure the reproducibility of experiments that make use of these libraries internally

    The standard library random package
    The numpy random package 
    NOTE: The random seed state for numpy is not threadsafe, meaning it is shared across different threads run concurrently. 
    See https://stackoverflow.com/questions/31057197/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener
    """

    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # See https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.set_seed(random_seed)

    if config.USE_GPU:
        # See  https://github.com/NVIDIA/tensorflow-determinism
        os.environ['TF_DETERMINISTIC_OPS'] = '1'


def setup_logger():
    """
    Sets up a logger for tracking experiment info. See constants.py in config folder to specify the level, format, and log file to write to
    """

    logging.basicConfig(filename=config.LOGFILE,
                        level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    return logging.getLogger(__name__)

def log_message(logger, msg):
    """
    Logs the provided message a well as printing it to STD_OUT
    """

    #TODO:


def get_column_positions(column_headings, feature_list):
    """
    Return a list of the corresponding position within column_headings of each feature in feature_list.
    This can be used to get the position of lists of features for use in pipelines and transformers where
    one cannot rely on the input being a pandas DataFrame.
    """

    feature_to_pos = {}
    for i in range(len(column_headings)):
        feature_to_pos[column_headings[i]] = i
    feature_positions = [feature_to_pos[feature] for feature in feature_list] 
    
    return feature_positions


def shuffle_targets(targets_to_shuffle):
    """
    Shuffle the provided targets randomly
    """
    print("first 10 training rows of targets...")
    print(targets_to_shuffle[0:10])
    shuffled_targets = shuffle(targets_to_shuffle)
    print("first 10 rows of targets shuffled...")
    print(shuffled_targets[0:10])

    return shuffled_targets


def get_cv_procedure():
    """
    Returns the cross validation procedure to use based on the current configuration settings
    """

    # NOTE: See the following for a good visualization of the effect of different types of cross validation procedures: https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
    if config.IS_TIME_SERIES:
        if config.DO_EXPANDING_WINDOW_VALIDATION:
            # The sklearn TimeSeriesSplit uses an expanding window of train-test splits. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn-model-selection-timeseriessplit
            cv_procedure = TimeSeriesSplit(n_splits=config.K)
        else:
            # A custom class. See docstring in process_data.py for more detail
            cv_procedure = SlidingWindowTimeSeriesSplit(
                n_splits=config.K)

    else:
        if config.DO_REPEATED_K_FOLD:
            # We can use repeated k-fold cross validation with multiple splits of the data in order to get a more robust estimate. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html#sklearn-model-selection-repeatedstratifiedkfold
            cv_procedure = RepeatedStratifiedKFold(
                n_splits=config.K, n_repeats=config.REPEATS, random_state=config.RANDOM_SEED)
        else:
            # See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
            cv_procedure = StratifiedKFold(
                n_splits=config.K, shuffle=True, random_state=config.RANDOM_SEED)

    return cv_procedure




