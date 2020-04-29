import random
import config
import numpy as np
import pandas as pd
import logging

def freeze_random_generators(random_seed):
    """
    Configures all of the random processed that depend on following packages,
    so that they start from the provided random_seed variable.

    This is done to ensure the reproducibility of experiments that make use of these libraries internally

    The standard library random package
    The numpy random package 
    NOTE: The random seed state for numpy is not threadsafe, meaning it is shared across different threads run concurrently. See https://stackoverflow.com/questions/31057197/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener
    TODO: Tensorflow See https://stackoverflow.com/a/59722305/5106849
    """

    random.seed(random_seed)
    np.random.seed(random_seed)


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




