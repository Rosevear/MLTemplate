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
    The numpy random package NOTE: The random seed state for numpy is not threadsafe, meaning it is shared across different threads run concurrently. See https://stackoverflow.com/questions/31057197/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener
    TODO: Tensorflow
    """

    random.seed(random_seed)
    np.random.seed(random_seed)


def load_csv(path, logger):
    """
    Loads the data stored in the csv file present on the provided path
    and returns the corresponding Pandas dataframe object (or series if the data is one-dimensional)
    """

    logger.info('Loading data set {}'.format(path))
    try:
        return pd.read_csv(path, header=0, squeeze=True)
    except Exception as e:
        logger.info(
            'There was a problem loading the data set {}: {}'.format(path, str(e)))


def save_csv(path, data, encoding, keep_indexes, logger):
    """
    Saves the provided pandas dataframe object in csv file format to the interim data directory, in the provided encoding, with/without the in-memory indexes of the data frame object, as specified by keep_indexes.
    """

    try:
        data.to_csv(path, encoding='utf-8', index=False)
    except Exception as e:
        logger.info(
            'There was a problem saving the data set {}: {}'.format(path, str(e)))

def setup_logger():
    """
    Sets up a logger for tracking experiment info. See constants.py in config folder to specify the level, format, and log file to write to
    """

    logging.basicConfig(filename=config.LOGFILE,
                        level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    return logging.getLogger(__name__)




