# -*- coding: utf-8 -*-
import os
from pathlib import Path
from pandas import read_csv
import logging
import config
import utils


"""
Runs data processing scripts to turn raw data from (../raw) into
cleaned data ready to be analyzed(saved in ../processed).
"""
    
def load_csv(path):
    """
    Loads the data stored in the csv file present on the provided path
    and returns the corresponding Pandas dataframe object (or series if the data is one-dimensional)
    """

    logger.info('Loading data set {}'.format(path))
    try:
        return read_csv(path, header=0, squeeze=True)
    except Exception as e:
        logger.info('There was a problem loading the data set {}: {}'.format(path, str(e)))

if __name__ == '__main__':

    #Set random seeds for reproducibility
    utils.freeze_random_generators(config.RANDOM_SEED)

    logging.basicConfig(filename=config.DATA_LOGFILE,
                        level=logging.INFO, format=config.LOG_FORMAT)
    logger = logging.getLogger(__name__)

    data_source = config.RAW_DATA_DIR / config.CUR_RAW_DATA_FILE

    logger.info('Starting the data processing pipeline for the data set {}'.format(
        config.CUR_RAW_DATA_FILE))
    data_frame = load_csv(data_source)



