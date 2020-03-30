# -*- coding: utf-8 -*-
import os
from pathlib import Path
import logging
import config
import utils
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


"""
Runs data processing scripts to turn raw data from (../raw) into
cleaned data ready to be analyzed(saved in ../processed).
"""
    
def process_data(data):
    """
    Processes the provided pandas dataframe object by: currently doing nothing
    """

    try:
        return data
    except Exception as e:
        logger.info(
            'There was a problem processing the data set: {} {}'.format(config.CUR_DATA_FILE, str(e)))


if __name__ == '__main__':

    #Set random seeds for reproducibility
    utils.freeze_random_generators(config.RANDOM_SEED)

    #Logging setup
    logging.basicConfig(filename=config.DATA_LOGFILE,
                        level=logging.INFO, format=config.LOG_FORMAT)
    logger = logging.getLogger(__name__)

    print("Loading the raw data set...")
    data_source = config.RAW_DATA_DIR / config.CUR_DATA_FILE
    logger.info('Starting the data processing pipeline for the data set {}'.format(
        config.CUR_DATA_FILE))
    data = utils.load_csv(data_source, logger)
    
    #Preprocess the data:
    print("Fetching the first few rows of unprocessed data to display...")
    print(data.head(10))
    data = process_data(data)
    
    print("Fetching the first few rows of processed data to display...")
    print(data.head(10))

    print("Saving the raw data to the interim directory...")
    save_location = config.INTERIM_DATA_DIR / config.CUR_DATA_FILE
    utils.save_csv(save_location, data, 'utf-8', False, logger)



