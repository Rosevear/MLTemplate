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
    
def load_csv(path):
    """
    Loads the data stored in the csv file present on the provided path
    and returns the corresponding Pandas dataframe object (or series if the data is one-dimensional)
    """

    logger.info('Loading data set {}'.format(path))
    try:
        return pd.read_csv(path, header=0, squeeze=True)
    except Exception as e:
        logger.info('There was a problem loading the data set {}: {}'.format(path, str(e)))

def process_data(data):
    """
    Processes the provided pandas dataframe object by:
    
    Parsing the dates into separate columns and deleting the original READ_DTM column
    Removing the METER_ID column
    """

    try:
        data['DAY'] = pd.DatetimeIndex(data['READ_DTM']).day
        data['MONTH'] = pd.DatetimeIndex(data['READ_DTM']).month
        del data['READ_DTM']
        del data['METER_ID']
        return data
    except Exception as e:
        logger.info(
            'There was a problem processing the data set: {}'.format(str(e)))

if __name__ == '__main__':

    #Set random seeds for reproducibility
    utils.freeze_random_generators(config.RANDOM_SEED)

    #Logging setup
    logging.basicConfig(filename=config.DATA_LOGFILE,
                        level=logging.INFO, format=config.LOG_FORMAT)
    logger = logging.getLogger(__name__)

    #Load raw Data
    data_source = config.RAW_DATA_DIR / config.CUR_RAW_DATA_FILE
    logger.info('Starting the data processing pipeline for the data set {}'.format(
        config.CUR_RAW_DATA_FILE))
    data = load_csv(data_source)
    
    data = process_data(data)
    
    print("Fetching the first few rows of data to display...")
    print(data.head(10))

    #SavE the transformed data set
    data.to_csv(config.PROCESSED_DATA_DIR / config.CUR_RAW_DATA_FILE, encoding='utf-8', index=False)



