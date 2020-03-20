# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import load_dataset
import pandas
import constants


def main(data_filepath):
    """ Runs data processing scripts to turn raw data from source_file into
        cleaned data ready to be analyzed (saved in destination_dir).
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #Load the raw data
    data_frame = load_dataset.load_csv(data_filepath)

    shift_size = 1
    logger.info(
        'Removing missing values and shifting data by {} steps for time series forecasting'.format(shift_size))
    shift(data_frame, shift_size)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=constants.DATA_LOGFILE, level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    source_file = constants.RAW_DATA_DIR / constants.CUR_DATA_FILE
    main(source_file)
