import config
import utils
import logging
import pandas as pd

def construct_features_BTU(data):
    """
    Processes the provided pandas dataframe object by:
    
    Parsing the dates into separate columns (month and day) and deleting the original READ_DTM and METER_ID columns
    """

    try:
        data['DAY'] = pd.DatetimeIndex(data['READ_DTM']).day
        data['MONTH'] = pd.DatetimeIndex(data['READ_DTM']).month
        del data['READ_DTM']
        del data['METER_ID']
        return data
    except Exception as e:
        logger.info(
            'There was a problem constructing the feature vector for the provided data set: {}'.format(str(e)))

def construct_features_BTU_1(data):
    """
     Processes the provided pandas dataframe object by:

    Deleting the original METER_ID, LOCATION_NO, BILLING_CYCLE, COMMENTS, and DAYS_FROM_BILLDT columns
    """

    try:
        del data['METER_ID']
        del data['LOCATION_NO']
        del data['BILLING_CYCLE']
        del data['COMMENTS']
        del data['DAYS_FROM_BILLDT']
        return data
    except Exception as e:
        logger.info(
        'There was a problem constructing the feature vector for the provided data set: {}'.format(str(e)))

def construct_time_series_features_BTU_1(data):
    """
   Deleting the original METER_ID, LOCATION_NO, BILLING_CYCLE, COMMENTS, and DAYS_FROM_BILLDT columns
    Constructing a time series index out of the year, month, day, hour, minute, second columns
    Sorting by the time series index
    """

    try:
        del data['METER_ID']
        del data['LOCATION_NO']
        del data['BILLING_CYCLE']
        del data['COMMENTS']
        del data['DAYS_FROM_BILLDT']

        data[config.TIME_SERIES_COLUMN] = pd.to_datetime(data[['YEAR', 'MONTH', 'DAY']])
        
        return data.sort_values(by=config.TIME_SERIES_COLUMN)
    except Exception as e:
        logger.info(
            'There was a problem constructing the feature vector for the provided data set: {}'.format(str(e)))


def construct_features_MH_1(data):
    """
     Processes the provided pandas dataframe object by:

    Deleting the original METER_ID, LOCATION_NO, BILLING_CYCLE, COMMENTS, and DAYS_FROM_BILLDT columns
    Constructing a time series index out of the year, month, day, hour, minute, second columns
    Sorting by the time series index
    """

    try:
        del data['METER_ID']
        del data['LOCATION_HASH']
        del data['BILLING_CYCLE']
        del data['COMMENTS']
        del data['DAYS_FROM_BILLDT']
        
        return data
    except Exception as e:
        logger.info(
            'There was a problem constructing the feature vector for the provided data set: {}'.format(str(e)))


if __name__ == '__main__':

    #Set random seeds for reproducibility
    utils.freeze_random_generators(config.RANDOM_SEED)

    logger = utils.setup_logger()

    print("Loading the interim data set...")
    data_source = config.RAW_DATA_DIR / config.CUR_DATA_FILE
    logger.info('Starting the featurization process for the data set {}'.format(
        config.CUR_DATA_FILE))
    data = utils.load_csv(data_source, logger)

    #Construct the unencoded feature vector: breaking features out, deleting unwanted columns, adding a target label, adding features etc...
    print("Fetching the first few rows of un-featurized data to display...")
    pd.set_option('display.max_rows', 500)
    print(data.head(10))
    data = construct_time_series_features_BTU_1(data)

    print("Fetching the first few rows of featurized data to display...")
    print(data.head(10))

    print("Saving the feature vectors to the processed directory...")
    file_name = config.CUR_DATA_FILE.split(".")[0]
    save_location = config.PROCESSED_DATA_DIR / (file_name + config.PROCESSED_FILE_SAVE_SUFFIX + config.PROCESSED_FILE_EXTENSION)
    utils.save_csv(save_location, data, 'utf-8', False, logger)
