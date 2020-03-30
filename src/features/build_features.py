import config
import utils
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def construct_features(data):
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

def encode_data(data, categorical_columns, logger):
    """
    Encodes ata for use in machine learning algorithms by: transforming categorical variables to 1-hot vectors 
    """
    try:
        transformer = ColumnTransformer(transformers=[('One Hot Encoding Transform for Categorical Data', OneHotEncoder(), categorical_columns)])
        return transformer.fit_transform(data)
    except Exception as e:
        logger.info(
            'There was a problem encoding the feature vector for the provided data set: {}'.format(str(e)))


if __name__ == '__main__':

    #Set random seeds for reproducibility
    utils.freeze_random_generators(config.RANDOM_SEED)

    #Logging setup
    logging.basicConfig(filename=config.DATA_LOGFILE,
                        level=logging.INFO, format=config.LOG_FORMAT)
    logger = logging.getLogger(__name__)

    print("Loading the interim data set...")
    data_source = config.RAW_DATA_DIR / config.CUR_DATA_FILE
    logger.info('Starting the featurization process for the data set {}'.format(
        config.CUR_DATA_FILE))
    data = utils.load_csv(data_source, logger)

    #Construct the unencoded feature vector: breaking features out, deleting unwanted columns, adding a target label, adding features etc...
    print("Fetching the first few rows of un-featurized data to display...")
    print(data.head(10))
    data = construct_features(data)

    print("Fetching the first few rows of featurized data to display...")
    print(data.head(10))

    print("Saving the feature vectors to the processed directory...")
    save_location = config.PROCESSED_DATA_DIR / config.CUR_DATA_FILE
    utils.save_csv(save_location, data, 'utf-8', False, logger)

    print("Encoding/formatting the data for ML algorithm usage")
    #We want to 1 hot encode the LOCATION_CLASS, PREV_STATUS, PREV_STATUS2, DAY, and MONTH features
    categorical_columns = [0, 4, 5, 10, 11]

    #TODO: Standardize or feature scale the numerical data for metric or gradient descent based methods. Need to do that inside the cross validation loop to prevent bias
    #We want to scale the READ_VALUE, PREV_READ, and PREV_READ2 features
    #numerical_columns = [1, 2, 3]

    data = encode_data(data, categorical_columns, logger)
    print("Displaying the first few rows of the encoded data...")
    print(data[0:10, :])
