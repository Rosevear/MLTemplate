import os
import dotenv
import pandas as pd
import config
import utils
import mlflow.sklearn
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder


def encode_data(data, categorical_columns, logger):
    """
    Encodes data for use in machine learning algorithms by: transforming categorical variables to 1-hot vectors
    NOTE: Data is assumed to be a numpy array or pandas dataframe
    """
    try:
        #We ask the transform to return sparse matrices in a compressed format for computationally efficiency: See https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning/
        transformer = ColumnTransformer(transformers=[('One Hot Encoding Transform for Categorical Data', OneHotEncoder(
            sparse=True), categorical_columns)],  remainder='passthrough')
        return transformer.fit_transform(data)
    except Exception as e:
        logger.info(
            'There was a problem encoding the feature vector for the provided data set: {}'.format(str(e)))

if __name__ == "__main__":

    #Load environment variables
    dotenv.load_dotenv(dotenv.find_dotenv())

    #Set random seeds for reproducibility
    utils.freeze_random_generators(config.RANDOM_SEED)
    
    logger = utils.setup_logger()

    print("Loading processed data for experiment...")
    data_source = config.PROCESSED_DATA_DIR / config.CUR_DATA_FILE
    data = utils.load_csv(data_source, logger)

    print("Encoding/formatting the data for ML algorithm usage...")
    
    #Columns witch which to use 1-hot encoding
    categorical_columns = ['LOCATION_CLASS',
                           'PREV_STATUS', 'PREV_STATUS2', 'DAY', 'MONTH']

    #TODO: Standardize or feature scale the numerical data for metric or gradient descent based methods. May need to do that inside the cross validation loop to prevent bias: double check this
    #We want to scale the READ_VALUE, PREV_READ, and PREV_READ2 features
    #numerical_columns = [1, 2, 3]

    data = encode_data(data, categorical_columns, logger)

    # X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    # y = np.array([0, 0, 1, 1, 1, 0])
    # lr = LogisticRegression()
    # lr.fit(X, y)
    # score = lr.score(X, y)
    # print("Score: %s" % score)
    # mlflow.log_metric("score", score)
    # mlflow.sklearn.log_model(lr, "model")
    # print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
