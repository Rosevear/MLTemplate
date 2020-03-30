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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

#TODO: Need to add standardization to column transformer
#TODO: Need to add column transformer steps to pipelines
#TODO: Need to get results for gridsearch cv and display them
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

    #Extract the data set targets as their own column and remove from the original dataset
    targets = data["TARGET"]
    del data["TARGET"]
    print("Displaying the first few rws of the data to be encoded")
    print(data.head(10))
    
    print("Encoding/formatting the data for ML algorithm usage...")
    
    #Columns witch which to use 1-hot encoding
    categorical_columns = ['LOCATION_CLASS',
                           'PREV_STATUS', 'PREV_STATUS2', 'DAY', 'MONTH']

    #TODO: Standardize or feature scale the numerical data for metric or gradient descent based methods. May need to do that inside the cross validation loop to prevent bias: double check this
    #We want to scale the READ_VALUE, PREV_READ, and PREV_READ2 features
    #numerical_columns = [1, 2, 3]

    data = encode_data(data, categorical_columns, logger)

    #The metrics to evaluate the algorithm on
    scores = config.METRIC_LIST

    #Split the data into train/validate and test
    X_train, X_test, y_train, y_test = train_test_split(data, targets,
                                                        train_size=config.TRAINING_SET_SIZE,
                                                        random_state=config.RANDOM_SEED,
                                                        shuffle=True,
                                                        stratify=None)
    
    #Initialize the classifiers
    clf1 = KNeighborsClassifier(algorithm='ball_tree', leaf_size=50)
    clf2 = DecisionTreeClassifier(random_state=config.RANDOM_SEED)

    #Standardization of the data: NOTE: Decision tree classifier does not require any such standardization, so we omit it here. See here for more on pipelines: https://scikit-learn.org/stable/modules/compose.html#pipeline
    pipe1 = Pipeline([('Standardization Step', StandardScaler()), #For more info on standardization: https://scikit-learn.org/stable/modules/preprocessing.html
                      ('Decision TRee Classifier', clf2)])

    #Setup hyper-parameters to search through
    #K-Nearest Neighbours: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.fit
    param_grid1 = [{'clf2__n_neighbors': list(range(1, 10)),
                    'clf2__p': [1, 2]}]

    #Decision Tree: https://scikit-learn.org/stable/modules/tree.html
    param_grid2 = [{'max_depth': list(range(1, 10)) + [None],
                    'criterion': ['gini', 'entropy']}]

    #Define the k-fold cross validation model evaluation procedure. See #See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    cv_procedure = KFold(n_splits=config.K, shuffle=True, random_state=config.RANDOM_SEED) 

    gridcvs = {}
    for score in scores:
        #Perform a grid search for each algorithm, to tune the hyper-paramters. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        for param_grid, estimator, name in zip((param_grid1, param_grid2), (pipe1, clf2), ('KNN', 'DTree')):
            gcv = GridSearchCV(estimator=estimator,  
                            param_grid=param_grid,
                            scoring=score,
                            n_jobs=1,
                            cv=cv_procedure,
                            verbose=1,
                            refit=False)
            gridcvs[name] = gcv


            #For the result of each grid_search, print out the results on the development set
            for name, gs_est in sorted(gridcvs.items()):
                print("PRINT OUT RESULTS HERE")
                # print("Best parameters set found on development set:")
                # print()
                # print(clf.best_params_)
                # print()
                # print("Grid scores on development set:")
                # print()
                # means = clf.cv_results_['mean_test_score']
                # stds = clf.cv_results_['std_test_score']
                # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                #     print("%0.3f (+/-%0.03f) for %r"
                #         % (mean, std * 2, params))
                # print()


