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
    targets = data["ACCEPTED"]
    del data["ACCEPTED"]
    print("Displaying the first few rows of the data...")
    print(data.head(10))

    print("Splitting the training data...")
    X_train, X_test, y_train, y_test = train_test_split(data, targets,
                                                        train_size=config.TRAINING_SET_SIZE,
                                                        random_state=config.RANDOM_SEED,
                                                        shuffle=True,
                                                        stratify=None)
    print("X_train dimensions (row, column): {}".format(X_train.shape))
    print("X_test data dimensions (row, column): {}".format(X_test.shape))
    print("y_train data dimensions (row, column): {}".format(y_train.shape))
    print("y_test data dimensions (row, column): {}".format(y_test.shape))
    
    #Initialize the classifiers
    #K-Nearest Neighbours: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.fit
    clf1 = KNeighborsClassifier(algorithm='ball_tree', leaf_size=50)

    #Decision Tree: https://scikit-learn.org/stable/modules/tree.html
    clf2 = DecisionTreeClassifier(random_state=config.RANDOM_SEED)

    print("Setting up the column transformer for use in pipelines...")
    categorical_columns = ['LOCATION_CLASS','YEAR', 'MONTH', 'DAY', 'DOW', 'PREVYEAR', 'PREVMONTH', 'PREVDAY', 'PREVDOW', 'PREVPREVYEAR', 'PREVPREVMONTH', 'PREVPREVDAY', 'PREVPREVDOW', 'UOM', 'PREV_STATUS', 'PREV_STATUS2']
    numerical_columns = ['READ_VALUE', 'PREV_READ', 'PREV_READ2']
    

    #This column transformer uses a 1-hot encoder for categorical data: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    #And a Standard Scaler for numerical interval data: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    #We ask the 1-hot encoder to return sparse matrices in a compressed format for computational efficiency: See https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning/
    transformer = ColumnTransformer(transformers=[('One Hot Encoding Transform for Categorical Data', OneHotEncoder(
        sparse=True), categorical_columns), ('Standardization For Interval Data', StandardScaler(), numerical_columns)],  remainder='passthrough')

    #Standardization of the data: NOTE: Decision tree classifier does not require any such standardization, so we omit it here. See here for more on pipelines: https://scikit-learn.org/stable/modules/compose.html#pipeline
    pipe1 = Pipeline([('Column Transformer', transformer),
                      ('KNN Classifier', clf1)])

    #Setup hyper-parameters to search through for each classifier
    param_grid1 = [{'clf1__n_neighbors': list(range(1, 10)),
                    'clf1__p': [1, 2]}]
    param_grid2 = [{'max_depth': list(range(1, 10)) + [None],
                    'criterion': ['gini', 'entropy']}]

    #Define the k-fold cross validation model evaluation procedure. See #See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    cv_procedure = KFold(n_splits=config.K, shuffle=True, random_state=config.RANDOM_SEED) 

    gridcvs = {}
    scores = config.METRIC_LIST
    # for score in scores:
    #     #Perform a grid search for each algorithm, to tune the hyper-paramters. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    #     for param_grid, estimator, name in zip((param_grid1, param_grid2), (pipe1, clf2), ('KNN', 'DTree')):
    #         gcv = GridSearchCV(estimator=estimator,  
    #                         param_grid=param_grid,
    #                         scoring=score,
    #                         n_jobs=1,
    #                         cv=cv_procedure,
    #                         verbose=1,
    #                         refit=False)
    #         gridcvs[name] = gcv


    #         #For the result of each grid_search, print out the results on the development set
    #         for name, gs_est in sorted(gridcvs.items()):
    #             print("PRINT OUT RESULTS HERE")
    #             # print("Best parameters set found on development set:")
    #             # print()
    #             # print(clf.best_params_)
    #             # print()
    #             # print("Grid scores on development set:")
    #             # print()
    #             # means = clf.cv_results_['mean_test_score']
    #             # stds = clf.cv_results_['std_test_score']
    #             # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #             #     print("%0.3f (+/-%0.03f) for %r"
    #             #         % (mean, std * 2, params))
    #             # print()


