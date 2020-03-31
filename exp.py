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
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
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
    clf1 = KNeighborsClassifier(algorithm='auto', leaf_size=50)

    #Decision Tree: https://scikit-learn.org/stable/modules/tree.html
    clf2 = DecisionTreeClassifier(random_state=config.RANDOM_SEED)

    print("Setting up the column transformer for use in pipelines...")
    categorical_columns = ['LOCATION_CLASS','YEAR', 'MONTH', 'DAY', 'DOW', 'PREVYEAR', 'PREVMONTH', 'PREVDAY', 'PREVDOW', 'PREVPREVYEAR', 'PREVPREVMONTH', 'PREVPREVDAY', 'PREVPREVDOW', 'UOM', 'PREV_STATUS', 'PREV_STATUS2']
    numerical_columns = ['READ_VALUE', 'PREV_READ', 'PREV_READ2']
    

    #This column transformer uses a 1-hot encoder for categorical data: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    #And a Standard Scaler for numerical interval data: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    #We can ask the 1-hot encoder to return sparse matrices in a compressed format for computational efficiency: See https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning/
    transformer = ColumnTransformer(transformers=[('One Hot Encoding Transform for Categorical Data', OneHotEncoder(
        sparse=False), categorical_columns), ('Standardization For Interval Data', StandardScaler(), numerical_columns)],  remainder='passthrough')

    #Standardization of the data: NOTE: Decision tree classifier does not require any such standardization, so we omit it here. See here for more on pipelines: https://scikit-learn.org/stable/modules/compose.html#pipeline
    pipe1 = Pipeline(steps=[('Column Transformer', transformer),
                      ('KNN Classifier', clf1)])

    #Setup hyper-parameters to search through for each classifier
    #NOTE: When using a pipeline as the estimator with GridSearchCV, the paramters need to be named according to a specific syntax of the form <pipeline_step_name>__<parameter>: value. See https://stackoverflow.com/questions/48726695/error-when-using-scikit-learn-to-use-pipelines
    param_grid1 = [{'KNN Classifier__n_neighbors': list(range(1, 10)),
                    'KNN Classifier__p': [1, 2]}] 
    param_grid2 = [{'max_depth': list(range(1, 10)) + [None],
                    'criterion': ['gini', 'entropy']}]

    #Define the k-fold cross validation model evaluation procedure. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    cv_procedure = StratifiedKFold(n_splits=config.K, shuffle=True, random_state=config.RANDOM_SEED) 

    gridcvs = {}
    scores = config.METRIC_LIST
    for score in scores:
        #Perform a grid search for each algorithm, to tune the hyper-parameters. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        for param_grid, estimator, name in zip((param_grid1, param_grid2), (pipe1, clf2), ('KNN', 'DTree')):
            print("Tuning hyper-paramters for {} classifier".format(name))
            gcv = GridSearchCV(estimator=estimator,  
                            param_grid=param_grid,
                            scoring=score,
                            n_jobs=-1,
                            cv=cv_procedure,
                            verbose=1,
                            refit=False,
                            return_train_score=True)
            gcv.fit(X_train, y_train) #NOTE: although we pass in X_train and y_train, this should be split into a train and dev set internally by the gridSearchCV cross validator settings
            gridcvs[name] = gcv


        #Run on the training set with all sets of paramters and print out the results
        for name, gs_est in sorted(gridcvs.items()):
            print("Best parameters found on development set for {}: {}".format(name, gs_est.best_params_))
            print("Best score found on development set for {}: {}".format(name, gs_est.best_score_))
            
            print("Grid scores on training set for {} classifier".format(name))
            training_cv_means = gs_est.cv_results_['mean_train_score']
            training_cv_stds = gs_est.cv_results_['std_train_score']
            for mean, std, params in zip(training_cv_means, training_cv_stds, gs_est.cv_results_['params']):
                print("Classifier: {}, Mean: {}, Standard Deviation: {}, Params: {}".format(name, mean, std, params))

            print("Grid scores on development set for : {} classifier".format(name))
            test_cv_means = gs_est.cv_results_['mean_test_score']
            test_cv_stds = gs_est.cv_results_['std_test_score']
            for mean, std, params in zip(test_cv_means, test_cv_stds, gs_est.cv_results_['params']):
                print("Classifier: {}, Mean: {}, Standard Deviation: {}, Params: {}".format(name, mean, std, params))

    #TODO: Estimate generalization performance with the best algorithm, features, and hyper-paramters on the held out test set (X_test and y_test)
    #NOTE: This is to be done as a final step ONLY once all of the prior modelling has been completed and we have the best model we think. This next step is to get an unbiased estimate of the models performance on unseen data


