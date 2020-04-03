import os
import dotenv
import pandas as pd
import config
import utils
import mlflow.sklearn
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, ShuffleSplit, GridSearchCV, train_test_split, validation_curve, learning_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from graphviz import Source
from datetime import datetime


def get_KNN_classifier_pipeline():
    """
    """

    #K-Nearest Neighbours: https://scikit-learn.org/stable/modules/neighbors.html#id6
    clf = KNeighborsClassifier(algorithm='auto', leaf_size=50)

    #This column transformer uses a 1-hot encoder for categorical data: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    #And a Standard Scaler for numerical interval data: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    #We can ask the 1-hot encoder to return sparse matrices in a compressed format for computational efficiency: See https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning/
    KNN_transformer = ColumnTransformer(transformers=[('One Hot Encoding Transform for Categorical Data', OneHotEncoder(
        sparse=False, handle_unknown='ignore'), config.CATEGORICAL_COLUMNS), ('Standardization For Interval Data', StandardScaler(), config.NUMERICAL_COLUMNS)],  remainder='passthrough')

    KNN_pipeline = Pipeline(steps=[('Column Transformer', KNN_transformer),
                            ('KNN_Classifier', clf)])
    
    return KNN_pipeline

def get_DT_classifier_pipeline():
    """
    """

    #Decision Tree: https://scikit-learn.org/stable/modules/tree.html
    clf = DecisionTreeClassifier(random_state=config.RANDOM_SEED)

    DT_transformer = ColumnTransformer(transformers=[('One Hot Encoding Transform for Categorical Data', OneHotEncoder(
        sparse=True, handle_unknown='ignore'), config.CATEGORICAL_COLUMNS), ('Standardization For Interval Data', StandardScaler(), config.NUMERICAL_COLUMNS)],  remainder='passthrough')

    DT_pipeline = Pipeline(steps=[('Column Transformer', DT_transformer),
                            ('Decision_Tree_Classifier', clf)])

    return DT_pipeline

def get_Logit_classifier_pipeline():
    """
    """

    #Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    clf = LogisticRegression(
        random_state=config.RANDOM_SEED, solver='liblinear')

    LOGIT_transformer = ColumnTransformer(transformers=[('One Hot Encoding Transform for Categorical Data', OneHotEncoder(
        sparse=True, handle_unknown='ignore'), config.CATEGORICAL_COLUMNS), ('Standardization For Interval Data', StandardScaler(), config.NUMERICAL_COLUMNS)],  remainder='passthrough')

    logit_classifier_pipeline = Pipeline(steps=[('Column Transformer', LOGIT_transformer),
                            ('Logit_Classifier', clf)])

    return logit_classifier_pipeline

if __name__ == "__main__":

    print("Starting the experiment at {}".format(datetime.now()))

    #Load environment variables
    dotenv.load_dotenv(dotenv.find_dotenv())

    #Set random seeds for reproducibility
    utils.freeze_random_generators(config.RANDOM_SEED)
    
    logger = utils.setup_logger()

    print("Loading processed data for the experiment...")
    data_source = config.PROCESSED_DATA_DIR / config.CUR_DATA_FILE
    data = utils.load_csv(data_source, logger)

    #Extract the data set targets as their own column and remove from the original dataset
    targets = data["ACCEPTED"]
    del data["ACCEPTED"]

    #Visual check on the data format
    print("Displaying the columns of the data...")
    print(list(data))
    print("Displaying the first few rows of the data...")
    print(data.head(10))

    print("Splitting the training data...")
    X_train, X_test, y_train, y_test = train_test_split(data, targets,
                                                        train_size=config.TRAINING_SET_SIZE,
                                                        random_state=config.RANDOM_SEED,
                                                        shuffle=True,
                                                        stratify=targets)
    print("X_train dimensions (row, column): {}".format(X_train.shape))
    print("X_test data dimensions (row, column): {}".format(X_test.shape))
    print("y_train data dimensions (row, column): {}".format(y_train.shape))
    print("y_test data dimensions (row, column): {}".format(y_test.shape))


    #Check that the accepted and rejected training targets are roughly equally balanced
    target_series = pd.Series(y_train)
    print('training set target counts \n {}'.format(target_series.value_counts()))

    #Display some of the data as a sanity check that it is in the desired format
    if config.VERBOSITY >= 2:
        KNN_transformer = get_KNN_classifier_pipeline().named_steps['Column Transformer']
        data_sample = data[0:3]
        data_sample = KNN_transformer.fit_transform(data_sample)
        print("Displaying the first few rows of data transformed by the KNN column transformer...")
        print(data_sample[0:3, :])


    #Setup classifier pipelines and hyper-parameters to search through for each classifier
    #NOTE: When using a pipeline as the estimator with GridSearchCV, the parameters need to be named according to a specific syntax of the form <pipeline_step_name>__<parameter>: value. See https://stackoverflow.com/questions/48726695/error-when-using-scikit-learn-to-use-pipelines
    KNN_pipeline = get_KNN_classifier_pipeline()
    KNN_paramS = [{'KNN_Classifier__n_neighbors': list(range(5, 6)),
                    'KNN_Classifier__p': [2]}] 
    
    DT_pipeline = get_DT_classifier_pipeline()
    DT_params = [{'Decision_Tree_Classifier__max_depth': list(range(1, 11)) + [None],
                    'Decision_Tree_Classifier__criterion': ['gini']}]

    LOGIT_pipeline = get_Logit_classifier_pipeline
    LOGIT_params = [{'Logit_Classifier__penalty': ['l2'],
                    'Logit_Classifier__C': np.power(10., np.arange(1))}]

    #Match up the pipelines with their respective hyper-parameter grid and name them
    #algorithm_param_combinations = zip((KNN_paramS, DT_params, LOGIT_params), (KNN_pipeline, DT_pipeline, LOGIT_pipeline), ('KNN', 'DTree', 'Logit'))
    #algorithm_param_combinations = [(LOGIT_params, LOGIT_pipeline, 'Logit')]
    algorithm_param_combinations = [(DT_params, DT_pipeline, 'DTree')]

    #Perform a grid search for each algorithm, to tune the hyper-parameters. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    if config.TUNE_HYPER_PARAMETERS:
        for score in config.METRIC_LIST:
            print("Optimizing classifiers for {} ".format(score))
            gridcvs = {}

            #Define the k-fold cross validation model evaluation procedure. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
            #cv_procedure = StratifiedKFold(n_splits=config.K, shuffle=True, random_state=config.RANDOM_SEED)
            cv_procedure = RepeatedStratifiedKFold(n_splits=config.K, n_repeats=config.REPEATS, random_state=config.RANDOM_SEED)

            #NOTE: For using cross validation with a single train test split when we want to avoid k-fold validation within gridSearchCV for faster (but less statistically reliable for small datasets) single split train/validate cycles. See https://stackoverflow.com/questions/29503689/how-to-run-gridsearchcv-without-cross-validation
            #cv_procedure = ShuffleSplit(train_size=config.TRAINING_SET_SIZE, n_splits=1, random_state=config.RANDOM_SEED)

            if config.VERBOSITY >= 2:
                print("Indices used for k fold cross validation")
                for train_index, test_index in cv_procedure.split(X_train, y_train):
                    print("TRAIN:", train_index, "TEST:", test_index)


            for param_grid, estimator, name in algorithm_param_combinations:
                print("Tuning hyper-parameters for {} classifier".format(name))
                gcv = GridSearchCV(estimator=estimator,  
                                param_grid=param_grid,
                                scoring=score,
                                n_jobs=-1,
                                cv=cv_procedure,
                                verbose=1,
                                refit=True,
                                return_train_score=config.RETURN_TRAIN_SCORES)
                gcv.fit(X_train, y_train) #NOTE: Although we pass in X_train and y_train, this should be split into a train and dev set internally by the gridSearchCV according to the cv argument
                gridcvs[name] = gcv


            #Get and print the results
            for name, gs_est in sorted(gridcvs.items()):

                print("Best parameters found on development set for {}: {}".format(name, gs_est.best_params_))
                print("Best score found on development set for {}: {}".format(name, gs_est.best_score_))

                #If we are using a decision tree we can create a graph of the tree to visualize how it makes decisions
                if name == "DTree":
                    tree_estimator = gs_est.best_estimator_.named_steps['Decision_Tree_Classifier']
                    graph_data = export_graphviz(tree_estimator, class_names=[str(class_val) for class_val in targets.unique()], filled=True, rounded=True)
                    graph = Source(graph_data, format="png")
                    graph.render("./reports/figures/DTPlot")

                #Find out what neighbours are being used to classify a random selection of training examples
                if name == "KNN":
                    knn_estimator = get_KNN_classifier_pipeline()
                    knn_estimator.set_params(KNN_Classifier__n_neighbors=5)
                    training_indices_sample = np.random.randint(0, X_train.shape[0], 10)
                    print("Get the nearest neighbours of a random sample of training points...")
                    #Since refit is true the nearest neighbors gridCV instance should be fit on all of the training data
                    #TODO: Use the random indices to index into the training data to get the points whose neighbors we want
                    #TODO: Pass that into get neighbors to get the relevant neighbors
                    #TODO: Transform the representation of the points and their neighbors back into their original format for easy viewing

                
                if config.RETURN_TRAIN_SCORES:
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
                print("Finished the hyper-parameter tuning!\n")
        


    cv_procedure = StratifiedKFold(n_splits=config.K, shuffle=True, random_state=config.RANDOM_SEED)

    #Plotstraining and validation set scores over after training over different sample sizes, to gauge how more data helps the algorithm. See https://scikit-learn.org/stable/modules/learning_curve.html
    if config.PLOT_LEARNING_CURVES:
        print('Training classifier for the learning curve...')
        cur_pipe = get_DT_classifier_pipeline()
        cur_pipe.set_params(Decision_Tree_Classifier__max_depth=50)
        train_sizes = np.arange(0.10, 1.10, 0.10)
        learning_plot = utils.plot_learning_curve(estimator=cur_pipe, title="DT learning curves", X=X_train, y=y_train, train_sizes=train_sizes, shuffle=True, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=1)
        learning_plot.show()

    #Plots the training and validation set scores for various values of a single hyper-parameter to explore its bias-variance trade off
    if config.PLOT_VALIDATION_CURVES:
        print("Training classifier for the validation curve...")
        param_name = 'Decision_Tree_Classifier__max_depth'
        param_range = np.array(list(range(1, 51)))
        validation_plot = utils.plot_validation_curve(estimator=cur_pipe, title="DT Validation Curve", X=X_train, y=y_train,
                                                        param_name=param_name, param_range=param_range, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=1)
        validation_plot.show()

    
    #TODO: Estimate generalization performance with the best algorithm, features, and hyper-parameters on the held out test set (X_test and y_test)
    #NOTE: This is to be done as a final step ONLY once all of the prior modelling has been completed and we have the best model we think. This next step is to get an unbiased estimate of the models performance on unseen data


