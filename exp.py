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
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, ShuffleSplit, GridSearchCV, train_test_split, validation_curve, learning_curve, cross_val_predict
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from graphviz import Source
from datetime import datetime


def get_KNN_classifier_pipeline():
    """
    #K-Nearest Neighbours: https://scikit-learn.org/stable/modules/neighbors.html#id6
    """
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
     https://scikit-learn.org/stable/modules/tree.html
    """

    clf = DecisionTreeClassifier(random_state=config.RANDOM_SEED)

    DT_transformer = ColumnTransformer(transformers=[('One Hot Encoding Transform for Categorical Data', OneHotEncoder(
        sparse=True, handle_unknown='ignore'), config.CATEGORICAL_COLUMNS), ('Standardization For Interval Data', StandardScaler(), config.NUMERICAL_COLUMNS)],  remainder='passthrough')

    DT_pipeline = Pipeline(steps=[('Column Transformer', DT_transformer),
                            ('Decision_Tree_Classifier', clf)])

    return DT_pipeline

def get_logit_classifier_pipeline():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    clf = LogisticRegression(
        random_state=config.RANDOM_SEED, solver='liblinear')

    logit_transformer = ColumnTransformer(transformers=[('One Hot Encoding Transform for Categorical Data', OneHotEncoder(
        sparse=True, handle_unknown='ignore'), config.CATEGORICAL_COLUMNS), ('Standardization For Interval Data', StandardScaler(), config.NUMERICAL_COLUMNS)],  remainder='passthrough')

    logit_classifier_pipeline = Pipeline(steps=[('Column Transformer', logit_transformer),
                            ('logit_Classifier', clf)])

    return logit_classifier_pipeline

def get_perceptron_classifier_pipeline():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
    """

    clf = Perceptron(random_state=config.RANDOM_SEED)

    perceptron_transformer = ColumnTransformer(transformers=[('One Hot Encoding Transform for Categorical Data', OneHotEncoder(
        sparse=True, handle_unknown='ignore'), config.CATEGORICAL_COLUMNS), ('Standardization For Interval Data', StandardScaler(), config.NUMERICAL_COLUMNS)],  remainder='passthrough')

    perceptron_classifier_pipeline = Pipeline(steps=[('Column Transformer', perceptron_transformer),
                                                ('perceptron_Classifier', clf)])

    return perceptron_classifier_pipeline

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
    num_samples = X_train.shape[0]
    num_features = X_train.shape[1]


    #Check that the accepted and rejected training targets are roughly equally balanced
    target_series = pd.Series(y_train)
    print('training set target counts \n {}'.format(target_series.value_counts()))
    if config.SHUFFLE_TARGETS:
        print("first 10 training rows of targets...")
        print(y_train[0:10])
        y_train = shuffle(y_train)
        print("first 10 rows of targets shuffled...")
        print(y_train[0:10])

    #Display some of the data as a sanity check that it is in the desired format
    if config.VERBOSITY >= 2:
        KNN_transformer = get_KNN_classifier_pipeline().named_steps['Column Transformer']
        data_sample = data[0:3]
        data_sample = KNN_transformer.fit_transform(data_sample)
        print("Displaying the first few rows of data transformed by the KNN column transformer...")
        print(data_sample[0:3, :])


    #Setup classifier pipelines and hyper-parameters to search through for tuning each classifier
    #NOTE: When using a pipeline as the estimator with GridSearchCV, the parameters need to be named according to a specific syntax of the form <pipeline_step_name>__<parameter>: value. See https://stackoverflow.com/questions/48726695/error-when-using-scikit-learn-to-use-pipelines
    KNN_pipeline = get_KNN_classifier_pipeline()
    KNN_params = [{'KNN_Classifier__n_neighbors': list(range(5, 6)),
                    'KNN_Classifier__p': [2]}] 
    
    DT_pipeline = get_DT_classifier_pipeline()
    DT_params = [{'Decision_Tree_Classifier__max_depth': list(range(50, 51)),
                    'Decision_Tree_Classifier__criterion': ['gini']}]

    logit_pipeline = get_logit_classifier_pipeline()
    logit_params = [{'logit_Classifier__penalty': ['l2'],
                    'logit_Classifier__C': np.power(10., np.arange(1))}]

    perceptron_pipeline = get_perceptron_classifier_pipeline()
    perceptron_params = [{'perceptron_classifier__': 'l1',
                          'perceptron_classifier__alpha': 0.0001,
                          'perceptron_classifier__fit_intercept': True,
                          'perceptron_classifier__max_iter': 1000,
                          'perceptron_classifier__tol': 1e-3,
                          'perceptron_classifier__eta0': 1,
                          'perceptron_classifier__early_stopping': True,
                          'perceptron_classifier__validation_fraction': 0.1,
                          'perceptron_classifier__n_iter_no_change': 5}]

    #Match up the pipelines with their respective hyper-parameter grid and name them
    #algorithm_param_combinations = zip((KNN_params, DT_params, logit_params), (KNN_pipeline, DT_pipeline, logit_pipeline), ('KNN', 'DTree', 'logit'))
    #algorithm_param_combinations = [(logit_params, logit_pipeline, 'logit')]
    algorithm_param_combinations = [(DT_params, DT_pipeline, 'DTree')]
    #algorithm_param_combinations = [(KNN_params, KNN_pipeline, 'KNN')]
    #algorithm_param_combinations = [(perceptron_params, perceptron_pipeline, 'perceptron')]

    #NOTE: See the following for a good visualization of the effect of different types of cross validation procedures: https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
    #Define the k-fold cross validation model evaluation procedure. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    cv_procedure = StratifiedKFold(n_splits=config.K, shuffle=True, random_state=config.RANDOM_SEED)
    
    #We can use repeated k-fold cross validation with multiple splits of the data in order to get a more robust estimate
    #cv_procedure = RepeatedStratifiedKFold(n_splits=config.K, n_repeats=config.REPEATS, random_state=config.RANDOM_SEED)

    #NOTE: This is for setting up a specific classifier in order to plot its learning/validation curves, confusion matrix, and generalization error
    cur_pipe = get_DT_classifier_pipeline()
    # Paramters that will be used in everything but the validation curve (since the latter plots the parameter itself on the x-axis)
    cur_pipe.set_params(Decision_Tree_Classifier__max_depth=50)

    #Learning curve plot configuration
    train_sizes = np.arange(0.05, 1.05, 0.05)
    learning_curve_title = "DT learning curves"

    #Validation curve plot configuration
    param_range = np.array(list(range(1, 51)))
    param_name = 'Decision_Tree_Classifier__max_depth'
    validation_curve_title = "DT Validation Curve"

    for score in config.METRIC_LIST:
        if config.ANALYZE_LEARNING:
            for _, estimator, name in algorithm_param_combinations:
                if name == 'DTree':
                    
                    #TODO: The estimator here does not want to fit for some reason. Getting error: AttributeError: 'DecisionTreeClassifier' object has no attribute 'transform'
                    estimator.set_params(
                        Decision_Tree_Classifier__max_depth=50).fit(X_train, y_train)
                    
                    graph_data = export_graphviz(decision_tree=estimator, filled=True, rounded=True)
                    graph = Source(graph_data, format="png")
                    graph.render("./reports/figures/DTPlot-{}".format(score))
                
                if name == 'KNN':
                    estimator.set_params(KNN_Classifier__n_neighbors=5)
                    estimator.fit(X_train, y_train)

                    #Extract the steps from the pipeline
                    knn_classifier = estimator.named_steps['KNN_Classifier']
                    knn_transformer = estimator.named_steps['Column Transformer']


                    #Index into the training set to retrieve a random sample of neighbours
                    sample_training_data_indices = np.random.randint(
                        0, num_samples, 10)
                    sample_training_data = X_train.to_numpy(
                    )[np.array([sample_training_data_indices])][0] #Returns a 3 dimensional array of (1, 10, 23), so we need to index into the first dimension to get the 10 examples with 23 features

                    #Transform the sample data back into a pandas dataframe for use with the current column transformer (it references columns by column name, which is not available to pure numpy arrays)
                    sample_training_data = pd.DataFrame(
                        sample_training_data, columns=X_train.columns)

                    #Transform the sample data via the pipeline to encode it properly for use by the algorithm
                    transformed_sample_data = knn_transformer .transform(
                        sample_training_data)

                    #Get the neighbors of the queried points
                    sample_training_data_neighbors = knn_classifier.kneighbors(
                        transformed_sample_data)

                    #TODO: Need to find a way to reverse the transform here as it looks like the column transformer and pipeline do not support that out of the box
                    #Reverse the encoding of data or easy inspection
                    original_training_data = knn_transformer.inverse_transform(
                        sample_training_data)
                    print("Training data sample...")
                    print(original_training_data)

                    #Reverse the encoding of neighbors to compare to their respective query points
                    neighbors_original_form = estimator.inverse_transform(
                        sample_training_data_neighbors)
                    print("Neighbors of training data samples...")
                    print(neighbors_original_form)
            
        #Perform a grid search for each algorithm, to tune the hyper-parameters. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        if config.TUNE_HYPER_PARAMETERS:
            print("Optimizing classifiers for {} ".format(score))
            gridcvs = {}

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

                #TODO: Remove this once we have the ANALYZE_learning working correctly
                #If we are using a decision tree we can create a graph of the tree to visualize how it makes decisions
                if name == "DTree":
                    print("Graphing the decision tree...")
                    tree_estimator = gs_est.best_estimator_.named_steps['Decision_Tree_Classifier']
                    graph_data = export_graphviz(decision_tree=tree_estimator, class_names=True, filled=True, rounded=True)
                    graph = Source(graph_data, format="png")
                    graph.render("./reports/figures/DTPlot-{}".format(score))

                print("Best parameters found on development set for {}: {}".format(name, gs_est.best_params_))
                print("Best score found on development set for {}: {}".format(name, gs_est.best_score_))
                
                if config.RETURN_TRAIN_SCORES:
                    print("Grid scores on training set for classifier: {} ".format(name))
                    training_cv_means = gs_est.cv_results_['mean_train_score']
                    training_cv_stds = gs_est.cv_results_['std_train_score']
                    for mean, std, params in zip(training_cv_means, training_cv_stds, gs_est.cv_results_['params']):
                        print("Classifier: {}, Mean: {}, Standard Deviation: {}, Params: {}".format(name, mean, std, params))

                print("Grid scores on development set for classifier: {}".format(name))
                test_cv_means = gs_est.cv_results_['mean_test_score']
                test_cv_stds = gs_est.cv_results_['std_test_score']
                for mean, std, params in zip(test_cv_means, test_cv_stds, gs_est.cv_results_['params']):
                    print("Classifier: {}, Mean: {}, Standard Deviation: {}, Params: {}".format(name, mean, std, params))
                print("Finished the hyper-parameter tuning!\n")
        
        
        #Plots training and validation set scores over after training over different sample sizes, to gauge how more data helps the algorithm. See https://scikit-learn.org/stable/modules/learning_curve.html
        if config.PLOT_LEARNING_CURVES:
            print('Training classifier for the learning curve...')
            learning_plot = utils.plot_learning_curve(estimator=cur_pipe, title=learning_curve_title, X=X_train, y=y_train, train_sizes=train_sizes, shuffle=True, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=1)
            learning_plot.show()

        #Plots the training and validation set scores for various values of a single hyper-parameter to explore its bias-variance trade off
        if config.PLOT_VALIDATION_CURVES:
            print("Training classifier for the validation curve...")
            
            validation_plot = utils.plot_validation_curve(estimator=cur_pipe, title=validation_curve_title, X=X_train, y=y_train, param_name=param_name, param_range=param_range, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=1)
            validation_plot.show()

    #Compute the confusion matrix for investigating the classification performance: https://en.wikipedia.org/wiki/Confusion_matrix
    #NOTE: This is a confusion matrix based on the predicted values generated during cross validation. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn-model-selection-cross-val-predict
    if config.COMPUTE_CONFUSION_MATRIX:
        predictions = cross_val_predict(estimator=cur_pipe, X=X_train, y=y_train, cv=cv_procedure, n_jobs=-1, verbose=1)
        unique, counts = np.unique(predictions, return_counts=True)
        prediction_counts = dict(zip(unique, counts))
        print("Classifier predictions...")
        print(prediction_counts)
        
        
        tn, fp, fn, tp = confusion_matrix(y_true=y_train, y_pred=predictions).ravel()
        print("Confusion matrix results...")
        print("True Positive Rate: {}".format(tp))
        print("False Positive Rate: {}".format(fp))
        print("True Negative Rate: {}".format(tn))
        print("False Negative Rate: {}".format(fn))

        # print("Plotting the confusion matrix...")
        # plot = plot_confusion_matrix(estimator=cur_pipe, X=X_train, y_true=y_train, normalize=True)
        # plt.show()

        
    #NOTE: This is to be done as a final step ONLY once all of the prior modelling has been completed and we have the best model we think. This next step is to get an unbiased estimate of the models performance on unseen data
    if config.EVALUATE_TEST_SET:
        print("Training and testing the generalization score for accuracy...")
        cur_pipe.fit(X_train, y_train)
        final_score = cur_pipe.score(X_test, y_test)
        print("Mean Generalization score: {} ".format(final_score))


