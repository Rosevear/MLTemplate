# Misc imports
import os
import dotenv
import pandas as pd
import config
import utils
import mlflow.sklearn
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Source
from datetime import datetime

#Scikit-learn imports
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import KFold, TimeSeriesSplit, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, ShuffleSplit, GridSearchCV, train_test_split, validation_curve, learning_curve, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

# Tensorflow imports
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def embed_classifier_in_pipeline(clf, sparse=False):
    """
    Embed the classifier clf into a machine learning pipeline

    Setting sparse to false will tell the 1-hot encoder to return matrices in a compressed format for computational efficiency: See https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning/
    """
    # This column transformer uses a 1-hot encoder for categorical data: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    one_hot_encoding_step = ('One Hot Encoding Transform for Categorical Data', OneHotEncoder(
        sparse=sparse, handle_unknown='ignore'), config.CATEGORICAL_COLUMNS)

    # And a Standard Scaler for numerical interval data: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    standardization_step = ('Standardization For Interval Data',
                            StandardScaler(), config.NUMERICAL_COLUMNS)

    # Transformer utility class to encode the inputs of different columns: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    transformer = ColumnTransformer(transformers=[
                                    one_hot_encoding_step, standardization_step],  remainder='passthrough')

    pipeline = Pipeline(steps=[('Column Transformer', transformer),
                               ('Classifier', clf)])

    return pipeline


def create_keras_model():
    """
    The tensorflow keras module provides a scikit-learn classifier wrapper class that implements the scikit-learn API. See here: https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/wrappers/scikit_learn.py#L191-L310
    """
	
    # Define the model structure 
    model = Sequential()
    model.add(Dense(units=100, input_dim=36, use_bias=True, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
	
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
    return model

def get_keras_classifier_pipelin(calibrate_probs=False, cv=None, method=None):
    """
    Keras Classifier: https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasClassifier
    """

    clf = KerasClassifier(build_fn=create_keras_model, epochs=150, batch_size=32)

    if calibrate_probs:
        if cv is None or method is None:
            raise ValueError("Pipelines which calibrate their probabilities require both a cv and method argument, but one or both are missing. Current arguments for cv and method parameters respectively: {}, {}".format(str(cv), method))
        clf = CalibratedClassifierCV(base_estimator=clf, method=method, cv=cv)

    return embed_classifier_in_pipeline(clf)

def get_dummy_classifier_pipeline(calibrate_probs=False, cv=None, method=None):
    """
    Dummy Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier
    """

    clf = DummyClassifier()

    if calibrate_probs:
        if cv is None or method is None:
            raise ValueError("Pipelines which calibrate their probabilities require both a cv and method argument, but one or both are missing. Current arguments for cv and method parameters respectively: {}, {}".format(str(cv), method))
        clf = CalibratedClassifierCV(base_estimator=clf, method=method, cv=cv)

    return embed_classifier_in_pipeline(clf)


def get_MLP_classifier_pipeline(calibrate_probs=False, cv=None, method=None):
    """
    MLP Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    """

    clf = clf = MLPClassifier(hidden_layer_sizes=(100,),
                              activation='relu',
                              solver='adam',
                              alpha=0.0001,  # L2 regularization parameter
                              batch_size='auto',
                              learning_rate='adaptive',
                              learning_rate_init=0.001,
                              power_t=0.5,  # Only used for invscaling option of learning_rate
                              momentum=0.9,  # Only used for SGD
                              nesterovs_momentum=True,  # Only used for SGD and momentum > 0
                              beta_1=0.9,  # Both betas are paramters for the Adam solver
                              beta_2=0.999,
                              epsilon=1e-8,  # Adam numerical stability constant
                              max_iter=200,
                              early_stopping=True,
                              verbose=False,
                              shuffle=True,
                              random_state=config.RANDOM_SEED)

    if calibrate_probs:
        if cv is None or method is None:
            raise ValueError("Pipelines which calibrate their probabilities require both a cv and method argument, but one or both are missing. Current arguments for cv and method parameters respectively: {}, {}".format(str(cv), method))
        clf = CalibratedClassifierCV(base_estimator=clf, method=method, cv=cv)

    return embed_classifier_in_pipeline(clf)


def get_KNN_classifier_pipeline(calibrate_probs=False, cv=None, method=None):
    """
    #K-Nearest Neighbours: https://scikit-learn.org/stable/modules/neighbors.html#id6
    """
    clf = KNeighborsClassifier(n_neighbors=5)

    if calibrate_probs:
        if cv is None or method is None:
            raise ValueError("Pipelines which calibrate their probabilities require both a cv and method argument, but one or both are missing. Current arguments for cv and method parameters respectively: {}, {}".format(str(cv), method))
        clf = CalibratedClassifierCV(base_estimator=clf, method=method, cv=cv)

    return embed_classifier_in_pipeline(clf)


def get_DT_classifier_pipeline(calibrate_probs=False, cv=None, method=None):
    """
     Decision Tree Classifier: https://scikit-learn.org/stable/modules/tree.html
    """

    clf = DecisionTreeClassifier(max_depth=20,
                                 random_state=config.RANDOM_SEED)

    if calibrate_probs:
        if cv is None or method is None:
            raise ValueError("Pipelines which calibrate their probabilities require both a cv and method argument, but one or both are missing. Current arguments for cv and method parameters respectively: {}, {}".format(str(cv), method))
        clf = CalibratedClassifierCV(base_estimator=clf, method=method, cv=cv)

    return embed_classifier_in_pipeline(clf)


def get_logit_classifier_pipeline(calibrate_probs=False, cv=None, method=None):
    """
    Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    clf = LogisticRegression()

    if calibrate_probs:
        if cv is None or method is None:
            raise ValueError("Pipelines which calibrate their probabilities require both a cv and method argument, but one or both are missing. Current arguments for cv and method parameters respectively: {}, {}".format(str(cv), method))
        clf = CalibratedClassifierCV(base_estimator=clf, method=method, cv=cv)

    return embed_classifier_in_pipeline(clf)


def get_perceptron_classifier_pipeline(calibrate_probs=False, cv=None, method=None):
    """
    Perceptron: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
    """
    
    clf = Perceptron(penalty='l1',
                     alpha=0.0001,
                     fit_intercept=True,
                     max_iter=1000,
                     tol=1e-3,
                     eta0=1,
                     early_stopping=True,
                     validation_fraction=0.1,
                     n_iter_no_change=5,
                     random_state=config.RANDOM_SEED)

    if calibrate_probs:
        if cv is None or method is None:
            raise ValueError("Pipelines which calibrate their probabilities require both a cv and method argument, but one or both are missing. Current arguments for cv and method parameters respectively: {}, {}".format(str(cv), method))
        clf = CalibratedClassifierCV(base_estimator=clf, method=method, cv=cv)

    return embed_classifier_in_pipeline(clf)

    

if __name__ == "__main__":

    print("Starting the experiment at {}".format(datetime.now()))
    
    #Avoid truncating columns in numpy arrays printed to std out (for easier visual inspection)
    np.set_printoptions(threshold=np.inf)

    #Load environment variables
    dotenv.load_dotenv(dotenv.find_dotenv())

    #Set random seeds for reproducibility
    utils.freeze_random_generators(config.RANDOM_SEED)
    
    logger = utils.setup_logger()

    data_source = config.PROCESSED_DATA_DIR / config.CUR_DATA_FILE
    print("Loading processed data for the experiment from {}".format(data_source))
    data = utils.load_csv(data_source, logger)

    #Extract the data set targets as their own column and remove from the original dataset
    targets = data[config.TARGET_COLUMN_NAME]
    del data[config.TARGET_COLUMN_NAME]

    if config.IS_TIME_SERIES:
        # Remove the date column used only for sorting the data
        del data[config.TIME_SERIES_COLUMN]
        # We want the test set to remain temporally ordered, so no shuffling
        shuffle_data = False
        stratify_by = None
        #Training sizes used by the learning curve plot in absolute terms: sklearn's TimeSeriesSplit cv underutilizes the provided data. See note plot_learning_curve function in exp_utils.py for more detail.
        if config.DO_EXPANDING_WINDOW_VALIDATION:
            train_sizes = np.array([1888, 3777, 5665, 7554, 9443, 11331, 13220, 15109, 16997, 18886, 20775, 22663, 24552, 26441, 28329, 30218, 32107, 33995, 35884, 37773])
        else:
            train_sizes = np.arange(0.05, 1.05, 0.05)
    else:
        shuffle_data = True
        stratify_by = targets
        train_sizes = np.arange(0.05, 1.00, 0.05)
    
    #Visual check on the data format
    print("Displaying the columns of the data...")
    print(list(data))
    print("Displaying the first few rows of the data...")
    print(data.head(10))
    
    print("Splitting the whole data set into a training set and a heldout test set...")
    X_train, X_test, y_train, y_test = train_test_split(data, targets,
                                                        train_size=config.TRAINING_SET_SIZE,
                                                        random_state=config.RANDOM_SEED,
                                                        shuffle=shuffle_data,
                                                        stratify=stratify_by)
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

        #Shuffle the original data-set for when cross-training
        targets = shuffle(targets)

    ####### CV SETUP START ###########
    #NOTE: See the following for a good visualization of the effect of different types of cross validation procedures: https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
    if config.IS_TIME_SERIES:
        if config.DO_EXPANDING_WINDOW_VALIDATION:
            #The sklearn TimeSeriesSplit uses an expanding window of train-test splits. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn-model-selection-timeseriessplit
            cv_procedure = TimeSeriesSplit(n_splits=config.K)
        else:
            #A custom class. See docstring in exp_utils.py for more detail
            cv_procedure = utils.SlidingWindowTimeSeriesSplit(
                n_splits=config.K)

    else:
        #Define the k-fold cross validation model evaluation procedure. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        cv_procedure = StratifiedKFold(
            n_splits=config.K, shuffle=True, random_state=config.RANDOM_SEED)

        #We can use repeated k-fold cross validation with multiple splits of the data in order to get a more robust estimate
        #cv_procedure = RepeatedStratifiedKFold(n_splits=config.K, n_repeats=config.REPEATS, random_state=config.RANDOM_SEED)

        # cv_procedure = RepeatedKFold(
        #     n_splits=config.K, n_repeats=config.REPEATS, random_state=config.RANDOM_SEED)

    if config.VERBOSE:
        print(
            'Current train test cross-validation split for the training data (using the first 100 row of data as an example of the split) ...')
        X_train_sample = X_train.iloc[0:100, :]
        y_train_sample = y_train.iloc[0:100]
        for train, test in cv_procedure.split(X_train_sample, y_train_sample):
            print("Train indices: {} Test indices: {}".format(train, test))

    ###### CV SETUP STOP #######

    ##### HYPER-PARAMETER TUNING SETUP START ##########
    #Setup classifier pipelines and hyper-parameters to search through for tuning each classifier
    #NOTE: When using a pipeline as the estimator with GridSearchCV, the parameters need to be named according to a specific syntax of the form <pipeline_step_name>__<parameter>: value. See https://stackoverflow.com/questions/48726695/error-when-using-scikit-learn-to-use-pipelines
    KNN_pipeline = get_KNN_classifier_pipeline()
    KNN_params = [{'Classifier__n_neighbors': list(range(5, 6)),
                    'Classifier__p': [2],''
                    'Classifier__random_state': config.RANDOM_SEED
                    }] 
    
    DT_pipeline = get_DT_classifier_pipeline()
    DT_params = [{'Classifier__max_depth': list(range(1, 21)),
                    'Classifier__criterion': ['gini'],
                    'Classifier__random_state': config.RANDOM_SEED
                    }]

    logit_pipeline = get_logit_classifier_pipeline()
    logit_params = [{'Classifier__penalty': ['l2'],
                    'Classifier__C': np.power(10., np.arange(1)),
                     'Classifier__random_state': config.RANDOM_SEED}]

    perceptron_pipeline = get_perceptron_classifier_pipeline()
    perceptron_params = [{'Classifier__penalty': ['l1', 'l2'],
                          'Classifier__alpha': np.arange(0.0000, 0.1, 0.005),
                          'Classifier__max_iter': np.arange(500, 10000, 500),
                          'Classifier__eta0': [0.01, 0.1, 1],
                          'Classifier__tol': [1e-3, None],
                          'Classifier__early_stopping': [True],
                          'Classifier__validation_fraction': [0.1],
                          'Classifier__shuffle': [True],
                          'Classifier__n_iter_no_change': [5, 10],
                          'Classifier__random_state': config.RANDOM_SEED}]

    #Match up the pipelines with their respective hyper-parameter grid and name them
    algorithm_param_combinations = zip((KNN_params, DT_params, logit_params), (KNN_pipeline, DT_pipeline, logit_pipeline), ('KNN', 'DTree', 'logit'))

    ####### HYPER-PARAMETER TUNING SETUP STOP ########

    ####### SPECIFIC CLASSIFIER SETUP START #######
    #Setup the classifier specific parameters for the learning and validation curves
    #NOTE: For each class the param_range varable is only used to setup a range for the validation curve to explore.
    print("Setting up the {} classifier...".format(
        config.CUR_CLASSIFIER))
    if config.CUR_CLASSIFIER == config.DUMMY:
        cur_pipe = get_dummy_classifier_pipeline("prior")
        cur_pipe_name = config.DUMMY

    elif config.CUR_CLASSIFIER == config.MLP:
        cur_pipe = get_MLP_classifier_pipeline(config.CALIBRATE_PROBABILITY, cv_procedure, 'isotonic')
        cur_pipe_name = config.MLP
        param_name = 'Classifier__alpha'
        param_range = 10.0 ** -np.arange(1, 7)

    elif config.CUR_CLASSIFIER == config.KNN:
        cur_pipe = get_KNN_classifier_pipeline(config.CALIBRATE_PROBABILITY, cv_procedure, 'isotonic')
        cur_pipe_name = config.KNN
        param_name = 'Classifier__n_neighbors'
        param_range = np.arange(1, 11, 1)

    elif config.CUR_CLASSIFIER == config.DT:
        cur_pipe = get_DT_classifier_pipeline(
            config.CALIBRATE_PROBABILITY, cv_procedure, 'isotonic')
        cur_pipe_name = config.DT
        param_name = 'Classifier__max_depth'
        param_range = np.arange(1, 51)

    elif config.CUR_CLASSIFIER == config.PERCEPTRON:
        cur_pipe = get_perceptron_classifier_pipeline(
            config.CALIBRATE_PROBABILITY, cv_procedure, 'isotonic')
        cur_pipe_name = config.PERCEPTRON
        param_name = 'Classifier__max_iter'
        param_range = np.arange(500, 10000, 500)

    elif config.CUR_CLASSIFIER == config.LOGISTIC:
        cur_pipe = get_logit_classifier_pipeline(
            config.CALIBRATE_PROBABILITY, cv_procedure, 'isotonic')
        cur_name = config.CUR_CLASSIFIER
        param_name = 'Classifier__C'
        param_range = np.arange(0.0, 1.1, 0.10)

    elif config.CUR_CLASSIFIER == config.KERAS:
        cur_pipe = get_keras_classifier_pipelin(config.CALIBRATE_PROBABILITY, cv_procedure, 'isotonic')
        cur_pipe_name = config.CUR_CLASSIFIER
        param_name = 'Classifier__batch_size'
        param_range = np.array([32, 64, 128])

    else:
        print("The current classifier {} is not recognized".format(config.CUR_CLASSIFIER))

    # Display some of the data as a sanity check that it is in the desired format
    if config.VERBOSE:
        cur_transformer = cur_pipe.named_steps['Column Transformer']
        data_sample = data[0:3]
        data_sample = cur_transformer.fit_transform(data_sample)
        print("Displaying the first few rows of data transformed by the column transformer...")
        print(data_sample[0:3, :])

    ######### SPECIFIC CLASSIFIER SETUP STOP ########


    for score in config.METRIC_LIST:

        ######## LEARNING ANALYSIS START #########
        if config.ANALYZE_LEARNING:

                cur_pipe.fit(X_train, y_train)

                if config.CUR_CLASSIFIER == config.DT:
                    print("Analyzing the Decision Tree algorithm...")
                    estimator = cur_pipe.named_steps['Classifier']
                    graph_data = export_graphviz(decision_tree=estimator, filled=True, rounded=True, class_names=True)
                    graph = Source(graph_data, format="png")
                    graph.render("./reports/figures/DTPlot-{}".format(score))
                
                elif config.CUR_CLASSIFIER == config.KNN:
                    print("Analyzing the KNN algorithm...")
                    cur_pipe.set_params(Classifier__n_neighbors=6)

                    #Extract the steps from the pipeline
                    knn_classifier = cur_pipe.named_steps['Classifier']
                    knn_transformer = cur_pipe.named_steps['Column Transformer']

                    #Index into the training set to retrieve a random sample of neighbours
                    num_points_to_query = 10
                    sample_training_data_indices = np.random.randint(
                        0, num_samples, num_points_to_query)

                    print("Sample data indices")
                    print(sample_training_data_indices)
                    
                    #Convert the training set to a numpy array for integer indexing extraction of query points
                    X_train_np = X_train.to_numpy()
                   
                    #Returns a 3 dimensional array of (1, 10, 23), so we need to index into the first dimension to get the 10 examples with 23 features
                    sample_training_data = X_train_np[np.array(
                        [sample_training_data_indices])][0]

                    print("Query points numpy")
                    print(sample_training_data)
                    
                    #Transform the sample data back into a pandas dataframe for use with the current column transformer (it references columns by column name, which is not available to pure numpy arrays)
                    sample_training_data_df = pd.DataFrame(
                        sample_training_data, columns=X_train.columns)

                    print("Query Points Data Frame format")
                    print(sample_training_data_df)

                    #Transform the sample data via the pipeline to encode it properly for use by the algorithm
                    transformed_sample_data = knn_transformer.transform(
                        sample_training_data_df)

                    #Get the neighbors of the queried points
                    #NOTE: The indices returned by the neighbor query correspond to the indices of the training set fit to the current model, so we can just index into X_train to get the human readable neighbours of the current queries
                    sample_training_data_neighbors_distance, sample_training_data_neighbors_indices = knn_classifier.kneighbors(
                        X=transformed_sample_data, return_distance=True)

                    print("Sample data neighbour indices")
                    print(sample_training_data_neighbors_indices)
                    
                    true_neighbours = sample_training_data_neighbors_indices[:, 1:]
                    print("True neighbour indices  excluding self")
                    print(true_neighbours)
                    
                    print("nearest neighbours")
                    nn = X_train_np[np.array(
                        [true_neighbours])][0]
                    print(nn)

                    print("Query points numpy for comparison")
                    print(sample_training_data)

                    print("Sample data neighbour distances")
                    print(sample_training_data_neighbors_distance[:, 1:])

                else:
                    print("There is no learning analysis path for the current classifier: {}".format(config.CUR_CLASSIFIER))

        ########## LEARNING ANALYSIS STOP ###########
        
        ########## GRID SEARCH START ###########
        #Perform a grid search for each algorithm, to tune the hyper-parameters. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        if config.TUNE_HYPER_PARAMETERS:
            print("Optimizing classifiers for {} ".format(score))
            gridcvs = {}

            if config.VERBOSE:
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
                                refit=False,
                                return_train_score=config.RETURN_TRAIN_SCORES)
                gcv.fit(X_train, y_train) #NOTE: Although we pass in X_train and y_train, this should be split into a train and dev set internally by the gridSearchCV according to the cv argument
                gridcvs[name] = gcv #This stores the grid search object in the dictionary, which stores the results for all of the paramters fitted for the current algorithm and parameter combos


            #Get and print the results
            for name, gs_est in sorted(gridcvs.items()):

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

        ####### GRID SEARCH STOP ##########


    ######### LEARNING AND VALIDATION CURVES START ###########
    #Plots training and validation set scores for training with different sample sizes, to gauge how more data helps the algorithm. See https://scikit-learn.org/stable/modules/learning_curve.html
    if config.PLOT_LEARNING_CURVES:
        print('Training {} classifier for the learning curve...'.format(config.CUR_CLASSIFIER))
        learning_curve_title = "{} Learning Curves".format(cur_pipe_name)
        learning_plot = utils.plot_learning_curve(estimator=cur_pipe, title=learning_curve_title, X=X_train,
                                                    y=y_train, train_sizes=train_sizes, shuffle=True, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=1)
        learning_plot.show()

    #Plots the training and validation set scores for various values of a single hyper-parameter to explore its bias-variance trade off
    if config.PLOT_VALIDATION_CURVES:
        print("Training {} classifier for the validation curve...".format(config.CUR_CLASSIFIER))
        validation_curve_title = "{} Validation Curve".format(
            cur_pipe_name)
        validation_plot = utils.plot_validation_curve(estimator=cur_pipe, title=validation_curve_title, X=X_train, y=y_train,
                                                        param_name=param_name, param_range=param_range, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=1)
        validation_plot.show()

    ########## LEARNING AND VALIDATION CURVES STOP ###########


    ########## CALIBRATION START ############

    if config.CALIBRATE_PROBABILITY:

        # Perform a kind of nested cross validation to get an a
        if config.CROSS_VALIDATE_CALIBRATION_PERFROMANCE:
            print("Cross validating calibrated {} classifier...".format(config.CUR_CLASSIFIER))
            cross_val_results = cross_validate(estimator=cur_pipe, X=X_train, y=y_train, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=0, return_train_score=True)
            print("Cross Validation Training set results. Mean: {} Standard Deviation: {}".format(np.mean(cross_val_results['train_score']), np.std(cross_val_results['train_score'])))
            print("Cross Validation Dev set results. Mean: {} Standard Deviation: {}".format(np.mean(cross_val_results['test_score']), np.std(cross_val_results['test_score'])))


        # Compute the calibration curves and get results for the held out test set
        print(
            "Splitting the training data into a training set and a held out calibration test set ...")
        X_calibration_train, X_calibration_test, y_calibration_train, y_calibration_test = train_test_split(X_train, y_train,
                                                                                                            train_size=config.TRAINING_SET_SIZE,
                                                                                                            random_state=config.RANDOM_SEED,
                                                                                                            shuffle=shuffle_data,
                                                                                                            stratify=stratify_by)
        print("X_calibration_train dimensions (row, column): {}".format(
            X_calibration_train.shape))
        print("X_calibration_test data dimensions (row, column): {}".format(
            X_calibration_test.shape))
        print("y_calibration_train data dimensions (row, column): {}".format(
            y_calibration_train.shape))
        print("y_calibration_test data dimensions (row, column): {}".format(
            y_calibration_test.shape))
        
        # Setup the calibrated and uncalibrated classifiers for plotting
        calibrated_classifier = cur_pipe
        calibrated_classifier_name = "Calibrated {}".format(config.CUR_CLASSIFIER)
        base_classifier = calibrated_classifier.named_steps['Classifier'].base_estimator
        base_classifier = embed_classifier_in_pipeline(base_classifier)
        base_classifier_name = "Uncalibrated {}".format(config.CUR_CLASSIFIER)
        
        print('Training the classifiers to plot for calibration...')
        base_classifier.fit(X_calibration_train, y_calibration_train)
        calibrated_classifier.fit(X_calibration_train, y_calibration_train)
        
        print('Plotting the classifier calibration curves...')
        classifiers_to_plot = [(base_classifier, base_classifier_name), (calibrated_classifier, calibrated_classifier_name)]
        utils.plot_calibration_curve(clf_list=classifiers_to_plot, X_test=X_calibration_test, y_test=y_calibration_test)
        

    ########## CALIBRATION STOP #############


    ######### CROSS VALIDATION SINGLE PARAMETER SETTING START ##########
    if config.CROSS_VALIDATE:
        print("Cross validating {} classifier...".format(config.CUR_CLASSIFIER))
        cross_val_results = cross_validate(estimator=cur_pipe, X=X_train, y=y_train, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=0, return_train_score=True)
        print("Cross Validation Training set results. Mean: {} Standard Deviation: {}".format(np.mean(cross_val_results['train_score']), np.std(cross_val_results['train_score'])))
        print("Cross Validation Dev set results. Mean: {} Standard Deviation: {}".format(np.mean(cross_val_results['test_score']), np.std(cross_val_results['test_score'])))

    ########### CROSS VALIDATION SINGLE PARAMETER SETTING STOP ##########



    ######## CROSS VAL CONFUSION MATRIX START ########
    #Compute the confusion matrix for investigating the classification performance: https://en.wikipedia.org/wiki/Confusion_matrix
    display_labels = np.array(config.CLASSES)
    cmap = 'viridis'
    ax = None
    xticks_rotation = 'horizontal'
    values_format = None
    include_values = True
    if config.COMPUTE_CROSS_VAL_CONFUSION_MATRIX:
        """"
        Since the plot_confusion_matrix function actually generates its own predictions based on the X value passed in, passing in the training set would not provide predictions of the folds used in cross-validation
        #So we get the values predicted during cross validation and get the confusion matrix from that
        #NOTE: This does not average the confusion matrix score across each fold, but instead just reports the total score across ALL predictions made, irrespective of which fold the prediction was made in.
        See for more detail: See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn-model-selection-cross-val-predict
        """
        print("Computing the confusion matrix for {} classifier".format(config.CUR_CLASSIFIER))
        predictions = cross_val_predict(estimator=cur_pipe, X=X_train, y=y_train, cv=cv_procedure, n_jobs=-1, verbose=1)
        unique, counts = np.unique(predictions, return_counts=True)
        prediction_counts = dict(zip(unique, counts))
        print("Classifier predictions...")
        print(prediction_counts)
        
        confusion_matrix = confusion_matrix(y_true=y_train, y_pred=predictions, normalize='all')
        tn, fp, fn, tp = confusion_matrix.ravel()
        print("Confusion matrix results...")
        print("True Positive Rate: {}".format(tp))
        print("False Positive Rate: {}".format(fp))
        print("True Negative Rate: {}".format(tn))
        print("False Negative Rate: {}".format(fn))

        #We need to bypass the predictions made in the plot_confusion_matrix function so we create the confusion matrix display directly. See https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/metrics/_plot/confusion_matrix.py#L119
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                      display_labels=display_labels)
        disp.plot(include_values=include_values,
                     cmap=cmap, ax=ax, xticks_rotation=xticks_rotation,
                     values_format=values_format)
        plt.show()

    ######### CROSS VALCONFUSION MATRIX STOP ##########


    ####### CROSS TRAIN START #######
    if config.IS_CROSS_TRAIN:
        cross_data_source = config.PROCESSED_DATA_DIR / config.CROSS_TEST_FILE
        print("Loading the cross-testing data from {}".format(cross_data_source))
        cross_data = utils.load_csv(cross_data_source, logger)
        cross_targets = cross_data[config.TARGET_COLUMN_NAME]
        del cross_data[config.TARGET_COLUMN_NAME]

        # Delete any columns in the cross testing data that are not present in the training data
        for cross_column in cross_data.columns:
            if not cross_column in data.columns:
                del cross_data[cross_column]

        print("Displaying the columns of the cross-data...")
        print(list(cross_data))

        print("Displaying the dimensions of the cross testing data...")
        print("Cross data feature vector dimensions (row, column): {}".format(
            cross_data.shape))
        print("Cross data target label dimensions (row, column): {}".format(
            cross_targets.shape))

        target_series = pd.Series(cross_targets)
        print('cross target counts \n {}'.format(target_series.value_counts()))

        print("Training on the data from {}".format(data_source))
        cur_pipe.fit(data, targets)
        
        print("Testing on the cross test data...")
        cross_test_score = cur_pipe.score(cross_data, cross_targets)
        
        print("Cross Test generalization score {}".format(cross_test_score))
        
        print("Plotting the confusion matrix for the cross-test ...")
        plot_confusion_matrix(estimator=cur_pipe, X=cross_data, y_true=cross_targets, labels=display_labels, normalize='all',
                              include_values=include_values, cmap=cmap, ax=ax, xticks_rotation=xticks_rotation, values_format=values_format)
        plt.show()
    ####### CROSS TRAIN STOP ######
        
    ######### GENERALIZATION TEST START #########
    #NOTE: This is to be done as a final step ONLY once all of the prior modelling has been completed and we have the best model we think we can get. This next step is to get an unbiased estimate of the model's performance on unseen data
    if config.EVALUATE_TEST_SET:
        print("Training and testing the generalization score for accuracy...")
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        cur_pipe.fit(X_train, y_train)
        
        final_score = cur_pipe.score(X_test, y_test)
        print("Test generalization score: {} ".format(final_score))
        
        print("Plotting the confusion matrix for the test set...")
        plot_confusion_matrix(estimator=cur_pipe, X=X_test, y_true=y_test, labels=display_labels, normalize='all', include_values=include_values, cmap=cmap, ax=ax, xticks_rotation=xticks_rotation, values_format=values_format)
        plt.show()
    ######### GENERALIZATION TEST STOP ########


