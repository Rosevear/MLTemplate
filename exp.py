# Misc imports
import os
import dotenv
import pandas as pd
import config
import utils
import models
import mlflow.sklearn
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Source
from datetime import datetime

#Scikit-learn imports
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import export_graphviz
from sklearn.model_selection import  GridSearchCV, train_test_split, validation_curve, cross_val_predict, cross_validate


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
    
    # Visual check on the data format
    print("Displaying the columns of the data...")
    print(list(data))
    print("Displaying the first few rows of the data...")
    print(data.head(10))

    print("Total size of data set: {} rows and {} columns".format(len(data.index), len(data.columns)))
    
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

    if config.SHUFFLE_TARGETS:
        y_train = utils.shuffle_targets(y_train)
        if config.IS_CROSS_TRAIN:
            # Shuffle the original data-set targets for when cross-training
            targets = utils.shuffle_targets(targets)

    cv_procedure = utils.get_cv_procedure()

    if config.VERBOSE:
        print(
            'Current train test cross-validation split for the training data (using the first 100 row of data as an example of the split) ...')
        X_train_sample = X_train.iloc[0:100, :]
        y_train_sample = y_train.iloc[0:100]
        for train, test in cv_procedure.split(X_train_sample, y_train_sample):
            print("Train indices: {} Test indices: {}".format(train, test))

        #Check that the accepted and rejected training targets are roughly equally balanced
        target_series = pd.Series(y_train)
        print('training set target counts \n {}'.format(target_series.value_counts()))

    ###### SPECIFIC CLASSIFIER SETUP START #######
    #Setup the classifier specific parameters for the learning and validation curves
    #NOTE: For each class the param_range varable is only used to setup a range for the validation curve to explore.
    print("Setting up the {} classifier...".format(
        config.CUR_CLASSIFIER))
    if config.CUR_CLASSIFIER == config.DUMMY:
        cur_pipe = models.get_dummy_classifier_pipeline(data)
        cur_pipe_name = config.DUMMY

    elif config.CUR_CLASSIFIER == config.MLP:
        cur_pipe = models.get_MLP_classifier_pipeline(data)
        cur_pipe_name = config.MLP
        param_name = '{}__alpha'.format(config.CLASSIFIER_STEP_NAME)
        param_range = 10.0 ** -np.arange(1, 7)

    elif config.CUR_CLASSIFIER == config.KNN:
        cur_pipe = models.get_KNN_classifier_pipeline(data)
        cur_pipe_name = config.KNN
        param_name = '{}__n_neighbors'.format(config.CLASSIFIER_STEP_NAME)
        param_range = np.arange(1, 11, 1)

    elif config.CUR_CLASSIFIER == config.DT:
        cur_pipe = models.get_DT_classifier_pipeline(data)
        cur_pipe_name = config.DT
        param_name = '{}__max_depth'.format(config.CLASSIFIER_STEP_NAME)
        param_range = np.arange(1, 51)

    elif config.CUR_CLASSIFIER == config.PERCEPTRON:
        cur_pipe = models.get_perceptron_classifier_pipeline(data)
        cur_pipe_name = config.PERCEPTRON
        param_name = '{}__max_iter'.format(config.CLASSIFIER_STEP_NAME)
        param_range = np.arange(500, 10000, 500)

    elif config.CUR_CLASSIFIER == config.LOGISTIC:
        cur_pipe = models.get_logit_classifier_pipeline(data)
        cur_name = config.CUR_CLASSIFIER
        param_name = '{}__C'.format(config.CLASSIFIER_STEP_NAME)
        param_range = np.arange(0.0, 1.1, 0.10)

    elif config.CUR_CLASSIFIER == config.KERAS_CLASSIFIER:
        cur_pipe = models.get_keras_classifier_pipeline(data)
        cur_pipe_name = config.CUR_CLASSIFIER
        param_name = '{}__batch_size'.format(config.CLASSIFIER_STEP_NAME)
        param_range = np.array([32, 64, 128])

    else:
        print("The current classifier {} is not recognized".format(
            config.CUR_CLASSIFIER))

    # Display some of the data as a sanity check that it is in the desired format
    if config.VERBOSE:
        cur_transformer = cur_pipe.named_steps[config.COLUMN_TRANSFORMER_STEP_NAME]
        data_sample = data[0:3]
        data_sample = cur_transformer.fit_transform(data_sample)
        print(
            "Displaying the first few rows of data transformed by the column transformer...")
        print(data_sample[0:3, :])

    ######### SPECIFIC CLASSIFIER SETUP STOP ########


    ##### HYPER-PARAMETER TUNING SETUP START ##########
    #Setup classifier pipelines and hyper-parameters to search through for tuning each classifier
    #NOTE: When using a pipeline as the estimator with GridSearchCV, the parameters need to be named according to a specific syntax of the form <pipeline_step_name>__<parameter>: value. 
    # See https://stackoverflow.com/questions/48726695/error-when-using-scikit-learn-to-use-pipelines
    KNN_pipeline = models.get_KNN_classifier_pipeline(data)
    KNN_params = [{'{}__n_neighbors'.format(config.CLASSIFIER_STEP_NAME): list(range(5, 6)),
                   '{}__p'.format(config.CLASSIFIER_STEP_NAME): [2], ''
                   '{}__random_state'.format(config.CLASSIFIER_STEP_NAME): config.RANDOM_SEED
                   }]

    DT_pipeline = models.get_DT_classifier_pipeline(data)
    DT_params = [{'{}__max_depth'.format(config.CLASSIFIER_STEP_NAME): list(range(1, 21)),
                  '{}__criterion'.format(config.CLASSIFIER_STEP_NAME): ['gini'],
                  '{}__random_state'.format(config.CLASSIFIER_STEP_NAME): config.RANDOM_SEED
                  }]

    logit_pipeline = models.get_logit_classifier_pipeline(data)
    logit_params = [{'{}__penalty'.format(config.CLASSIFIER_STEP_NAME): ['l2'],
                     '{}__C'.format(config.CLASSIFIER_STEP_NAME): np.power(10., np.arange(1)),
                     '{}__random_state'.format(config.CLASSIFIER_STEP_NAME): config.RANDOM_SEED}]

    perceptron_pipeline = models.get_perceptron_classifier_pipeline(data)
    perceptron_params = [{'{}__penalty'.format(config.CLASSIFIER_STEP_NAME): ['l1', 'l2'],
                          '{}__alpha'.format(config.CLASSIFIER_STEP_NAME): np.arange(0.0000, 0.1, 0.005),
                          '{}__max_iter'.format(config.CLASSIFIER_STEP_NAME): np.arange(500, 10000, 500),
                          '{}__eta0'.format(config.CLASSIFIER_STEP_NAME): [0.01, 0.1, 1],
                          '{}__tol'.format(config.CLASSIFIER_STEP_NAME): [1e-3, None],
                          '{}__early_stopping'.format(config.CLASSIFIER_STEP_NAME): [True],
                          '{}__validation_fraction'.format(config.CLASSIFIER_STEP_NAME): [0.1],
                          '{}__shuffle'.format(config.CLASSIFIER_STEP_NAME): [True],
                          '{}__n_iter_no_change'.format(config.CLASSIFIER_STEP_NAME): [5, 10],
                          '{}__random_state'.format(config.CLASSIFIER_STEP_NAME): config.RANDOM_SEED}]

    #Match up the pipelines with their respective hyper-parameter grid and name them
    algorithm_param_combinations = zip((KNN_params, DT_params, logit_params), (KNN_pipeline, DT_pipeline, logit_pipeline), ('KNN', 'DTree', 'logit'))

    ####### HYPER-PARAMETER TUNING SETUP STOP ########


    for score in config.METRIC_LIST:

        ######## LEARNING ANALYSIS START #########
        if config.ANALYZE_LEARNING:

                cur_pipe.fit(X_train, y_train)

                if config.CUR_CLASSIFIER == config.DT:
                    print("Analyzing the Decision Tree algorithm...")
                    estimator = cur_pipe.named_steps[config.CLASSIFIER_STEP_NAME]
                    graph_data = export_graphviz(decision_tree=estimator, filled=True, rounded=True, class_names=True)
                    graph = Source(graph_data, format="png")
                    graph.render("./reports/figures/DTPlot-{}".format(score))
                
                elif config.CUR_CLASSIFIER == config.KNN:
                    print("Analyzing the KNN algorithm...")
                    cur_pipe.set_params(Classifier__n_neighbors=6)

                    #Extract the steps from the pipeline
                    knn_classifier = cur_pipe.named_steps[config.CLASSIFIER_STEP_NAME]
                    knn_transformer = cur_pipe.named_steps[config.COLUMN_TRANSFORMER_STEP_NAME]

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
                                verbose=config.VERBOSE,
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
                                                    y=y_train, train_sizes=train_sizes, shuffle=True, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=config.VERBOSE)
        learning_plot.show()

    #Plots the training and validation set scores for various values of a single hyper-parameter to explore its bias-variance trade off
    if config.PLOT_VALIDATION_CURVES:
        print("Training {} classifier for the validation curve...".format(config.CUR_CLASSIFIER))
        validation_curve_title = "{} Validation Curve".format(
            cur_pipe_name)
        validation_plot = utils.plot_validation_curve(estimator=cur_pipe, title=validation_curve_title, X=X_train, y=y_train,
                                                        param_name=param_name, param_range=param_range, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=config.VERBOSE)
        validation_plot.show()

    ########## LEARNING AND VALIDATION CURVES STOP ###########


    ########## CALIBRATION START ############

    if config.CALIBRATE_PROBABILITY:

        # Perform a kind of nested cross validation to get an a
        if config.CROSS_VALIDATE_CALIBRATION_PERFROMANCE:
            print("Cross validating calibrated {} classifier...".format(config.CUR_CLASSIFIER))
            cross_val_results = cross_validate(estimator=cur_pipe, X=X_train, y=y_train, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=config.VERBOSE, return_train_score=config.RETURN_TRAIN_SCORES)
            
            if config.RETURN_TRAIN_SCORES:
                print("Cross Validation Training set results. Mean: {} Standard Deviation: {}".format(np.mean(cross_val_results['train_score']), np.std(cross_val_results['train_score'])))
            else:
                print("No training score computed to display...set RETURN_TRAIN_SCORES in constants.py to TRUE in order to compare scores on the dev and training sets...")
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
        base_classifier = models.create_classifier_pipeline(base_classifier, data)
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
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        print("Cross validating {} classifier...".format(config.CUR_CLASSIFIER))
        cross_val_results = cross_validate(estimator=cur_pipe, X=X_train, y=y_train, scoring=score, cv=cv_procedure, n_jobs=-1, verbose=config.VERBOSE, return_train_score=config.RETURN_TRAIN_SCORES)
        if config.RETURN_TRAIN_SCORES:
            print("Cross Validation Training set results. Mean: {} Standard Deviation: {}".format(np.mean(cross_val_results['train_score']), np.std(cross_val_results['train_score'])))
        else:
            print("No training score computed to display...set RETURN_TRAIN_SCORES in constants.py to TRUE in order to compare scores on the dev and training sets...")
        print("Cross Validation Dev set results. Mean: {} Standard Deviation: {}".format(np.mean(cross_val_results['test_score']), np.std(cross_val_results['test_score'])))

    ########### CROSS VALIDATION SINGLE PARAMETER SETTING STOP ##########

    ########### CROSS VAL CONFUSION MATRIX START ##########
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
        So we get the values predicted during cross validation and get the confusion matrix from that
        NOTE: This does not average the confusion matrix score across each fold, but instead just reports the total score across ALL predictions made, irrespective of which fold the prediction was made in.
        See for more detail: See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn-model-selection-cross-val-predict
        """
        print("Computing the confusion matrix for {} classifier".format(config.CUR_CLASSIFIER))
        predictions = cross_val_predict(estimator=cur_pipe, X=X_train, y=y_train, cv=cv_procedure, n_jobs=-1, verbose=config.VERBOSE)
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
        X_test = X_test.to_numpy()

        print("X-train shape {}".format(X_train.shape))
        print("Y train shape {}".format(y_train.shape))

        cur_pipe.fit(X_train, y_train)
        
        final_score = cur_pipe.score(X_test, y_test)
        print("Test generalization score: {} ".format(final_score))
        
        print("Plotting the confusion matrix for the test set...")

        # See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
        plot_confusion_matrix(estimator=cur_pipe, X=X_test, y_true=y_test, labels=display_labels, normalize='all', include_values=include_values, cmap=cmap, ax=ax, xticks_rotation=xticks_rotation, values_format=values_format)
        plt.show()
    ######### GENERALIZATION TEST STOP ########


