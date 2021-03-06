from pathlib import Path
import logging

############### DIRECTORY STRUCTURE ################
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data/raw"
INTERIM_DATA_DIR = ROOT_DIR / "data/interim"
PROCESSED_DATA_DIR = ROOT_DIR / "data/processed"
PROCESSED_FILE_SAVE_SUFFIX = "_time_series"
PROCESSED_FILE_EXTENSION = ".csv"
#CUR_DATA_FILE = "Oracle_BTU_Live_All_Data_2020-04-08.csv"
CUR_DATA_FILE = "Oracle_BTU_Live_All_Data_2020-04-08_time_series.csv"
#CROSS_TEST_FILE = "NPEIAllFailures.csv"
CROSS_TEST_FILE =  "Medicine_Hat_Full_Data_2020-04-09.csv"


############# CLASSIFIERS ##############
KNN = "KNN"
DT = "DT"
PERCEPTRON = "PERCEPTRON"
MLP = "MLP"
DUMMY = "DUMMY"
LOGISTIC = "LOGISTIC"
NB = "NAIVE BAYES"
PA = "PASSIVE AGRESSIVE"
SGD = "STOCHASIC GRADIENT DESCENT"

KERAS_CLASSIFIER = 'KERAS_CLASSIFIER'

# Whether the pipeline should try and infer the final input shape 
INFER_KERAS_INPUT_SHAPE = True

# The shape of the feature vector for Keras models needs to be specified prior to training
KERAS_INPUT_SHAPE = None

CUR_CLASSIFIER = SGD

####### PIPLINE STEP NAMES ########
COLUMN_TRANSFORMER_STEP_NAME = "Column Transformer"
CLASSIFIER_STEP_NAME = "Classifier"

####### RUNTIME ENV ######
# The random seed used to enforce determinism in the experiment. See exp_utils.py for more info.
RANDOM_SEED = 0

# Whether or not to use a gpu for training
USE_GPU = False #NOTE: THIS IS NOT IMPLEMENTED YET


############### EXPERIMENT CONFIG ###############

# What percentage of the data that should be used as the training set
TRAINING_SET_SIZE = 0.80

# The number of folds to use for K-Fold Cross Validation. See https://scikit-learn.org/stable/modules/cross_validation.html
K = 10

# The number of times to repeat k-fold cross validation with different randomized splits. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html
DO_REPEATED_K_FOLD = False
REPEATS = 10

#METRIC_LIST = ['accuracy', 'precision', 'recall']
METRIC_LIST = ['accuracy']

# Whether to diplay additional information during the running of the experiment, chiefly for diagnostic purposes
VERBOSE = False

# Whether or not to calibrate the probabilities emitted. See https://scikit-learn.org/stable/modules/calibration.html
CALIBRATE_PROBABILITY = False
CROSS_VALIDATE_CALIBRATION_PERFROMANCE = False
NUM_CALIBRATION_BINS = 10
CALIBRATION_METHOD = 'isotonic'

# Whether or not to run some specific analysis of the current classifier regarding how it learns
ANALYZE_LEARNING = False

# Whether or not to use GridSearchCV to find the best hyperparameters
TUNE_HYPER_PARAMETERS = False

# Whether to plot a learning curve to display how the algorithm fares given more training data
PLOT_LEARNING_CURVES = True

# Whether or not to plot a validation curve depict the bias-variance trade-off for a given hyper-parameter 
PLOT_VALIDATION_CURVES = False

# Whether or not to compute the confusion matrix yielded by the predictions made during cross validation
COMPUTE_CROSS_VAL_CONFUSION_MATRIX = False

# Whether or not to report a single cross validation score for a given set of parameters
CROSS_VALIDATE = True

# Whether or not to run the experiment with the held out test set to estimate generalization performance
EVALUATE_TEST_SET = False

# Whether or not to return the training scores used during learning
RETURN_TRAIN_SCORES = True

# Shuffle the targets of the training set in order to test if the algorithm is still getting some type of signal (probably a sign of data leakage if it is)
SHUFFLE_TARGETS = False

# Whether or not the data should be treated as time series for the purposes of splitting and evaluation
IS_TIME_SERIES = True
DO_EXPANDING_WINDOW_VALIDATION = True
TIME_SERIES_COLUMN = 'Date'

# Set whether or not the algorithm should be trained on one data set and tested on another
IS_CROSS_TRAIN = False

####### LOGGING CONFIG######
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO
LOGFILE = ROOT_DIR / "logs" / "data.log"

##### CLASSES #####
CLASSES = [1, 0]

####### DATA ########
CATEGORICAL_COLUMNS = ['LOCATION_CLASS',
'YEAR', 'MONTH', 'DAY', 'DOW', 
'PREVYEAR', 'PREVMONTH', 'PREVDAY','PREVDOW', 'PREV_UOM', 
'PREVPREVYEAR', 'PREVPREVMONTH', 'PREVPREVDAY', 'PREVPREVDOW', 'PREVPREV_UOM', 
'UOM', 
'PREV_STATUS', 'PREV_STATUS2', 
'NEXTYEAR', 'NEXTMONTH', 'NEXTDAY', 'NEXTDOW', 'NEXT_UOM', 'NEXT_STATUS',
'METER_TP',
'READ_HR', 'READ_MIN', 'READ_SECOND']

NUMERICAL_COLUMNS = ['READ_VALUE', 'PREV_READ', 'PREV_READ2', 'NEXT_READ_VALUE']

#ORDINAL_COLUMNS = ['READ_HR', 'READ_MIN', 'READ_SECOND']

BINARY_COLUMNS = ['HILO', 'REGISTERINCREASING', 'SUMCHECK', 'MAXDAILYUSAGE']

TARGET_COLUMN_NAME = 'Status'

