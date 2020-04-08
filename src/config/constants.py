from pathlib import Path
import logging

###############DIRECTORY STRUCTURE################
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data/raw"
INTERIM_DATA_DIR = ROOT_DIR / "data/interim"
PROCESSED_DATA_DIR = ROOT_DIR /  "data/processed"
CUR_DATA_FILE = "Oracle_BTU_Live_All_Data_2020-04-08.csv"


###############EXPERIMENT CONFIG###############
TRAINING_SET_SIZE = 0.80
RANDOM_SEED = 0
K = 10  # The number of folds to use for K-Fold Cross Validation. See https://scikit-learn.org/stable/modules/cross_validation.html
REPEATS = 10  # The number of times to repeat k-fold cross validation with different randomized splits. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html
#METRIC_LIST = ['accuracy', 'precision', 'recall']
METRIC_LIST = ['accuracy']
VERBOSITY = 1


# Whether or not we want to train on a specific set of parameters in order to analyze the learning method of an algorithm
ANALYZE_LEARNING = True
TUNE_HYPER_PARAMETERS = False
PLOT_LEARNING_CURVES = True
PLOT_VALIDATION_CURVES = True
COMPUTE_CONFUSION_MATRIX = True

#Whether or not to run the experiment with the held out test set to estimate generalization performance
EVALUATE_TEST_SET = False

#Whether or not to return the training scores during gridSearchCV
RETURN_TRAIN_SCORES = True

#Shuffle the targets of the training set in order to test if the algorithm is still getting some type of signal (probably a sign of data leakage)
SHUFFLE_TARGETS = False

####### LOGGING CONFIG######
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO
LOGFILE = ROOT_DIR / "logs" / "data.log"

#######DATA########
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

