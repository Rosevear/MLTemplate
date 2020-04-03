from pathlib import Path
import logging

###############DIRECTORY STRUCTURE################
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data/raw"
INTERIM_DATA_DIR = ROOT_DIR / "data/interim"
PROCESSED_DATA_DIR = ROOT_DIR /  "data/processed"
CUR_DATA_FILE = "BTUData.csv"


###############EXPERIMENT CONFIG###############
TRAINING_SET_SIZE = 0.80
RANDOM_SEED = 0
K = 10  # The number of folds to use for K-Fold Cross Validation. See https://scikit-learn.org/stable/modules/cross_validation.html
REPEATS = 10  # The number of times to repeat k-fold cross validation with different randomized splits. See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html
METRIC_LIST = ['accuracy']
TUNE_HYPER_PARAMETERS = True
PLOT_LEARNING_CURVES = True
PLOT_VALIDATION_CURVES = True
RETURN_TRAIN_SCORES = True

####### LOGGING CONFIG######
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO
LOGFILE = ROOT_DIR / "logs" / "data.log"

#######DATA ########
CATEGORICAL_COLUMNS = ['LOCATION_CLASS', 'YEAR', 'MONTH', 'DAY', 'DOW', 'PREVYEAR', 'PREVMONTH', 'PREVDAY',
                       'PREVDOW', 'PREVPREVYEAR', 'PREVPREVMONTH', 'PREVPREVDAY', 'PREVPREVDOW', 'UOM', 'PREV_STATUS', 'PREV_STATUS2']
NUMERICAL_COLUMNS = ['READ_VALUE', 'PREV_READ', 'PREV_READ2']

