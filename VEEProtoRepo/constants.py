from pathlib import Path

###############DIRECTORY STRUCTURE################
ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT_DIR / "data/raw"
INTERIM_DATA_DIR = ROOT_DIR / "data/interim"
PROCESSED_DATA_DIR = ROOT_DIR /  "data/processed"
CUR_RAW_DATA_FILE = ""
CUR_PROCESSED_DATA_FILE = ""
DATA_LOGFILE = "data.log"

###############EXPERIMENT CONFIG###############
TRAINING_SET_SIZE = 0.80
RANDOM_SEED = 0
PREDICTION_STEP  = 1 #How far into the future we want to forecast

