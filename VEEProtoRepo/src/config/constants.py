from pathlib import Path

###############DIRECTORY STRUCTURE################
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data/raw"
INTERIM_DATA_DIR = ROOT_DIR / "data/interim"
PROCESSED_DATA_DIR = ROOT_DIR /  "data/processed"
CUR_RAW_DATA_FILE = "BTUAcceptedSample.csv"
CUR_PROCESSED_DATA_FILE = ""
DATA_LOGFILE = ROOT_DIR / "logs" / "data.log"


###############EXPERIMENT CONFIG###############
TRAINING_SET_SIZE = 0.80
RANDOM_SEED = 0

####### LOGGING CONFIG######
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
