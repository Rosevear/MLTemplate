import numpy as np
import pandas
import constants
from src.data import load_dataset
from src.models.baseline_predictions import random_model_predictions, persistence_model_predictions, mean_regressor_model_predictions, drift_model_predictions
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.ar_model import AR, ARResults

