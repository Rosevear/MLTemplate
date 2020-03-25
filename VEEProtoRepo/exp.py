# import numpy as np
# import pandas
# import constants
# from src.data import load_dataset
# from src.models.baseline_predictions import random_model_predictions, persistence_model_predictions, mean_regressor_model_predictions, drift_model_predictions
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_absolute_error
# from statsmodels.tsa.ar_model import AR, ARResults

#Load environment variables
# project_dir = constants.ROOT_DIR
# dotenv_path = os.path.join(project_dir, '.env')
# dotenv.load_dotenv(dotenv_path)

from __future__ import print_function
import constants
import mlflow.sklearn
import mlflow
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import dotenv


if __name__ == "__main__":
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
