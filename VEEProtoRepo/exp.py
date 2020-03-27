import mlflow.sklearn
import mlflow
import numpy as np
import os
import dotenv
import pandas
import config
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":

    #Load environment variables
    dotenv.load_dotenv(dotenv.find_dotenv())

    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
