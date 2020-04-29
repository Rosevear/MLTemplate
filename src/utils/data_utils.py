
import config
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator


def load_csv(path, logger):
    """
    Loads the data stored in the csv file present on the provided path
    and returns the corresponding Pandas dataframe object (or series if the data is one-dimensional)
    """

    logger.info('Loading data set {}'.format(path))
    try:
        return pd.read_csv(path, header=0, squeeze=True)
    except Exception as e:
        logger.info(
            'There was a problem loading the data set {}: {}'.format(path, str(e)))


def save_csv(path, data, encoding, keep_indexes, logger):
    """
    Saves the provided pandas dataframe object in csv file format to the interim data directory, in the provided encoding, with/without the in-memory indexes of the data frame object, as specified by keep_indexes.
    """

    try:
        data.to_csv(path, encoding='utf-8', index=False)
    except Exception as e:
        logger.info(
            'There was a problem saving the data set {}: {}'.format(path, str(e)))

#NOTE: Code sourced from here and modified: https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
class SlidingWindowTimeSeriesSplit():
    """
    Constructs a sliding window of folds from the data that respects its original ordering, with each fold being broken down into a 80/20 train-test split.

    For example, for a provided training set with 100 samples the output will be as follows:

    Train Indices[0 1 2 3 4 5 6 7] Test indices: [8 9]
    Train indices: [10 11 12 13 14 15 16 17] Test indices: [18 19]
    Train indices: [20 21 22 23 24 25 26 27] Test indices: [28 29]
    Train indices: [30 31 32 33 34 35 36 37] Test indices: [38 39]
    Train indices: [40 41 42 43 44 45 46 47] Test indices: [48 49]
    Train indices: [50 51 52 53 54 55 56 57] Test indices: [58 59]
    Train indices: [60 61 62 63 64 65 66 67] Test indices: [68 69]
    Train indices: [70 71 72 73 74 75 76 77] Test indices: [78 79]
    Train indices: [80 81 82 83 84 85 86 87] Test indices: [88 89]
    Train indices: [90 91 92 93 94 95 96 97] Test indices: [98 99]
    """

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):

        n_folds = self.n_splits + 1
        n_samples = X.shape[0]
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))

        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(config.TRAINING_SET_SIZE * (stop - start)) + start
            yield indices[start: mid], indices[mid: stop]

class Pipeline_Spy(BaseEstimator, TransformerMixin):
    """
    To use as a pseudo-step in an sklearn pipline, in order to spy on the transformations made to the data prior to being passed to the final estimator
    """

    def transform(self, X):
        self.shape = X.shape
        
        print("Shape of the transformed data after passing through the pipeline: {} ".format(self.shape))

        if config.VERBOSE:
            print("Sample of the data which is of type {}".format(X.dtype))
            print(X[0, :])
        
        return X

    def fit(self, X, y=None, **fit_params):
        
        return self

    def fit_transform(self, X, y=None, **fit_params):

        return self.fit(X).transform(X)
