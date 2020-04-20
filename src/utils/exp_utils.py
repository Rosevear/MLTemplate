import random
import config
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import learning_curve, validation_curve, cross_validate
import matplotlib.pyplot as plt

def freeze_random_generators(random_seed):
    """
    Configures all of the random processed that depend on following packages,
    so that they start from the provided random_seed variable.

    This is done to ensure the reproducibility of experiments that make use of these libraries internally

    The standard library random package
    The numpy random package NOTE: The random seed state for numpy is not threadsafe, meaning it is shared across different threads run concurrently. See https://stackoverflow.com/questions/31057197/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener
    TODO: Tensorflow
    """

    random.seed(random_seed)
    np.random.seed(random_seed)


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

def setup_logger():
    """
    Sets up a logger for tracking experiment info. See constants.py in config folder to specify the level, format, and log file to write to
    """

    logging.basicConfig(filename=config.LOGFILE,
                        level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    return logging.getLogger(__name__)

def log_message(logger, msg):
    """
    Logs the provided message a well as printing it to STD_OUT
    """

    #TODO:

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
        n_samples = len(X)
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

#NOTE: This code has been modified from its original from after taken from here: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(estimator, title, X, y, train_sizes, shuffle, scoring, cv, n_jobs, verbose=1, axes=None, ylim=None):

    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score: {}".format(scoring))

    # See the following for a visual description of the difference between expanding and sliding window validation for time-series data: https://stats.stackexchange.com/a/326247/176281
    if config.IS_TIME_SERIES and config.DO_EXPANDING_WINDOW_VALIDATION:
        """
        NOTE: The TimeSeriesSplit object does not appear to play nicely with the learning_curve function; it uses the size of the first train-test split created by TimeSeriesSplit to determine the maximum number of training examples, 
        which results in the learning curve not making use of all of the the available data. This is likely because it expects a k-fold validator, which does not have to respect temporal ordering and can therefore always combine all 
        of the non-test set folds in each train-test split resulting in all cross-validations being performed on training sets of the same size. In contrast, performing an expanding window validation starts of with a small training set 
        that grows over time within the cross-validation procedure, so the learning curve plotting functions assume that the first train-test split size is representative of all of them. The code below manually sp;its up the data 
        into different sizes and uses an expanding window validation on each data set size, up to the maximum amount of data available.
        """
        train_scores = np.empty((len(train_sizes), config.K))
        test_scores = np.empty((len(train_sizes), config.K))
        fit_times = np.empty((len(train_sizes), config.K))
        
        for i in range(len(train_sizes)):
            cur_train_size = train_sizes[i]
            cur_train_data = X.iloc[0:cur_train_size + 1, :]
            
            if config.VERBOSE:
                print('Time Series expanding window split indices for the current subsection of data...')
                for train, test in cv.split(cur_train_data):
                    print("Train indices: {} Test indices: {}".format(train, test))
            
            cur_train_targets = y.iloc[0:cur_train_size + 1]
            cross_val_results = cross_validate(estimator=estimator, X=cur_train_data, y=cur_train_targets, scoring=scoring, cv=cv, n_jobs=-1, verbose=0, return_train_score=True)
            train_scores[i: i + 1, :] = cross_val_results['train_score']
            test_scores[i: i + 1, :] = cross_val_results['test_score']
            fit_times[i: i + 1, :] = cross_val_results['fit_time']
            
    else:
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator=estimator, X=X, y=y, train_sizes=train_sizes, shuffle=shuffle, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose, exploit_incremental_learning=False, return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score: {}".format(scoring))
    axes[2].set_title("Performance of the model")

    return plt

def plot_validation_curve(estimator, X, y, param_name, param_range, scoring, cv, n_jobs, verbose, title):
    """
    """

    train_scores, test_scores = validation_curve(estimator=estimator, X=X, y=y, param_name=param_name,
                                                 param_range=param_range, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score: {}".format(scoring))
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    
    return plt





