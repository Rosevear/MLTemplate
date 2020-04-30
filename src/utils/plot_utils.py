import config
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve, cross_validate
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

#NOTE: This code has been modified from its original from after taken from here: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(estimator, title, X, y, train_sizes, shuffle, scoring, cv, n_jobs, verbose=config.VERBOSE, axes=None, ylim=None):

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
        that grows over time within the cross-validation procedure, so the learning curve plotting functions assume that the first train-test split size is representative of all of them. The code below manually splits up the data 
        into different sizes and uses an expanding window validation on each data set size, up to the maximum amount of data available.
        """
        train_scores = np.empty((len(train_sizes), config.K))
        test_scores = np.empty((len(train_sizes), config.K))
        fit_times = np.empty((len(train_sizes), config.K))
        
        for i in range(len(train_sizes)):
            cur_train_size = train_sizes[i]
            cur_train_data = X.iloc[0:cur_train_size + 1, :]
            cur_train_targets = y.iloc[0:cur_train_size + 1]
            
            cross_val_results = cross_validate(estimator=estimator, X=cur_train_data, y=cur_train_targets, scoring=scoring, cv=cv, n_jobs=-1, verbose=config.VERBOSE, return_train_score=True)
            
            train_scores[i, :] = cross_val_results['train_score']
            test_scores[i, :] = cross_val_results['test_score']
            fit_times[i, :] = cross_val_results['fit_time']
            
    else:
        # See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html
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
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html
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

#NOTE: Code taken from here and modified to fit its present purpose: https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py
def plot_calibration_curve(clf_list, X_test, y_test):
    
    """
    Plots calibration curves and computes summary statistics for each classifier provided in clf_list, where clf_list is a list of tuples in the form (estimator, str)
    Assumes that the provided classifiers have already been fit on a training and (where applicable) calibration set
    See https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
    """

    plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for clf, name in clf_list:
        predicted_probs = clf.predict_proba(X_test)
        prob_pos = predicted_probs[:, 1] 
        clf_score = brier_score_loss(y_test, prob_pos)

        print("Classes: {}".format(
            clf.named_steps['Classifier'].classes_))
        positive_class_prediction_probs = np.array(
            [prob for prob in predicted_probs[:, 1] if prob > 0.50])
        negative_class_prediction_probs = np.array(
            [prob for prob in predicted_probs[:, 0] if prob > 0.50])

        print("{} Classifier Positive Class Probabilites Mean: {}, Standard Deviation {}".format(name,
            np.mean(positive_class_prediction_probs), np.std(positive_class_prediction_probs)))
        print("{} Classifier Negative Class Probabilites Mean: {}, Standard Deviation {}".format(name,
            np.mean(negative_class_prediction_probs), np.std(negative_class_prediction_probs)))
        print("{} Classifier Accuracy Score: {}".format(name,
            clf.score(X_test, y_test)))
        print("{} Classifier Brier Score {}".format(name, clf_score))

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=config.NUM_CALIBRATION_BINS)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=config.NUM_CALIBRATION_BINS, label=name,
                histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

    plt.show()
